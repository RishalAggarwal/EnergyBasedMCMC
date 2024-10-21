import time
from typing import Any, Dict, Optional

import hydra
import matplotlib.pyplot as plt
import numpy as np
import ot as pot
import torch
from hydra.utils import get_original_cwd
from lightning import LightningModule
from lightning.pytorch.loggers import WandbLogger
from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
)
from torchmetrics import MeanMetric

from dem.energies.base_energy_function import BaseEnergyFunction
from dem.utils.data_utils import remove_mean
from dem.utils.logging_utils import fig_to_image

from .components.clipper import Clipper
from .components.cnf import CNF
from .components.distribution_distances import compute_distribution_distances
from .components.ema import EMAWrapper
from .components.lambda_weighter import BaseLambdaWeighter
from .components.mlp import TimeConder
from .components.noise_schedules import BaseNoiseSchedule
from .components.replay_buffer_timed import ReplayBufferTimed
from .components.scaling_wrapper import ScalingWrapper
from .components.score_estimator import estimate_grad_Rt, wrap_for_richardsons,simple_score_estimator
from .components.score_scaler import BaseScoreScaler
from .components.sde_integration import integrate_sde
from .components.sdes import ScoreBiasedVEReverseSDE
from .dem_module import get_wandb_logger,t_stratified_loss
from .components.sde_integration import euler_maruyama_step
from .components.beta import BaseBetaFunction
from typing import List, Tuple
from torch.utils.data import DataLoader

class BYOBLitModule(LightningModule):
    """Example of a `LightningModule` for MNIST classification.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        energy_function: BaseEnergyFunction,
        noise_schedule: BaseNoiseSchedule,
        beta_function: BaseBetaFunction,
        lambda_weighter: BaseLambdaWeighter,
        buffer: ReplayBufferTimed,
        num_init_samples: int,
        num_estimator_mc_samples: int,
        num_samples_to_generate_per_epoch: int,
        num_samples_to_sample_from_buffer: int,
        num_samples_to_save: int,
        eval_batch_size: int,
        num_integration_steps: int,
        lr_scheduler_update_frequency: int,
        nll_with_cfm: bool,
        nll_with_dem: bool,
        nll_on_buffer: bool,
        logz_with_cfm: bool,
        cfm_sigma: float,
        cfm_prior_std: float,
        use_otcfm: bool,
        nll_integration_method: str,
        use_richardsons: bool,
        compile: bool,
        prioritize_cfm_training_samples: bool = False,
        input_scaling_factor: Optional[float] = None,
        output_scaling_factor: Optional[float] = None,
        clipper: Optional[Clipper] = None,
        score_scaler: Optional[BaseScoreScaler] = None,
        partial_prior=None,
        clipper_gen: Optional[Clipper] = None,
        diffusion_scale=1.0,
        cfm_loss_weight=1.0,
        use_ema=False,
        use_exact_likelihood=False,
        debug_use_train_data=False,
        init_from_prior=False,
        compute_nll_on_train_data=False,
        use_buffer=True,
        tol=1e-5,
        version=1,
        negative_time=False,
        num_negative_time_steps=100,
        num_samples_to_generate_for_eval=1024,
        true_score_only=False,
        intermediate_steps=1,
    ) -> None:
        
        """Initialize a `MNISTLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        :param buffer: Buffer of sampled objects
        """
        super().__init__()
        # Seems to slow things down
        # torch.set_float32_matmul_precision('high')

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net(energy_function=energy_function)
        self.cfm_net = net(energy_function=energy_function)

        if use_ema:
            self.net = EMAWrapper(self.net)
            self.cfm_net = EMAWrapper(self.cfm_net)
        if input_scaling_factor is not None or output_scaling_factor is not None:
            self.net = ScalingWrapper(self.net, input_scaling_factor, output_scaling_factor)

            self.cfm_net = ScalingWrapper(
                self.cfm_net, input_scaling_factor, output_scaling_factor
            )
        self.score_scaler = None
        if score_scaler is not None:
            self.score_scaler = self.hparams.score_scaler(noise_schedule)

            self.net = self.score_scaler.wrap_model_for_unscaling(self.net)
            self.cfm_net = self.score_scaler.wrap_model_for_unscaling(self.cfm_net)

        self.energy_function = energy_function
        self.noise_schedule = noise_schedule
        self.buffer = buffer
        self.dim = self.energy_function.dimensionality

        
        grad_fxn = simple_score_estimator
        self.clipper = clipper
        self.clipped_grad_fxn = self.clipper.wrap_grad_fxn(grad_fxn)
        self.beta_function = beta_function(self.noise_schedule)
        self.true_score_only = true_score_only
        self.reverse_sde = ScoreBiasedVEReverseSDE(self.net, self.noise_schedule,self.energy_function,self.beta_function,self.clipped_grad_fxn,self.true_score_only)
        
        self.byob_train_loss = MeanMetric()
        self.cfm_train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.val_nll_logdetjac = MeanMetric()
        self.test_nll_logdetjac = MeanMetric()
        self.val_nll_log_p_1 = MeanMetric()
        self.test_nll_log_p_1 = MeanMetric()
        self.val_nll = MeanMetric()
        self.test_nll = MeanMetric()
        self.val_nfe = MeanMetric()
        self.test_nfe = MeanMetric()
        self.val_energy_w2 = MeanMetric()
        self.val_dist_w2 = MeanMetric()
        self.val_dist_total_var = MeanMetric()

        self.val_dem_nll_logdetjac = MeanMetric()
        self.test_dem_nll_logdetjac = MeanMetric()
        self.val_dem_nll_log_p_1 = MeanMetric()
        self.test_dem_nll_log_p_1 = MeanMetric()
        self.val_dem_nll = MeanMetric()
        self.test_dem_nll = MeanMetric()
        self.val_dem_nfe = MeanMetric()
        self.test_dem_nfe = MeanMetric()
        self.val_dem_logz = MeanMetric()
        self.val_logz = MeanMetric()
        self.test_dem_logz = MeanMetric()
        self.test_logz = MeanMetric()

        self.val_buffer_nll_logdetjac = MeanMetric()
        self.val_buffer_nll_log_p_1 = MeanMetric()
        self.val_buffer_nll = MeanMetric()
        self.val_buffer_nfe = MeanMetric()
        self.val_buffer_logz = MeanMetric()
        self.test_buffer_nll_logdetjac = MeanMetric()
        self.test_buffer_nll_log_p_1 = MeanMetric()
        self.test_buffer_nll = MeanMetric()
        self.test_buffer_nfe = MeanMetric()
        self.test_buffer_logz = MeanMetric()

        self.val_train_nll_logdetjac = MeanMetric()
        self.val_train_nll_log_p_1 = MeanMetric()
        self.val_train_nll = MeanMetric()
        self.val_train_nfe = MeanMetric()
        self.val_train_logz = MeanMetric()
        self.test_train_nll_logdetjac = MeanMetric()
        self.test_train_nll_log_p_1 = MeanMetric()
        self.test_train_nll = MeanMetric()
        self.test_train_nfe = MeanMetric()
        self.test_train_logz = MeanMetric()

        self.num_init_samples = num_init_samples

        self.num_samples_to_generate_per_epoch = num_samples_to_generate_per_epoch
        self.num_samples_to_sample_from_buffer = num_samples_to_sample_from_buffer
        self.num_integration_steps = num_integration_steps
        self.num_samples_to_save = num_samples_to_save
        self.eval_batch_size = eval_batch_size

        self.lambda_weighter = self.hparams.lambda_weighter(self.noise_schedule)

        self.last_samples = None
        self.last_energies = None
        self.eval_step_outputs = []

        self.partial_prior = partial_prior

        self.clipper_gen = clipper_gen

        self.diffusion_scale = diffusion_scale
        self.init_from_prior = init_from_prior
        self.num_samples_to_generate_for_eval = num_samples_to_generate_for_eval
        self.intermediate_steps = intermediate_steps
        

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(t, x)
    
    def training_step(self, batch, batch_idx):
        if self.true_score_only:
            return None
        loss = 0.0
        iter_samples,_,times,_= self.buffer.sample(self.num_samples_to_sample_from_buffer)
        byob_loss = self.get_loss(times,iter_samples)
        self.log_dict(
                t_stratified_loss(times, byob_loss, loss_name="train/stratified/byob_loss")
            )
        byob_loss=byob_loss.mean()
        loss = loss + byob_loss

        # update and log metrics
        self.byob_train_loss(byob_loss)
        self.log(
            "train/byob_loss",
            self.byob_train_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return loss
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        optimizer.step(closure=optimizer_closure)
        if self.hparams.use_ema:
            self.net.update_ema()
            if self.should_train_cfm(batch_idx):
                self.cfm_net.update_ema()

    def get_loss(self, times, samples):
        
        samples_proposed,forward_drift =euler_maruyama_step(self.reverse_sde, times, samples, 1 / self.num_integration_steps)
        samples_backward,backward_drift =euler_maruyama_step(self.reverse_sde, times, samples_proposed, 1 / self.num_integration_steps)
        weight_t=(1/(self.reverse_sde.g(times,samples_proposed)**2 * 1 / self.num_integration_steps)).squeeze(-1)
        fwd_diff  = samples_proposed - (samples + forward_drift)
        fwd_log_prob = - 0.5 * (fwd_diff * fwd_diff).sum(dim=-1)
        fwd_log_prob=fwd_log_prob.unsqueeze(-1)
        fwd_log_prob = fwd_log_prob * weight_t
        bwd_diff = samples - (samples_proposed + backward_drift)
        bwd_log_prob=- 0.5 * (bwd_diff * bwd_diff).sum(dim=-1)
        bwd_log_prob=bwd_log_prob.unsqueeze(-1)
        bwd_log_prob = bwd_log_prob * weight_t
        U_proposed=self.energy_function(samples_proposed).unsqueeze(-1)
        U_current=self.energy_function(samples).unsqueeze(-1)
        beta_t=self.beta_function(times)
        loss = -(beta_t*(U_proposed-U_current)+bwd_log_prob-fwd_log_prob)
        return (loss**2).squeeze()

    def generate_samples(
        self,
        reverse_sde: ScoreBiasedVEReverseSDE = None,
        num_samples: Optional[int] = None,
        return_full_trajectory: bool = False,
        diffusion_scale=1.0,
        negative_time=False,
        return_time=True
    ) -> torch.Tensor:
        num_samples = num_samples or self.num_samples_to_generate_per_epoch

        samples = self.prior.sample(num_samples)

        return self.integrate(
            reverse_sde=reverse_sde,
            samples=samples,
            reverse_time=True,
            return_full_trajectory=return_full_trajectory,
            diffusion_scale=diffusion_scale,
            negative_time=negative_time,
            return_time=return_time,
        )
    
    def integrate(
        self,
        reverse_sde: ScoreBiasedVEReverseSDE = None,
        samples: torch.Tensor = None,
        reverse_time=True,
        return_full_trajectory=False,
        diffusion_scale=1.0,
        no_grad=True,
        negative_time=False,
        return_time=True
    ) -> torch.Tensor:
        trajectory = integrate_sde(
            reverse_sde or self.reverse_sde,
            samples,
            self.num_integration_steps,
            self.energy_function,
            diffusion_scale=diffusion_scale,
            reverse_time=reverse_time,
            no_grad=no_grad,
            negative_time=negative_time,
            num_negative_time_steps=self.hparams.num_negative_time_steps,
            return_time=return_time,
            sequential=True,
            beta_function=self.beta_function,
            intermediate_steps=self.intermediate_steps
        )
        if return_full_trajectory:
            return trajectory

        return trajectory[-1]

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        if self.clipper_gen is not None:
            reverse_sde = ScoreBiasedVEReverseSDE(
                self.clipper_gen.wrap_grad_fxn(self.net), self.noise_schedule,self.energy_function,self.beta_function,self.clipped_grad_fxn
            )
            samples,times = self.generate_samples(
                reverse_sde=reverse_sde, diffusion_scale=self.diffusion_scale,return_full_trajectory=True,return_time=True
            )
        else:
            samples,times= self.generate_samples(diffusion_scale=self.diffusion_scale,return_full_trajectory=True,return_time=True)

        energies=[]
        for i in range(samples.shape[1]):
            energies.append(self.energy_function(samples[:,i,:]))
        energies=torch.stack(energies)
        self.buffer.add(samples,energies,times)
        self.last_samples,self.last_energies=self.buffer.get_last_n_inserted(self.num_samples_to_generate_per_epoch)

        self._log_energy_w2(prefix="val")

        if self.energy_function.is_molecule:
            self._log_dist_w2(prefix="val")
            self._log_dist_total_var(prefix="val")
        

    def _log_energy_w2(self, prefix="val"):
        if prefix == "test":
            data_set = self.energy_function.sample_val_set(self.eval_batch_size)
            generated_samples = self.generate_samples(
                num_samples=self.eval_batch_size, diffusion_scale=self.diffusion_scale
            )
            generated_energies = self.energy_function(generated_samples)
        else:
            if len(self.buffer) < self.eval_batch_size:
                return
            data_set = self.energy_function.sample_test_set(self.eval_batch_size)
            _, generated_energies = self.buffer.get_last_n_inserted(self.eval_batch_size)

        energies = self.energy_function(self.energy_function.normalize(data_set))
        energy_w2 = pot.emd2_1d(energies.cpu().numpy(), generated_energies.cpu().numpy())

        self.log(
            f"{prefix}/energy_w2",
            self.val_energy_w2(energy_w2),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def _log_dist_w2(self, prefix="val"):
        if prefix == "test":
            data_set = self.energy_function.sample_val_set(self.eval_batch_size)
            generated_samples = self.generate_samples(
                num_samples=self.eval_batch_size, diffusion_scale=self.diffusion_scale
            )
        else:
            if len(self.buffer) < self.eval_batch_size:
                return
            data_set = self.energy_function.sample_test_set(self.eval_batch_size)
            generated_samples, _ = self.buffer.get_last_n_inserted(self.eval_batch_size)

        dist_w2 = pot.emd2_1d(
            self.energy_function.interatomic_dist(generated_samples).cpu().numpy().reshape(-1),
            self.energy_function.interatomic_dist(data_set).cpu().numpy().reshape(-1),
        )
        self.log(
            f"{prefix}/dist_w2",
            self.val_dist_w2(dist_w2),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def _log_dist_total_var(self, prefix="val"):
        if prefix == "test":
            data_set = self.energy_function.sample_val_set(self.eval_batch_size)
            generated_samples = self.generate_samples(
                num_samples=self.eval_batch_size, diffusion_scale=self.diffusion_scale
            )
        else:
            if len(self.buffer) < self.eval_batch_size:
                return
            data_set = self.energy_function.sample_test_set(self.eval_batch_size)
            generated_samples, _ = self.buffer.get_last_n_inserted(self.eval_batch_size)

        generated_samples_dists = (
            self.energy_function.interatomic_dist(generated_samples).cpu().numpy().reshape(-1),
        )
        data_set_dists = self.energy_function.interatomic_dist(data_set).cpu().numpy().reshape(-1)

        H_data_set, x_data_set = np.histogram(data_set_dists, bins=200)
        H_generated_samples, _ = np.histogram(generated_samples_dists, bins=(x_data_set))
        total_var = (
            0.5
            * np.abs(
                H_data_set / H_data_set.sum() - H_generated_samples / H_generated_samples.sum()
            ).sum()
        )

        self.log(
            f"{prefix}/dist_total_var",
            self.val_dist_total_var(total_var),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
    
    def eval_step(self, prefix: str, batch: Optional[List[torch.Tensor]], batch_idx: int) -> None:
        # Determine number of samples to generate
        if prefix == "val":
            n_samples = self.num_samples_to_generate_per_epoch
        elif prefix == "test":
            n_samples = self.num_samples_to_generate_for_eval
        else:
            raise ValueError(f"Invalid prefix: {prefix}")
        
        # Generate samples
        if self.num_samples_to_generate_per_epoch >= n_samples:
            random_indices = torch.randperm(n_samples)
            samples = self.last_samples[random_indices]
            energies = self.last_energies[random_indices]
        else:
            samples = self.generate_samples(
                num_samples=n_samples, diffusion_scale=self.diffusion_scale
            )
            energies = self.energy_function(samples)
        
        
        
        # Get ground truth samples
        gt_samples = self.energy_function.sample_val_set(num_points=n_samples            
        )
        to_log = {"data_0": gt_samples,
            "gen_0": samples,
        }


        '''names, dists = compute_distribution_distances(
            pred=samples[:, None],
            true=gt_samples[:, None],
            energy_function=self.energy_function,
        )
        names = [f"{prefix}/{name}" for name in names]
        d = dict(zip(names, dists))
        self.log_dict(d, sync_dist=True)'''
        #   > from energy class
        self.eval_step_outputs.append(to_log)

    def validation_step(self, batch: List[torch.Tensor], batch_idx: int) -> None:
        if not self.trainer.sanity_checking:
            self.eval_step("val", batch, batch_idx)

    def test_step(self, batch: List[torch.Tensor], batch_idx: int) -> None:
        if not self.trainer.sanity_checking:
            self.eval_step("test", batch, batch_idx)

    '''def train_dataloader(self):
        return self.dataset.train_dataloader(**self.hparams.loader)'''

    def val_dataloader(self):
        # Dummy validation dataset
        loader_kwargs = {k: v for k, v in self.hparams.loader.items() if k != "batch_size"}
        return DataLoader([(torch.empty(1), torch.empty(1))], batch_size=1, **loader_kwargs)
    
    def test_dataloader(self):
        # Dummy test dataset
        loader_kwargs = {k: v for k, v in self.hparams.loader.items() if k != "batch_size"}
        return DataLoader([(torch.empty(1), torch.empty(1))], batch_size=1, **loader_kwargs)
    
    def eval_epoch_end(self, prefix: str):
        wandb_logger = get_wandb_logger(self.loggers)
        outputs = {
            k: torch.cat([dic[k] for dic in self.eval_step_outputs], dim=0)
            for k in self.eval_step_outputs[0]
        }
        self.energy_function.log_on_epoch_end(
                self.last_samples,
                self.last_energies,
                wandb_logger,
            )
        if "data_0" in outputs:
            # pad with time dimension 1
            names, dists = compute_distribution_distances(
                self.energy_function.unnormalize(outputs["gen_0"])[:, None],
                outputs["data_0"][:, None],
                self.energy_function,
            )
            names = [f"{prefix}/{name}" for name in names]
            d = dict(zip(names, dists))
            self.log_dict(d, sync_dist=True)

        self.eval_step_outputs.clear()

    def on_validation_epoch_end(self) -> None:
        self.eval_epoch_end("val")

    def on_test_epoch_end(self) -> None:
        wandb_logger = get_wandb_logger(self.loggers)

        self.eval_epoch_end("test")
        batch_size = 1000
        final_samples = []
        n_batches = self.num_samples_to_save // batch_size
        print("Generating samples")
        for i in range(n_batches):
            start = time.time()
            samples = self.generate_samples(
                num_samples=batch_size,
                diffusion_scale=self.diffusion_scale,
                negative_time=self.hparams.negative_time,
            )
            final_samples.append(samples)
            end = time.time()
            print(f"batch {i} took {end - start:0.2f}s")

            if i == 0:
                self.energy_function.log_on_epoch_end(
                    samples,
                    self.energy_function(samples),
                    wandb_logger,
                )

        final_samples = torch.cat(final_samples, dim=0)
        output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        path = f"{output_dir}/samples_{self.num_samples_to_save}.pt"
        torch.save(final_samples, path)
        print(f"Saving samples to {path}")
        import os

        os.makedirs(self.energy_function.name, exist_ok=True)
        path2 = f"{self.energy_function.name}/samples_{self.hparams.version}_{self.num_samples_to_save}.pt"
        torch.save(final_samples, path2)
        print(f"Saving samples to {path2}")



    def setup(self, stage: Optional[str] = None) -> None:

        self.net = self.net.to(self.device)
        reverse_sde = ScoreBiasedVEReverseSDE(self.net, self.noise_schedule,self.energy_function,self.beta_function,self.clipped_grad_fxn,self.true_score_only)
        self.prior = self.partial_prior(device=self.device)
        init_states,init_times = self.generate_samples(
                reverse_sde, self.num_init_samples, diffusion_scale=self.diffusion_scale,return_full_trajectory=True,return_time=True
            )
        init_energies=[]
        for i in range(init_states.shape[1]):
            init_energies.append(self.energy_function(init_states[:,i,:]))
        init_energies=torch.stack(init_energies)
        self.buffer.add(init_states, init_energies, init_times)
        self.last_samples,self.last_energies=self.buffer.get_last_n_inserted(self.num_samples_to_generate_per_epoch)
        if self.hparams.compile and stage=='fit':
            self.net = torch.compile(self.net)


    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": self.hparams.lr_scheduler_update_frequency,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = BYOBLitModule(
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )
