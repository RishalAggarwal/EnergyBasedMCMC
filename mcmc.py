import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tqdm
from typing import Optional, Tuple, Union
import argparse
import sys
import copy
from dem.energies.gmm_energy import GMM
from dem.energies.lennardjones_energy import LennardJonesPotential, LennardJonesEnergy
from dem.energies.multi_double_well_energy import MultiDoubleWellEnergy
import wandb
from dem.models.components.distribution_distances import compute_distribution_distances
from dem.models.components.optimal_transport import wasserstein
from dem.utils.data_utils import remove_mean
from ot import emd2_1d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser=argparse.ArgumentParser(description="Run MCMC sampling on a given energy function")
    parser.add_argument("--energy", type=str, default="gmm", help="Energy function to sample from")
    parser.add_argument("--n_chains", type=int, default=1000, help="Number of chains to run")
    parser.add_argument("--n_steps", type=int, default=1000, help="Number of steps to run")
    parser.add_argument("--step_size", type=float, default=0.1, help="Step size for the MCMC sampler")
    parser.add_argument("--run_name", type=str, default="", help="Name of the run")
    parser.add_argument("--box_size", type=float, default=3.0, help="Size of the box to sample from")
    parser.add_argument("--max_step",type=float,default=1.25,help="Maximum step size for the MCMC sampler")
    parser.add_argument("--plot_frequency",type=int,default=100,help="how frequently to plot distributions")
    parser.add_argument("--save_endpoints",type=bool,default=False,help="Save the endpoints of the MCMC sampler")
    parser.add_argument("--endpoint_name",type=str,default="mcmc_long.npy",help="Name of the file to save the endpoints")
    parser.add_argument("--num_equilibriation_steps",type=int,default=1000,help="Number of equilibriation steps")
    parser.add_argument("--checkpoint",type=str,default=None,help="Path to the checkpoint to load")
    return parser.parse_args()

def initialize_energy(args):
    if args.energy == "gmm":
        energy = GMM(
        plotting_buffer_sample_size=1000,
        should_unnormalize=False,
        data_normalization_factor=1,
        train_set_size=1000,
        test_set_size=1000,
        val_set_size=1000,
        device=device)
    elif args.energy == "dw4":
        energy= MultiDoubleWellEnergy(
            dimensionality= 8,
            n_particles= 4,
            data_from_efm= True,
            data_path= "data/test_split_DW4.npy",
            data_path_train= "data/train_split_DW4.npy",
            data_path_val= "data/val_split_DW4.npy",
            device=device,
            data_normalization_factor= 1.0,
            is_molecule= True)
    elif args.energy == "lj13":
        energy = LennardJonesEnergy(
            dimensionality= 39,
            n_particles= 13,
            data_path= "data/test_split_LJ13-1000.npy",
            data_path_train= "data/train_split_LJ13-1000.npy",
            data_path_val= "data/test_split_LJ13-1000.npy",
            device=device,
            data_normalization_factor= 1.0,
            is_molecule= True)
    elif args.energy == "lj55":
        energy = LennardJonesEnergy(
            dimensionality= 165,
            n_particles= 55,
            data_path= "data/test_split_LJ55-1000-part1.npy",
            data_path_train= "data/train_split_LJ55-1000-part1.npy",
            data_path_val= "data/test_split_LJ55-1000-part1.npy",
            device=device,
            data_normalization_factor= 1.0,
            is_molecule= True)
    test_set = energy.setup_test_set()
    test_set=test_set[np.random.choice(test_set.shape[0],1000,replace=False)]
    return energy, test_set

def _calc_dist_total_var(
    gt_samples: torch.Tensor,
    latest_samples: torch.Tensor,
    system_shape: Union[torch.Size, Tuple[int]],
    energy_function    ) -> torch.Tensor:
    x = gt_samples.view(*system_shape)
    y = latest_samples.view(*system_shape) if latest_samples.device == "cpu" else latest_samples.detach().cpu().view(*system_shape)
    
    if x.size(-1) == 3:
        generated_samples_dists = (
            energy_function.interatomic_dist(x).cpu().numpy().reshape(-1),
        )
        data_set_dists = energy_function.interatomic_dist(y).cpu().numpy().reshape(-1)

        H_data_set, x_data_set = torch.histogram(data_set_dists, bins=200)
        H_generated_samples, _ = torch.histogram(generated_samples_dists, bins=(x_data_set))
        total_var = (
            0.5
            * torch.abs(
                H_data_set / H_data_set.sum() - H_generated_samples / H_generated_samples.sum()
            ).sum()
        )
    else:
        H_data_set_x, x_data_set = torch.histogram(x[:, 0], bins=200)
        H_data_set_y, _ = torch.histogram(x[:, 1], bins=(x_data_set))
        H_generated_samples_x, _ = torch.histogram(y[:, 0], bins=(x_data_set))
        H_generated_samples_y, _ = torch.histogram(y[:, 1], bins=(x_data_set))
        total_var = (
            0.5
            * (
                torch.abs(
                    H_data_set_x / H_data_set_x.sum() - H_generated_samples_x / H_generated_samples_x.sum()
                ) +
                torch.abs(
                    H_data_set_y / H_data_set_y.sum() - H_generated_samples_y / H_generated_samples_y.sum()
                )
            ).sum()
        )
    return total_var

class scaled_energy_function:
    def __init__(self,energy,beta,oscillator_scale=1):
        self.energy=copy.deepcopy(energy)
        if type(energy)==LennardJonesEnergy:
            self.energy.lennard_jones._oscillator_scale=oscillator_scale
        self.beta=beta
    def __getattr__(self, name):
        return getattr(self.energy, name)
    def __call__(self,x):
        return self.beta*self.energy(x)
    
def get_hist_metrics(samples,test,prefix="",range=(0,7)):
    metrics={}
    samples_hist,_=np.histogram(samples.view(-1),bins=100,density=True,range=range)
    test_hist,_=np.histogram(test.view(-1),bins=100,density=True,range=range)
    #w2 distance
    w2=emd2_1d(samples.view(-1),test.view(-1),metric="euclidean")
    #l2 distance
    l2_distance = np.linalg.norm(samples_hist - test_hist)
    metrics[prefix+"L2 Distance"]=l2_distance
    metrics[prefix+"W2 Distance"]=w2
    return metrics
    

def compute_metrics(energy,initial_samples,test_set,plot_figs=True):
    log_metrics={}
    current_samples_idx=torch.multinomial(torch.ones(initial_samples.size()[0]),test_set.size()[0], replacement=False)
    current_samples=initial_samples[current_samples_idx]
    if energy.is_molecule:
        current_samples=remove_mean(current_samples,energy.n_particles,energy.n_spatial_dim)
        test_set=remove_mean(test_set,energy.n_particles,energy.n_spatial_dim)
        dist_samples = energy.interatomic_dist(current_samples).detach().cpu()
        dist_test = energy.interatomic_dist(test_set).detach().cpu()
        
        dist_metrics=get_hist_metrics(dist_samples,dist_test,prefix="Interatomic Distance ",range=(0,7))
        if energy.n_particles == 13:
            min_energy = -60
            max_energy = 0
        elif energy.n_particles == 55:
            min_energy = -380
            max_energy = -180
        else:
            min_energy = -26
            max_energy = 0
        energy_samples = -energy(current_samples).detach().cpu()
        energy_test = -energy(test_set).detach().cpu()
        energy_metrics=get_hist_metrics(energy_samples,energy_test,prefix="Energy ",range=(min_energy,max_energy))
        if plot_figs:
            fig, axs = plt.subplots(1, 2, figsize=(12, 4))
            axs[0].hist(
            dist_samples.view(-1),
            bins=100,
            alpha=0.5,
            density=True,
            histtype="step",
            linewidth=4,
            )
            axs[0].hist(
            dist_test.view(-1),
            bins=100,
            alpha=0.5,
            density=True,
            histtype="step",
            linewidth=4,
            )
            axs[0].set_xlabel("Interatomic distance")
            axs[0].legend(["generated data", "test data"])
            axs[1].hist(
                energy_test.cpu(),
                bins=100,
                density=True,
                alpha=0.4,
                range=(min_energy, max_energy),
                color="g",
                histtype="step",
                linewidth=4,
                label="test data",
            )
            axs[1].hist(
                energy_samples.cpu(),
                bins=100,
                density=True,
                alpha=0.4,
                range=(min_energy, max_energy),
                color="r",
                histtype="step",
                linewidth=4,
                label="generated data",
            )
            axs[1].set_xlabel("Energy")
            axs[1].legend()
            log_metrics["Ensemble Observations"]=wandb.Image(fig)
        log_metrics.update(dist_metrics)
        log_metrics.update(energy_metrics)
    current_samples=current_samples.cpu()
    test_set=test_set.cpu()
    w2=wasserstein(test_set,current_samples,power=2)
    tv=_calc_dist_total_var(test_set,current_samples,current_samples.shape,energy)
    log_metrics["W2"]=w2
    log_metrics["Total Variation"]=tv
    wandb.log(log_metrics)

def mc_step(energy,initial_samples,initital_energies,step_size):
    new_samples=initial_samples+torch.randn_like(initial_samples,device=device)*step_size
    new_energies=energy(new_samples)
    acceptances=torch.exp(new_energies-initital_energies)>=torch.rand_like(new_energies,device=device)
    initial_samples[acceptances]=new_samples[acceptances]
    initital_energies[acceptances]=new_energies[acceptances]
    return initial_samples,initital_energies

def ess(log_w):
    w = torch.exp(log_w)
    return torch.sum(w)**2 / torch.sum(w ** 2)

def reweighted_samples(energy_function, x):
    log_weights=energy_function(x)
    log_weights=log_weights-torch.logsumexp(log_weights,dim=0)
    weights=torch.exp(log_weights)
    x_idx=torch.multinomial(weights,x.shape[0],replacement=True)
    x=x[x_idx]
    return x

def equilibriation(args,energy_function,initial_samples,test_set,step_size,max_step):
    betas=[0,0.2,0.4,0.6,0.8,1.0]
    if energy_function.n_particles==55:
        betas=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    for i in range(1,len(betas)):
        energy_function_reweighting=scaled_energy_function(energy_function,betas[1],0)
        energy_function_current=scaled_energy_function(energy_function,betas[i],1/betas[i])
        initial_samples=reweighted_samples(energy_function_reweighting,initial_samples)
        initial_energies=energy_function_current(initial_samples)
        for j in tqdm.tqdm(range(args.num_equilibriation_steps)):
            initial_samples,initial_energies=mc_step(energy_function_current,initial_samples,initial_energies,min(step_size/betas[i],max_step))
            if j%100==0:
                plot_figs=True if j%args.plot_frequency==0 else False
                compute_metrics(energy_function,initial_samples,test_set,plot_figs)
    return initial_samples

def run_mcmc(args,energy,num_steps,step_size, test_set):
    if args.checkpoint is None:
        if args.energy == "gmm":
            initial_samples = (torch.rand((args.n_chains,)+test_set.size()[1:],device=device)*2-1)*45
            initial_samples=reweighted_samples(energy,initial_samples)
        else:
            initial_samples = (torch.rand((args.n_chains,)+test_set.size()[1:],device=device)*2-1)*args.box_size
            if args.energy == "lj13" or args.energy == "lj55":
                initial_samples = (torch.randn((args.n_chains,)+test_set.size()[1:],device=device))
            initial_samples=remove_mean(initial_samples,energy.n_particles,energy.n_spatial_dim)
            test_set=remove_mean(test_set,energy.n_particles,energy.n_spatial_dim)
            initial_samples=equilibriation(args,energy,initial_samples,test_set,step_size,args.max_step)
            #energy=scaled_energy_function(energy,1,0)
            #initial_samples=reweighted_samples(energy,initial_samples)
            num_steps=num_steps-(5*args.num_equilibriation_steps)
            if energy.n_particles==55:
                num_steps=num_steps-(5*args.num_equilibriation_steps)
    else:
        initial_samples=np.load(args.checkpoint)
        initial_samples=torch.tensor(initial_samples,device=device)
    #num_steps=num_steps-10000 if energy.is_molecule else num_steps
    energy=scaled_energy_function(energy,1,1)
    initital_energies=energy(initial_samples)
    for i in tqdm.tqdm(range(num_steps)):
        initial_samples,initital_energies=mc_step(energy,initial_samples,initital_energies,step_size)
        if i%100==0:
            plot_figs=True if i%args.plot_frequency==0 else False
            compute_metrics(energy,initial_samples,test_set,plot_figs)
    return initial_samples
def main():
    args=parse_args()
    energy,test_set=initialize_energy(args)
    wandb.init(project="byob-mcmc",name=args.run_name)
    x=run_mcmc(args,energy,args.n_steps,args.step_size,test_set)
    if args.save_endpoints:
        x=x.detach().cpu().numpy()
        np.save(args.endpoint_name,x)


if __name__=="__main__":
    main()

            
            
    
