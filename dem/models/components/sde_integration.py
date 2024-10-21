from contextlib import contextmanager

import numpy as np
import torch

from dem.energies.base_energy_function import BaseEnergyFunction
from dem.models.components.sdes import VEReverseSDE
from dem.utils.data_utils import remove_mean


@contextmanager
def conditional_no_grad(condition):
    if condition:
        with torch.no_grad():
            yield
    else:
        yield


def grad_E(x, energy_function):
    with torch.enable_grad():
        x = x.requires_grad_()
        return torch.autograd.grad(torch.sum(energy_function(x)), x)[0].detach()


def negative_time_descent(x, energy_function, num_steps, dt=1e-4):
    samples = []
    for _ in range(num_steps):
        drift = grad_E(x, energy_function)
        x = x + drift * dt

        if energy_function.is_molecule:
            x = remove_mean(x, energy_function.n_particles, energy_function.n_spatial_dim)

        samples.append(x)
    return torch.stack(samples)

def ess(log_w):
    w = torch.exp(log_w)
    return torch.sum(w)**2 / torch.sum(w ** 2)


def euler_maruyama_step(
    sde: VEReverseSDE, t: torch.Tensor, x: torch.Tensor, dt: float, diffusion_scale: float=1.0): 
    drift = sde.f(t, x) * dt
    diffusion = diffusion_scale * sde.g(t, x) * np.sqrt(dt) * torch.randn_like(x)

    # Update the state    
    x_next = x + drift + diffusion
    return x_next, drift


def integrate_pfode(
    sde: VEReverseSDE,
    x0: torch.Tensor,
    num_integration_steps: int,
    reverse_time: bool = True,
):
    start_time = 1.0 if reverse_time else 0.0
    end_time = 1.0 - start_time

    times = torch.linspace(start_time, end_time, num_integration_steps + 1, device=x0.device)[:-1]

    x = x0
    samples = []
    with torch.no_grad():
        for t in times:
            x, f = euler_maruyama_step(sde, t, x, 1 / num_integration_steps)
            samples.append(x)

    return torch.stack(samples)


def integrate_sde(
    sde: VEReverseSDE,
    x0: torch.Tensor,
    num_integration_steps: int,
    energy_function: BaseEnergyFunction,
    reverse_time: bool = True,
    diffusion_scale=1.0,
    no_grad=True,
    time_range=1.0,
    negative_time=False,
    num_negative_time_steps=100,
    return_time=False,
    sequential=False,
    beta_function=None,
    intermediate_steps=1
):
    start_time = time_range if reverse_time else 0.0
    end_time = time_range - start_time

    times = torch.linspace(start_time, end_time, num_integration_steps + 1, device=x0.device)[:-1]

    x = x0
    samples = []

    with conditional_no_grad(no_grad):
        for t in times:
            beta_t=beta_function(t)
            if len(t.shape) >1:
                t = t.squeeze()
            if sequential:
                log_w = energy_function(x)*beta_t
                if ess(log_w)<len(x)*0.6:
                    # resample with replacement
                    x = x[torch.multinomial(torch.exp(log_w),len(x),replacement=True)] 
            for _ in range(intermediate_steps):
                x, f = euler_maruyama_step(
                    sde, t, x, time_range / num_integration_steps, diffusion_scale)
            if energy_function.is_molecule:
                x = remove_mean(x, energy_function.n_particles, energy_function.n_spatial_dim)
            samples.append(x)

    samples = torch.stack(samples)
    if negative_time:
        print("doing negative time descent...")
        samples_langevin = negative_time_descent(
            x, energy_function, num_steps=num_negative_time_steps
        )
        samples = torch.concatenate((samples, samples_langevin), axis=0)

    if return_time:
        return samples, times
    return samples
