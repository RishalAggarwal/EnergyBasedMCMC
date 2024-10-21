import math
from typing import Dict,List,Union
from omegaconf import ListConfig

import torch

from torch.distributions import constraints


class Prior:
    def __init__(self, dim, scale, device="cpu"):
        self.dim = dim
        self.scale = scale
        self.dist = torch.distributions.MultivariateNormal(
            torch.zeros(dim, device=device),
            torch.eye(dim, device=device) * (scale**2),
        )

    def log_prob(self, x):
        return self.dist.log_prob(x)

    def sample(self, n_samples):
        return self.dist.sample((n_samples,))


class MeanFreePrior(torch.distributions.Distribution):
    arg_constraints: Dict[str, constraints.Constraint] = {}

    def __init__(self, n_particles, spatial_dim, scale, device="cpu"):
        super().__init__()
        self.n_particles = n_particles
        self.spatial_dim = spatial_dim
        self.dim = n_particles * spatial_dim
        self.scale = scale
        self.device = device

    def log_prob(self, x):
        x = x.reshape(-1, self.n_particles, self.spatial_dim)
        N, D = x.shape[-2:]

        # r is invariant to a basis change in the relevant hyperplane.
        r2 = torch.sum(x**2, dim=(-1, -2)) / self.scale**2

        # The relevant hyperplane is (N-1) * D dimensional.
        degrees_of_freedom = (N - 1) * D

        # Normalizing constant and logpx are computed:
        log_normalizing_constant = (
            -0.5 * degrees_of_freedom * math.log(2 * torch.pi * self.scale**2)
        )
        log_px = -0.5 * r2 + log_normalizing_constant
        return log_px

    def sample(self, n_samples):
        samples = torch.randn(n_samples, self.dim, device=self.device) * self.scale
        samples = samples.reshape(-1, self.n_particles, self.spatial_dim)
        samples = samples - samples.mean(-2, keepdims=True)
        return samples.reshape(-1, self.n_particles * self.spatial_dim)

class Uniform(Prior):
    
    def __init__(self, low: Union[float, List[float]], high: Union[float, List[float]],dim,n_particles,device='cpu'):
        super().__init__(dim=dim,scale=high-low,device=device)
        self.dim = dim
        self.n_particles = n_particles
        low_is_list = isinstance(low, list) or isinstance(low, ListConfig)
        high_is_list = isinstance(high, list) or isinstance(high, ListConfig)
        self.low = torch.tensor(low) if low_is_list else torch.tensor([low] * self.n_particles*self.dim)
        self.high = torch.tensor(high) if high_is_list else torch.tensor([high] *self.n_particles* self.dim)
        assert self.low.shape == self.high.shape == (self.dim,)
        assert torch.all(self.low < self.high)
        self.low = self.low.to(device)
        self.high = self.high.to(device)
        self.dist = torch.distributions.Uniform(self.low, self.high)
        self.device = device

    def sample(self, n_samples: int, device: str = "cpu"):
        samples = self.dist.sample((n_samples,))
        samples = samples.reshape(-1,self.n_particles* self.dim)
        if self.n_particles > 1:
            samples = samples.reshape(-1, self.n_particles, self.dim)         
        return samples

    def log_prob(self, x: torch.Tensor):
        # TODO: fix control flow for batched inputs
        samples = x.reshape(-1, self.n_particles* self.dim)
        log_probs = self.dist.log_prob(samples)
        return log_probs