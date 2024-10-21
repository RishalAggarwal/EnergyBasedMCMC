import torch
from .noise_schedules import BaseNoiseSchedule

class BaseBetaFunction():
    def __init__(self,noise_schedule:BaseNoiseSchedule):
        self.noise_schedule = noise_schedule
        

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ClampedAnnealedBeta(BaseBetaFunction):
    
    def __init__(self,noise_schedule:BaseNoiseSchedule,normalization_factor:float=None):
        super().__init__(noise_schedule)
        self.normalization_factor = normalization_factor

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        factor=self.normalization_factor if self.normalization_factor is not None else 1
        beta=1/(self.noise_schedule.g(t)*factor)
        beta = torch.clamp(beta,0,1)
        return beta