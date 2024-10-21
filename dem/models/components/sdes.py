import torch


class SDE(torch.nn.Module):
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, drift, diffusion):
        super().__init__()
        self.drift = drift
        self.diffusion = diffusion

    def f(self, t, x):
        if t.dim() == 0:
            # repeat the same time for all points if we have a scalar time
            t = t * torch.ones(x.shape[0]).to(x.device)

        return self.drift(t, x)

    def g(self, t, x):
        return self.diffusion(t, x)


class VEReverseSDE(torch.nn.Module):
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, score, noise_schedule):
        super().__init__()
        self.score = score
        self.noise_schedule = noise_schedule

    def f(self, t, x):
        if t.dim() == 0:
            # repeat the same time for all points if we have a scalar time
            t = t * torch.ones(x.shape[0]).to(x.device)

        score = self.score(t, x)
        return self.g(t, x).pow(2) * score

    def g(self, t, x):
        g = self.noise_schedule.g(t)
        return g.unsqueeze(1) if g.ndim > 0 else torch.full_like(x, g)
    
class ScoreBiasedVEReverseSDE(torch.nn.Module):
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, score, noise_schedule,energy_function,beta_function,grad_fxn,true_score_only=False):
        super().__init__()
        self.score = score
        self.noise_schedule = noise_schedule
        self.energy_function = energy_function
        self.beta_function = beta_function
        self.grad_fxn = grad_fxn
        self.true_score_only=true_score_only

    def f(self, t, x):
        if t.dim() == 0:
            # repeat the same time for all points if we have a scalar time
            t = t * torch.ones(x.shape[0]).to(x.device)
        beta_t=self.beta_function(t)
        with torch.set_grad_enabled(True):
            x.requires_grad_(True)
            score_true = self.grad_fxn(t,x,self.energy_function,self.noise_schedule,100)
        score_true = score_true*beta_t.unsqueeze(1)/2
        score_true = self.energy_function.normalize(score_true)
        x=self.energy_function.normalize(x)
        if self.true_score_only:
            score = score_true
        else:
            score = self.score(t, x)
            score = score + score_true
        return self.g(t, x).pow(2) * score

    def g(self, t, x):
        g = self.noise_schedule.g(t)
        return g.unsqueeze(1) if g.ndim > 0 else torch.full_like(x, g)


class RegVEReverseSDE(VEReverseSDE):
    def f(self, t, x):
        dx = super().f(t, x[..., :-1])
        quad_reg = 0.5 * dx.pow(2).sum(dim=-1, keepdim=True)
        return torch.cat([dx, quad_reg], dim=-1)

    def g(self, t, x):
        g = self.noise_schedule.g(t)
        if g.ndim > 0:
            return g.unsqueeze(1)
        return torch.cat([torch.full_like(x[..., :-1], g), torch.zeros_like(x[..., -1:])], dim=-1)
