import torch

class NoiseScheduler:
    def __init__(self, timesteps=1000):
        self.timesteps = timesteps
        self.betas = torch.linspace(1e-4, 0.02, timesteps)
        self.alphas = 1. - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)

    def get_noisy_image(self, x, t):
        # Ensure scheduler tensors are on the same device as x and t
        alpha = self.alpha_cumprod.to(x.device)
        
        sqrt_alpha = alpha[t]**0.5
        sqrt_one_minus_alpha = (1 - alpha[t])**0.5

        noise = torch.randn_like(x)
        x_t = sqrt_alpha[:, None, None, None] * x + sqrt_one_minus_alpha[:, None, None, None] * noise
        return x_t, noise

