import torch

class DiffusionScheduler:
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.num_timesteps = num_timesteps
        self.beta_schedule = torch.linspace(beta_start, beta_end, num_timesteps)  # Linear schedule
        self.alpha = 1.0 - self.beta_schedule
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)  # Cumulative product of alpha

    def forward_process(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1, 1)  # Reshape for broadcasting
        x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise
        return x_t, noise

    def reverse_process(self, x_t, t, model_pred):
        beta_t = self.beta_schedule[t].view(-1, 1, 1, 1)
        alpha_t = self.alpha[t].view(-1, 1, 1, 1)
        alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1, 1)

        mean = (1 / torch.sqrt(alpha_t)) * (x_t - (beta_t / torch.sqrt(1 - alpha_bar_t)) * model_pred)
        std_dev = torch.sqrt(beta_t)
        noise = torch.randn_like(x_t)
        x_prev = mean + std_dev * noise
        return x_prev

    def sample(self, model, shape, device='cpu'):
        x_t = torch.randn(shape, device=device)
        for t in reversed(range(self.num_timesteps)):
            t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)
            model_pred = model(x_t, t_tensor)  # Predict noise using trained model
            x_t = self.reverse_process(x_t, t_tensor, model_pred)
        return x_t
