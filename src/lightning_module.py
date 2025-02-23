import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from models.unet import UNetDenoiser as UNET

class DiffusionScheduler:
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02):
        """
        Initialize the diffusion scheduler with linear beta schedule
        """
        self.num_timesteps = num_timesteps
        self.register_buffers(beta_start, beta_end)

    def register_buffers(self, beta_start, beta_end):
        """
        Pre-calculate diffusion parameters
        """
        # Define beta schedule
        self.betas = torch.linspace(beta_start, beta_end, self.num_timesteps)
        
        # Calculate alphas
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculate diffusion parameters
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1. / self.alphas)
        
        # Calculate posterior variance
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def to(self, device):
        """
        Move all tensors to specified device
        """
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.sqrt_recip_alphas = self.sqrt_recip_alphas.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        return self

    def forward_process(self, x_0, t, noise=None):
        """
        Forward diffusion process: q(x_t | x_0)
        """
        if noise is None:
            noise = torch.randn_like(x_0)
            
        # Extract the proper timestep parameters
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        
        # Compute noisy image
        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        return x_t, noise

    def reverse_process(self, model_output, x_t, t, use_clipped_model_output=True):
        """
        Reverse diffusion process: p(x_{t-1} | x_t)
        """
        # Extract parameters for current timestep
        beta_t = extract(self.betas, t, x_t.shape)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        
        # Predict x_0
        if use_clipped_model_output:
            pred_x_0 = sqrt_recip_alphas_t * (x_t - sqrt_one_minus_alphas_cumprod_t * model_output)
            pred_x_0 = torch.clamp(pred_x_0, -1., 1.)
        else:
            pred_x_0 = model_output

        # Calculate mean for posterior q(x_{t-1} | x_t, x_0)
        posterior_mean = (
            extract(self.alphas_cumprod_prev, t, x_t.shape).sqrt() * beta_t * pred_x_0 +
            extract(self.alphas, t, x_t.shape).sqrt() * (1 - extract(self.alphas_cumprod_prev, t, x_t.shape)) * x_t
        ) / (1 - extract(self.alphas_cumprod, t, x_t.shape))

        # Add noise for timesteps > 0
        posterior_variance_t = extract(self.posterior_variance, t, x_t.shape)
        noise = torch.randn_like(x_t) if t[0] > 0 else torch.zeros_like(x_t)
        
        return posterior_mean + torch.sqrt(posterior_variance_t) * noise

def extract(a, t, x_shape):
    """
    Extract appropriate entries from a for a batch of indices t
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


class LitModel(pl.LightningModule):
    def __init__(self, in_channels=3, out_channels=3, lr=1e-3, 
                 num_timesteps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize model and scheduler
        self.model = UNET(in_channels, out_channels)
        self.scheduler = DiffusionScheduler(num_timesteps, beta_start, beta_end)
        
    def setup(self, stage=None):
        # Move scheduler to correct device when setup is called
        self.scheduler = self.scheduler.to(self.device)

    def forward(self, x, t):
        return self.model(x, t)

    def training_step(self, batch, batch_idx):
        x_0 = batch
        batch_size = x_0.shape[0]
        
        # Generate random timesteps
        t = torch.randint(0, self.hparams.num_timesteps, (batch_size,), device=self.device)
        
        # Forward diffusion process
        x_t, noise = self.scheduler.forward_process(x_0, t)
        
        # Predict noise
        noise_pred = self(x_t, t)
        
        # Calculate loss
        loss = (noise - noise_pred).pow(2).mean()
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x_0 = batch
        batch_size = x_0.shape[0]
        
        # Generate random timesteps
        t = torch.randint(0, self.hparams.num_timesteps, (batch_size,), device=self.device)
        
        # Forward diffusion process
        x_t, noise = self.scheduler.forward_process(x_0, t)
        
        # Predict noise
        noise_pred = self(x_t, t)
        
        # Calculate loss
        loss = (noise_pred - noise).pow(2).mean()
        self.log('val_loss', loss, prog_bar=True)

        # Generate and log images
        if batch_idx == 0:
            generated_images = self.sample_images(batch_size)
            # self.logger.experiment.add_images('generated_images', generated_images, self.current_epoch)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.hparams.num_timesteps, eta_min=1e-6)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }

    @torch.no_grad()
    def sample_images(self, batch_size):
        # Start from random noise
        x_t = torch.randn(batch_size, self.hparams.in_channels, 64, 64, device=self.device)
        
        # Gradually denoise the image
        for t in reversed(range(self.hparams.num_timesteps)):
            t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            noise_pred = self(x_t, t_tensor)
            x_t = self.scheduler.reverse_process(x_t, t_tensor, noise_pred)
            
        return x_t.clamp(0, 1)