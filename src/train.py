import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from torchvision.utils import save_image

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

# Import your Lightning diffusion model (adjust the import if necessary)
from lightning_module import LitModel
# Optional: Set random seed for reproducibility
# pl.seed_everything(42)

# Custom dataset wrapper for Hugging Face datasets
class HFImageDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        # Change the key below if your dataset returns the image under a different key.
        item = self.hf_dataset[idx]
        image = item["pixel_values"]  # PIL Image (or adjust key if needed)
        if self.transform:
            image = self.transform(image)
        return image
    
# Custom callback for image sampling
class SamplingCallback(pl.Callback):
    def __init__(self, sample_every_n_epochs=1, num_samples=16):
        super().__init__()
        self.sample_every_n_epochs = sample_every_n_epochs
        self.num_samples = num_samples
        os.makedirs("samples", exist_ok=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.sample_every_n_epochs == 0:
            # Generate samples
            samples = pl_module.sample_images(batch_size=self.num_samples)
            
            # Save to disk
            save_image(
                samples,
                f"samples/epoch_{trainer.current_epoch:03d}.png",
                nrow=4,
                normalize=True
            )
            
            # Log to wandb if available
            if isinstance(trainer.logger, WandbLogger):
                trainer.logger.log_image(
                    key=f"generated_images",
                    images=[samples],
                    caption=[f"Epoch {trainer.current_epoch}"]
                )

def main():
    # ----------------------------
    # Data Loading and Transforms
    # ----------------------------
    hf_dataset_train = load_dataset("nzm97/lsun_bedroom_64x64", split="train")
    hf_dataset_val   = load_dataset("nzm97/lsun_bedroom_64x64", split="test")

    data_transform = transforms.Compose([
        transforms.ToTensor(),  # Converts a PIL Image to a float tensor in [0,1]
    ])

    train_dataset = HFImageDataset(hf_dataset_train, transform=data_transform)
    val_dataset   = HFImageDataset(hf_dataset_val, transform=data_transform)

    batch_size = 64
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    # ----------------------------
    # Model, Logger, and Checkpoints
    # ----------------------------
    wandb_logger = WandbLogger(project="diffusion_lsund", name="lsun_bedroom_run")

    callbacks =  [
        ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            filename="{epoch:02d}-{val_loss:.2f}",
            dirpath="checkpoints",
        ),
        SamplingCallback(
            sample_every_n_epochs=1,  # Sample every epoch
            num_samples=16
        )
    ]

    model = LitModel(
        in_channels=3,
        out_channels=3,
        lr=1e-3,
        num_timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02
    )

    # ----------------------------
    # Trainer Setup and Fit
    # ----------------------------
    trainer = pl.Trainer(
        max_epochs=50,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else None,
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=50,
    )

    trainer.fit(model, 
                train_dataloaders=train_loader, 
                val_dataloaders=val_loader)

    # ----------------------------
    # Sampling and Image Saving
    # ----------------------------
    sample_images = model.sample_images(batch_size=16)
    
    from torchvision.utils import save_image
    os.makedirs("samples", exist_ok=True)
    save_image(sample_images, "samples/final_generated.png", nrow=4)
    print("Sample image saved at samples/final_generated.png")

if __name__ == "__main__":
    main()