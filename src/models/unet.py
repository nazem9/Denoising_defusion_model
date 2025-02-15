import torch
import torch.nn as nn
import torch.nn.functional as F

# A double–conv layer with kernel_size=3, padding=1 so output spatial dims match input
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

# Down block: performs max–pool then double conv
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),  # reduce HxW by a factor of 2
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

# Up block: upsample, then concatenate with skip, then double conv.
# The upsampling can be done by either ConvTranspose2d or bilinear interpolation.
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:   # use the normal upsample and then a conv to reduce channel dims
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:          # or use the learned ConvTranspose2d for upsampling.
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        # after concatenation, the number of input channels is (skip connection channels + upsampled channels)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # x1 is the upsampled input from the layer below and x2 is the skip connection.
        x1 = self.up(x1)
        # Sometimes, the upsampled feature map does not perfectly match the shape of x2.
        # We therefore pad x1 on both sides.
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # Concatenate along the channel dimension.
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# Final output convolution: here we use kernel_size=1 to map to desired number of classes.
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        return self.conv(x)

# The actual U‑Net model. Here we assume an encoder with four down blocks before the bottom.
class UNET(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, bilinear=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        # Initial block (no pooling)
        self.inc = DoubleConv(n_channels, 64)
        # Down–sampling path
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        # Optionally, if we are using bilinear upsampling
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        # Up–sampling path (each Up receives the concatenated tensor from the encoder)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        # Final 1x1 convolution to get the desired number of classes.
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # Encoder: with skip connections
        x1 = self.inc(x)     # size: original resolution
        x2 = self.down1(x1)  # size reduced by /2
        x3 = self.down2(x2)  # /4
        x4 = self.down3(x3)  # /8
        x5 = self.down4(x4)  # /16 (lowest resolution)
        # Decoder: up–sample and concatenate with encoder features.
        x = self.up1(x5, x4)  # /8
        x = self.up2(x, x3)   # /4
        x = self.up3(x, x2)   # /2
        x = self.up4(x, x1)   # back to original resolution
        logits = self.outc(x)
        return logits

# Example usage:
if __name__ == '__main__':
    # Create a dummy input tensor (batch_size, channels, height, width)
    x = torch.randn(1, 3, 64, 64)
    # Instantiate the model
    model = UNET(n_channels=3, n_classes=2, bilinear=True)
    # Forward pass
    output = model(x)
    print("Output shape:", output.shape)