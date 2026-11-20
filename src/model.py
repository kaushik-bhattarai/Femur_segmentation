"""2D U-Net written from scratch (Ronneberger et al., 2015) with configurable depth/width.

Shape walk-through for input (B, 1, 256, 256), base_channels=32, depth=4:
  enc1: 32 @256   enc2: 64 @128   enc3: 128 @64   enc4: 256 @32   bottleneck: 512 @16
  dec4: 256 @32   dec3: 128 @64   dec2: 64 @128   dec1: 32 @256   head: 1x1 conv -> (B, 1, 256, 256)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_norm(kind, channels):
    if kind == "batch":
        return nn.BatchNorm2d(channels)
    if kind == "instance":
        return nn.InstanceNorm2d(channels, affine=True)
    if kind == "group":
        groups = 8 if channels % 8 == 0 else 1
        return nn.GroupNorm(groups, channels)
    raise ValueError(f"Unknown norm: {kind}")


class DoubleConv(nn.Module):
    """(Conv3x3 -> Norm -> ReLU) x 2. Padding=1 keeps the spatial size unchanged."""

    def __init__(self, in_ch, out_ch, norm="batch"):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            make_norm(norm, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            make_norm(norm, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """MaxPool 2x2 (halves H, W) then DoubleConv."""

    def __init__(self, in_ch, out_ch, norm):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch, norm)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    """Upsample x2, concatenate the encoder skip features, then DoubleConv."""

    def __init__(self, in_ch, skip_ch, out_ch, norm, mode="transpose"):
        super().__init__()
        if mode == "transpose":
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        elif mode == "bilinear":
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(in_ch, in_ch // 2, kernel_size=1),
            )
        else:
            raise ValueError(f"Unknown upsample mode: {mode}")
        self.conv = DoubleConv(in_ch // 2 + skip_ch, out_ch, norm)

    def forward(self, x, skip):
        x = self.up(x)
        # Handle odd input sizes: pad x so it matches the skip tensor.
        dy, dx = skip.size(2) - x.size(2), skip.size(3) - x.size(3)
        if dy != 0 or dx != 0:
            x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return self.conv(torch.cat([skip, x], dim=1))


class UNet2D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=32, depth=4,
                 norm="batch", upsample="transpose"):
        super().__init__()
        chs = [base_channels * 2 ** i for i in range(depth + 1)]  # e.g. [32, 64, 128, 256, 512]
        self.inc = DoubleConv(in_channels, chs[0], norm)
        self.downs = nn.ModuleList([Down(chs[i], chs[i + 1], norm) for i in range(depth)])
        self.ups = nn.ModuleList([
            Up(chs[i + 1], chs[i], chs[i], norm, upsample) for i in reversed(range(depth))
        ])
        self.head = nn.Conv2d(chs[0], out_channels, kernel_size=1)  # 1x1 conv -> class logits

    def forward(self, x):
        skips = [self.inc(x)]
        for down in self.downs:
            skips.append(down(skips[-1]))
        x = skips.pop()  # bottleneck
        for up in self.ups:
            x = up(x, skips.pop())
        return self.head(x)  # raw logits (apply sigmoid outside / inside the loss)


def build_model(cfg_model):
    return UNet2D(
        in_channels=1, out_channels=1,
        base_channels=cfg_model["base_channels"], depth=cfg_model["depth"],
        norm=cfg_model["norm"], upsample=cfg_model["upsample"],
    )


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
