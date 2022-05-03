import torch
import torch.nn as nn
import torch.nn.functional as F

import einops
from .registry import register
from ..layers.utils import _grot90
from ..layers.P4Conv import P4ConvZ2, P4ConvP4
from ..layers.P4ConvTranspose import P4ConvTransposeP4


class DoubleConv(nn.Module):
    """(convolution => [BN] => SiLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, input=False):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        if input:
            conv = P4ConvZ2(
                in_channels, mid_channels, kernel_size=3, padding=1, bias=False
            )
        else:
            conv = P4ConvP4(
                in_channels, mid_channels, kernel_size=3, padding=1, bias=False
            )
        self.double_conv = nn.Sequential(
            conv,
            nn.BatchNorm3d(mid_channels),
            nn.SiLU(inplace=True),
            P4ConvP4(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d((1, 2, 2)), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.up = P4ConvTransposeP4(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = P4ConvP4(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class EquivariantUnet(nn.Module):
    def __init__(self, n_channels, out_channels, bilinear=False):
        super(EquivariantUnet, self).__init__()
        self.n_channels = n_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64, input=True)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, out_channels)

    def forward(self, x, t=None):
        if t is not None:
            b, h, w = x.shape[0], x.shape[-2], x.shape[-1]
            t = einops.repeat(t, "b -> b 1 h w", b=b, h=h, w=w)
            x = torch.cat([x, t], dim=1)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


def test():
    print("Double conv test")
    conv2 = DoubleConv(3, 10)
    x = torch.rand(7, 3, 4, 256, 256)
    y = conv2(x)
    print(x.size(), "->", y.size())
    for k in range(1, 4):
        y2 = _grot90(conv2(_grot90(x, k)), -k)
        print(k, (y - y2).abs().max().item())

    print("Down test")
    conv2 = Down(3, 10)
    x = torch.rand(7, 3, 4, 256, 256)
    y = conv2(x)
    print(x.size(), "->", y.size())
    for k in range(1, 4):
        y2 = _grot90(conv2(_grot90(x, k)), -k)
        print(k, (y - y2).abs().max().item())

    print("Equivariant Unet test")
    conv2 = EquivariantUnet(3, 3)
    x = torch.rand(7, 3, 256, 256)
    y = conv2(x)
    print(x.size(), "->", y.size())
    for k in range(1, 4):
        y2 = _grot90(conv2(torch.rot90(x, k, (2, 3))), -k)
        print(k, (y - y2).abs().max().item())


@register
def p4_unet(args):
    return EquivariantUnet(args.num_channels, args.num_channels)
