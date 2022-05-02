import torch.nn as nn
from .registry import register


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


@register
def convmixer(args):
    return nn.Sequential(
        nn.Conv2d(1, args.dim, kernel_size=args.patch_size, stride=args.patch_size),
        nn.GELU(),
        nn.BatchNorm2d(args.dim),
        *[
            nn.Sequential(
                Residual(
                    nn.Sequential(
                        nn.Conv2d(
                            args.dim,
                            args.dim,
                            args.kernel_size,
                            groups=args.dim,
                            padding="same",
                        ),
                        nn.GELU(),
                        nn.BatchNorm2d(args.dim),
                    )
                ),
                nn.Conv2d(args.dim, args.dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(args.dim),
            )
            for i in range(args.depth)
        ],
        nn.ConvTranspose2d(
            args.dim, 1, kernel_size=args.patch_size, stride=args.patch_size
        )
    )
