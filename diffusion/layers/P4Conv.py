import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from .utils import _grot90
from einops import rearrange


class P4ConvZ2(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.use_bias = bias

        # i/p stabilizer no for Z2 -> P4 is 1
        self.input_stabilizer = 1
        # o/p stabilizer is 4, as there are 4 rotations
        self.output_stabilizer = 4

        in_chns = in_channels // groups
        out_chns = out_channels
        self.weight = nn.Parameter(
            torch.FloatTensor(out_chns, in_chns, *self.kernel_size),
            requires_grad=True,
        )
        if bias:
            self.bias = nn.Parameter(
                torch.FloatTensor(out_channels), requires_grad=True
            )
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.xavier_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def transform_kernel(
        self,
        kernel: torch.tensor,
    ) -> torch.tensor:
        # get all possible rotated kernels and stack them
        kernels = [torch.rot90(kernel, k, (2, 3)) for k in range(4)]
        kernels = torch.stack(kernels, axis=1)
        kernels = rearrange(
            kernels, "o ot i k1 k2 -> (o ot) i k1 k2", ot=self.output_stabilizer
        )
        return kernels

    def forward(
        self,
        x: torch.tensor,
    ) -> torch.tensor:
        weights = self.transform_kernel(self.weight)
        output = F.conv2d(
            x,
            weights,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        # separate the output stabilizers and output channels
        output = rearrange(
            output, "b (c ot) h w -> b c ot h w", ot=self.output_stabilizer
        )
        if self.bias is not None:
            output += self.bias[None, :, None, None, None]

        return output


class P4ConvP4(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.use_bias = bias

        # i/p stabilizer no for P4 -> P4 is 4
        self.input_stabilizer = 4
        # o/p stabilizer is 4, as there are 4 rotations
        self.output_stabilizer = 4

        in_chns = in_channels // groups
        out_chns = out_channels
        self.weight = nn.Parameter(
            torch.FloatTensor(
                out_chns, in_chns, self.input_stabilizer, *self.kernel_size
            ),
            requires_grad=True,
        )
        if bias:
            self.bias = nn.Parameter(
                torch.FloatTensor(out_channels), requires_grad=True
            )
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.xavier_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def transform_kernel(
        self,
        kernel: torch.tensor,
    ) -> torch.tensor:
        # get all possible rotated kernels and stack them
        kernels = [
            rearrange(
                _grot90(kernel, k),
                "o i it h w -> o (i it) h w",
                it=self.input_stabilizer,
            )
            for k in range(4)
        ]
        return rearrange(
            torch.stack(kernels, 1),
            "o ot i h w -> (o ot) i h w",
            ot=self.output_stabilizer,
        )

    def forward(
        self,
        x: torch.tensor,
    ) -> torch.tensor:
        weights = self.transform_kernel(self.weight)
        x = rearrange(x, "b c it h w -> b (c it) h w", it=self.input_stabilizer)
        output = F.conv2d(
            x,
            weights,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        # separate the output stabilizers and output channels
        output = rearrange(
            output, "b (c ot) h w -> b c ot h w", ot=self.output_stabilizer
        )
        if self.bias is not None:
            output += self.bias[None, :, None, None, None]

        return output
