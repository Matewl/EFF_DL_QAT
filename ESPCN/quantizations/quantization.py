import math
from itertools import product

import torch
import torch.nn.functional as F
from torch import nn

from .lsq import LSQQuantizer
from .pact import PACTActivation, SymmetricWeightQuantizer
from .apot import APoTQuantizer

def build_activation_quantizer(
    method: str,
    bits: int,
    signed: bool,
) -> nn.Module | None:
    method = method.lower()
    if method == "pact":
        return PACTActivation(bits=bits, signed=signed)
    if method == "lsq":
        return LSQQuantizer(bit=bits, signed=False, per_channel=False)
    if method == "apot":
        return None

def build_weight_quantizer(
    method: str,
    bits: int,
) -> nn.Module:
    method = method.lower()
    if method == "pact":
        return SymmetricWeightQuantizer(bits=bits)
    if method == "lsq":
        return LSQQuantizer(bit=bits, signed=True, per_channel=True, ch_axis=0)
    if method == "qdrop":
        return APoTQuantizer(bits=bits, groups=2, signed=True)
        
class QuantConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        bias=True,
        act_bits: int = 8,
        weight_bits: int = 8,
        signed: bool = False,
        quant_method: str = "pact",
    ):
        super().__init__()
        self.quant_method = quant_method.lower()
        self.act_quant = build_activation_quantizer(
            self.quant_method,
            act_bits,
            signed,
        )

        self.weight_quant = build_weight_quantizer(
            self.quant_method,
            weight_bits,
        )

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_q = self.act_quant(x) if self.act_quant else x
        w_q = self.weight_quant(self.conv.weight) if self.weight_quant else self.conv.weight
        return F.conv2d(
            x_q,
            w_q,
            self.conv.bias,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups,
        )


class QuantizedESPCN(nn.Module):
    def __init__(
        self,
        upscale_factor: int = 2,
        signed: bool = True,
        quant_method: str = "pact",
        act_bits: int = 8,
        weight_bits: int = 8,
    ):
        super().__init__()
        self.quant_method = quant_method.lower()
        self.conv1 = QuantConv2d(3, 64, 5, 1, 2,
            signed=signed,
            quant_method=self.quant_method,
            act_bits=act_bits,
            weight_bits=weight_bits,
        )
        self.conv2 = QuantConv2d(64, 32, 3, 1, 1,
            signed=signed,
            quant_method=self.quant_method,
            act_bits=act_bits,
            weight_bits=weight_bits,
        )
        self.conv3 = QuantConv2d(32, 3 * (upscale_factor ** 2), 3, 1, 1,
            signed=signed,
            quant_method=self.quant_method,
            act_bits=act_bits,
            weight_bits=weight_bits,
        )
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.act = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = torch.sigmoid(self.pixel_shuffle(self.conv3(x)))
        return x