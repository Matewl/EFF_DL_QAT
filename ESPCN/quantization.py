import torch
from torch import nn
import torch.nn.functional as F

class PACTActivation(nn.Module):
    def __init__(self, bits: int = 8, eps: float = 1e-6, signed: bool = False):
        super().__init__()
        self.bits = bits
        self.eps = eps
        self.alpha = nn.Parameter(torch.tensor(float(6.)))
        self.signed = signed

    def forward(self, x):
        alpha = torch.clamp(self.alpha, min=self.eps)        
        if self.signed:
            x_clipped = torch.clamp(x, min=-alpha, max=alpha)
            qmax = 2 ** (self.bits - 1) - 1
            scale = alpha / qmax
            x_q = torch.round(x_clipped / scale).clamp(-qmax, qmax) * scale
        else:
            x_clipped = torch.clamp(x, min=0.0)
            x_clipped = torch.minimum(x_clipped, alpha)
            q_levels = 2 ** self.bits - 1
            scale = alpha / q_levels
            x_q = torch.round(x_clipped / scale) * scale
        x_q = x_clipped + (x_q - x_clipped).detach()
        return x_q


class SymmetricWeightQuantizer(nn.Module):
    def __init__(self, bits: int = 8, eps: float = 1e-8):
        super().__init__()
        self.bits = bits
        self.eps = eps

    def forward(self, w):
        qmax = 2 ** (self.bits - 1) - 1
        max_val = torch.clamp(w.abs().max(), min=self.eps)
        scale = max_val / qmax

        w_q = torch.round(w / scale).clamp(-qmax, qmax) * scale
        w_q = w + (w_q - w).detach()
        return w_q


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
    ):
        super().__init__()
        self.act_quant = PACTActivation(bits=act_bits, signed=signed)
        self.weight_quant = SymmetricWeightQuantizer(bits=weight_bits)

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, x):
        x_q = self.act_quant(x)
        # w_q = self.conv.weight
        w_q = self.weight_quant(self.conv.weight)

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
        signed: bool = False
    ):
        super(QuantizedESPCN, self).__init__()
        self.conv1 = QuantConv2d(3, 64, 5, 1, 2, signed=signed)
        self.conv2 = QuantConv2d(64, 32, 3, 1, 1, signed=signed)
        self.conv3 = QuantConv2d(32, 3 * (upscale_factor ** 2), 3, 1, 1, signed=signed)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        
        if not signed:
            self.act = nn.Tanh()
        else:
            self.act = nn.ReLU() 

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = nn.Sigmoid()(self.pixel_shuffle(self.conv3(x)))
        return x