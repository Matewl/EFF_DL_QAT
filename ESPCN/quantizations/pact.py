import torch
from torch import nn

class PACTActivation(nn.Module):
    def __init__(self, bits: int = 8, eps: float = 1e-6, signed: bool = False):
        super().__init__()
        self.bits = bits
        self.eps = eps
        self.alpha = nn.Parameter(torch.tensor(2.0))
        self.signed = signed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        return x_clipped + (x_q - x_clipped).detach()


class SymmetricWeightQuantizer(nn.Module):
    def __init__(self, bits: int = 8, eps: float = 1e-8):
        super().__init__()
        self.bits = bits
        self.eps = eps

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        qmax = 2 ** (self.bits - 1) - 1
        max_val = torch.clamp(w.abs().max(), min=self.eps)
        scale = max_val / qmax
        w_q = torch.round(w / scale).clamp(-qmax, qmax) * scale
        return w + (w_q - w).detach()
