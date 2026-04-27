import torch
from torch import nn

class GradScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None


class RoundPass(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def grad_scale(x, scale):
    return GradScale.apply(x, scale)


def round_pass(x):
    return RoundPass.apply(x)


class LSQQuantizer(nn.Module):
    def __init__(self, bit=8, signed=True, per_channel=False, ch_axis=0, eps=1e-8):
        super().__init__()
        self.bit = bit
        self.signed = signed
        self.per_channel = per_channel
        self.ch_axis = ch_axis
        self.eps = eps
        self.initialized = False
        self.s = nn.Parameter(torch.tensor(1.0))

        if signed:
            self.qn = -(2 ** (bit - 1))
            self.qp = 2 ** (bit - 1) - 1
        else:
            self.qn = 0
            self.qp = 2 ** bit - 1

    def _init_s(self, x):
        with torch.no_grad():
            if self.per_channel:
                dims = [i for i in range(x.ndim) if i != self.ch_axis]
                mean = x.detach().abs().mean(dim=dims, keepdim=True)
                s = 2 * mean / (self.qp ** 0.5)
                self.s = nn.Parameter(s.clamp(min=self.eps))
            else:
                s = 2 * x.detach().abs().mean() / (self.qp ** 0.5)
                self.s.data = s.clamp(min=self.eps)
            self.initialized = True

    def forward(self, x):
        if not self.initialized:
            self._init_s(x)

        s = self.s.clamp(min=self.eps)

        # LSQ gradient scaling: 1 / sqrt(N * Qp)
        n = x.numel() if not self.per_channel else x[0].numel()
        scale = 1.0 / ((n * self.qp) ** 0.5)
        s = grad_scale(s, scale)

        x_q = torch.clamp(x / s, self.qn, self.qp)
        x_q = round_pass(x_q)
        return x_q * s

