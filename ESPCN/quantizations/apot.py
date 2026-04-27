import torch
from torch import nn

class APoTQuantizer(nn.Module):
    """
    Additive Powers-of-Two quantizer with STE.

    Для bits=4 и groups=2 уровни строятся как суммы:
        s1 * 2^-k1 + s2 * 2^-k2
    """
    def __init__(self, bits=4, groups=2, signed=True, eps=1e-8):
        super().__init__()
        self.bits = bits
        self.groups = groups
        self.signed = signed
        self.eps = eps

        self.register_buffer("levels", self._make_levels())

    def _make_levels(self):
        # Число базовых степеней двойки на группу
        n_terms = max(1, self.bits // self.groups)

        base = torch.tensor([2.0 ** (-i) for i in range(n_terms)])

        # Все суммы additive powers-of-two
        levels = torch.zeros(1)
        for _ in range(self.groups):
            levels = (levels[:, None] + base[None, :]).reshape(-1)

        levels = torch.unique(levels)
        levels = levels / levels.max()

        levels = torch.cat([torch.zeros(1), levels])

        if self.signed:
            levels = torch.cat([-levels.flip(0), levels[1:]])

        return torch.unique(levels).sort()[0]

    def forward(self, x):
        scale = x.detach().abs().max().clamp(min=self.eps)
        x_norm = (x / scale).clamp(-1.0 if self.signed else 0.0, 1.0)

        # nearest APoT level
        dist = (x_norm[..., None] - self.levels).abs()
        idx = dist.argmin(dim=-1)
        x_q = self.levels[idx] * scale

        # Straight-Through Estimator
        return x + (x_q - x).detach()
