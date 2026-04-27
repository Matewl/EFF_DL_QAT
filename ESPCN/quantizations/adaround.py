from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy


def symmetric_quant_params(w, num_bits=8, eps=1e-8):
    qmax = 2 ** (num_bits - 1) - 1
    scale = w.abs().max() / qmax
    scale = torch.clamp(scale, min=eps)
    return scale, qmax


def ste_round(x):
    return (x.round() - x).detach() + x


class AdaRoundQuantizer(nn.Module):
    def __init__(self, weight, num_bits=8, zeta=1.1, gamma=-0.1):
        super().__init__()

        self.num_bits = num_bits
        self.zeta = zeta
        self.gamma = gamma

        scale, qmax = symmetric_quant_params(weight, num_bits)
        self.register_buffer("scale", scale)
        self.qmin = -qmax - 1
        self.qmax = qmax

        w_scaled = weight / scale
        self.register_buffer("w_floor", torch.floor(w_scaled))
        rest = w_scaled - torch.floor(w_scaled)

        # inverse sigmoid initialization
        rest = torch.clamp(rest, 1e-6, 1 - 1e-6)
        alpha = -torch.log((self.zeta - self.gamma) / (rest - self.gamma) - 1)
        self.alpha = nn.Parameter(alpha)

    def rectified_sigmoid(self):
        h = torch.sigmoid(self.alpha) * (self.zeta - self.gamma) + self.gamma
        return torch.clamp(h, 0, 1)

    def forward(self, hard=False):
        if hard:
            h = (self.alpha >= 0).float()
        else:
            h = self.rectified_sigmoid()

        w_q = self.w_floor + h
        w_q = torch.clamp(w_q, self.qmin, self.qmax)
        return w_q * self.scale

    def round_loss(self):
        h = self.rectified_sigmoid()
        return torch.sum(1 - torch.abs(2 * h - 1))


class QuantConv2dAdaRound(nn.Module):
    def __init__(self, conv: nn.Conv2d, num_bits=8):
        super().__init__()

        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups

        self.weight_quantizer = AdaRoundQuantizer(conv.weight.data, num_bits)

        if conv.bias is not None:
            self.bias = nn.Parameter(conv.bias.data.clone(), requires_grad=False)
        else:
            self.bias = None

    def forward(self, x, hard=False):
        w_q = self.weight_quantizer(hard=hard)
        return F.conv2d(
            x,
            w_q,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )



def replace_conv_with_adaround(module, num_bits=8):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d):
            setattr(module, name, QuantConv2dAdaRound(child, num_bits))
        else:
            replace_conv_with_adaround(child, num_bits)


def set_hard_rounding(module, hard=True):
    module._adaround_hard = hard


@torch.no_grad()
def collect_inputs(model, dataloader, device, max_batches=16):
    xs = []
    for i, batch in tqdm(enumerate(dataloader), desc="collecting inputs"):
        if i >= max_batches:
            break

        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch

        xs.append(x.to(device))

    return torch.cat(xs, dim=0)


def forward_adaround(model, x, hard=False):
    def patch_forward(m):
        if isinstance(m, QuantConv2dAdaRound):
            return lambda inp, m=m: QuantConv2dAdaRound.forward(m, inp, hard=hard)
        return None

    hooks = []
    old_forwards = {}

    for m in model.modules():
        if isinstance(m, QuantConv2dAdaRound):
            old_forwards[m] = m.forward
            m.forward = patch_forward(m)

    y = model(x)

    for m, old_fwd in old_forwards.items():
        m.forward = old_fwd

    return y


def adaround_quantize_espcn(
    trained_model,
    calibration_loader,
    device="cuda",
    num_bits=8,
    iters=5000,
    lr=1e-3,
    reg_weight=0.01,
    max_calib_batches=16,
):
    fp_model = deepcopy(trained_model).to(device).eval()
    q_model = deepcopy(trained_model).to(device).eval()

    replace_conv_with_adaround(q_model, num_bits=num_bits)

    calib_x = collect_inputs(
        fp_model,
        calibration_loader,
        device=device,
        max_batches=max_calib_batches,
    )

    with torch.no_grad():
        target_y = fp_model(calib_x)

    params = []
    for m in q_model.modules():
        if isinstance(m, QuantConv2dAdaRound):
            params.append(m.weight_quantizer.alpha)

    optimizer = torch.optim.Adam(params, lr=lr)

    for step in tqdm(range(iters), desc="train adaround"):
        optimizer.zero_grad()

        pred_y = forward_adaround(q_model, calib_x, hard=False)

        recon_loss = F.mse_loss(pred_y, target_y)

        round_loss = 0.0
        for m in q_model.modules():
            if isinstance(m, QuantConv2dAdaRound):
                round_loss = round_loss + m.weight_quantizer.round_loss()

        temp = max(0.0, 1.0 - step / iters)
        loss = recon_loss + reg_weight * temp * round_loss

        loss.backward()
        optimizer.step()
    for m in q_model.modules():
        if isinstance(m, QuantConv2dAdaRound):
            with torch.no_grad():
                q_weight = m.weight_quantizer(hard=True)
                m.weight_quantizer.w_floor.copy_(torch.floor(q_weight / m.weight_quantizer.scale))

    return q_model.eval()


# -----------------------------
# Example usage
# -----------------------------

# espcn = ESPCN(...)
# espcn.load_state_dict(torch.load("espcn_fp32.pth"))
#
# q_espcn = adaround_quantize_espcn(
#     trained_model=espcn,
#     calibration_loader=calib_loader,
#     device="cuda",
#     num_bits=8,
#     iters=5000,
#     lr=1e-3,
# )
#
# torch.save(q_espcn.state_dict(), "espcn_adaround_int8_weights.pth")