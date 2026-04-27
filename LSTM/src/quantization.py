import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools

class WeightParametrization(nn.Module):
    def __init__(self, quantizer):
        super().__init__()
        self.quantizer = quantizer

    def forward(self, weight):
        return self.quantizer(weight)

# STE 
def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad

def round_pass(x):
    y = torch.round(x)
    return (y - x).detach() + x

class LSQQuantizer(nn.Module):
    def __init__(self, bit_width=8, per_channel=False, num_channels=1, **kwargs):
        super().__init__()
        self.bit_width = bit_width
        self.qmax = 2**(bit_width - 1) - 1
        self.qmin = -2**(bit_width - 1)
        shape = (num_channels, 1) if per_channel else (1,)
        self.s = nn.Parameter(torch.ones(shape))
        self.initialized = False

    def forward(self, x):
        if not self.initialized:
            if self.s.numel() == 1:
                # Per-tensor квантизация
                # Берем среднее абсолютно по всем измерениям тензора
                init_val = x.detach().abs().mean() * 2 / (self.qmax**0.5)
                self.s.data.fill_(torch.clamp(init_val, min=1e-4))
            else:
                # Per-channel квантизация
                # Оставляем dim=0 (выходные каналы), усредняем по остальным
                dims = list(range(1, x.dim())) if x.dim() > 1 else [0]
                init_val = x.detach().abs().mean(dim=dims, keepdim=True) * 2 / (self.qmax**0.5)
                self.s.data.copy_(torch.clamp(init_val.view(self.s.shape), min=1e-4))
            self.initialized = True
            
        s_pos = self.s.abs() + 1e-5
        
        # Защита от деления на 0 при вычислении n_el
        n_el = x.numel() / x.shape[0] if len(self.s) > 1 else x.numel()
        
        g = grad_scale(s_pos, 1.0 / (n_el * self.qmax)**0.5)
        return torch.clamp(round_pass(x / g), self.qmin, self.qmax) * s_pos


class PACTQuantizer(nn.Module):
    def __init__(self, bit_width=8, init_alpha=1.0, **kwargs):
        super().__init__()
        self.bit_width = bit_width
        self.qmax = 2**(bit_width - 1) - 1
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))
        self.initialized = False

    def forward(self, x):
        if not self.initialized:
            init_val = max(self.alpha.item(), x.detach().abs().max().item())
            self.alpha.data.copy_(torch.tensor(init_val))
            self.initialized = True
            
        alpha_pos = self.alpha.abs() + 1e-5
        x_clipped = torch.clamp(x, -alpha_pos, alpha_pos)
        scale = alpha_pos / self.qmax
        x_q = round_pass(x_clipped / scale) * scale
        return x_q

class APoTQuantizer(nn.Module):
    def __init__(self, bit_width=8, apot_m=2, weight_norm=False, **kwargs):
        super().__init__()
        self.bit_width = bit_width
        self.apot_m = apot_m
        self.apot_k = bit_width // apot_m # m * k = bits
        self.weight_norm = weight_norm
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.initialized = False
        self.alpha.register_hook(lambda grad: torch.clamp(grad, -0.01, 0.01))

        self.register_buffer('codebook', self._build_levels())

    def _build_levels(self):
        base_powers = [2**(-i) for i in range(2**self.apot_k - 1)]
        base_levels = torch.tensor([0.0] + base_powers)
        
        combinations = list(itertools.product(base_levels.tolist(), repeat=self.apot_m))
        levels = torch.tensor([sum(c) for c in combinations])
        
        levels = torch.unique(levels)
        levels = levels / levels.max()
        codebook = torch.cat([-levels[levels > 0], torch.tensor([0.0]), levels[levels > 0]])
        return codebook.sort().values

    def forward(self, x):
        if self.weight_norm and self.training:
            mean = x.mean()
            std = x.std() + 1e-5
            x = (x - mean) / std

        if not self.initialized:
            init_val = x.detach().abs().max()
            self.alpha.data.copy_(torch.clamp(init_val, min=1e-4))
            self.initialized = True
            
        alpha_pos = self.alpha.abs() + 1e-5
        x_norm = torch.clamp(x / alpha_pos, -1.0, 1.0)
        
        shape = x_norm.shape
        x_flat = x_norm.view(-1, 1)
        cb = self.codebook.view(1, -1)
        idx = torch.argmin(torch.abs(x_flat - cb), dim=1)
        x_q_norm = self.codebook[idx].view(shape)
        
        x_q_norm = (x_q_norm - x_norm).detach() + x_norm
        return x_q_norm * alpha_pos

class DSQQuantizer(nn.Module):
    def __init__(self, bit_width=8, temperature=0.1, **kwargs):
        super().__init__()
        self.bit_width = bit_width
        self.qmax = 2**(bit_width - 1) - 1
        self.qmin = -2**(bit_width - 1)
        self.temperature = temperature
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.initialized = False

    def forward(self, x):
        if not self.initialized:
            init_val = x.detach().abs().max() / self.qmax
            self.alpha.data.copy_(torch.clamp(init_val, min=1e-4))
            self.initialized = True
            
        scale = self.alpha.abs() + 1e-5
        x_scaled = x / scale
        x_clipped = torch.clamp(x_scaled, self.qmin, self.qmax)
        
        x_floor = torch.floor(x_clipped)
        residual = x_clipped - x_floor - 0.5
        soft_round = x_floor + 0.5 + 0.5 * torch.tanh(residual / self.temperature)
        
        x_q = soft_round + (torch.round(x_clipped) - soft_round).detach()
        return x_q * scale


class AdaRoundQuantizer(nn.Module):
    def __init__(self, bit_width=8, reg_lambda=0.01, weight_shape=None, **kwargs):
        super().__init__()
        self.bit_width = bit_width
        self.qmax = 2**(bit_width - 1) - 1
        self.qmin = -2**(bit_width - 1)
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.reg_lambda = reg_lambda
        self.initialized = False
        
        if weight_shape is not None:
            self.V = nn.Parameter(torch.zeros(weight_shape))
        else:
            self.V = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        if not self.initialized:
            init_val = x.detach().abs().max() / self.qmax
            self.scale.data.copy_(torch.clamp(init_val, min=1e-4))
            
            s_pos = self.scale.abs() + 1e-5
            x_scaled = x / s_pos
            x_floor = torch.floor(x_scaled)
            diff = x_scaled - x_floor

            target = (diff + 0.1) / 1.2
            target = torch.clamp(target, 1e-4, 1.0 - 1e-4) 
            v_init = -torch.log((1.0 / target) - 1.0)
        
            self.V.data = v_init.clone().detach()
                
            self.initialized = True
            
        s_pos = self.scale.abs() + 1e-5
        x_scaled = x / s_pos
        x_floor = torch.floor(x_scaled)
        
        h_v = torch.clamp(torch.sigmoid(self.V) * 1.2 - 0.1, 0.0, 1.0)
        
        if self.training:
            x_q = torch.clamp(x_floor + h_v, self.qmin, self.qmax)
        else:
            h_v_hard = (h_v >= 0.5).float()
            x_q = torch.clamp(x_floor + h_v_hard, self.qmin, self.qmax)
        
        return x_q * s_pos


class QuantizedLinear(nn.Linear):
    def __init__(self, in_f, out_f, bias=True, quantizer_type='none', bit_width=8, **kwargs):
        super().__init__(in_f, out_f, bias)
        self.quantizer_type = quantizer_type
        
        if quantizer_type == 'none':
            self.w_q, self.a_q = nn.Identity(), nn.Identity()
        else:
            q_map = {
                'lsq': LSQQuantizer,
                'pact': PACTQuantizer,
                'apot': APoTQuantizer,
                'dsq': DSQQuantizer,
                'adaround': AdaRoundQuantizer
            }
            if quantizer_type not in q_map:
                raise ValueError(f"Unknown quantizer: {quantizer_type}")
            
            QClass = q_map[quantizer_type]
            
            safe_w_kwargs = kwargs.copy()
            safe_w_kwargs.pop('per_channel', None)
            safe_w_kwargs.pop('num_channels', None)
            safe_w_kwargs.pop('weight_shape', None)
            
            self.w_q = QClass(
                bit_width=bit_width, 
                per_channel=True, 
                num_channels=out_f, 
                weight_shape=(out_f, in_f), 
                **safe_w_kwargs
            )
            
            safe_a_kwargs = kwargs.copy()
            safe_a_kwargs.pop('per_channel', None)
            safe_a_kwargs.pop('num_channels', None)
            safe_a_kwargs.pop('weight_shape', None)
            
            if quantizer_type == 'adaround':
                self.a_q = LSQQuantizer(bit_width=bit_width, **safe_a_kwargs)
            else:
                self.a_q = QClass(bit_width=bit_width, **safe_a_kwargs)

    def forward(self, x):
        return F.linear(self.a_q(x), self.w_q(self.weight), self.bias)


class QuantizedEmbedding(nn.Embedding):
    def __init__(self, num_emb, emb_dim, quantizer_type='none', bit_width=8, **kwargs):
        super().__init__(num_emb, emb_dim)
        self.quantizer_type = quantizer_type
        
        if quantizer_type != 'none':
            q_map = {
                'lsq': LSQQuantizer, 
                'pact': PACTQuantizer, 
                'apot': APoTQuantizer, 
                'dsq': DSQQuantizer, 
                'adaround': AdaRoundQuantizer
            }
            QClass = q_map[quantizer_type]
            
            safe_kwargs = kwargs.copy()
            safe_kwargs.pop('per_channel', None)
            safe_kwargs.pop('num_channels', None)
            safe_kwargs.pop('weight_shape', None)
            
            self.w_q = QClass(
                bit_width=bit_width, 
                per_channel=True, 
                num_channels=num_emb, 
                weight_shape=(num_emb, emb_dim), 
                **safe_kwargs
            )
        else:
            self.w_q = nn.Identity()

    def forward(self, x):
        return F.embedding(
            x, self.w_q(self.weight), self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse
        )


def get_quantizer(quantizer_type, bit_width, **kwargs):
    quantizers = {
        'lsq': LSQQuantizer,
        'pact': PACTQuantizer,
        'apot': APoTQuantizer,
        'dsq': DSQQuantizer,
        'adaround': AdaRoundQuantizer,
        'none': nn.Identity
    }
    q_class = quantizers.get(quantizer_type.lower(), nn.Identity)
    if q_class == nn.Identity:
        return q_class()
    return q_class(bit_width=bit_width, **kwargs)