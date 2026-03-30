import math
import numpy as np
import torch

def pil_to_tensor(img):
    arr = np.array(img).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=-1)
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)

def calc_psnr(sr, hr, shave: int = 0) -> float:
    if sr.dim() == 3:
        sr = sr.unsqueeze(0)
    if hr.dim() == 3:
        hr = hr.unsqueeze(0)

    sr = sr.clamp(0.0, 1.0)
    hr = hr.clamp(0.0, 1.0)

    if shave > 0:
        sr = sr[:, :, shave:-shave, shave:-shave]
        hr = hr[:, :, shave:-shave, shave:-shave]

    mse = torch.mean((sr - hr) ** 2).item()
    if mse == 0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)