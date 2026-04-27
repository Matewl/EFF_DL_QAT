import math
import numpy as np
import torch

def pil_to_tensor(img):
    arr = np.array(img).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=-1)
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)