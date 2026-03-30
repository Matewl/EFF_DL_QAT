import torch
from torch import nn
import torch.nn.functional as F
import lightning as L
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

class ESPCN(nn.Module):
    def __init__(self, upscale_factor):
        super(ESPCN, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, 5, 1, 2)
        self.conv2 = nn.Conv2d(64, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(32, 3 * (upscale_factor ** 2), 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = F.tanh(self.conv1(x))
        x = F.tanh(self.conv2(x))
        x = F.sigmoid(self.pixel_shuffle(self.conv3(x)))
        return x