from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

class QDropFakeQuant(nn.Module):
    def __init__(self, observer, quant_min=0, quant_max=255, drop_prob=0.5):
        super().__init__()
        self.observer = observer
        self.quant_min = quant_min
        self.quant_max = quant_max
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.training:
            if torch.rand(1).item() < self.drop_prob:
                return x

        scale, zero_point = self.observer(x)
        x_int = torch.clamp(torch.round(x / scale + zero_point),
                            self.quant_min, self.quant_max)
        x_dequant = (x_int - zero_point) * scale
        return x_dequant
    
class MinMaxObserver(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        min_val = x.min()
        max_val = x.max()

        scale = (max_val - min_val) / 255.0
        scale = torch.clamp(scale, min=self.eps)

        zero_point = -min_val / scale
        zero_point = torch.clamp(zero_point, 0, 255)

        return scale, zero_point

class QDropConv2d(nn.Module):
    def __init__(self, conv: nn.Conv2d, drop_prob=0.5):
        super().__init__()
        self.conv = conv

        self.weight_quant = QDropFakeQuant(MinMaxObserver(), drop_prob=drop_prob)
        self.act_quant = QDropFakeQuant(MinMaxObserver(), drop_prob=drop_prob)

    def forward(self, x):
        w_q = self.weight_quant(self.conv.weight)
        x = F.conv2d(x, w_q, self.conv.bias,
                     stride=self.conv.stride,
                     padding=self.conv.padding)

        x = self.act_quant(x)
        return x

class QDropESPCN(nn.Module):
    def __init__(self, pretrained_model, drop_prob=0.5):
        super().__init__()

        self.conv1 = QDropConv2d(pretrained_model.conv1, drop_prob)
        self.conv2 = QDropConv2d(pretrained_model.conv2, drop_prob)
        self.conv3 = QDropConv2d(pretrained_model.conv3, drop_prob)

        self.pixel_shuffle = pretrained_model.pixel_shuffle

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pixel_shuffle(self.conv3(x))
        return x

def train_q_drop(pretrained_model, dataloader, device):
    model = QDropESPCN(pretrained_model, drop_prob=0.5)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    for epoch in tqdm(range(150), desc="training"):
        for lr, hr in dataloader:
            lr = lr.to(device)
            hr = hr.to(device)
            sr = model(lr)

            loss = F.mse_loss(sr, hr)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()

    for m in model.modules():
        if isinstance(m, QDropFakeQuant):
            m.drop_prob = 0.0
    return model