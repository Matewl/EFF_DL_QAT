import time
import copy
import argparse
from pathlib import Path

from tqdm import tqdm

import torch
from torch import nn
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import torch.ao.quantization as tq

from dataset import ESPCNDataModule
from lightning_module import LitESPCN
from model import ESPCN

@torch.no_grad()
def export_pact_to_float(pact_model: nn.Module, upscale_factor: int = 2) -> nn.Module:
    m = ESPCN(upscale_factor=upscale_factor)

    m.conv1.weight.copy_(pact_model.conv1.conv.weight)
    m.conv1.bias.copy_(pact_model.conv1.conv.bias)

    m.conv2.weight.copy_(pact_model.conv2.conv.weight)
    m.conv2.bias.copy_(pact_model.conv2.conv.bias)

    m.conv3.weight.copy_(pact_model.conv3.conv.weight)
    m.conv3.bias.copy_(pact_model.conv3.conv.bias)

    return m.eval()


class QuantizedConvFloatActESPCN(nn.Module):
    def __init__(self, fp32_model: ESPCN):
        super().__init__()

        self.quant1 = tq.QuantStub()
        self.conv1 = copy.deepcopy(fp32_model.conv1)
        self.dequant1 = tq.DeQuantStub()

        self.quant2 = tq.QuantStub()
        self.conv2 = copy.deepcopy(fp32_model.conv2)
        self.dequant2 = tq.DeQuantStub()

        self.quant3 = tq.QuantStub()
        self.conv3 = copy.deepcopy(fp32_model.conv3)
        self.dequant3 = tq.DeQuantStub()

        self.pixel_shuffle = copy.deepcopy(fp32_model.pixel_shuffle)
        self.act = nn.Tanh()

    def forward(self, x):
        x = self.quant1(x)
        x = self.conv1(x)
        x = self.dequant1(x)
        x = self.act(x)

        x = self.quant2(x)
        x = self.conv2(x)
        x = self.dequant2(x)
        x = self.act(x)

        x = self.quant3(x)
        x = self.conv3(x)
        x = self.dequant3(x)

        x = self.pixel_shuffle(x)
        x = torch.sigmoid(x)
        return x


@torch.no_grad()
def convert_to_real_int8(
    pact_model: nn.Module,
    calibration_loader,
    upscale_factor: int = 2,
    num_calib_batches: int = 100,
    backend: str | None = None,
):

    if backend is None:
        backend = "x86" if "x86" in torch.backends.quantized.supported_engines else "qnnpack"

    torch.backends.quantized.engine = backend

    fp32_model = export_pact_to_float(pact_model, upscale_factor=upscale_factor).cpu().eval()
    qmodel = QuantizedConvFloatActESPCN(fp32_model).cpu().eval()

    qmodel.qconfig = tq.get_default_qconfig(backend)

    prepared = tq.prepare(qmodel, inplace=False)
    for i, batch in enumerate(tqdm(calibration_loader, desc="INT8 calibration")):
        if i >= num_calib_batches:
            break

        lr = batch[0] if isinstance(batch, (tuple, list)) else batch
        lr = lr.cpu()

        prepared(lr)

    int8_model = tq.convert(prepared, inplace=False).eval()

    return fp32_model, int8_model


@torch.no_grad()
def evaluate_quality(model, dataloader, device="cpu"):
    model = model.to(device).eval()

    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    for lr, hr in tqdm(dataloader, desc="Quality"):
        lr = lr.to(device)
        hr = hr.to(device)
        sr = model(lr)

        psnr.update(sr, hr)
        ssim.update(sr, hr)

    return {
        "psnr": float(psnr.compute().cpu()),
        "ssim": float(ssim.compute().cpu()),
    }


@torch.no_grad()
def benchmark_cpu(
    model,
    dataloader,
    num_batches: int = 100,
    warmup_batches: int = 20,
    num_threads: int | None = None,
):
    if num_threads is not None:
        torch.set_num_threads(num_threads)

    model = model.cpu().eval()

    for i, batch in enumerate(dataloader):
        if i >= warmup_batches:
            break
        lr = batch[0] if isinstance(batch, (tuple, list)) else batch
        model(lr.cpu())

    total_time = 0.0
    total_images = 0
    total_batches = 0

    for i, batch in enumerate(tqdm(dataloader, desc="CPU benchmark")):
        if i >= num_batches:
            break

        lr = batch[0] if isinstance(batch, (tuple, list)) else batch
        lr = lr.cpu()

        start = time.perf_counter()
        model(lr)
        end = time.perf_counter()

        total_time += end - start
        total_images += lr.shape[0]
        total_batches += 1

    return {
        "avg_batch_latency_ms": 1000.0 * total_time / max(total_batches, 1),
        "avg_image_latency_ms": 1000.0 * total_time / max(total_images, 1),
        "fps": total_images / max(total_time, 1e-12),
        "num_threads": torch.get_num_threads(),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate ESPCN with PyTorch Lightning.")
    parser.add_argument("--data-root", type=Path, default=Path("ESPCN/data"))
    parser.add_argument("--scale", type=int, default=2, choices=[2, 3, 4])
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--best-path", type=Path)


def convert_and_compare():
    args = parse_args()
    datamodule = ESPCNDataModule(
        data_root=args.data_root,
        scale=args.scale,
        patch_size=args.patch_size,
        batch_size=1,
        load_immediatly=False,
        augment=False,
    )
    datamodule.setup(stage="test")
    
    model = LitESPCN.load_from_checkpoint(args.best_path).model
    model = model.to("cpu")
    model.cpu().eval()
    
    test_loaders = datamodule.test_dataloader()
    for test_loader in test_loaders:
        print(f"For Set{len(test_loader)}")
        fp32_model, int8_model = convert_to_real_int8(
            pact_model=model,
            calibration_loader=test_loader,
            upscale_factor=args.scale,
            num_calib_batches=100,
        )


        fp32_quality = evaluate_quality(fp32_model, test_loader, device="cpu")
        int8_quality = evaluate_quality(int8_model, test_loader, device="cpu")

        fp32_speed = benchmark_cpu(fp32_model, test_loader, num_batches=100)
        int8_speed = benchmark_cpu(int8_model, test_loader, num_batches=100)

        print("FP32 quality:", fp32_quality)
        print("INT8 quality:", int8_quality)

        print("FP32 speed:", fp32_speed)
        print("INT8 speed:", int8_speed)

        print("Speedup FPS:", int8_speed["fps"] / fp32_speed["fps"])
        print("PSNR drop:", fp32_quality["psnr"] - int8_quality["psnr"])
        print("SSIM drop:", fp32_quality["ssim"] - int8_quality["ssim"])

