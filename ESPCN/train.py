import argparse
from pathlib import Path

import torch
from torch import nn
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

from dataset import ESPCNDataModule
from lightning_module import LitESPCN
from quantizations.adaround import adaround_quantize_espcn
from quantizations.qdrop import train_q_drop

def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate ESPCN with PyTorch Lightning.")
    parser.add_argument("--data-root", type=Path, default=Path("ESPCN/data"))
    parser.add_argument("--scale", type=int, default=2, choices=[2, 3, 4])
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--quant-method",
        type=str,
        default="wo_quant",
        choices=["wo_quant", "pact", "lsq", "adaround", "apot", "qdrop"],
    )
    parser.add_argument("--signed", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--act-bits", type=int, default=8)
    parser.add_argument("--weight-bits", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load-immediatly", action="store_true")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=str, default="1")
    parser.add_argument("--precision", type=str, default="32-true")
    parser.add_argument("--default-root-dir", type=Path, default=Path("ESPCN/runs"))
    parser.add_argument("--experiment-name", type=str, default="espcn")
    parser.add_argument("--logger", type=str, default="tensorboard", choices=["tensorboard", "csv"])
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--test-only", action="store_true")
    return parser.parse_args()


def parse_devices(devices: str):
    if devices == "auto":
        return devices
    if devices.isdigit():
        return int(devices)
    return devices


def build_logger(logger_name: str, save_dir: Path, experiment_name: str):
    if logger_name == "tensorboard":
        return TensorBoardLogger(save_dir=str(save_dir), name=experiment_name)
    return CSVLogger(save_dir=str(save_dir), name=experiment_name)


def test(model: nn.Module, dataloader, device):
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    for lr, hr in dataloader:
        lr = lr.to(device)
        hr = hr.to(device)
        sr = model(lr)

        psnr.update(sr, hr)
        ssim.update(sr, hr)

    return {
        "psnr": float(psnr.compute()),
        "ssim": float(ssim.compute()),
    }


def main():
    args = parse_args()
    L.seed_everything(args.seed, workers=True)

    datamodule = ESPCNDataModule(
        data_root=args.data_root,
        scale=args.scale,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        load_immediatly=args.load_immediatly,
        augment=not args.no_augment,
    )
    datamodule.setup(stage="fit")

    model = LitESPCN(
        scale=args.scale,
        signed=args.signed,
        quant_method=args.quant_method,
        act_bits=args.act_bits,
        weight_bits=args.weight_bits,
        lr=args.lr,
        weight_decay=args.weight_decay,
        test_dataset_names=datamodule.test_dataset_names,
    )

    logger = build_logger(args.logger, args.default_root_dir, args.experiment_name)
    checkpoint_callback = ModelCheckpoint(
        monitor="val/psnr",
        mode="max",
        save_top_k=1,
        save_last=True,
        filename="best-{epoch:02d}",
        auto_insert_metric_name=False,
    )

    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=parse_devices(args.devices),
        precision=args.precision,
        default_root_dir=str(args.default_root_dir),
        logger=logger,
        callbacks=[
            checkpoint_callback,
            LearningRateMonitor(logging_interval="epoch"),
        ],
        log_every_n_steps=10,
    )

    ckpt_path = args.checkpoint_path
    if not args.test_only and args.quant_method != "adaround" and args.quant_method != "qdrop":
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        ckpt_path = checkpoint_callback.best_model_path or "best"
    
    lit_model = LitESPCN.load_from_checkpoint(ckpt_path)
    device = lit_model.device
    model = lit_model.model
    if args.quant_method == "adaround":
        model = adaround_quantize_espcn(model, datamodule.train_dataloader(), device=device)
    if args.quant_method == "qdrop":
        model = train_q_drop(model, datamodule.train_dataloader(), device)

    for loader in datamodule.test_dataloader():
        res = test(model, loader, device)
        print(f"For Set{len(loader)}:")
        print(f"PSNR: {res['psnr']}")
        print(f"SSIM: {res['ssim']}")


if __name__ == "__main__":
    main()
