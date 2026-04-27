import math

import torch
import torch.nn.functional as F
from torch import nn

import lightning as L
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from quantizations.quantization import QuantizedESPCN
from model import ESPCN

class LitESPCN(L.LightningModule):
    def __init__(
        self,
        scale: int,
        signed: bool = True,
        quant_method: str = "pact",
        act_bits: int = 8,
        weight_bits: int = 8,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        test_dataset_names: list[str] | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.quant_method = quant_method
        self.lr = lr
        self.weight_decay = weight_decay
        self.test_dataset_names = test_dataset_names or ["set5", "set14"]
        if quant_method == "wo_quant" or quant_method == "adaround" or quant_method == "qdrop":
            self.model = ESPCN(upscale_factor=scale)
        else:                
            self.model = QuantizedESPCN(
                upscale_factor=scale,
                signed=signed,
                quant_method=quant_method,
                act_bits=act_bits,
                weight_bits=weight_bits,
            )

        self.criterion = nn.MSELoss()

        self.val_psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.val_ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

        self.test_metrics = {
            name: {
                "psnr": PeakSignalNoiseRatio(data_range=1.0).to(self.device),
                "ssim": StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device),
            }
            for name in self.test_dataset_names
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _shared_step(self, batch):
        lr, hr = batch
        sr = self(lr)
        loss = self.criterion(sr, hr)

        return loss, hr, sr

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._shared_step(batch)
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch[0].size(0),
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, sr, hr = self._shared_step(batch)
        self.val_psnr.update(sr, hr)
        self.val_ssim.update(sr, hr)

        self.log(
            "val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=1
        )

    def on_validation_epoch_end(self):
        psnr = self.val_psnr.compute()
        ssim = self.val_ssim.compute()

        self.log("val/psnr", psnr, prog_bar=True, sync_dist=False)
        self.log("val/ssim", ssim, prog_bar=True, sync_dist=False)

        self.val_psnr.reset()
        self.val_ssim.reset()

    def test_step(self, batch, batch_idx, dataloader_idx: int = 0):
        dataset_name = self.test_dataset_names[dataloader_idx]
        _, sr, hr = self._shared_step(batch)

        self.test_metrics[dataset_name]["psnr"].update(sr, hr)
        self.test_metrics[dataset_name]["ssim"].update(sr, hr)

    def on_test_epoch_end(self):
        for dataset_name in self.test_metrics:
            psnr = self.test_metrics[dataset_name]["psnr"].compute()
            ssim = self.test_metrics[dataset_name]["ssim"].compute()

            self.log(f"test/{dataset_name}/psnr", psnr, prog_bar=True, sync_dist=False)
            self.log(f"val/{dataset_name}/ssim", ssim, prog_bar=True, sync_dist=False)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer
