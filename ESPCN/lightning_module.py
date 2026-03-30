import torch
from torch import nn
import torch.nn.functional as F
import lightning as L
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from model import ESPCN
from quantization import QuantizedESPCN

class LitESPCN(L.LightningModule):
    def __init__(
        self,
        scale: int,
        quant: bool = False,
        signed: bool = False,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        log_images_every_n_epochs: int = 5,

    ):
        super().__init__()
        self.save_hyperparameters()

        if not quant:
            self.model = ESPCN(scale)
        else:
            self.model = QuantizedESPCN(scale, signed)

        self.criterion = nn.MSELoss()

        self.val_psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.val_ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.test_psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.test_ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _shared_step(self, batch):
        lr, hr = batch
        sr = self(lr)
        loss = self.criterion(sr, hr)
        return loss, sr, hr

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._shared_step(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, sr, hr = self._shared_step(batch)

        self.val_psnr.update(sr, hr)
        self.val_ssim.update(sr, hr)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=1)

    def on_validation_epoch_end(self):
        psnr = self.val_psnr.compute()
        ssim = self.val_ssim.compute()

        self.log("val/psnr", psnr, prog_bar=True, sync_dist=False)
        self.log("val/ssim", ssim, prog_bar=True, sync_dist=False)

        if self.hparams.quant:
            self.log("pact/alpha1", self.model.conv1.act_quant.alpha.detach(), on_epoch=True, prog_bar=True, sync_dist=False)
            self.log("pact/alpha2", self.model.conv2.act_quant.alpha.detach(), on_epoch=True, prog_bar=True, sync_dist=False)
            self.log("pact/alpha3", self.model.conv3.act_quant.alpha.detach(), on_epoch=True, prog_bar=True, sync_dist=False)

        self.val_psnr.reset()
        self.val_ssim.reset()

    def test_step(self, batch, batch_idx):
        loss, sr, hr = self._shared_step(batch)

        self.test_psnr.update(sr, hr)
        self.test_ssim.update(sr, hr)

        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=1)

    def on_test_epoch_end(self):
        psnr = self.test_psnr.compute()
        ssim = self.test_ssim.compute()

        self.log("test/psnr", psnr, prog_bar=True, sync_dist=False)
        self.log("test/ssim", ssim, prog_bar=True, sync_dist=False)

        self.test_psnr.reset()
        self.test_ssim.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        return {
            "optimizer": optimizer,
        }
