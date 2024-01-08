import pytorch_lightning as pl
import torchmetrics
from torch import nn, optim

from .unet import UNet


class LitUNet(pl.LightningModule):
    def __init__(self, n_channels, n_classes, lr=None):
        super().__init__()

        self.model = UNet(n_channels, n_classes)
        self.lr = lr or 0.1

        self.train_acc = torchmetrics.Accuracy("binary")
        self.val_acc = torchmetrics.Accuracy("binary")
        self.val_iou = torchmetrics.JaccardIndex("binary")

        self.save_hyperparameters()

    def forward(self, x):
        return self.model.forward(x)

    def training_step(self, batch, batch_idx):
        img, msk = batch
        logits = self.model(img)

        loss = nn.functional.binary_cross_entropy_with_logits(logits, msk)

        self.train_acc(logits, msk.long())

        self.log("train/loss", loss, prog_bar=True)
        self.log("train/acc", self.train_acc, on_epoch=True, on_step=False)

        return loss

    def validation_step(self, batch, batch_idx):
        img, msk = batch
        logits = self.model(img)

        val_loss = nn.functional.binary_cross_entropy_with_logits(logits, msk)

        self.val_acc(logits, msk.long())
        self.val_iou(logits, msk.long())

        self.log("val/loss", val_loss, prog_bar=True)
        self.log("val/acc", self.val_acc, on_epoch=True)
        self.log("val/iou", self.val_iou, on_epoch=True)

        return val_loss

    def configure_optimizers(self):
        optimizer = optim.SGD(
            self.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-4
        )
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=20, verbose=True
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": lr_scheduler, "monitor": "val/loss"},
        }
