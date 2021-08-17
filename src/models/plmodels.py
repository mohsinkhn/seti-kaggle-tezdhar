from typing import Any, List

import numpy as np
import torch
from pytorch_lightning import LightningModule
from scipy.stats import beta as betad
from torchmetrics import Accuracy
from torchmetrics.functional import auroc

from src.models.modules import simple_bb


class LitModel(LightningModule):
    """
    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        base_model: str = "SimpleBB",
        backbone: str = "efficientnet_b0",
        dropout: float = 0.5,
        use_mixup: bool = False,
        mixup_untied: bool = True,
        mixup_alpha: float = 1.0,
        mixup_beta: float = 1.0,
        mixup_ualpha: float = 1.0,
        mixup_ubeta: float = 0.5,
        multiobjective: bool = False,
        **kwargs
    ):
        super().__init__()

        self.save_hyperparameters()
        self.model = getattr(simple_bb, base_model)(hparams=self.hparams)

        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()

    def forward(self, x1, x2):
        if self.hparams.multiobjective:
            return self.model(x1, x2)
        else:
            return self.model(x1)

    def step(self, batch, use_mixup):
        x, y, x2 = batch["im"], batch["label"], batch["im2"]
        x, y1 = self._mixup_data(x, y, use_mixup)
        logits = self.forward(x, x2)
        logits = torch.clamp(logits, -10, 10)
        logits = logits.view(-1)
        y = y.view(-1)
        loss = self.criterion(logits, y1)
        preds = torch.sigmoid(logits)
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, y = self.step(batch, self.hparams["use_mixup"])
        y = y > 0.5
        acc = self.train_accuracy(preds, y.to(torch.long))
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", acc, on_step=True, on_epoch=True, prog_bar=False)
        return {"loss": loss, "preds": preds, "targets": y}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        auc = self._get_auroc(outputs)
        self.log("train/auc", auc)

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch, False)
        acc = self.val_accuracy(preds, targets.to(torch.long))
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=False)
        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        auc = self._get_auroc(outputs)
        self.log("val/auc", auc)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch, False)
        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        preds = torch.cat([o["preds"] for o in outputs], 0).cpu().numpy()
        self.log("outputs", preds)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=10)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def _get_auroc(self, outputs):
        preds = torch.cat([o["preds"] for o in outputs], 0)
        targets = torch.cat([o["targets"] for o in outputs], 0)
        return auroc(preds, targets.to(torch.long), pos_label=1)

    def _mixup_data(self, x, t, use_mixup):
        """Returns mixed inputs, pairs of targets, and lambda"""
        if not use_mixup:
            return x, t

        alpha = self.hparams["mixup_alpha"]
        beta = self.hparams["mixup_beta"]
        ualpha = self.hparams["mixup_ualpha"]
        ubeta = self.hparams["mixup_ubeta"]

        untied = self.hparams["mixup_untied"]

        if alpha > 0:
            dist1 = betad(alpha, beta)
            dist2 = betad(ualpha, ubeta)
            dist3 = betad(ubeta, ualpha)
            rng = np.random.rand()
            lam1 = dist1.ppf(rng)
            if untied:
                lam2 = dist2.ppf(rng)
                lam3 = dist3.ppf(rng)
            else:
                lam2 = lam1
        else:
            lam1 = 1
            lam2 = 1

        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(self.device)
        mixed_x = lam1 * x + (1 - lam1) * x[index, :]
        if untied:
            mixed_y = torch.clamp((lam2 * t) + (1 - lam3) * t[index], 0, 1)  # 0.3, 1 --> 0.5, 1; 0.7, 0 --> 0.5, 1
        else:
            mixed_y = lam2 * t + (1 - lam2) * t[index]
        return mixed_x, mixed_y
