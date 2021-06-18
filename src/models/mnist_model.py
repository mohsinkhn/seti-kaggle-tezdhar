from typing import Any, List

import numpy as np
import torch
from pytorch_lightning import LightningModule
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
        base_model: str = 'SimpleBB',
        backbone: str = "efficientnet_b0",
        dropout: float = 0.5,
        lr: float = 0.001,
        weight_decay: float = 0.0005,
        epochs: int = 10,
        batch_size: int = 16,
        train_samples: int = 40000,
        use_mixup: bool = False,
        mixup_union: bool = False,
        mixup_alpha: float = 1.0,
        **kwargs
    ):
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()

        self.model = getattr(simple_bb, base_model)(hparams=self.hparams)

        # loss function
        self.criterion = torch.nn.BCEWithLogitsLoss()

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()  # pos_label=1, compute_on_step=False)
        # self.test_accuracy = AUROC(compute_on_step=False)

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def step(self, batch: Any, use_mixup: bool):
        x, y = batch['t1'], batch['label']
        x1, y1, y2, alpha = mixup_data(use_mixup, x, y, self.hparams['mixup_alpha'], self.hparams['mixup_union'], self.device)
        logits = self.forward(x1)
        logits = logits.view(-1)  # [::2]
        y = y.view(-1)  # [::2]
        criterion = get_criterion(use_mixup, self.criterion)
        loss = criterion(logits, y1, y2, alpha)
        preds = torch.sigmoid(logits)
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch, self.hparams['use_mixup'])

        # log train metrics
        acc = self.train_accuracy(preds, targets.to(torch.long))
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", acc, on_step=True, on_epoch=True, prog_bar=False)

        # we can return here dict with any tensors
        # and then read it in some callback or in training_epoch_end() below
        # remember to always return loss from training_step, or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        auc = self._get_auroc(outputs)
        self.log("train/auc", auc)

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch, False)

        # log val metrics
        acc = self.val_accuracy(preds, targets.to(torch.long))
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=False)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        auc = self._get_auroc(outputs)
        self.log("val_auc", auc)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch, False)
        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        preds = torch.cat([o["preds"] for o in outputs], 0).cpu().numpy()

        self.log('outputs', preds)

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = torch.optim.AdamW(
            params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.lr,
            pct_start=0.1,
            div_factor=100,
            final_div_factor=100,
            epochs=self.hparams.epochs,
            steps_per_epoch=self.hparams.train_samples // self.hparams.batch_size,
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def _get_auroc(self, outputs):
        preds = torch.cat([o["preds"] for o in outputs], 0)
        targets = torch.cat([o["targets"] for o in outputs], 0)
        return auroc(preds, targets.to(torch.long), pos_label=1)


class LitModel2(LightningModule):
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
        base_model: str = 'SimpleBB',
        backbone: str = "efficientnet_b0",
        dropout: float = 0.0,
        lr: float = 0.001,
        weight_decay: float = 0.0005,
        epochs: int = 10,
        batch_size: int = 16,
        train_samples: int = 40000,
        use_mixup: bool = False,
        mixup_union: bool = False,
        mixup_alpha: float = 1.0,
        **kwargs
    ):
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()

        self.model = getattr(simple_bb, base_model)(hparams=self.hparams)

        # loss function
        self.criterion = torch.nn.BCEWithLogitsLoss()

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()  # pos_label=1, compute_on_step=False)
        # self.test_accuracy = AUROC(compute_on_step=False)

    def forward(self, x):
        return self.model(x)

    def step(self, batch: Any, use_mixup: bool):
        t1, t2, b1, y = batch['t1'], batch['t2'], batch['b1'], batch['label']
        x1, x2, x3, y1, y2, alpha = mixup_data2(use_mixup, t1, t2, b1, y, self.hparams['mixup_alpha'],
                                                self.hparams['mixup_union'], self.device)
        logits1 = self.forward(x1)
        logits2 = self.forward(x2)
        logits1 = logits1.view(-1)
        logits2 = logits2.view(-1)

        y = y.view(-1)  # [::2]
        criterion = get_criterion(use_mixup, self.criterion)
        loss1 = criterion(logits1, y1, y2, alpha)
        loss2 = criterion(logits2, y1, y2, alpha)
        loss = 0.3 * loss1 + 0.7 * loss2
        preds = torch.sigmoid(logits2) * 0.7 + torch.sigmoid(logits1) * 0.3
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch, self.hparams['use_mixup'])

        # log train metrics
        acc = self.train_accuracy(preds, targets.to(torch.long))
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", acc, on_step=True, on_epoch=True, prog_bar=False)

        # we can return here dict with any tensors
        # and then read it in some callback or in training_epoch_end() below
        # remember to always return loss from training_step, or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        auc = self._get_auroc(outputs)
        self.log("train/auc", auc)

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch, False)

        # log val metrics
        acc = self.val_accuracy(preds, targets.to(torch.long))
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=False)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        auc = self._get_auroc(outputs)
        self.log("val_auc", auc)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch, False)

        # log test metrics
        # acc = self.test_accuracy(preds, targets)
        # self.log("test/loss", loss, on_step=False, on_epoch=True)
        # self.log("test/acc", acc, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": preds}

    # def test_epoch_end(self, outputs: List[Any]):
    #    return outputs


    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = torch.optim.AdamW(
            params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.lr,
            pct_start=0.1,
            div_factor=100,
            final_div_factor=100,
            epochs=self.hparams.epochs,
            steps_per_epoch=self.hparams.train_samples // self.hparams.batch_size,
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def _get_auroc(self, outputs):
        preds = torch.cat([o["preds"] for o in outputs], 0)
        targets = torch.cat([o["targets"] for o in outputs], 0)
        return auroc(preds, targets.to(torch.long), pos_label=1)


class LitModel3(LightningModule):
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
        base_model: str = 'Res3inpBB',
        backbone: str = "efficientnet_b0",
        dropout: float = 0.0,
        lr: float = 0.001,
        weight_decay: float = 0.0005,
        epochs: int = 10,
        batch_size: int = 16,
        train_samples: int = 40000,
        use_mixup: bool = False,
        mixup_union: bool = False,
        mixup_alpha: float = 1.0,
        **kwargs
    ):
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()

        self.model = getattr(simple_bb, base_model)(hparams=self.hparams)

        # loss function
        self.criterion = torch.nn.BCEWithLogitsLoss()

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()  # pos_label=1, compute_on_step=False)
        # self.test_accuracy = AUROC(compute_on_step=False)

    def forward(self, x1, x2, x3):
        return self.model(x1, x2, x3)

    def step(self, batch: Any, use_mixup: bool):
        t1, t2, t3, y = batch['t1'], batch['t2'], batch['t3'], batch['label']
        x1, x2, x3, y1, y2, alpha = mixup_data2(use_mixup, t1, t2, t3, y, self.hparams['mixup_alpha'],
                                                self.hparams['mixup_union'], self.device)
        logits1 = self.forward(x1, x2, x3)
        logits1 = logits1.view(-1)
        y = y.view(-1)  # [::2]
        criterion = get_criterion(use_mixup, self.criterion)
        loss = criterion(logits1, y1, y2, alpha)
        preds = torch.sigmoid(logits1)
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch, self.hparams['use_mixup'])

        # log train metrics
        acc = self.train_accuracy(preds, targets.to(torch.long))
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", acc, on_step=True, on_epoch=True, prog_bar=False)

        # we can return here dict with any tensors
        # and then read it in some callback or in training_epoch_end() below
        # remember to always return loss from training_step, or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        auc = self._get_auroc(outputs)
        self.log("train/auc", auc)

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch, False)

        # log val metrics
        acc = self.val_accuracy(preds, targets.to(torch.long))
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=False)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        auc = self._get_auroc(outputs)
        self.log("val_auc", auc)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch, False)

        # log test metrics
        # acc = self.test_accuracy(preds, targets)
        # self.log("test/loss", loss, on_step=False, on_epoch=True)
        # self.log("test/acc", acc, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": preds}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = torch.optim.AdamW(
            params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.lr,
            pct_start=0.1,
            div_factor=100,
            final_div_factor=100,
            epochs=self.hparams.epochs,
            steps_per_epoch=self.hparams.train_samples // self.hparams.batch_size,
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def _get_auroc(self, outputs):
        preds = torch.cat([o["preds"] for o in outputs], 0)
        targets = torch.cat([o["targets"] for o in outputs], 0)
        return auroc(preds, targets.to(torch.long), pos_label=1)


def mixup_data(use_mixup, x, t, alpha=1.0, mixup_union=False, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if not use_mixup:
        return x, t, None, None

    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    t_a, t_b = t, t[index]
    if mixup_union:
        t_a = t_b = torch.where(lam * t_a + (1 - lam) * t_b > 0.3, 1, 0).to(torch.float)
    return mixed_x, t_a, t_b, lam


def mixup_data2(use_mixup, x1, x2, x3, t, alpha=1.0, mixup_union=False, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if not use_mixup:
        return x1, x2, x3, t, None, None

    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x1.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x1 = lam * x1 + (1 - lam) * x1[index, :]
    mixed_x2 = lam * x2 + (1 - lam) * x2[index, :]
    mixed_x3 = lam * x3 + (1 - lam) * x3[index, :]

    t_a, t_b = t, t[index]
    if mixup_union:
        t_a = t_b = torch.where(lam * t_a + (1 - lam) * t_b > 0.3, 1, 0).to(torch.float)
    return mixed_x1, mixed_x2, mixed_x3, t_a, t_b, lam


def get_criterion(use_mixup, loss_func):

    def mixup_criterion(pred, t_a, t_b, lam):
        return lam * loss_func(pred, t_a) + (1 - lam) * loss_func(pred, t_b)

    def single_criterion(pred, t_a, t_b, lam):
        return loss_func(pred, t_a)

    if use_mixup:
        return mixup_criterion
    else:
        return single_criterion