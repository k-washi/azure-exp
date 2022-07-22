import torch
from pytorch_lightning import LightningModule
import torchmetrics
import numpy as np

from timm.scheduler import CosineLRScheduler
from torch import nn
from torch.optim import AdamW
from neptune.new.types import File
from src.yogacls.model.util import load_weight

from src.util.plot import create_fig_of_confmat

# ref: https://docs.neptune.ai/integrations-and-supported-tools/model-training/pytorch-lightning

def _model_setup(model_name, num_classes, ckpt_path=None, head_ignore=False):
    if model_name == "lightvit_tiny":
        from src.yogacls.model.lightvit import load_lightvit_tiny
        model = load_lightvit_tiny(num_classes=int(num_classes))
    else:
        raise NotImplementedError(f"{model_name}は、モデルの準備ができません。")
    if ckpt_path is not None:
        print(ckpt_path)
        model = load_weight(model, ckpt_path, head_ignore)
    return model

class YogaPoseClassifier(LightningModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.ml_cfg = cfg.ml
        
        mcfg = self.ml_cfg.model
        self.model = _model_setup(
            mcfg.name,
            self.ml_cfg.num_classes,
            mcfg.pretrained,
            mcfg.load_head_ignore
        )
        
        self.val_confusion = torchmetrics.classification.ConfusionMatrix(num_classes=self.ml_cfg.num_classes)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.ml_cfg.label_smooth)
        
        
    
    def forward(self, x):
        return self.model(x)
    
    def _step(self, batch):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        predicted = torch.argmax(logits, dim=1)
        acc = (predicted == y).sum() / y.size(0)
        return loss, acc.item(), y, predicted
    
    def training_step(self, batch, batch_idx):
        loss, acc, _, _ = self._step(batch)
        self.log(
            "metrics/batch/train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "metrics/batch/train_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc, targets, preds = self._step(batch)
        self.log(
            "metrics/batch/val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "metrics/batch/val_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        
        self.val_confusion.update(preds, targets)
        return {"val_loss":loss, "val_acc":acc}

    def validation_epoch_end(self, outputs) -> None:
        # Confusion matrix
        conf_mat = self.val_confusion.compute().detach().cpu().numpy().astype(np.int)
        fig = create_fig_of_confmat(conf_mat, class_num=self.ml_cfg.num_classes)
        

        self.logger.experiment["val/confusion_matrix"].log(File.as_image(fig))
        self.val_confusion.confmat *= 0
        
        
    def test_step(self, batch, batch_idx):
        loss, acc, _, _ = self._step(batch)
        self.log(
            "metrics/batch/test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "metrics/batch/test_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return {"test_loss":loss, "test_acc":acc}

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]

        # https://tma15.github.io/blog/2021/09/17/deep-learningbert%E5%AD%A6%E7%BF%92%E6%99%82%E3%81%ABbias%E3%82%84layer-normalization%E3%82%92weight-decay%E3%81%97%E3%81%AA%E3%81%84%E7%90%86%E7%94%B1/#weight-decay%E3%81%AE%E5%AF%BE%E8%B1%A1%E5%A4%96%E3%81%A8%E3%81%AA%E3%82%8B%E3%83%91%E3%83%A9%E3%83%A1%E3%83%BC%E3%82%BF
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.ml_cfg.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.ml_cfg.learning_rate,
            eps=self.ml_cfg.adam_epsilon,
        )
        
        self.scheduler = CosineLRScheduler(
            self.optimizer,
            t_initial=self.ml_cfg.scheduler.t_initial,
            cycle_mul=self.ml_cfg.scheduler.t_mul,
            cycle_decay=self.ml_cfg.scheduler.decay_rate,
            warmup_t=self.ml_cfg.scheduler.warm_up_t,
            warmup_lr_init=self.ml_cfg.scheduler.warm_up_init,
            warmup_prefix=self.ml_cfg.scheduler.warmup_prefix,
        )
        return [self.optimizer], [
            {
                "scheduler": self.scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": self.ml_cfg.scheduler.monitor,
            }
        ]

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        # timm's scheduler need the epoch value
        scheduler.step(epoch=self.current_epoch)


