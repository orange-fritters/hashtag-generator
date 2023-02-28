import torch
import torch.nn as nn
from torch.nn import functional as nnf
from .dataloader import HarrisonDataset
from .model import TagModel
from torch.utils.data import DataLoader
import pytorch_lightning as pl


class HarrisonModel(pl.LightningModule):
    def __init__(
            self,
            data_path: str,
            prefix_length: int,
            dim_clip: int = 512,
            dim_embedding: int = 512,
            num_layers: int = 4,
            dim_feedforward: int = 512,
            nhead: int = 4,
            dropout: float = 0.1,
            lr: float = 1e-4,
            batch_size: int = 16,
            num_workers: int = 4,
            normalize_prefix: bool = False,
    ):
        super(HarrisonModel, self).__init__()
        self.save_hyperparameters()
        self.dataset = HarrisonDataset(
            data_path=data_path,
            prefix_length=prefix_length,
            normalize_prefix=normalize_prefix,
        )
        self.model = TagModel(
            dim_clip=dim_clip,
            dim_embedding=dim_embedding,
            prefix_length=prefix_length,
            num_layers=num_layers,
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        tokens, mask, embed = batch
        out = self.model(embed)
        loss = nn.CrossEntropyLoss()(out, tokens)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        tokens, mask, embed = batch
        out = self.model(embed)
        loss = nn.CrossEntropyLoss()(out, tokens)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.hparams.lr,
        )
        return optimizer

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            drop_last=True,
        )
