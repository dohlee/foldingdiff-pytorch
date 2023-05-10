import torch
import torch.nn as nn
import wandb

from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from .data import FoldingDiffDataset
from .foldingdiff_pytorch import FoldingDiff

T = 1000
bsz = 16  # should be clarified

wandb.init()

train_set = FoldingDiffDataset(T=T)
val_set = FoldingDiffDataset(T=T)

model = FoldingDiff()

train_dataloader = DataLoader(
    train_set,
    batch_size=bsz,
    shuffle=True,
    drop_last=True,
    num_workers=4,
    pin_memory=True,
)
val_dataloader = DataLoader(
    val_set,
    batch_size=bsz,
    shuffle=False,
    drop_last=False,
    num_workers=4,
)

trainer = Trainer()
trainer.fit(
    model,
    train_dataloader,
    val_dataloader,
)