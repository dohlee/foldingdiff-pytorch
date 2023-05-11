import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl

from torch.optim.lr_scheduler import _LRScheduler


class LinearAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, num_annealing_steps, num_total_steps):
        self.num_annealing_steps = num_annealing_steps
        self.num_total_steps = num_total_steps

        super().__init__(optimizer)

    def get_lr(self):
        if self._step_count <= self.num_annealing_steps:
            return [
                base_lr * self._step_count / self.num_annealing_steps
                for base_lr in self.base_lrs
            ]
        else:
            return [
                base_lr
                * (self.num_total_steps - self._step_count)
                / (self.num_total_steps - self.num_annealing_steps)
                for base_lr in self.base_lrs
            ]

class RandomFourierFeatures(nn.Module):
    def __init__(self):
        super().__init__()

        self.w = nn.Linear(1, 192)
        nn.init.normal_(self.time_w.weight, std=2 * torch.pi)
        self.time_embed = nn.Sequential()
    
    def forward(self, t):
        t = self.w(t)
        return torch.cat([ torch.sin(t) , torch.cos(t) ], axis=-1)


class FoldingDiff(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.upscale = nn.Linear(6, 384)
        self.time_embed = RandomFourierFeatures()

        # TODO: Transformer with relative positional encoding here

        self.head = nn.Sequential(
            nn.Linear(384, 384),
            nn.GELU(),
            nn.LayerNorm(384),
            nn.Linear(384, 6),
        )

    def forward(self, x, t):
        x = self.upscale(x) + self.time_embed(t)

        # TODO: forward pass here

        pass

    def training_step(self, batch, batch_idx):
        x = batch["x"]
        t = batch["t"]
        eps = batch["eps"]

        out = self(x, t)
        loss = F.smooth_l1_loss(out, eps, reduction="mean", beta=5.0 / torch.pi)

        self.log_dict(
            {"train/loss": "loss"}, prog_bar=True, on_step=True, on_epoch=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, t, eps = batch["x"], batch["t"], batch["eps"]

        out = self(x, t)
        loss = F.smooth_l1_loss(out, eps, reduction="mean", beta=5.0 / torch.pi)
        self.log_dict({"val/loss": "loss"}, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-3)
        scheduler = LinearAnnealingLR(
            optimizer, num_annealing_steps=1000, num_total_steps=10000
        )

        return [optimizer], [scheduler]
