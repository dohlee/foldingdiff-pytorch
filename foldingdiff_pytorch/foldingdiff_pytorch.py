import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl

from torch.optim.lr_scheduler import _LRScheduler
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertEncoder

import foldingdiff_pytorch.loss as loss


class LinearAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, num_annealing_steps, num_total_steps):
        self.num_annealing_steps = num_annealing_steps
        self.num_total_steps = num_total_steps

        super().__init__(optimizer)

    def get_lr(self):
        if self._step_count <= self.num_annealing_steps:
            return [
                base_lr * self._step_count / self.num_annealing_steps for base_lr in self.base_lrs
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

        self.w = nn.Linear(1, 192, bias=False)
        nn.init.normal_(self.w.weight, std=2 * torch.pi)
        self.w.weight.requires_grad = False

    def forward(self, t):
        t = self.w(t.float())
        return torch.cat([torch.sin(t), torch.cos(t)], axis=-1)


class FoldingDiff(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.upscale = nn.Linear(6, 384)
        self.time_embed = RandomFourierFeatures()

        config = BertConfig(
            hidden_size=384,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=384 * 2,
            max_position_embeddings=512,
            hidden_dropout_prob=0.1,
            position_embedding_type="relative_key",
        )
        self.encoder = BertEncoder(config)

        self.head = nn.Sequential(
            nn.Linear(384, 384),
            nn.GELU(),
            nn.LayerNorm(384),
            nn.Linear(384, 6),
        )

        self.criterion = loss.WrappedSmoothL1Loss(beta=0.1 * torch.pi)

    def forward(self, x, t):
        x = self.upscale(x) + self.time_embed(t).unsqueeze(1)

        bert_output = self.encoder(x)
        return self.head(bert_output.last_hidden_state)

    def training_step(self, batch, batch_idx):
        x, t, eps = batch["x"], batch["t"], batch["eps"]

        out = self(x, t)
        loss = self.criterion(out, eps)

        self.log_dict({"train/loss": loss}, prog_bar=True, on_step=True, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, t, eps = batch["x"], batch["t"], batch["eps"]

        out = self(x, t)
        loss = self.criterion(out, eps)

        self.log_dict({"val/loss": loss}, prog_bar=True, on_step=False, on_epoch=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-3)
        scheduler = LinearAnnealingLR(optimizer, num_annealing_steps=1000, num_total_steps=10000)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}


if __name__ == "__main__":
    model = FoldingDiff()

    x = torch.randn(8, 128, 6)
    t = torch.tensor([[0.0]])
    print(model(x, t).shape)
