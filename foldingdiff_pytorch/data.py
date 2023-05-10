import math
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torch import pi as PI

class FoldingDiffDataset(Dataset):
    def __init__(self, T, s=8e-3):
        self.T = T

        # Cosine variance schedule
        t = torch.arange(T + 1)
        f_t = torch.cos((t / T + s) / (1 + s) * math.pi / 2.0).square()
        self.alpha_bar = f_t / f_t[0]
        self.beta = torch.clip(
            1 - self.alpha_bar[1:] / self.alpha_bar[:-1], min=0.001, max=0.999
        )
        self.alpha = 1 - self.beta

        self.alpha_bar_sqrt = torch.sqrt(self.alpha_bar)
        self.one_minus_alpha_bar_sqrt = torch.sqrt(1 - self.alpha_bar)

    def _wrap(self, x):
        return torch.remainder(x + PI, 2 * PI) - PI

    def __len__(self):
        pass

    def __getitem__(self, idx):
        x0 = self.tensor[idx].float()
        t = torch.randint(0, self.T, (1,)).long()
        eps = torch.randn(x0.shape)

        x = x0 * self.alpha_bar_sqrt[t] + eps * self.one_minus_alpha_bar_sqrt[t]
        x = self._wrap(x)

        return {'x': x, 't': t, 'eps': eps}
