import math
import torch
import torch.nn as nn
import random
import numpy as np

from torch.utils.data import Dataset, DataLoader
from pathlib import Path

import foldingdiff_pytorch.util as util


class FoldingDiffDataset(Dataset):
    def __init__(self, meta, data_dir, T, mu=None, s=8e-3, max_len=128):
        self.meta = meta
        self.records = meta.to_records()

        self.data_dir = Path(data_dir)
        self.T = T
        self.max_len = max_len

        # Cosine variance schedule
        t = torch.arange(T + 1)
        f_t = torch.cos((t / T + s) / (1 + s) * math.pi / 2.0).square()
        self.alpha_bar = f_t / f_t[0]
        self.beta = torch.clip(1 - self.alpha_bar[1:] / self.alpha_bar[:-1], min=0.001, max=0.999)
        self.alpha = 1 - self.beta

        self.alpha_bar_sqrt = torch.sqrt(self.alpha_bar)
        self.one_minus_alpha_bar_sqrt = torch.sqrt(1 - self.alpha_bar)

        if mu is None:
            feats = []
            for r in self.records:
                angles = np.load(self.data_dir / f'{r.id}.npy')
                angles = np.nan_to_num(angles)
                feats.append(angles)

            feats = np.concatenate(feats, axis=0)
            self.mu = torch.tensor( feats.mean(axis=0) ).float()
        else:
            self.mu = torch.tensor(mu).float()
    
    def get_mu(self):
        return self.mu

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        r = self.records[idx]
        x0 = torch.tensor(np.load(self.data_dir / f"{r.id}.npy")).float()
        x0.nan_to_num_(0.0)
        x0 = util.wrap(x0 - self.mu)

        n_residues = len(x0)
        if n_residues < self.max_len:
            x0 = torch.cat([x0, torch.zeros([self.max_len - n_residues, 6])], axis=0)
        elif n_residues > self.max_len:
            start_idx = random.randint(0, n_residues - self.max_len)
            x0 = x0[start_idx : start_idx + self.max_len]

        t = torch.randint(0, self.T, (1,)).long()
        eps = torch.randn(x0.shape)

        x = x0 * self.alpha_bar_sqrt[t] + eps * self.one_minus_alpha_bar_sqrt[t]
        x = util.wrap(x)

        return {"x": x, "t": t, "eps": eps}


if __name__ == "__main__":
    import pandas as pd

    dataset = FoldingDiffDataset(
        meta=pd.read_csv("../data/meta.csv"),
        data_dir="../data/npy",
        T=1000,
    )

    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    for batch in loader:
        print(batch["x"].shape)
        print(batch["t"].shape)
        print(batch["eps"].shape)
        break
