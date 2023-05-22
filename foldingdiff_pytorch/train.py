import argparse
import torch
import torch.nn as nn
import pandas as pd
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from .data import FoldingDiffDataset
from .foldingdiff_pytorch import FoldingDiff


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", type=str, default="../data/meta.csv")
    parser.add_argument("--data_dir", type=str, default="../data/npy")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--timesteps", type=int, default=1000)
    return parser.parse_args()


def main():
    wandb_logger = pl.loggers.WandbLogger(
        project="foldingdiff-pytorch", entity="dohlee"
    )

    args = parse_arguments()
    wandb_logger.log_hyperparams(args)

    meta = pd.read_csv(args.meta).sample(frac=1.0, random_state=42)
    N = len(meta)

    train_meta, val_meta = meta.iloc[: int(0.8 * N)], meta.iloc[int(0.8 * N) :]

    train_set = FoldingDiffDataset(
        meta=train_meta, data_dir=args.data_dir, T=args.timesteps
    )
    mu = train_set.get_mu()
    wandb_logger.log_hyperparams({"mu": mu})

    val_set = FoldingDiffDataset(
        meta=val_meta, data_dir=args.data_dir, T=args.timesteps, mu=mu
    )

    model = FoldingDiff()

    train_dataloader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=4,
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=1500,
        logger=wandb_logger,
        callbacks=[
            pl.callbacks.LearningRateMonitor(logging_interval="step"),
        ],
    )
    trainer.fit(
        model,
        train_dataloader,
        val_dataloader,
    )


if __name__ == "__main__":
    main()
