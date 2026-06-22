from __future__ import annotations

import lightning as pl
from torch.utils.data import DataLoader


class PrebuiltLoaderDataModule(pl.LightningDataModule):
    def __init__(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        super().__init__()
        self._train_loader = train_loader
        self._val_loader = val_loader

    def train_dataloader(self) -> DataLoader:
        return self._train_loader

    def val_dataloader(self) -> DataLoader:
        return self._val_loader


__all__ = ["PrebuiltLoaderDataModule"]
