import logging
from data.dataloader_csi import SolarDataset
_LOG = logging.getLogger(__name__)
_LOG.setLevel(logging.DEBUG)
from typing import Any, Dict, Optional
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import torch
from pathlib import Path
import pandas as pd
class SIS_DataModule(LightningDataModule):
    def __init__(
        self,
        dataset: Dict[str, Any],
        batch_size: int = 16,
        num_workers: int = 4,
        pin_memory: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.dataset_config = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def _validate_config(self):
        required_keys = ['data_path', 'years', 'input_len', 'pred_len']
        for key in required_keys:
            if key not in self.dataset_config:
                raise ValueError(f"Missing required config key: {key}")
        data_path = Path(self.dataset_config['data_path'])
        if not data_path.exists():
            raise FileNotFoundError(f"Data path does not exist: {data_path}")
        _LOG.info(f"Data configuration validated successfully")
        _LOG.debug(f"Dataset config: {self.dataset_config}")

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`."""
        _LOG.info(f"Setting up data module for stage: {stage}")
        try:
            if not self.data_train and not self.data_val and not self.data_test:
                _LOG.debug("Initializing datasets...")
                self.data_train = SolarDataset(
                    **self.dataset_config,
                    mode="train",
                )
                _LOG.info(f"Training dataset initialized with {len(self.data_train)} samples")
                self.data_val = SolarDataset(
                    **self.dataset_config,
                    mode="val",
                )
                _LOG.info(f"Validation dataset initialized with {len(self.data_val)} samples")
                self.data_test = SolarDataset(
                    **self.dataset_config,
                    mode="test",
                )
                _LOG.info(f"Test dataset initialized with {len(self.data_test)} samples")
        except Exception as e:
            _LOG.error(f"Error setting up datasets: {str(e)}")
            raise

    def train_dataloader(self):
        if self.data_train is None:
            _LOG.info("Train dataset not initialized, calling setup()")
            self.setup(stage='fit')

        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            drop_last=True
        )

    def val_dataloader(self):
        if self.data_val is None:
            _LOG.info("Validation dataset not initialized, calling setup()")
            self.setup(stage='fit')

        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=True
        )

    def test_dataloader(self):
        if self.data_test is None:
            _LOG.info("Test dataset not initialized, calling setup()")
            self.setup(stage='test')

        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    @property
    def num_samples(self):
        """Returns the number of samples in all datasets."""
        return {
            'train': self.num_train_samples,
            'val': self.num_val_samples,
            'test': self.num_test_samples
        }

    @property
    def num_train_samples(self):
        if self.data_train is None:
            self.setup(stage='fit')
        return len(self.data_train) if self.data_train else 0

    @property
    def num_val_samples(self):
        if self.data_val is None:
            self.setup(stage='fit')
        return len(self.data_val) if self.data_val else 0

    @property
    def num_test_samples(self):
        if self.data_test is None:
            self.setup(stage='test')
        return len(self.data_test) if self.data_test else 0


def verify_dataloader_shapes(config):
    # 创建 datamodule
    datamodule = SIS_DataModule(
        dataset=config,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory']
    )
    datamodule.setup()
    print(f"Train samples: {datamodule.num_train_samples}")
    print(f"Validation samples: {datamodule.num_val_samples}")

    # 取一个 train batch 看 shape
    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))

    print("\n[Train batch shapes]")
    for k, v in batch.items():
        if hasattr(v, "shape"):
            print(f"{k:15s}: {tuple(v.shape)}")

    # 取一个 val batch 看 shape（如果你有 val_dataloader）
    if hasattr(datamodule, "val_dataloader"):
        val_loader = datamodule.val_dataloader()
        val_batch = next(iter(val_loader))

        print("\n[Val batch shapes]")
        for k, v in val_batch.items():
            if hasattr(v, "shape"):
                print(f"{k:15s}: {tuple(v.shape)}")



if __name__ == '__main__':
    dataset_config = {
        "data_path": "/home/joty/code/flow_matching/data",
        "years": {
            "train": [2017, 2018, 2019, 2020],
            "val": [2021],
            "test": [2022]
        },
        "input_len": 8,
        "pred_len": 1,
        "stride": 1,
        "use_possible_starts": True,
        "batch_size": 16,
        "num_workers": 10,
        "pin_memory": True
    }

    verify_dataloader_shapes(dataset_config)