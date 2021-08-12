from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from pytorch_lightning import LightningDataModule
from sklearn import model_selection
from torch.utils.data import Subset, DataLoader, Dataset

from src.datamodules.datasets import setidata


class SetiDataModule(LightningDataModule):
    """
    LightningDataModule for SETI dataset.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        train_files: List[str] = ["train_labels.csv"],
        train_data_dirs: List[str] = ["./data"],
        train_folders: List[str] = ["train"],
        test_files: List[str] = ["sample_submission.csv"],
        test_data_dirs: List[str] = ["./data"],
        test_folders: List[str] = ["test"],
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        use_data_for_val='first',
        val_fold: int = 0,
        dataset: str = 'SetiDataNp',
        validation_scheme: str = 'KFold',
        validation_kws=None,
        train_transforms=None,
        test_transforms=None,
        **kwargs,
    ):
        super().__init__()
        self.train_files = train_files
        self.train_data_dirs = train_data_dirs
        self.train_folders = train_folders
        self.test_files = test_files
        self.test_data_dirs = test_data_dirs
        self.test_folders = test_folders

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.use_data_for_val = use_data_for_val
        self.validation_scheme = validation_scheme
        self.validation_kws = validation_kws
        if self.validation_kws is None:
            self.validation_kws = {}
        self.val_fold = val_fold

        self.train_transforms = train_transforms
        self.test_transforms = test_transforms

        self.dataset = dataset

        self.train_data: Optional[Dataset] = None
        self.val_data: Optional[Dataset] = None
        self.test_data: Optional[Dataset] = None

    @property
    def num_classes(self) -> int:
        return 1

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        # IMPORTANT; Use only first file for taking out validation
        if self.use_data_for_val == 'first':
            splitter = getattr(model_selection, self.validation_scheme)(**self.validation_kws)
            train_data = pd.read_csv(Path(self.train_data_dirs[0]) / self.train_files[0])
            cv_idx = list(splitter.split(train_data, y=train_data.target))
            train_idx, val_idx = cv_idx[self.val_fold]
            train_indices_list = [train_idx]
            val_indices_list = [val_idx]
        else:
            raise NotImplementedError("Multiple data validation not implemented yet.")

        ds_cls = getattr(setidata, self.dataset)
        self.train_data = ds_cls(
            self.train_files,
            self.train_data_dirs,
            self.train_folders,
            train_indices_list,
            train=True,
            transforms=self.train_transforms,
        )

        self.val_data = ds_cls(
            [self.train_files[0]],
            [self.train_data_dirs[0]],
            [self.train_folders[0]],
            val_indices_list,
            train=True,
            transforms=self.test_transforms,
        )

        self.test_data = ds_cls(
            self.test_files,
            self.test_data_dirs,
            self.test_folders,
            None,
            train=False,
            transforms=self.test_transforms,
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_data,
            batch_size=self.batch_size * 2,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_data,
            batch_size=self.batch_size * 2,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            shuffle=False,
        )


class AEDataModule(LightningDataModule):
    """
    LightningDataModule for Autoencoder.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        train_file: str = "train_labels.csv",
        test_file: str = "sample_submission.csv",
        data_dir: str = "./data/",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        val_fold: int = 0,
        dataset: str = 'SingleImageData',
        validation_scheme: str = 'KFold',
        validation_kws=None,
        train_transforms=None,
        test_transforms=None,
        sum_cells=8,
        **kwargs,
    ):
        super().__init__()
        self.train_file = train_file
        self.test_file = test_file
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.val_fold = val_fold
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms
        self.dataset = dataset
        self.validation_scheme = validation_scheme
        self.validation_kws = validation_kws
        if self.validation_kws is None:
            self.validation_kws = {}

        self.train_data: Optional[Dataset] = None
        self.val_data: Optional[Dataset] = None
        self.test_data: Optional[Dataset] = None

    @property
    def num_classes(self) -> int:
        return 1

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        splitter = getattr(model_selection, self.validation_scheme)(**self.validation_kws)
        train_data = pd.read_csv(Path(self.data_dir) / self.train_file)
        cv_idx = list(splitter.split(train_data, y=train_data.target))
        train_idx, val_idx = cv_idx[self.val_fold]

        ds_cls = getattr(setidata, self.dataset)
        self.train_data = ds_cls(
            self.train_file,
            self.data_dir,
            train_idx,
            train=True,
            transforms=self.train_transforms,
            sum_cells=self.sum_cells,
        )

        self.val_data = ds_cls(
            self.train_file,
            self.data_dir,
            val_idx,
            train=True,
            transforms=self.test_transforms,
            sum_cells=self.sum_cells,
        )

        self.test_data = ds_cls(
            self.test_file,
            self.data_dir,
            None,
            train=False,
            transforms=self.test_transforms,
            sum_cells=self.sum_cells,
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_data,
            batch_size=self.batch_size * 2,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_data,
            batch_size=self.batch_size * 2,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            shuffle=False,
        )
