from typing import Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import KFold
from torch.utils.data import Subset, DataLoader, Dataset
from torchvision.transforms import transforms


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
        train_file: str = "train_labels.csv",
        test_file: str = "test.csv",
        data_dir: str = "./data/",
        val_splits: int = 5,
        val_fold: int = 0,
        kfold_split_seed: int = 12345786,
        power: float = 0.5,
        add_bg: bool = False,
        add_wpb_noise: bool = False,
        specaug: bool = False,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        image_size: list = [256, 64],
        use_mixup: bool = False,
        stacking: str = 'frequency',
        **kwargs,
    ):
        super().__init__()
        self.train_file = train_file
        self.test_file = test_file
        self.data_dir = data_dir
        self.val_splits = val_splits
        self.val_fold = val_fold
        self.kfold_split_seed = kfold_split_seed
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.power = power
        self.add_bg = add_bg
        self.add_wpb_noise = add_wpb_noise
        self.specaug = specaug
        self.image_size = tuple(image_size)
        self.stacking = stacking
        self.use_mixup = use_mixup

        # self.dims is returned when you call datamodule.size()
        self.dims = (3, 256, 256)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self) -> int:
        return 1

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        cvidx = list(
            KFold(
                shuffle=True, random_state=self.kfold_split_seed, n_splits=int(self.val_splits)
            ).split(list(range(50165)))
        )
        train_idx, valid_idx = cvidx[self.val_fold]
        self.data_train = SetiDataAug(
            self.train_file,
            self.data_dir,
            train_idx,
            train=True,
            power=self.power,
            add_bg=self.add_bg,
            add_wpb_noise=self.add_wpb_noise,
            specaug=self.specaug,
            valid=False,
            image_size=self.image_size,
            stacking=self.stacking,
            use_mixup=self.use_mixup
        )
        self.data_val = SetiDataAug(
            self.train_file,
            self.data_dir,
            valid_idx,
            train=True,
            power=self.power,
            add_bg=False,
            add_wpb_noise=False,
            specaug=False,
            valid=True,
            image_size=self.image_size,
            stacking=self.stacking,
            use_mixup=False,
        )

        self.data_test = SetiDataAug(
            self.test_file,
            self.data_dir,
            None,
            train=False,
            power=self.power,
            add_bg=False,
            add_wpb_noise=False,
            specaug=False,
            valid=True,
            image_size=self.image_size,
            stacking=self.stacking,
            use_mixup=False
        )
        # self.data_train = Subset(trainset, cvidx[self.val_fold][0])
        # self.data_val = Subset(trainset, cvidx[self.val_fold][1])

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            shuffle=False,
        )


class SetiDataModule2(LightningDataModule):
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
        train_file: str = "train_labels.csv",
        test_file: str = "test.csv",
        data_dir: str = "./data/",
        val_splits: int = 5,
        val_fold: int = 0,
        kfold_split_seed: int = 12345786,
        power: float = 0.5,
        add_bg: bool = False,
        add_wpb_noise: bool = False,
        specaug: bool = False,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        image_size: list = [256, 64],
        use_mixup: bool = False,
        stacking: str = 'frequency',
        **kwargs,
    ):
        super().__init__()
        self.train_file = train_file
        self.test_file = test_file
        self.data_dir = data_dir
        self.val_splits = val_splits
        self.val_fold = val_fold
        self.kfold_split_seed = kfold_split_seed
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.power = power
        self.add_bg = add_bg
        self.add_wpb_noise = add_wpb_noise
        self.specaug = specaug
        self.image_size = tuple(image_size)
        self.stacking = stacking
        self.use_mixup = use_mixup

        # self.dims is returned when you call datamodule.size()
        self.dims = (3, 786, 786)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self) -> int:
        return 1

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        cvidx = list(
            KFold(
                shuffle=True, random_state=self.kfold_split_seed, n_splits=int(self.val_splits)
            ).split(list(range(50165)))
        )
        train_idx, valid_idx = cvidx[self.val_fold]
        self.data_train = SetiDataAug2(
            self.train_file,
            self.data_dir,
            train_idx,
            train=True,
            power=self.power,
            add_bg=self.add_bg,
            add_wpb_noise=self.add_wpb_noise,
            specaug=self.specaug,
            valid=False,
            image_size=self.image_size,
            use_mixup=self.use_mixup
        )
        self.data_val = SetiDataAug2(
            self.train_file,
            self.data_dir,
            valid_idx,
            train=True,
            power=self.power,
            add_bg=False,
            add_wpb_noise=False,
            specaug=False,
            valid=True,
            image_size=self.image_size,
            use_mixup=False,
        )

        self.data_test = SetiDataAug2(
            self.test_file,
            self.data_dir,
            None,
            train=False,
            power=self.power,
            add_bg=False,
            add_wpb_noise=False,
            specaug=False,
            valid=True,
            image_size=self.image_size,
            use_mixup=False
        )
        # self.data_train = Subset(trainset, cvidx[self.val_fold][0])
        # self.data_val = Subset(trainset, cvidx[self.val_fold][1])

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            shuffle=False,
        )


class SetiDataModule3(LightningDataModule):
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
        train_file: str = "train_labels.csv",
        test_file: str = "test.csv",
        data_dir: str = "./data/",
        val_splits: int = 5,
        val_fold: int = 0,
        kfold_split_seed: int = 12345786,
        power: float = 0.5,
        add_bg: bool = False,
        add_wpb_noise: bool = False,
        specaug: bool = False,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        image_size: list = [256, 64],
        use_mixup: bool = False,
        stacking: str = 'frequency',
        **kwargs,
    ):
        super().__init__()
        self.train_file = train_file
        self.test_file = test_file
        self.data_dir = data_dir
        self.val_splits = val_splits
        self.val_fold = val_fold
        self.kfold_split_seed = kfold_split_seed
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.power = power
        self.add_bg = add_bg
        self.add_wpb_noise = add_wpb_noise
        self.specaug = specaug
        self.image_size = tuple(image_size)
        self.stacking = stacking
        self.use_mixup = use_mixup

        # self.dims is returned when you call datamodule.size()
        self.dims = (3, 786, 786)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self) -> int:
        return 1

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        cvidx = list(
            KFold(
                shuffle=True, random_state=self.kfold_split_seed, n_splits=int(self.val_splits)
            ).split(list(range(50165)))
        )
        train_idx, valid_idx = cvidx[self.val_fold]
        self.data_train = SetiDataAug3(
            self.train_file,
            self.data_dir,
            train_idx,
            train=True,
            power=self.power,
            add_bg=self.add_bg,
            add_wpb_noise=self.add_wpb_noise,
            specaug=self.specaug,
            valid=False,
            image_size=self.image_size,
            use_mixup=self.use_mixup
        )
        self.data_val = SetiDataAug3(
            self.train_file,
            self.data_dir,
            valid_idx,
            train=True,
            power=self.power,
            add_bg=False,
            add_wpb_noise=False,
            specaug=False,
            valid=True,
            image_size=self.image_size,
            use_mixup=False,
        )

        self.data_test = SetiDataAug3(
            self.test_file,
            self.data_dir,
            None,
            train=False,
            power=self.power,
            add_bg=False,
            add_wpb_noise=False,
            specaug=False,
            valid=True,
            image_size=self.image_size,
            use_mixup=False
        )
        # self.data_train = Subset(trainset, cvidx[self.val_fold][0])
        # self.data_val = Subset(trainset, cvidx[self.val_fold][1])

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            shuffle=False,
        )


class SetiDataModule4(LightningDataModule):
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
        train_file: str = "train_labels.csv",
        test_file: str = "test.csv",
        data_dir: str = "./data/",
        val_splits: int = 5,
        dataset: int = 4,
        val_fold: int = 0,
        kfold_split_seed: int = 12345786,
        power: float = 0.5,
        add_bg: bool = False,
        add_wpb_noise: bool = False,
        specaug: bool = False,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        image_size: list = [256, 64],
        use_mixup: bool = False,
        stacking: str = 'frequency',
        freq_substract: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.train_file = train_file
        self.test_file = test_file
        self.data_dir = data_dir
        self.val_splits = val_splits
        self.val_fold = val_fold
        self.kfold_split_seed = kfold_split_seed
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.power = power
        self.add_bg = add_bg
        self.add_wpb_noise = add_wpb_noise
        self.specaug = specaug
        self.image_size = tuple(image_size)
        self.stacking = stacking
        self.use_mixup = use_mixup
        self.freq_substract = freq_substract
        if dataset == 4:
            self.dt = SetiDataAug4
        elif dataset == 5:
            self.dt = SetiDataAug5
        # self.dims is returned when you call datamodule.size()
        self.dims = (3, 786, 786)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self) -> int:
        return 1

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        cvidx = list(
            KFold(
                shuffle=True, random_state=self.kfold_split_seed, n_splits=int(self.val_splits)
            ).split(list(range(50165)))
        )
        train_idx, valid_idx = cvidx[self.val_fold]
        self.data_train = self.dt(
            self.train_file,
            self.data_dir,
            train_idx,
            train=True,
            power=self.power,
            add_bg=self.add_bg,
            add_wpb_noise=self.add_wpb_noise,
            specaug=self.specaug,
            valid=False,
            image_size=self.image_size,
            use_mixup=self.use_mixup
        )
        self.data_val = self.dt(
            self.train_file,
            self.data_dir,
            valid_idx,
            train=True,
            power=self.power,
            add_bg=False,
            add_wpb_noise=False,
            specaug=False,
            valid=True,
            image_size=self.image_size,
            use_mixup=False,
            freq_substract=self.freq_substract
        )

        self.data_test = self.dt(
            self.test_file,
            self.data_dir,
            None,
            train=False,
            power=self.power,
            add_bg=False,
            add_wpb_noise=False,
            specaug=False,
            valid=True,
            image_size=self.image_size,
            use_mixup=False,
            freq_substract=self.freq_substract
        )
        # self.data_train = Subset(trainset, cvidx[self.val_fold][0])
        # self.data_val = Subset(trainset, cvidx[self.val_fold][1])

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            shuffle=False,
        )


class SetiData(Dataset):
    def __init__(self, file_path, data_path, transform=None, train=False):
        super().__init__()
        df = pd.read_csv(Path(data_path) / file_path)
        self.data_path = data_path
        self.transforms = transforms
        self.train = train
        self.ids = df.id.values
        self.labels = None
        if train:
            self.labels = df.target.values.astype(np.float32)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        nid = self.ids[idx]
        if self.train:
            im = np.load(str(Path(self.data_path) / "train" / f"{nid[0]}" / f"{nid}.npy")).astype(
                np.float32
            )
            label = self.labels[idx]
        else:
            im = np.load(str(Path(self.data_path) / "test" / f"{nid[0]}" / f"{nid}.npy")).astype(
                np.float32
            )
            label = -1

        im_target = im[[0, 2, 4], :, :]
        return im_target, label


class SetiDataAug(Dataset):
    def __init__(
        self,
        file_path,
        data_path,
        indices=None,
        train=False,
        add_bg=False,
        add_wpb_noise=False,
        specaug=False,
        add_flip=False,
        stacking='frequency',
        freq_substract=False,
        power=0.5,
        valid=True,
        use_mixup=False,
        image_size=(128, 410)
    ):
        super().__init__()
        df = pd.read_csv(Path(data_path) / file_path)
        if indices is not None:
            df = df.iloc[indices]
        self.data_path = data_path
        self.train = train
        self.add_bg = add_bg
        self.add_wpb_noise = add_wpb_noise
        self.specaug = specaug
        self.add_flip = add_flip
        self.power = power
        self.valid = valid
        self.image_size = image_size
        self.freq_substract = freq_substract
        self.use_mixup = use_mixup
        self.ids = df.id.values
        self.labels = None
        if train:
            self.labels = df.target.values.astype(np.float32)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        nid = self.ids[idx]

        if self.train:
            im = np.load(str(Path(self.data_path) / "train" / f"{nid[0]}" / f"{nid}.npy")).astype(
                np.float32
            )
            label = self.labels[idx]

        else:
            im = np.load(str(Path(self.data_path) / "test" / f"{nid[0]}" / f"{nid}.npy")).astype(
                np.float32
            )
            label = 0.0

        if self.freq_substract:
            im_target = np.vstack((im[0] - np.mean(im[0], axis=0, keepdims=True),
                                    im[2] - np.mean(im[2], axis=0, keepdims=True),
                                    im[4] - np.mean(im[4], axis=0, keepdims=True)))
        else:
            im_target = np.vstack((im[0], im[2], im[4]))

        if not self.valid:
            im_target = self._preprocess(im_target, power=max(0.1, self.power + (np.random.rand() / 2 - 0.25)))
        else:
            im_target = self._preprocess(im_target, power=self.power)

        # if not self.valid and (np.random.rand() < 0.1) and self.use_mixup:
        #     idx2 = np.random.choice(np.where(self.labels == 1)[0], 1)[0]
        #     label2 = self.labels[idx2]
        #     nid2 = self.ids[idx2]
        #     im2 = np.load(str(Path(self.data_path) / "train" / f"{nid2[0]}" / f"{nid2}.npy")).astype(
        #         np.float32
        #     )
        #     if self.stacking == 'frequency':
        #         im_target2 = np.vstack((im2[0], im2[2], im2[4]))
        #     else:
        #         im_target2 = np.stack((im2[0], im2[2], im2[4]))

        #     im_target2 = self._preprocess(im_target2, power=self.power)

        #     alpha = np.random.uniform(0.5, 1.0, 1)[0]  # (np.random.beta(1, 1), 0.3, 0.7)
        #     im_target = (1 - alpha) * im_target + alpha * im_target2
        #     label = 1.0
            # label = min(1, label)

        if self.specaug:
            if np.random.rand() > 0.5:
                h, w = im_target.shape[1:]
                fw = int(0.05 * h)
                fi = int(np.random.rand() * (h - fw))
                im_target[:, fi: fi + fw, :] = 0

        if self.add_flip:
            if np.random.rand() > 0.25:
                im_target = im_target[:, :, ::-1]

            if np.random.rand() > 0.25:
                im_target = im_target[:, ::-1, :]

        return {'t1': im_target, 'label': label}

    def _preprocess(self, im, power):
        tmp = (im - im.min()) / (im.max() - im.min())
        tmp = tmp ** power
        tmp *= 255
        tmp = tmp.astype(np.uint8)
        if self.stacking == 'frequency':
            tmp = cv2.resize(tmp, self.image_size, interpolation=cv2.INTER_AREA)
        else:
            tmp = cv2.resize(np.moveaxis(tmp, 0, 2), self.image_size, interpolation=cv2.INTER_CUBIC)
            tmp = np.moveaxis(tmp, 2, 0)
        tmp = tmp / 255.0
        tmp = tmp.astype(np.float32)
        if self.stacking == 'frequency':
            tmp = np.stack((tmp, tmp, tmp))
        # return tmp
        return (tmp - 0.5) / 0.5


class SetiDataAug2(Dataset):
    def __init__(
        self,
        file_path,
        data_path,
        indices=None,
        train=False,
        add_bg=False,
        add_wpb_noise=False,
        specaug=False,
        add_flip=False,
        power=0.5,
        valid=True,
        use_mixup=False,
        image_size=(128, 410)
    ):
        super().__init__()
        df = pd.read_csv(Path(data_path) / file_path)
        if indices is not None:
            df = df.iloc[indices]
        self.data_path = data_path
        self.train = train
        self.add_bg = add_bg
        self.add_wpb_noise = add_wpb_noise
        self.specaug = specaug
        self.add_flip = add_flip
        self.power = power
        self.valid = valid
        self.image_size = image_size
        self.use_mixup = use_mixup
        self.ids = df.id.values
        self.labels = None
        if train:
            self.labels = df.target.values.astype(np.float32)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        nid = self.ids[idx]

        if self.train:
            im = np.load(str(Path(self.data_path) / "train" / f"{nid[0]}" / f"{nid}.npy")).astype(
                np.float32
            )
            label = self.labels[idx]

        else:
            im = np.load(str(Path(self.data_path) / "test" / f"{nid[0]}" / f"{nid}.npy")).astype(
                np.float32
            )
            label = 0.0

        im_target1 = np.vstack((im[0], im[2], im[4]))
        im_target2 = np.vstack((im[0] - np.mean(im[0], axis=0, keepdims=True),
                                im[2] - np.mean(im[2], axis=0, keepdims=True),
                                im[4] - np.mean(im[4], axis=0, keepdims=True)))
        im_background1 = np.vstack((im[1] - np.mean(im[1], axis=0, keepdims=True),
                                im[3] - np.mean(im[3], axis=0, keepdims=True),
                                im[5] - np.mean(im[5], axis=0, keepdims=True)))

        if not self.valid:
            new_power = max(0.1, self.power + (np.random.rand() / 2 - 0.25))
            im_target = self._preprocess(im_target1, im_background1, im_target2, power=new_power)

        else:
            im_target = self._preprocess(im_target1, im_background1, im_target2, power=self.power)

        if self.specaug and not self.valid:
            if np.random.rand() > 0.5:
                h, w = im_target.shape[1:]
                fw = int(0.02 * h)
                fi = int(np.random.rand() * (h - fw))
                im_target[:, fi: fi + fw, :] = 0

        if self.add_flip and not self.valid:
            if np.random.rand() > 0.25:
                im_target = im_target[:, :, ::-1]

            if np.random.rand() > 0.25:
                im_target = im_target[:, ::-1, :]

        return {'t1': im_target, 'label': label}

    def _preprocess(self, im, im2, im3, power):
        def resize(im):
            a = Image.fromarray(im.astype(np.float32))
            return np.asarray(a.resize(self.image_size, Image.LANCZOS))

        im, im2, im3 = resize(im), resize(im2), resize(im3)
        im_del, im_min = im.min(), im.max() - im.min()
        im2_del, im2_min = im2.min(), im2.max() - im2.min()
        im = ((im - im_min) / im_del)**power * im_del + im_min
        im2 = ((im2 - im2_min) / im2_del)**power * im2_del + im2_min
        return np.stack((im, im2, im3))


class SetiDataAug3(Dataset):
    def __init__(
        self,
        file_path,
        data_path,
        indices=None,
        train=False,
        add_bg=False,
        add_wpb_noise=False,
        specaug=False,
        add_flip=False,
        power=0.5,
        valid=True,
        use_mixup=False,
        image_size=(128, 410)
    ):
        super().__init__()
        df = pd.read_csv(Path(data_path) / file_path)
        if indices is not None:
            df = df.iloc[indices]
        self.data_path = data_path
        self.train = train
        self.add_bg = add_bg
        self.add_wpb_noise = add_wpb_noise
        self.specaug = specaug
        self.add_flip = add_flip
        self.power = power
        self.valid = valid
        self.image_size = image_size
        self.use_mixup = use_mixup
        self.ids = df.id.values
        self.labels = None
        if train:
            self.labels = df.target.values.astype(np.float32)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        nid = self.ids[idx]

        if self.train:
            im = np.load(str(Path(self.data_path) / "train" / f"{nid[0]}" / f"{nid}.npy")).astype(
                np.float32
            )
            label = self.labels[idx]

        else:
            im = np.load(str(Path(self.data_path) / "test" / f"{nid[0]}" / f"{nid}.npy")).astype(
                np.float32
            )
            label = 0.0

        im_target1 = np.vstack((im[0], im[2], im[4]))
        im_background1 = np.vstack((im[1], im[3], im[5]))
        im_target2 = np.vstack((im[0] - np.mean(im[0], axis=0, keepdims=True),
                                im[2] - np.mean(im[2], axis=0, keepdims=True),
                                im[4] - np.mean(im[3], axis=0, keepdims=True)))

        if not self.valid:
            new_power = max(0.1, self.power + (np.random.rand() / 2 - 0.25))
            im_target1 = self._preprocess(im_target1, power=new_power)
            im_background1 = self._preprocess(im_background1, power=new_power)

        else:
            im_target1 = self._preprocess(im_target1, power=self.power)
            im_background1 = self._preprocess(im_background1, power=self.power)

        im_target2 = self._preprocess(im_target2, power=1.0)

        if self.add_flip and not self.valid:
            if np.random.rand() > 0.25:
                im_target1 = im_target1[:, :, ::-1]
                im_target2 = im_target2[:, :, ::-1]
                im_background1 = im_background1[:, :, ::-1]

            if np.random.rand() > 0.25:
                im_target1 = im_target1[:, ::-1, :]
                im_target2 = im_target2[:, ::-1, :]
                im_background1 = im_background1[:, ::-1, :]

        return {'t1': im_target1, 't2': im_target2, 'b1': im_background1, 'label': label}

    def _preprocess(self, im, power):
        def resize(im):
            a = Image.fromarray(im.astype(np.float32), mode='F')
            return np.asarray(a.resize(self.image_size, Image.LANCZOS))

        im = resize(im)
        im_del, im_min = im.min(), im.max() - im.min()
        im = ((im - im_min) / im_del)**power * im_del + im_min
        return np.stack((im, im, im))


class SetiDataAug4(Dataset):
    def __init__(
        self,
        file_path,
        data_path,
        indices=None,
        train=False,
        add_bg=False,
        add_wpb_noise=False,
        specaug=False,
        add_flip=False,
        stacking='frequency',
        freq_substract=False,
        power=0.5,
        valid=True,
        use_mixup=False,
        image_size=(128, 410)
    ):
        super().__init__()
        df = pd.read_csv(Path(data_path) / file_path)
        if indices is not None:
            df = df.iloc[indices]
        self.data_path = data_path
        self.train = train
        self.add_bg = add_bg
        self.add_wpb_noise = add_wpb_noise
        self.specaug = specaug
        self.add_flip = add_flip
        self.power = power
        self.valid = valid
        self.image_size = image_size
        self.freq_substract = freq_substract
        self.use_mixup = use_mixup
        self.ids = df.id.values
        self.labels = None
        if train:
            self.labels = df.target.values.astype(np.float32)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        nid = self.ids[idx]

        if self.train:
            im = np.load(str(Path(self.data_path) / "train" / f"{nid[0]}" / f"{nid}.npy")).astype(
                np.float32
            )
            label = self.labels[idx]

        else:
            im = np.load(str(Path(self.data_path) / "test" / f"{nid[0]}" / f"{nid}.npy")).astype(
                np.float32
            )
            label = 0.0

        if self.freq_substract:
            im_target = np.vstack((im[0] - np.mean(im[0], axis=0, keepdims=True),
                                   im[2] - np.mean(im[2], axis=0, keepdims=True),
                                   im[4] - np.mean(im[4], axis=0, keepdims=True)))
        else:
            im_target = np.vstack((im[0], im[2], im[4]))

        if not self.valid:
            im_target = self._preprocess(im_target, power=max(0.1, self.power + (np.random.rand() / 2 - 0.25)))
        else:
            im_target = self._preprocess(im_target, power=self.power)

        if self.specaug and not self.valid:
            if np.random.rand() > 0.5:
                h, w = im_target.shape[1:]
                fw = int(0.05 * h)
                fi = int(np.random.rand() * (h - fw))
                im_target[:, fi: fi + fw, :] = 0

        if self.add_flip and not self.valid:
            if np.random.rand() > 0.25:
                im_target = im_target[:, :, ::-1]

            if np.random.rand() > 0.25:
                im_target = im_target[:, ::-1, :]

        return {'t1': im_target, 'label': label}

    def _preprocess(self, im, power):
        signs = np.sign(im)
        im = np.abs(im) ** power
        im = signs * im
        tmp = np.stack((im, im, im))
        return tmp


class SetiDataAug5(Dataset):
    def __init__(
        self,
        file_path,
        data_path,
        indices=None,
        train=False,
        add_bg=False,
        add_wpb_noise=False,
        specaug=False,
        add_flip=False,
        stacking='frequency',
        freq_substract=False,
        power=0.5,
        valid=True,
        use_mixup=False,
        image_size=(128, 410)
    ):
        super().__init__()
        df = pd.read_csv(Path(data_path) / file_path)
        if indices is not None:
            df = df.iloc[indices]
        self.data_path = data_path
        self.train = train
        self.add_bg = add_bg
        self.add_wpb_noise = add_wpb_noise
        self.specaug = specaug
        self.add_flip = add_flip
        self.power = power
        self.valid = valid
        self.image_size = image_size
        self.freq_substract = freq_substract
        self.use_mixup = use_mixup
        self.ids = df.id.values
        self.labels = None
        if train:
            self.labels = df.target.values.astype(np.float32)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        nid = self.ids[idx]

        if self.train:
            im = np.load(str(Path(self.data_path) / "train" / f"{nid[0]}" / f"{nid}.npy")).astype(
                np.float32
            )
            label = self.labels[idx]

        else:
            im = np.load(str(Path(self.data_path) / "test" / f"{nid[0]}" / f"{nid}.npy")).astype(
                np.float32
            )
            label = 0.0

        if self.freq_substract:
            im_target1 = im[0] - np.mean(im[0], axis=0, keepdims=True)
            im_target2 = im[2] - np.mean(im[2], axis=0, keepdims=True)
            im_target3 = im[4] - np.mean(im[4], axis=0, keepdims=True)
        else:
            im_target1 = im[0]
            im_target2 = im[2]
            im_target3 = im[4]

        if not self.valid:
            new_power = max(0.1, self.power + (np.random.rand() / 2 - 0.25))
        else:
            new_power = self.power
        im_target1 = self._preprocess(im_target1, power=new_power)
        im_target2 = self._preprocess(im_target2, power=new_power)
        im_target3 = self._preprocess(im_target3, power=new_power)

        if self.specaug and not self.valid:
            if np.random.rand() > 0.5:
                h, w = im_target1.shape[1:]
                fw = int(0.05 * h)
                fi = int(np.random.rand() * (h - fw))
                im_target1[:, fi: fi + fw, :] = 0
                fi = int(np.random.rand() * (h - fw))
                im_target2[:, fi: fi + fw, :] = 0
                fi = int(np.random.rand() * (h - fw))
                im_target3[:, fi: fi + fw, :] = 0

        if self.add_flip and not self.valid:
            if np.random.rand() > 0.25:
                im_target1 = im_target1[:, :, ::-1]
                im_target2 = im_target2[:, :, ::-1]
                im_target3 = im_target3[:, :, ::-1]

            if np.random.rand() > 0.25:
                im_target1 = im_target1[:, ::-1, :]
                im_target2 = im_target2[:, ::-1, :]
                im_target3 = im_target3[:, ::-1, :]

        return {'t1': np.concatenate((im_target1, im_target2, im_target3), axis=0), 't2': im_target2, 't3': im_target3, 'label': label}

    def _preprocess(self, im, power):
        signs = np.sign(im)
        im = np.abs(im) ** power
        im = signs * im
        tmp = np.stack((im, im, im))
        return tmp