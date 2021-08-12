from typing import Any

import albumentations
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from src.utils import utils

log = utils.get_logger(__name__)


class SetiDataNp(Dataset):
    def __init__(
        self,
        file_paths: list = None,
        data_paths: list = None,
        image_folders: list = None,
        indices_list: list = None,
        train: bool = False,
        transforms: Any = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        df = []
        for i, (data_path, file_path, folder) in enumerate(zip(data_paths, file_paths, image_folders)):
            df_ = pd.read_csv(Path(data_path) / file_path)
            df_['folder'] = folder
            df_['data_path'] = data_path
            if indices_list is not None and len(indices_list) > i:
                indices = indices_list[i]
                if indices is not None:
                    df_ = df_.iloc[indices]
            df.append(df_)
            del df_
        df = pd.concat(df)
        self.train = train
        self.ids = df.id.values
        self.data_path = df.data_path.values
        self.folders = df.folder.values
        self.transforms = transforms
        self.labels = None
        if train:
            self.labels = df.target.values.astype(np.float32)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        nid = self.ids[idx]
        folder = self.folders[idx]
        data_path = self.data_path[idx]
        im = self._load_np(nid, data_path, folder).astype(np.float32)
        label = self.labels[idx] if self.labels is not None else 0.0
        if self.transforms is not None:
            im, label = self.transforms(im, label)
        else:
            im = np.vstack((im[0], im[2], im[4]))
        if im.ndim == 2:
            im = np.expand_dims(im, 0)
        im = im.copy()
        return {'im': im, 'label': label, 'im2': [1]}

    def _load_np(self, fileid, data_path, folder):
        return np.load(str(Path(data_path) / folder / f"{fileid[0]}" / f"{fileid}.npy"), allow_pickle=False)


class SingleImageData(Dataset):
    def __init__(
        self,
        file_path,
        data_path,
        indices=None,
        train=True,
        transforms=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        df = pd.read_csv(Path(data_path) / file_path)
        if indices is not None:
            df = df.iloc[indices]
        self.data_path = data_path
        self.train = train
        self.ids = df.id.values
        self.transforms = transforms

    def __len__(self):
        return len(self.ids) * 6

    def __getitem__(self, idx):
        row_idx, ch_idx = divmod(idx, 6)
        nid = self.ids[row_idx]
        im = self._load_np(nid).astype(np.float32)[ch_idx]
        im -= im.min()
        im /= im.max()
        im = np.stack((im, im, im))
        if self.transforms is not None:
            im = np.moveaxis(im, 0, 2)
            im = self.transforms(image=im)["image"]
            im = np.moveaxis(im, 2, 0)

        return im, im  # {'im': }

    def _load_np(self, idx):
        flag = "train" if self.train else "test"
        return np.load(str(Path(self.data_path) / flag / f"{idx[0]}" / f"{idx}.npy"), allow_pickle=False)


class EmbeddingData(Dataset):
    def __init__(
        self,
        numpy_file,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.data = np.load(numpy_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        arr = self.data[idx]
        return arr
