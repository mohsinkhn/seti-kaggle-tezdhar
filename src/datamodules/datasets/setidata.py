import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class SetiDataNp(Dataset):
    def __init__(
        self,
        file_path,
        data_path,
        indices=None,
        train=False,
        valid=True,
        norm=1,
        power=0.5,
        transforms=None,
        preload=True,
        channels=1,
        min_clip=-5,
        max_clip=5,
        **kwargs
    ):
        super().__init__(**kwargs)
        df = pd.read_csv(Path(data_path) / file_path)
        if indices is not None:
            df = df.iloc[indices]
        self.data_path = data_path
        self.train = train
        self.valid = valid
        self.ids = df.id.values
        self.arrs = None
        self.preload = preload
        self.channels = channels
        self.min_clip = min_clip
        self.max_clip = max_clip
        if preload:
            self.arrs = self.get_arrs(df.id.values)
        self.labels = None
        if train:
            self.labels = df.target.values.astype(np.float32)
        self.norm = norm
        self.power = power
        self.transforms = transforms

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        if self.preload:
            im = self.arrs[idx].astype(np.float32)
        else:
            nid = self.ids[idx]
            im = self._load_np(nid).astype(np.float32)

        label = self.labels[idx] if self.train else 0.0

        if not self.valid:
            new_power = max(0.1, self.power + (np.random.rand() / 2 - 0.25))
        else:
            new_power = self.power
        im = self._amplify(im, new_power)

        if self.transforms is not None:
            im = self.transforms(image=np.moveaxis(im, 0, 2))['image']
            im = np.moveaxis(im, 2, 0)
        return {'im': im, 'label': label}

    def get_arrs(self, ids):
        arrs = np.zeros(shape=(len(ids), 3, 273, 256), dtype=np.float16)
        for i, idx in tqdm(enumerate(ids)):
            arrs[i] = self._load_np(idx)
        return arrs

    def _amplify(self, im, power):
        im1 = self._normalize_stack(im, self.norm, power)
        if self.channels == 1:
            im = np.expand_dims(im1, 0)
        if self.channels == 111:
            im = np.stack((im1, im1, im1))
        if self.channels == 111222:
            im2 = self._normalize_stack(im, 2, power)
            im = np.stack((im1, im1, im1, im2, im2, im2))
        if self.channels == 123:
            im2 = self._normalize_stack(im, 2, power)
            im3 = self._normalize_stack(im, 3, power)
            im = np.stack((im1, im2, im3))
        return im

    def _load_np(self, idx):
        flag = "train" if self.train else "test"
        return np.load(str(Path(self.data_path) / flag / f"{idx[0]}" / f"{idx}.npy"), allow_pickle=False)[[0, 2, 4]]

    def _normalize_stack(self, im, norm, power):
        im0, im2, im4 = np.clip(im, self.min_clip, self.max_clip)

        if norm == 2:  # substract - column wise mean
            im0 -= np.mean(im0, axis=0, keepdims=True)
            im2 -= np.mean(im2, axis=0, keepdims=True)
            im4 -= np.mean(im4, axis=0, keepdims=True)
        elif norm == 3:  # substract row mean followed by column mean
            im0 -= np.mean(im0, axis=1, keepdims=True)
            im2 -= np.mean(im2, axis=1, keepdims=True)
            im4 -= np.mean(im4, axis=1, keepdims=True)
            im0 -= np.mean(im0, axis=0, keepdims=True)
            im2 -= np.mean(im2, axis=0, keepdims=True)
            im4 -= np.mean(im4, axis=0, keepdims=True)
        elif norm == 4:  # Remove signals present in background
            raise NotImplementedError
        elif norm == 5:  # NMF to clean signals
            raise NotImplementedError

        im0 = ((im0 - im0.min()) / (im0.max() - im0.min())) ** power
        im2 = ((im2 - im2.min()) / (im2.max() - im2.min())) ** power
        im4 = ((im4 - im4.min()) / (im4.max() - im4.min())) ** power
        return np.vstack((im0, im2, im4))
