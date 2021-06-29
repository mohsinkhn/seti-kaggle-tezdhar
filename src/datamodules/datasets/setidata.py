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
        multiobjective=False,
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
        self.multiobjective = multiobjective
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
        im2 = self._amplify(im[1:], new_power)
        im = self._amplify(im, new_power)
        if self.transforms is not None:
            if self.multiobjective:
                imm = np.vstack((np.moveaxis(im, 0, 2), np.moveaxis(im2, 0, 2)))
                imm = self.transforms(image=imm)['image']
                im, im2 = np.moveaxis(imm[:306], 2, 0), np.moveaxis(imm[306:], 2, 0)
            else:
                im = self.transforms(image=np.moveaxis(im, 0, 2))['image']
                im = np.moveaxis(im, 2, 0)
        return {'im': im, 'label': label, 'im2': im2}

    def get_arrs(self, ids):
        arrs = np.zeros(shape=(len(ids), 3, 273, 256), dtype=np.float16)
        for i, idx in tqdm(enumerate(ids)):
            arrs[i] = self._load_np(idx)
        return arrs

    def _amplify(self, im, power):
        im1 = self._normalize_stack(im, self.norm, power)
        if self.channels == 1:
            im = np.expand_dims(im1, 0)
        elif self.channels == 111:
            im = np.stack((im1, im1, im1))
        elif self.channels == 111222:
            im2 = self._normalize_stack(im, 2, power)
            im = np.stack((im1, im1, im1, im2, im2, im2))
        elif self.channels == 123:
            im2 = self._normalize_stack(im, 2, power)
            im3 = self._normalize_stack(im, 3, power)
            im = np.stack((im1, im2, im3))
        else:
            return im1
        return im

    def _load_np(self, idx):
        flag = "train" if self.train else "test"
        return np.load(str(Path(self.data_path) / flag / f"{idx[0]}" / f"{idx}.npy"), allow_pickle=False)

    def _normalize_stack(self, im, norm, power):
        im0, im2, im4 = np.clip(im[[0, 2, 4]], self.min_clip, self.max_clip)

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
            im0 = ((im0 - im0.min()) / (im0.max() - im0.min()))
            im2 = ((im2 - im2.min()) / (im2.max() - im2.min()))
            im4 = ((im4 - im4.min()) / (im4.max() - im4.min()))
            im0 = im0[:272].reshape((8, 34, 256), order='F')
            im0 = ((im0 ** power)).sum(axis=0)
            im2 = im2[:272].reshape((8, 34, 256), order='F')
            im2 = ((im2 ** power)).sum(axis=0)
            im4 = im4[:272].reshape((8, 34, 256), order='F')
            im4 = ((im4 ** power)).sum(axis=0)

        elif norm == 5:  # Remove signals present in background
            im1, im3, im5 = im[[1, 3, 5]]
            im0 = ((im0 - im0.min()) / (im0.max() - im0.min()))
            im1 = ((im1 - im1.min()) / (im1.max() - im1.min()))
            im2 = ((im2 - im2.min()) / (im2.max() - im2.min()))
            im3 = ((im3 - im3.min()) / (im3.max() - im3.min()))
            im4 = ((im4 - im4.min()) / (im4.max() - im4.min()))
            im5 = ((im5 - im5.min()) / (im5.max() - im5.min()))

            im0 = im0[:272].reshape((8, 34, 256), order='F')
            im0 = ((im0 ** power)).sum(axis=0)
            im1 = im1[:272].reshape((8, 34, 256), order='F')
            im1 = ((im1 ** power)).sum(axis=0)

            im2 = im2[:272].reshape((8, 34, 256), order='F')
            im2 = ((im2 ** power)).sum(axis=0)
            im3 = im3[:272].reshape((8, 34, 256), order='F')
            im3 = ((im3 ** power)).sum(axis=0)

            im4 = im4[:272].reshape((8, 34, 256), order='F')
            im4 = ((im4 ** power)).sum(axis=0)
            im5 = im5[:272].reshape((8, 34, 256), order='F')
            im5 = ((im5 ** power)).sum(axis=0)
            im0 = np.stack((np.vstack((im0, im2, im4)), np.vstack((im1, im3, im5))))
            # im2 = np.stack((im2, im3))
            # im4 = np.stack((im4, im5))

        elif norm == 6:  # NMF to clean signals
            raise NotImplementedError

        if (norm not in (4, 5)):
            im0 = ((im0 - im0.min()) / (im0.max() - im0.min())) ** power
            im2 = ((im2 - im2.min()) / (im2.max() - im2.min())) ** power
            im4 = ((im4 - im4.min()) / (im4.max() - im4.min())) ** power

        if norm == 5:
            return im0  # np.concatenate((im0, im2, im4), 0)
        else:
            return np.vstack((im0, im2, im4))
