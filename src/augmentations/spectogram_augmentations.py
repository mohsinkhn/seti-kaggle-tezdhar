"""Augmentations for spectograms"""
from abc import abstractmethod

import cv2
import numpy as np
from pathlib import Path
import PIL
from PIL import Image


class RandomTransform:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, y):
        return self.transform(image, y)

    def transform(self, image, y):
        if np.random.rand() < self.p:
            return self._transform(image, y)
        return image, y

    @abstractmethod
    def _transform(self, image, y):
        pass


class Compose:
    def __init__(self, transform_list):
        self.transform_list = transform_list

    def __call__(self, image, y):
        out = image.copy()
        for tfm in self.transform_list:
            out, y = tfm(out, y)
        return out, y


class RobustScaling(RandomTransform):
    def __init__(self, min_percentile=0.025, max_percentile=0.975, p=1.0):
        super().__init__(p=p)
        self.min_percentile = min_percentile
        self.max_percentile = max_percentile

    def _transform(self, image, y):
        image = image.copy()
        min_value = np.percentile(image, self.min_percentile * 100)
        max_value = np.percentile(image, self.max_percentile * 100)
        image[image < min_value] = min_value
        image[image > max_value] = max_value
        median_value = np.median(image)
        image -= median_value
        image /= (max_value - min_value)
        image *= 2
        return image, y


class PowerTransform(RandomTransform):
    def __init__(self, power=(0.75, 1.0), p=1.0):
        super().__init__(p=p)
        self.power = power

    def _transform(self, image, y):
        min_value = np.min(image)
        max_value = np.max(image)
        if (type(self.power) == float) | (type(self.power) == int):
            power = self.power
        else:
            power = np.random.uniform(low=self.power[0], high=self.power[1])
        norm_image = ((image - min_value) / max_value) ** power
        return norm_image, y


class Resize(RandomTransform):
    def __init__(self, height=256, width=256, resample=PIL.Image.BILINEAR, p=1.0):
        super().__init__(p=p)
        self.height = height
        self.width = width
        self.resample = resample
        self.p = p

    def _transform(self, image, y):
        original_dtype = image.dtype
        image = Image.fromarray(image)
        image = image.resize(size=(self.width, self.height), resample=self.resample)
        image = np.array(image)
        return image.astype(original_dtype), y


class ResizeMulti(RandomTransform):
    def __init__(self, height=256, width=256, resample=PIL.Image.BILINEAR, p=1.0):
        super().__init__(p=p)
        self.height = height
        self.width = width
        self.resample = resample
        self.p = p

    def _transform(self, image, y):
        frames = image.shape[0]
        out_image = []
        for frame in range(frames):
            im = image[frame]
            fimage = Image.fromarray(im)
            fimage = fimage.resize(size=(self.width, self.height), resample=self.resample)
            fimage = np.array(fimage)
            out_image.append(fimage)
        return np.asarray(out_image).astype(np.float32), y


class Brightness(RandomTransform):
    def __init__(self, factor=0.1, p=0.5):
        super().__init__(p=p)
        self.factor = factor

    def _transform(self, image, y):
        mean = (np.random.rand() - 0.5) * self.factor
        return image - mean, y


class Flip(RandomTransform):
    def __init__(self, axis=1, p=0.5):
        super().__init__(p=p)
        self.p = p
        self.axis = axis

    def _transform(self, image, y):
        image = image.copy()
        if self.axis == 0:
            return image[::-1], y
        elif self.axis == 1:
            return image[:, ::-1], y
        else:
            return image[:, :, ::-1], y


class Roll(RandomTransform):
    def __init__(self, axis=1, fraction=0.2, p=0.5):
        super().__init__(p=p)
        self.axis = axis
        self.fraction = fraction

    def _transform(self, image, y):
        fraction = np.random.uniform(low=-self.fraction, high=self.fraction)
        shift = int(image.shape[self.axis] * fraction)
        return np.roll(image, shift=shift, axis=self.axis), y


# class RollPixel(RandomTransform):
#     def __init__(self, axis=1, p=0.5):
#         super().__init__(p=p)
#         self.axis = axis

#     def _transform(self, image, y):
#         fraction = np.random.uniform(low=-self.fraction, high=self.fraction)
#         shift = int(image.shape[self.axis] * fraction)
#         return np.roll(image, shift=shift, axis=self.axis), y


class TemporallyStackONChannels(RandomTransform):
    def __init__(self, p=1.0):
        super().__init__(p=p)

    def _transform(self, image, y):
        return np.vstack((image[0], image[2], image[4])), y


class RemoveFrequencyMean(RandomTransform):
    def _transform(self, image, y):
        return image - image.mean(axis=2, keepdims=True), y


class RemoveTimeFrequencyMean(RandomTransform):
    def _transform(self, image, y):
        return image - image.mean(axis=1, keepdims=True) - image.mean(axis=2, keepdims=True), y


class TemporallyStackMultiChannels(RandomTransform):
    def _transform(self, image, y):
        return np.concatenate((image[0], image[1], image[2], image[3], image[4], image[5]), 1), y


class TemporallyStackChannels(RandomTransform):
    def _transform(self, image, y):
        return np.vstack(image), y


class SpecAug(RandomTransform):
    def __init__(self, num_consecutive_freq=1, num_patches=1, axis=1, p=1.0):
        super().__init__(p=p)
        self.num_consecutive_freq = num_consecutive_freq
        self.num_patches = num_patches
        self.axis = axis

    def _transform(self, image, y):
        image = image.copy()
        num_patches = np.random.choice(list(range(1, self.num_patches)), 1)[0]
        indices = np.random.choice(list(range(image.shape[self.axis])), self.num_patches)
        for idx in indices:
            num_consecutive_freq = np.random.choice(list(range(1, self.num_consecutive_freq)), 1)[0]
            if self.axis == 0:
                image[idx:idx + num_consecutive_freq] = 0
            elif self.axis == 1:
                image[:, idx:idx + num_consecutive_freq] = 0
            else:
                image[:, :, idx:idx + num_consecutive_freq] = 0
        return image, y


class ClipVal(RandomTransform):
    def __init__(self, min_val=-2, max_val=2, p=1.0):
        self.min_val = min_val
        self.max_val = max_val

    def _transform(self, image, y):
        image = image.copy()
        return np.clip(image, self.min_val, self.max_val), y


class AddChannelInfo(RandomTransform):
    def _transform(self, image, y):
        im_len = image.shape[0] // 6
        onindicator = np.zeros_like(image, dtype=np.float32)
        onindicator[:im_len] = 1.0
        onindicator[2 * im_len:3 * im_len] = 1.0
        onindicator[4 * im_len:5 * im_len] = 1.0
        image_mean = np.tile(image.mean(0, keepdims=True), (image.shape[0], 1))
        return np.stack((image, onindicator, image_mean)), y


class AddChannelMulti(RandomTransform):
    def _transform(self, image, y):
        frames = image.shape[0]
        out_images = []
        for frame in range(frames):
            im1 = image[frame][1:][::2]
            im2 = image[frame][:-1][::2]
            onindicator = np.zeros_like(im1, dtype=np.float32)
            if frame % 2 == 0:
                onindicator[:, :] = 1.0
            im = np.stack((im1, im2, onindicator))
            out_images.append(im)
        return np.array(out_images), y


class AddChannelMulti2(RandomTransform):
    def _transform(self, image, y):
        frames = image.shape[0]
        out_images = []
        for frame in range(frames):
            im1 = image[frame]
            onindicator = np.zeros_like(im1, dtype=np.float32)
            if frame % 2 == 0:
                onindicator[:, :] = 1.0
            image_mean = np.tile(im1.mean(0, keepdims=True), (im1.shape[0], 1))
            im = np.stack((im1, image_mean, onindicator))
            out_images.append(im)
        return np.array(out_images), y


class SwapOnOff(RandomTransform):
    def _transform(self, image, y):
        return np.roll(image, shift=-1, axis=0), 0.0


class VerticalShift(RandomTransform):
    def _transform(self, image, y):
        return np.roll(image, shift=2, axis=0), y


class MaskOneSignal(RandomTransform):
    def _transform(self, image, y):
        image = image.copy()
        ii = np.random.choice([0, 2, 4], 1)[0]
        jj = np.random.choice([1, 3, 5], 1)[0]
        image[ii] = image[jj]
        return image, y


class LabelSmoothing(RandomTransform):
    def __init__(self, d=0.95, p=1.0):
        super().__init__(p=p)
        self.d = d

    def _transform(self, image, y):
        return image,  y * self.d + (1 - y) * (1 - self.d)


class MovePixel(RandomTransform):
    def _transform(self, image, y):
        pass


class LogScaler(RandomTransform):
    def _transform(self, image, y):
        frames = image.shape[0]
        out_image = []
        for frame in range(frames):
            im = image[frame]
            im -= im.mean()
            im /= im.std()
            im -= im.min()
            im = np.log1p(im)
            im -= np.median(im)
            out_image.append(im)
        return np.array(out_image), y


class MoreSignals(RandomTransform):
    def __init__(self, p=0.2):
        super().__init__(p=p)
        search_path = "/home/mohsin_okcredit_in/projects/seti-kaggle-tezdhar/data/primary_small/train/{folder}"
        folders = ["narrowband", "narrowbanddrd", "brightpixel",
                   "squarepulsednarrowband", "squiggle",
                   "squigglesquarepulsednarrowband"]
        files = []
        for folder in folders:
            folder_files = Path(search_path.format(folder=folder)).glob("*.png")
            files.extend(list(folder_files))
        self.signal_arr = [self._load_file(file) for file in files]
        self.noise_arr = [self._load_file(file) for file in Path(search_path.format(folder='noise')).glob("*.png")] * 6
        self.num_signals = len(self.signal_arr)

    def _transform(self, image, y):
        idx1 = np.random.choice([0, 2, 4], 1)[0]
        aug_im_idx = np.random.choice(list(range(self.num_signals)), 1)[0]
        aug_im = self.signal_arr[aug_im_idx]
        lam = np.clip(np.random.rand(), 0.25, 0.75)
        non_sig_idx = set([0, 2, 4])
        image[idx1] = image[idx1] + aug_im * lam
        non_sig_idx -= set([idx1])
        if np.random.rand() < 0.5:
            if idx1 < 4:
                idx2 = idx1 + 2
                aug_im = np.roll(aug_im, int(np.random.rand() * 0.2 * aug_im.shape[1]), axis=1)
                image[idx2] = image[idx2] + aug_im
                non_sig_idx -= set([idx2])

                if np.random.rand() < 0.5:
                    if idx2 < 4:
                        idx2 = idx2 + 2
                        aug_im = np.roll(aug_im, int(np.random.rand() * 0.4 * aug_im.shape[1]), axis=1)
                        image[idx2] = image[idx2] + aug_im
                        non_sig_idx -= set([idx1])
        if len(non_sig_idx) > 0:
            for idx in non_sig_idx:
                image[idx] = image[idx] + self.noise_arr[aug_im_idx]
        return image, 1.0

    def _load_file(self, filepath):
        im = cv2.imread(str(filepath))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im = cv2.resize(im, (256, 273)).astype(np.float32)
        im -= im.mean()
        im /= im.std()
        return im


class MoreSignals2(RandomTransform):
    def __init__(self, p=0.2):
        super().__init__(p=p)
        search_path = "/home/mohsin_okcredit_in/projects/seti-kaggle-tezdhar/data/primary_small/train/{folder}"
        folders = ["narrowband", "narrowbanddrd", "brightpixel",
                   "squarepulsednarrowband", "squiggle",
                   "squigglesquarepulsednarrowband"]
        files = []
        for folder in folders:
            folder_files = Path(search_path.format(folder=folder)).glob("*.png")
            files.extend(list(folder_files))
        self.signal_arr = [self._load_file(file) for file in files]
        self.noise_arr = [self._load_file(file) for file in Path(search_path.format(folder='noise')).glob("*.png")] * 6
        self.num_signals = len(self.signal_arr)

    def _transform(self, image, y):
        idx1 = np.random.choice([0, 2, 4], 1)[0]
        aug_im_idx = np.random.choice(list(range(self.num_signals)), 1)[0]
        aug_im = self.signal_arr[aug_im_idx]
        lam = np.random.rand() / 0.5 + 0.5
        non_sig_idx = set([0, 2, 4])
        image[idx1] = image[idx1] * (1 - lam) + aug_im * lam
        non_sig_idx -= set([idx1])
        if np.random.rand() < 0.5:
            if idx1 < 4:
                idx2 = idx1 + 2
                aug_im = np.roll(aug_im, int(np.random.rand() * 0.2 * aug_im.shape[1]), axis=1)
                image[idx2] = image[idx2] * (1 - lam) + aug_im * lam
                non_sig_idx -= set([idx2])

                if np.random.rand() < 0.5:
                    if idx2 < 4:
                        idx2 = idx2 + 2
                        aug_im = np.roll(aug_im, int(np.random.rand() * 0.4 * aug_im.shape[1]), axis=1)
                        image[idx2] = image[idx2] * (1 - lam) + aug_im * lam
                        non_sig_idx -= set([idx1])
        if len(non_sig_idx) > 0:
            for idx in list(non_sig_idx) + [1, 3, 5]:
                image[idx] = image[idx] * (1 - lam) + self.noise_arr[aug_im_idx] * lam
        return image, (1 - lam) * y + lam

    def _load_file(self, filepath):
        im = cv2.imread(str(filepath))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im = cv2.resize(im, (256, 273)).astype(np.float32)
        im -= im.mean()
        im /= im.std()
        return im