"""Data generators and prep."""

import os
from random import randint

import numpy as np
import pandas as pd

import cv2
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import RandomSampler

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def prep_data(batch_size, threads, use_cuda):
    """"Prepare DataLoaders for training.

    Args:
        batch_size (int): Number of image samples to be propagated through network.
        threads (int): Number of threads/workers to load image samples.
        use_cuda (bool): Use CUDA or not (for pin_memory in DataLoaders).

    Returns:
        Training DataLoader, Validation DataLoader
    """
    full_train = pd.read_json(os.path.join(THIS_DIR, '..', 'input', 'train.json'))

    full_train['inc_angle'] = pd.to_numeric(full_train['inc_angle'],
                                            errors='coerce')
    full_train['band_1'] = full_train['band_1'].apply(
        lambda x: np.array(x).reshape(75, 75))
    full_train['band_2'] = full_train['band_2'].apply(
        lambda x: np.array(x).reshape(75, 75))

    train_ds = full_train.sample(frac=0.8).dropna()
    val_ds = full_train[~full_train.isin(train_ds)].dropna()

    # training data
    train_ds = ImageDataset(train_ds, include_target=True, u=0.5,
                            X_transform=[random_horizontal_flip,
                                         random_vertical_flip,
                                         random_crop])

    train_loader = DataLoader(
        train_ds,
        batch_size,
        sampler=RandomSampler(train_ds),
        num_workers=threads,
        pin_memory=use_cuda
    )

    # validation data
    val_ds = ImageDataset(val_ds, include_target=True, u=0.5,
                          X_transform=None) # no transforms for validation set

    val_loader = DataLoader(
        val_ds,
        batch_size,
        sampler=RandomSampler(val_ds),
        num_workers=threads,
        pin_memory=use_cuda
    )

    return train_loader, val_loader


class ImageDataset(Dataset):
    """A pytorch Dataset for the data.

    Args:
        X_data (DataFrame): Pandas df of the prepared data.
        include_target (bool): True if train and False if test.
        u (float): Arg in X_transform functions indication probability of
            performing the transformation(s).
        X_transform (list): List of image transformation functions.
    """
    def __init__(self, X_data, include_target, u=0.5, X_transform=None):
        self.X_data = X_data
        self.include_target = include_target
        self.u = u
        self.X_transform = X_transform

    def __getitem__(self, index):
        """Return a dictionary of data for each sample."""
        # for some reason this is necessary in order to
        # not generate the same data each iter
        np.random.seed()

        # get 2 channels of our image
        img1 = self.X_data.iloc[index]['band_1']
        img2 = self.X_data.iloc[index]['band_2']

        # image shape = (75, 75, 2)
        img = np.stack([img1, img2], axis=2)

        # get angle and img_name
        angle = self.X_data.iloc[index]['inc_angle']
        img_id = self.X_data.iloc[index]['id']

        # perform augmentation
        if self.X_transform is not None:
            for txfrm in self.X_transform:
                img = txfrm(img, **{'u' : self.u})

        # reshape image for pytorch
        img = img.transpose((2, 0, 1))
        img_numpy = img.astype(np.float32)

        # convert image to tensor
        img_torch = torch.from_numpy(img_numpy)

        # so our loader will yield dictionary wi such fields:
        dict_ = {
            'img': img_torch,
            'id': img_id,
            'angle': angle,
            'img_np': img_numpy
        }

        # if train - then also include target
        if self.include_target:
            target = self.X_data.iloc[index]['is_iceberg']
            dict_['target'] = target

        return dict_

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.X_data)


def random_vertical_flip(img, u=0.5):
    """Flip coin and perform vertical flip."""
    if np.random.random() < u:
        img = cv2.flip(img, 0)
    return img


def random_horizontal_flip(img, u=0.5):
    """Flip coin and perform horizontal flip."""
    if np.random.random() < u:
        img = cv2.flip(img, 1)
    return img


def random_crop(img, u=0.5):
    """Flip coin and perform crop and resize."""
    if np.random.random() < u:
        y1, y2 = randint(0, 15), randint(60, 75)
        x1, x2 = randint(0, 15), randint(60, 75)
        img = img[y1:y2, x1:x2, :]
        img = cv2.resize(img, (75, 75), interpolation=cv2.INTER_CUBIC)
    return img


def rotate(image, angle, center=None, scale=1.0, u=0.5):
    """Flip coin and perform rotation."""
    if np.random.random() < u:
        (h, w) = image.shape[:2]

        if center is None:
            center = (w / 2, h / 2)

        # perform the rotation
        M = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.warpAffine(image, M, (w, h))

    return rotated
