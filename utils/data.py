import os
import cv2
import random
import numpy as np
from torch.utils.data import Dataset
import albumentations as albu
from albumentations.pytorch import ToTensorV2

__all__ = ['STS2DDataset', 'train_transforms', 'valid_transforms', 'infer_transforms']


class STS2DDataset(Dataset):
    def __init__(self, data_dir, filenames, transform=None):
        super(STS2DDataset, self).__init__()
        self.data_dir = data_dir
        self.filenames = filenames
        self.transform = transform

    def __getitem__(self, index):
        image_path = os.path.join(self.data_dir, 'image', self.filenames[index])
        image = cv2.imread(image_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = {'image': image}

        mask_path = os.path.join(self.data_dir, 'mask', self.filenames[index])
        mask = cv2.imread(mask_path, 0)
        mask = np.where(mask > 127, 1, 0).astype(np.float32)
        result['mask'] = mask

        if self.transform is not None:
            result = self.transform(**result)

        return result

    def __len__(self):
        return len(self.filenames)


# Transforms
def crop_transforms(image_size: int = 224):
    ratio = random.random() * 3 + 1  # [1, 4)
    max_size = int(image_size * ratio)
    scaled_crop = albu.Compose([
        albu.SmallestMaxSize(max_size, p=1),
        albu.RandomCrop(image_size, image_size, p=1),
    ])

    direct_crop = albu.Compose([
        albu.RandomCrop(image_size, image_size, p=1),
    ])

    trans = [
        albu.OneOf([scaled_crop, direct_crop], p=1),
    ]
    return trans


def hard_transforms():
    trans = [
        albu.RandomRotate90(),
        albu.Flip(),
        albu.ShiftScaleRotate(),
        # albu.Cutout(),
        albu.CoarseDropout(),
        albu.RandomBrightnessContrast(p=0.3),
        albu.HueSaturationValue(p=0.3),
        albu.GridDistortion(p=0.3),
    ]
    return trans


def post_transforms():
    trans = [
        albu.Normalize(),
        ToTensorV2(),
    ]
    return trans


def train_transforms(image_size: int = 224):
    return albu.Compose([
        *crop_transforms(image_size),
        *hard_transforms(),
        *post_transforms(),
    ])


def valid_transforms():
    return albu.Compose([
        *post_transforms(),
    ])


def infer_transforms():
    return albu.Compose([
        *post_transforms()
    ])
