import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
import os
import albumentations as A
from albumentations.pytorch import ToTensor
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
from torch.utils.data import DataLoader
from config import TRAIN_FILE, MEAN, STD

# https://www.kaggle.com/rishabhiitbhu/unet-starter-kernel-pytorch-lb-0-88

def make_mask(row_id, df):
    '''Given a row index, return image_id and mask (256, 1600, 4) from the dataframe `df`'''
    fname = df.iloc[row_id].name
    labels = df.iloc[row_id][:4]
    masks = np.zeros((256, 1600, 4), dtype=np.float32) # float32 is V.Imp
    # 4:class 1～4 (ch:0～3)

    for idx, label in enumerate(labels.values):
        if label != -1:
            label = label.split(" ")
            positions = map(int, label[0::2])
            length = map(int, label[1::2])
            mask = np.zeros(256 * 1600, dtype=np.uint8)
            for pos, le in zip(positions, length):
                mask[pos:(pos + le)] = 1
            masks[:, :, idx] = mask.reshape(256, 1600, order='F')
    return fname, masks

class SteelDatasetNoAug(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df
        self.mean = MEAN
        self.std = STD
        self.fnames = self.df.index.tolist()

    def __getitem__(self, idx):
        image_id, mask = make_mask(idx, self.df)
        image_path = os.path.join("train_images",  image_id)
        img = cv2.imread(image_path)
        mask = torch.FloatTensor(mask).permute(2, 1, 0)
        img = torch.FloatTensor(img).permute(2, 1, 0)
        return img, mask

    def __len__(self):
        return len(self.fnames)

class SteelDataset(torch.utils.data.Dataset):
    def __init__(self, df, training):
        self.training = training
        self.df = df
        self.fnames = self.df.index.tolist()
        self.transforms = self.get_transforms()

    def __getitem__(self, idx):
        image_id, mask = make_mask(idx, self.df)
        image_path = os.path.join("train_images",  image_id)
        img = cv2.imread(image_path)
        augd = self.transforms(image=img, mask=mask)
        img = augd['image']
        mask = augd['mask']
        mask = mask[0].permute(2, 0, 1)
        return img, mask

    def get_transforms(self):
        transforms = []
        if self.training:
            flips = [A.HorizontalFlip(0.5), A.VerticalFlip(0.5)]
            bright = [A.RandomBrightness(p=0.5, limit=0.4)]
            transforms.extend(flips)
            transforms.extend(bright)
        transforms.extend([A.Normalize(mean=MEAN, std=STD), ToTensor()])
        transforms = A.Compose(transforms)
        return transforms

    def __len__(self):
        return len(self.fnames)

class SteelTestDataset(torch.utils.data.Dataset):
    def __init__(self, dir='test_images/'):
        self.dir = dir
        self.df = pd.read_csv('sample_submission.csv')
        self.df['images'] = self.df['ImageId_ClassId'].apply(lambda x: x.split('.')[0])
        self.images = [x + '.jpg' for x in list(self.df['images'].unique())]
        self.transforms = self.get_transforms()
        print(len(self.images))

    def __getitem__(self, idx):
        image = self.images[idx]
        image_path = os.path.join(self.dir,  image)
        img = cv2.imread(image_path)
        img = self.transforms(image=img)['image']
        return image, img

    def get_transforms(self):
        transforms = [A.Normalize(mean=MEAN, std=STD), ToTensor()]
        transforms = A.Compose(transforms)
        return transforms

    def __len__(self):
        return len(self.images)

class DataLoaderFactory():
    def __init__(self, infile=None):
        self.df_file = TRAIN_FILE if not infile else infile

    def gen(self, batch_size=8, test_size=0.2):
        df = pd.read_csv(self.df_file)
        df = df.replace(np.nan, -1)
        df['ImageId'], df['ClassId'] = zip(*df['ImageId_ClassId'].str.split('_'))
        df['ClassId'] = df['ClassId'].astype(int)
        df = df.pivot(index='ImageId',columns='ClassId',values='EncodedPixels')
        df['defects'] = df.count(axis=1)
        df_train, df_cv = train_test_split(df, test_size=test_size, stratify=df['defects'], random_state=42)
        print(len(df), len(df_train), len(df_cv))

        train_dataset = SteelDataset(df_train, training=True)
        cv_dataset = SteelDataset(df_cv, training=False)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        cv_loader = DataLoader(cv_dataset, batch_size=batch_size, shuffle=True)
        torch.save(train_loader, 'train_loader.pt')
        torch.save(cv_loader, 'cv_loader.pt')