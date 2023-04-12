#!/usr/bin/env python
# encoding: utf-8

"""
@auther: hangyuli
@desc: code for Facial expression data loader
@time: 2019/11/14
"""

from torchvision import transforms as T
from torch.utils.data import Dataset
import numpy as np
import cv2
from PIL import Image
import os
import csv
import h5py
import pandas as pd

def img_loader(path):
    try:
        with open(path, 'rb') as f:
            img = cv2.imread(path)
            img = Image.fromarray(img)
            return img
    except IOError:
        print('Cannot load image ' + path)


class Dataset_RAF(Dataset):
    def __init__(self, root, file_list, transform=None, loader=img_loader):

        self.root = root
        self.transform = transform
        self.loader = loader

        image_list = []
        label_list = []
        finegain_list = []

        with open(file_list) as f:
            img_label_list = f.read().splitlines()
        for info in img_label_list:
            image_path, label_name, fine_gain = info.split(' ')
            image_list.append(image_path)
            label_list.append(int(label_name))
            finegain_list.append(int(fine_gain))

        self.image_list = image_list
        self.label_list = label_list
        self.finegain_list = finegain_list
        self.class_nums = len(np.unique(self.label_list))
        print("dataset size: ", len(self.image_list), '/', self.class_nums)

    def __getitem__(self, index):
        img_path = self.image_list[index]
        label = self.label_list[index]
        fine_gain = self.finegain_list[index]

        img = self.loader(os.path.join(self.root, img_path))
        if self.transform is not None:
            img = self.transform(img)

        return img, label, fine_gain

    def __len__(self):
        return len(self.image_list)


class Dataset_AffectNet(Dataset):
    def __init__(self, root, file_list, transform=None, loader=img_loader):

        self.root = root
        self.transform = transform
        self.loader = loader

        image_list = []
        label_list = []
        finegain_list = []

        with open(file_list, 'r') as csvin:
            data = csv.reader(csvin)
            for row in data:
                image_list.append(row[0])
                label_list.append(int(row[-2]))
                finegain_list.append(int(row[-1]))

        self.image_list = image_list
        self.label_list = label_list
        self.finegain_list = finegain_list
        self.class_nums = len(np.unique(self.label_list))
        print("dataset size: ", len(self.image_list), '/', self.class_nums)

    def __getitem__(self, index):
        img_path = self.image_list[index]
        label = self.label_list[index]
        fine_gain = self.finegain_list[index]

        img = self.loader(os.path.join(self.root, img_path))
        if self.transform is not None:
            img = self.transform(img)

        return img, label, fine_gain

    def __len__(self):
        return len(self.image_list)

class FERPlus(Dataset):
    def __init__(self, file_name, split='Training', transform=None):

        self.transform = transform
        self.split = split
        self.data = h5py.File(file_name, 'r', driver='core')

        if self.split == 'Training':
            self.train_data = self.data['Training_pixel']
            self.train_labels = self.data['Training_label']
            self.train_coarse = self.data['Training_coarse']
            self.train_data = np.asarray(self.train_data)
            self.train_data = self.train_data.reshape((-1, 48, 48))

        elif self.split == 'PublicTest':
            self.PublicTest_data = self.data['PublicTest_pixel']
            self.PublicTest_labels = self.data['PublicTest_label']
            self.PublicTest_coarse = self.data['PublicTest_coarse']
            self.PublicTest_data = np.asarray(self.PublicTest_data)
            self.PublicTest_data = self.PublicTest_data.reshape((-1, 48, 48))

        else:
            self.PrivateTest_data = self.data['PrivateTest_pixel']
            self.PrivateTest_labels = self.data['PrivateTest_label']
            self.PrivateTest_coarse = self.data['PrivateTest_coarse']
            self.PrivateTest_data = np.asarray(self.PrivateTest_data)
            self.PrivateTest_data = self.PrivateTest_data.reshape((-1, 48, 48))


    def __getitem__(self, index):
        if self.split == 'Training':
            img, label, fine_gain = self.train_data[index], self.train_labels[index], self.train_coarse[index]
        elif self.split == 'PublicTest':
            img, label, fine_gain = self.PublicTest_data[index], self.PublicTest_labels[index], self.PublicTest_coarse[index]
        else:
            img, label, fine_gain = self.PrivateTest_data[index], self.PrivateTest_labels[index], self.PrivateTest_coarse[index]

        img = img[:, :, np.newaxis]
        img = np.concatenate((img, img, img), axis=2)
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, label, fine_gain

    def __len__(self):
        if self.split == 'Training':
            return len(self.train_data)
        elif self.split == 'PublicTest':
            return len(self.PublicTest_data)
        else:
            return len(self.PrivateTest_data)

class Dataset_Pose(Dataset):
    def __init__(self, root, file_list, transform=None, loader=img_loader):

        self.root = root
        self.transform = transform
        self.loader = loader

        image_list = []
        label_list = []
        finegain_list = []

        with open(file_list) as f:
            img_label_list = f.read().splitlines()
        for info in img_label_list:
            label_name, image_path = info.split(' ')
            image_list.append(image_path)
            label_list.append(int(label_name))
            finegain_list.append(int(label_name))

        self.image_list = image_list
        self.label_list = label_list
        self.finegain_list = finegain_list
        self.class_nums = len(np.unique(self.label_list))
        print("dataset size: ", len(self.image_list), '/', self.class_nums)

    def __getitem__(self, index):
        img_path = self.image_list[index]
        label = self.label_list[index]
        fine_gain = self.finegain_list[index]

        img = self.loader(os.path.join(self.root, img_path))
        if self.transform is not None:
            img = self.transform(img)

        return img, label, fine_gain

    def __len__(self):
        return len(self.image_list)

