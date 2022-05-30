import torch
import os
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
import pickle
import scipy.io as scio

class DatasetPorcessing_flickr_mat(Dataset):
    def __init__(self, train_data, train_label, transform=None, is_train=False):
        self.train_data = train_data.transpose(3, 0, 1, 2)
        self.is_train = is_train
        self.transform = transform
        self.labels = torch.tensor(train_label).float()
    def __getitem__(self, index):
        img = Image.fromarray(self.train_data[index])
        if self.transform is not None:
            img1 = self.transform(img)
            img2 = self.transform(img)
        label = self.labels[index]
        if self.is_train:
            return img1, img2, label, label, index
        else:
            return img1, label, label, index
    def __len__(self):
        return self.labels.shape[0]

class DatasetPorcessing_nus_h5(Dataset):
    def __init__(self, train_data, train_label, transform=None, is_train=False):
        self.train_data = train_data
        self.is_train = is_train
        self.transform = transform
        self.labels = torch.tensor(train_label).float()
    def __getitem__(self, index):
        img = Image.fromarray(self.train_data[index])
        if self.transform is not None:
            img1 = self.transform(img)
            img2 = self.transform(img)
        label = self.labels[index]
        if self.is_train:
            return img1, img2, label, label, index
        else:
            return img1, label, label, index
    def __len__(self):
        return self.labels.shape[0]
class DatasetPorcessing_mat(Dataset):
    def __init__(self, train_data, train_label, transform=None, is_train=False):
        self.train_data = train_data.transpose(3, 0, 1, 2)
        self.is_train = is_train
        self.transform = transform
        num_class = int(train_label.max() + 1)
        self.labels = torch.zeros((train_label.shape[0], num_class))
        for i in range(train_label.shape[0]):
            self.labels[i, int(train_label[i])] = 1
    def __getitem__(self, index):
        img = Image.fromarray(self.train_data[index])
        if self.transform is not None:
            img1 = self.transform(img)
            img2 = self.transform(img)
        label = self.labels[index]
        if self.is_train:
            return img1, img2, label, label, index
        else:
            return img1, label, label, index
    def __len__(self):
        return self.labels.shape[0]



if __name__=='__main__':
    dd = DatasetPorcessing('xxx', 'database')
    dd.get_labels()
    print(dd.num_class)