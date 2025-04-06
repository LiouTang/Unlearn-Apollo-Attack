import os
import random
import numpy as np

import torch
from torch.utils.data import Dataset

from .partial_dataset import PartialDataset

import pickle as pkl
import utils

def generate_dataset(N, num_classes=4):
    utils.random_seed()
    train_dataset = []
    train_labels = []

    valid_dataset = []
    valid_labels = []
    for i in range(N):
        for _ in range(50):
            a = np.random.uniform( (2 * np.pi) / N * i, (2 * np.pi) / N * (i + 1) )
            r = np.random.uniform(0.1, 1)
            train_dataset.append([r * np.cos(a), r * np.sin(a)])
            train_labels.append((i % num_classes))
        
        for _ in range(20):
            a = np.random.uniform( (2 * np.pi) / N * i, (2 * np.pi) / N * (i + 1) )
            r = np.random.uniform(0.1, 1)
            valid_dataset.append([r * np.cos(a), r * np.sin(a)])
            valid_labels.append((i % num_classes))
    return np.array(train_dataset), np.array(train_labels), np.array(valid_dataset), np.array(valid_labels)

class ToyDataset(Dataset):
    """Dataset class for ImageNet"""
    def __init__(self, dataset, labels, train=False):
        super(ToyDataset, self).__init__()
        assert(len(dataset) == len(labels))
        self.data = dataset
        self.targets = labels

        self.transform = None
        self.train = train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.Tensor(self.data)[idx], torch.Tensor(self.targets)[idx]

class PartialToyData(PartialDataset):
    def __init__(self, root, img_size=0):
        super().__init__("ToyData", root, img_size, "Partial")
        train_dataset, train_labels, valid_dataset, valid_labels = generate_dataset(N=4)

        self.train_dataset = ToyDataset(train_dataset, train_labels, train=True)
        self.valid_dataset = ToyDataset(valid_dataset, valid_labels, train=False)