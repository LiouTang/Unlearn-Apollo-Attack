import os
import random
import numpy as np

import torch
from torch.utils.data import Dataset, ConcatDataset, Subset

from .pretrain_dataset import PretrainDataset, train_shadow_split
from .unlearn_dataset import UnLearnDataset, replace_indexes

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

class PretrainToyData(PretrainDataset):
    def __init__(self, root, img_size=0):
        super().__init__("ToyData", root, img_size, "pretrain")

    def get_datasets(self):
        data_path = os.path.join(self.root, "ToyData.tar.gz")
        train_dataset, train_labels, valid_dataset, valid_labels = generate_dataset(N=4)
        
        self.train_dataset_full = ToyDataset(train_dataset, train_labels, train=True)
        train_idx, shadow_idx = train_shadow_split(self.train_dataset_full)
        self.train_dataset, self.shadow_dataset = replace_indexes(self.train_dataset_full, train_idx), replace_indexes(self.train_dataset_full, shadow_idx)
        self.valid_dataset = ToyDataset(valid_dataset, valid_labels, train=False)

        return self.train_dataset, self.shadow_dataset, self.valid_dataset

class RandomUnlearnToyData(UnLearnDataset):
    def __init__(self, root, img_size=0):
        super().__init__("ToyData", root, img_size, "random")

    def get_datasets(self):
        data_path = os.path.join(self.root, "ToyData.tar.gz")
        train_dataset, train_labels, valid_dataset, valid_labels = generate_dataset(N=4)
        
        self.train_dataset_full = ToyDataset(train_dataset, train_labels, train=True)
        train_idx, shadow_idx = train_shadow_split(self.train_dataset_full)
        self.train_dataset, self.shadow_dataset = replace_indexes(self.train_dataset_full, train_idx), replace_indexes(self.train_dataset_full, shadow_idx)
        self.valid_dataset = ToyDataset(valid_dataset, valid_labels, train=False)

        return self.train_dataset, self.shadow_dataset, self.valid_dataset

    def random_split(self, forget_perc):
        # random_indexes_path = os.path.join(save_path, "random_idx.npy")
        random_indexes = self.get_random_indexes()
        forget_len = int(len(self.train_dataset) * forget_perc)
        forget_train_indexes = random_indexes[:forget_len]
        retain_train_indexes = random_indexes[forget_len:]
        print(forget_train_indexes, retain_train_indexes)
        self.forget_trainset = replace_indexes(self.train_dataset, forget_train_indexes)
        self.retain_trainset = replace_indexes(self.train_dataset, retain_train_indexes)
        
        return self.forget_trainset, self.retain_trainset
    
    def get_random_indexes(self):
        train_len = len(self.train_dataset)
        random_indexes = list(range(train_len))
        random.shuffle(random_indexes)
        random_indexes = np.array(random_indexes)
        return random_indexes