import torch
import torch.nn as nn
from torch.utils.data import Dataset

import copy
import numpy as np

import utils

class PartialDataset:
    def __init__(self, dataset_name, root, img_size, setting="Partial"):
        super().__init__()
        self.dataset_name = dataset_name
        self.root = root
        self.img_size = img_size
        self.setting = setting
        self.train_dataset = Dataset
        self.valid_dataset = Dataset

    def get_subset(self, dataset=None, idx=None) -> Dataset:
        new_dataset = copy.deepcopy(dataset)
        new_dataset.data = dataset.data[idx]
        try:
            new_dataset.targets = np.array(dataset.targets)[idx]
        except:
            new_dataset.labels = np.array(dataset.labels)[idx]
        return new_dataset

    def set_train_shadow_idx(self, size_train, size_shadow=0, num_shadow=0, split="half", seed=42):
        utils.random_seed(seed)

        N = len(self.train_dataset)
        full_idx = np.arange(N)
        np.random.shuffle(full_idx)

        train_idx, shadow_idx = full_idx[:(N // 2)], full_idx[(N // 2):]
        train_idx = np.random.choice(train_idx, size=size_train, replace=False)

        shadow_idx_collection = dict()
        if (split == "half"):
            for i in range(num_shadow):
                shadow_idx_collection[i] = np.random.choice(shadow_idx, size=size_shadow, replace=False)
        elif (split == "full"):
            for i in range(num_shadow):
                shadow_idx_collection[i] = np.random.choice(full_idx, size=size_shadow, replace=False)

        self.train_idx = train_idx
        self.shadow_idx_collection = shadow_idx_collection

        print("train:", len(train_idx), train_idx[:5])

    def set_unlearn_idx(self, un_perc=None, un_class=None, seed=42):
        utils.random_seed(seed)

        temp_train_idx = self.train_idx
        np.random.shuffle(temp_train_idx)
        if (un_perc != None):
            un_len = int(len(temp_train_idx) * un_perc)
            unlearn_idx, retain_idx = temp_train_idx[:un_len], temp_train_idx[un_len:]

        if (un_class != None):
            try:
                un_mask = np.array(self.train_dataset.targets)[temp_train_idx] == 0
            except:
                un_mask = np.array(self.train_dataset.labels)[temp_train_idx] == 0
            unlearn_idx, retain_idx = temp_train_idx[un_mask], temp_train_idx[np.logical_not(un_mask)]

        print("unlearn:", len(unlearn_idx), unlearn_idx[:5], "retain", len(retain_idx), retain_idx[:5])
        self.unlearn_idx = unlearn_idx
        self.retain_idx = retain_idx
    
    def set_ULiRA_idx(self, num_shadow=0, seed=42):
        utils.random_seed(seed)

        N = len(self.train_dataset)
        full_idx = np.arange(N)
        ULiRA_idx_collection = dict()
        for i in range(num_shadow):
            ULiRA_idx_collection[i] = np.random.choice(full_idx, size=int(N // 2), replace=False)
            print("U-LiRA: ", i, ULiRA_idx_collection[i])

        self.ULiRA_idx_collection = ULiRA_idx_collection
