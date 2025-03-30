import os 
import random 
import numpy as np 

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR100

from .partial_dataset import PartialDataset

class PartialCIFAR100(PartialDataset):
    def __init__(self, root, img_size=32):
        super().__init__("CIFAR100", root, img_size, "Partial")
        self.transform_train = transforms.Compose([transforms.ToTensor(), transforms.Resize(img_size)])
        self.transform_valid = transforms.Compose([transforms.ToTensor(), transforms.Resize(img_size)])

        self.train_dataset = CIFAR100(root=self.root, train=True,  download=True, transform=self.transform_train)
        self.valid_dataset = CIFAR100(root=self.root, train=False, download=True, transform=self.transform_valid)