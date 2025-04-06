import os 
import random 
import numpy as np 

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import SVHN

from .partial_dataset import PartialDataset

class PartialSVHN(PartialDataset):
    def __init__(self, root, img_size=32):
        super().__init__("SVHN", root, img_size, "Partial")
        self.transform_train = transforms.Compose([transforms.ToTensor(), transforms.Resize(img_size, antialias=True)])
        self.transform_valid = transforms.Compose([transforms.ToTensor(), transforms.Resize(img_size, antialias=True)])

        self.train_dataset = SVHN(root=self.root, split='train', download=True, transform=self.transform_train)
        self.valid_dataset = SVHN(root=self.root, split='test',  download=True, transform=self.transform_valid)