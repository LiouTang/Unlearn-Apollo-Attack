import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .attack_framework import Attack_Framework

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ULiRA(Attack_Framework):
    def __init__(self, shadow_models, args):
        super().__init__(shadow_models, args)

    def get_unlearned_model(self, idx):
        return

    def update_atk_summary(self, target_input, target_label, idx):
        self.summary[idx] = {
            "p_class_include"   : 0,
            "p_class_exclude"   : 0
        }
        return None