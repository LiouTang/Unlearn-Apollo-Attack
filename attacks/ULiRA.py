import os
import numpy as np
import pandas as pd
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