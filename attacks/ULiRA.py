import os
import numpy as np
from tqdm import tqdm
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from .attack_framework import Attack_Framework
from models import create_model
import unlearn

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ULiRA(Attack_Framework):
    def __init__(self, shadow_models, args):
        super().__init__(shadow_models, args)

    def get_unlearned_model(self, i: int):
        unlearned_model = create_model(model_name=self.args.shadow_model, num_classes=self.args.num_classes)
        weights_path = os.path.join(self.args.shadow_path, f"unlearned_{i}.pth.tar")

        if os.path.exists(weights_path):
            unlearned_model.load_state_dict(torch.load(weights_path, map_location=DEVICE, weights_only=True))
            unlearned_model.to(DEVICE)
            unlearned_model.eval()
            return unlearned_model
        else:
            forget_idx = set(self.idxs["unlearn"]).intersection(self.shadow_col[i])
            retain_idx = set(self.idxs["unlearn"]).difference(self.shadow_col[i])

            forget_set = self.dataset.get_subset(self.dataset.train_dataset, forget_idx)
            retain_set = self.dataset.get_subset(self.dataset.train_dataset, retain_idx)

            forget_loader = DataLoader(forget_set, batch_size=self.unlearn_args.batch_size, shuffle=True, num_workers=4)
            retain_loader = DataLoader(retain_set, batch_size=self.unlearn_args.batch_size, shuffle=True, num_workers=4)

            unlearn_dataloaders = OrderedDict(
                forget_train = forget_loader, retain_train = retain_loader,
                forget_valid = None, retain_valid = None,
            )

            unlearn_method = unlearn.create_unlearn_method(self.unlearn_args.unlearn)(self.shadow_models[i], self.CE, weights_path, self.unlearn_args)
            unlearn_method.prepare_unlearn(unlearn_dataloaders)
            unlearned_model = unlearn_method.get_unlearned_model()
            return unlearned_model

    def update_atk_summary(self, target_input, target_label, idx):
        p_in, p_ex = {}, {}
        for i in self.include:
            model = self.get_unlearned_model(i)
            with torch.no_grad():
                output = model(target_input)
            print(">>>>>>>>>>>>>", output.shape)
            exit()
        for i in self.exclude:
            model = self.get_unlearned_model(i)
            with torch.no_grad():
                output = model(target_input)
        self.summary[idx] = {
            "p_in"   : p_in,
            "p_ex"   : p_ex,
        }
        return None