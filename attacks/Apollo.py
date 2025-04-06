import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .attack_framework import Attack_Framework

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Apollo(Attack_Framework):
    def __init__(self, shadow_models, args):
        super().__init__(shadow_models, args)

    def get_near_miss_label(self, target_input, target_label):
        min_loss_label = None
        min_loss = None
        for label in range(self.args.num_classes):
            if (label == target_label.item()):
                continue

            sum_loss = 0
            for i in range(len(self.exclude)):
                adv_output = self.shadow_models[i](target_input)
                loss = self.CE(adv_output, torch.Tensor([label]).to(torch.int64).to(DEVICE))
                sum_loss += loss.item()

            if (min_loss == None):
                min_loss, min_loss_label = sum_loss, label
            elif (sum_loss < min_loss):
                min_loss, min_loss_label = sum_loss, label
        return torch.Tensor([min_loss_label]).to(torch.int64).to(DEVICE)

    def Under_Un_Adv(self, target_input, target_label):
        # Under-Unlearning: Given target (x, y), adv. input x',
        # unlearned model (x in unlearned set) x' --> y
        # retrained model (x not in unlearned set) x' --> y' --> Find the nearest (x', y')
        adv_input = target_input.detach().clone().to(DEVICE)
        adv_input.requires_grad = True
        adv_label = self.get_near_miss_label(target_input, target_label)
        optimizer = torch.optim.SGD([adv_input], lr=self.args.lr)

        for epoch in range(self.args.atk_epochs):
            loss = 0.0
            optimizer.zero_grad()
            loss_exclude, loss_d, loss_db = 0.0, 0.0, 0.0

            # Only consider exclude
            for i in self.exclude:
                adv_output = self.shadow_models[i](adv_input)
                loss_exclude += F.cross_entropy(adv_output, adv_label)
                loss_db += adv_output.topk(2)[1][0, 0] - adv_output.topk(2)[1][0, 1]
            
            loss_d = F.mse_loss(adv_input, target_input)
            loss = self.args.w[0] * loss_exclude / len(self.exclude) + \
                   self.args.w[1] * loss_d + \
                   self.args.w[2] * loss_db / len(self.exclude)
            loss.backward()
            optimizer.step()
            torch.clamp(adv_input, min=0, max=1) # Image data
            if (loss.item() < .2):
                break

        if (self.args.debug):
            include_labels, exclude_labels = self.get_labels(adv_input)
            print(self.include, self.exclude)
            print(">>>>>>>>>>", target_label, adv_label)
            print("include_labels: ", include_labels)
            print("exclude_labels: ", exclude_labels)
        return adv_input.clone().detach()

    def Over_Un_Adv(self, target_input, target_label):
        return