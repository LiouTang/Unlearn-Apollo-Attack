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
    def __init__(self, dataset, shadow_models, args, idxs, shadow_col, unlearn_args):
        super().__init__(dataset, shadow_models, args, idxs, shadow_col, unlearn_args)

    def get_near_miss_label(self, target_input, target_label):
        min_loss_label = None
        min_loss = None
        for label in range(self.args.num_classes):
            if (label == target_label.item()):
                continue

            sum_loss = 0
            for i in self.exclude:
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
        optimizer = torch.optim.SGD([adv_input], lr=self.args.atk_lr)

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
        adv_input = adv_input.clone().detach()

        if (self.args.debug):
            include_labels, exclude_labels = self.get_labels(adv_input)
            print(self.include, self.exclude)
            print(">>>>>>>>>>", target_label, adv_label)
            print("include_labels: ", include_labels)
            print("exclude_labels: ", exclude_labels)
        return adv_input, torch.norm(target_input - adv_input, p=2)

    def Over_Un_Adv(self, target_input, target_label):
        # Over-Unlearning: Given target (x, y), adv. input x',
        # unlearned model (x in unlearned set) x' --> y'
        # retrained model (x not in unlearned set) x' --> y --> Find the nearest (x', y)
        adv_input = target_input.detach().clone().to(DEVICE)
        adv_input.requires_grad = True
        adv_label = target_label.to(torch.int64).to(DEVICE)
        optimizer = torch.optim.SGD([adv_input], lr=self.args.atk_lr)

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
        adv_input = adv_input.clone().detach()
        return adv_input, torch.norm(target_input - adv_input, p=2)

    def update_atk_summary(self, target_input, target_label, idx):
        under_adv_input, under_dist = self.Under_Un_Adv(target_input, target_label)
        over_adv_input,  over_dist  = self.Over_Un_Adv(target_input, target_label)
        self.summary[idx] = {
            "under_adv_input"   : under_adv_input,
            "under_dist"        : under_dist,
            "over_adv_input"    : over_adv_input,
            "over_dist"         : over_dist
        }
        return None