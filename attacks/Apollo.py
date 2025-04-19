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
        self.unlearned_shadow_models = nn.ModuleList()
        for i in range(self.args.num_shadow):
            self.unlearned_shadow_models.append( self.get_unlearned_model(i) )
        self.max_dist = 0.0

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

            loss_include, loss_exclude, loss_d = 0.0, 0.0, 0.0
            # unlearned model (x in unlearned set) x' --> y
            for i in self.include:
                adv_output = self.unlearned_shadow_models[i](adv_input)
                loss_include += F.cross_entropy(adv_output, target_label)
            # retrained model (x not in unlearned set) x' --> y'
            for i in self.exclude:
                adv_output = self.shadow_models[i](adv_input)
                loss_exclude += F.cross_entropy(adv_output, adv_label)
            # distance to target (locality)
            loss_d = F.mse_loss(adv_input, target_input)

            loss = self.args.w[0] * loss_include / len(self.include) + \
                   self.args.w[1] * loss_exclude / len(self.exclude) + \
                   self.args.w[2] * loss_d
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
        return adv_input, torch.norm(target_input - adv_input, p=2).item()

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

            loss_include, loss_exclude, loss_d = 0.0, 0.0, 0.0
            # unlearned model (x in unlearned set) x' --> y'
            for i in self.include:
                adv_output = self.unlearned_shadow_models[i](adv_input)
                loss_include += F.cross_entropy(adv_output, adv_label)
            # retrained model (x not in unlearned set) x' --> y
            for i in self.exclude:
                adv_output = self.shadow_models[i](adv_input)
                loss_exclude += F.cross_entropy(adv_output, target_label)
            # distance to target (locality)
            loss_d = F.mse_loss(adv_input, target_input)

            loss = self.args.w[0] * loss_include / len(self.include) + \
                   self.args.w[1] * loss_exclude / len(self.exclude) + \
                   self.args.w[2] * loss_d
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
        return adv_input, torch.norm(target_input - adv_input, p=2).item()

    def update_atk_summary(self, name, target_input, target_label, idx):
        if (not name in self.summary):
            self.summary[name] = dict()
        under_adv_input, under_dist = self.Under_Un_Adv(target_input, target_label)
        over_adv_input,  over_dist  = self.Over_Un_Adv(target_input, target_label)
        self.max_dist = max(max(self.max_dist, under_dist), over_dist)
        self.summary[name][idx] = {
            "target_input"      : target_input,
            "target_label"      : target_label,
            "under_adv_input"   : under_adv_input,
            "over_adv_input"    : over_adv_input,
        }
        return None

    def get_results(self, target_model, **kwargs):
        return eval(f"self.get_results_{kwargs['type']}")(target_model)

    def get_results_Under(self, target_model):
        tp, fp, fn, tn = [], [], [], []

        print("Calculating Results!")
        for eps in tqdm(np.arange(0, self.max_dist * 2, 1e-3)):
            _tp, _fp, _fn, _tn = 0, 0, 0, 0
            for name in ["unlearn", "valid"]:
                inputs = torch.cat([self.summary[name][i]["target_input"] for i in self.summary[name]], dim=0)
                gt     = torch.cat([self.summary[name][i]["target_label"] for i in self.summary[name]], dim=0)

                under_adv = torch.cat([self.summary[name][i]["under_adv_input"] for i in self.summary[name]], dim=0)
                diff, _ = normalize(under_adv - inputs)
                under_eps = inputs + diff * eps

                with torch.no_grad():
                    under_outputs = target_model(under_eps)
                under_pred = under_outputs.max(1)[1]
                # print(eps, name,  under_pred)

                if (name == "unlearn"):
                    _tp += np.sum( (under_pred.cpu().numpy() == gt.cpu().numpy()) )
                    _fn += np.sum( (under_pred.cpu().numpy() != gt.cpu().numpy()) )
                else:
                    _fp += np.sum( (under_pred.cpu().numpy() == gt.cpu().numpy()) )
                    _tn += np.sum( (under_pred.cpu().numpy() != gt.cpu().numpy()) )
            tp.append(_tp)
            fp.append(_fp)
            fn.append(_fn)
            tn.append(_tn)
        return np.array(tp), np.array(fp), np.array(fn), np.array(tn)

    def get_results_Over(self, target_model):
        tp, fp, fn, tn = [], [], [], []

        print("Calculating Results!")
        for eps in tqdm(np.arange(0, self.max_dist * 2, 1e-3)):
            _tp, _fp, _fn, _tn = 0, 0, 0, 0
            for name in ["unlearn", "valid"]:
                inputs = torch.cat([self.summary[name][i]["target_input"] for i in self.summary[name]], dim=0)
                gt     = torch.cat([self.summary[name][i]["target_label"] for i in self.summary[name]], dim=0)

                over_adv  = torch.cat([self.summary[name][i]["over_adv_input"]  for i in self.summary[name]], dim=0)
                diff, _ = normalize(over_adv - inputs)
                over_eps = inputs + diff * eps

                with torch.no_grad():
                    over_outputs  = target_model(over_eps)
                over_pred  = over_outputs.max(1)[1]
                # print(eps, name, over_pred)

                if (name == "unlearn"):
                    _tp += np.sum( (over_pred.cpu().numpy() != gt.cpu().numpy()) )
                    _fn += np.sum( (over_pred.cpu().numpy() == gt.cpu().numpy()) )
                else:
                    _fp += np.sum( (over_pred.cpu().numpy() != gt.cpu().numpy()) )
                    _tn += np.sum( (over_pred.cpu().numpy() == gt.cpu().numpy()) )
            tp.append(_tp)
            fp.append(_fp)
            fn.append(_fn)
            tn.append(_tn)
        return np.array(tp), np.array(fp), np.array(fn), np.array(tn)

def normalize(tensor):
    norm = torch.norm(tensor, p=2, dim=-1, keepdim=True)
    return tensor / (norm + 1e-9), norm