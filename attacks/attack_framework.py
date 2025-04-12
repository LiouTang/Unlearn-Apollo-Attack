import torch
import torch.nn as nn
import copy
from collections import OrderedDict
from dataset import PartialDataset

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Attack_Framework():
    def __init__(self, dataset : PartialDataset, shadow_models : nn.ModuleList, args, idxs: OrderedDict, shadow_col: dict[list], unlearn_args):
        self.dataset = dataset
        self.shadow_models = shadow_models
        self.include, self.exclude = [], []
        self.args = args
        self.idxs = idxs
        self.shadow_col = {}
        for i in shadow_col:
            self.shadow_col[i] = set(shadow_col[i])
        self.unlearn_args = unlearn_args

        self.summary = dict()
        self.CE = nn.CrossEntropyLoss()

    def set_include_exclude(self, target_idx):
        include, exclude = [], []
        for i in range(self.args.num_shadow):
            if (target_idx in set(self.shadow_col[i])):
                include.append(i)
            else:
                exclude.append(i)
        if self.args.debug:
            print("target idx:", target_idx, include, exclude)
        self.include, self.exclude = include, exclude

    def get_labels(self, input): # For Debugging Purposes
        include_labels = []
        for i in self.include:
            with torch.no_grad():
                adv_output = self.shadow_models[i](input)
            pred_label = adv_output.max(1)[1]
            include_labels.append(pred_label.item())
        exclude_labels = []
        for i in self.exclude:
            with torch.no_grad():
                adv_output = self.shadow_models[i](input)
            pred_label = adv_output.max(1)[1]
            exclude_labels.append(pred_label.item())
        return include_labels, exclude_labels

    def update_atk_summary(self, name, target_input, target_label, idx) -> dict:
        return {}
    def get_atk_summary(self):
        summary = self.summary.copy()
        self.summary = dict()
        return summary

    def get_results(self, target_model):
        return
