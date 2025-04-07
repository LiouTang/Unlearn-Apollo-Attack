import torch
import torch.nn as nn

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Attack_Framework():
    def __init__(self, shadow_models, args):
        self.num_shadow = args.num_shadow
        self.shadow_models = shadow_models
        self.include, self.exclude = [], []
        self.args = args
        self.summary = dict()

        self.CE = nn.CrossEntropyLoss()

    def set_include_exclude(self, target_idx, shadow_idx_collection):
        include, exclude = [], []
        for i in range(self.num_shadow):
            if (target_idx in set(shadow_idx_collection[i])):
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
