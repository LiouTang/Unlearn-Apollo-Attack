import os
import numpy as np
from scipy.stats import norm
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
    def __init__(self, target_model, dataset, shadow_models, args, idxs, shadow_col, unlearn_args):
        super().__init__(target_model, dataset, shadow_models, args, idxs, shadow_col, unlearn_args)
        self.unlearned_shadow_models = nn.ModuleList()
        for i in range(self.args.num_shadow):
            self.unlearned_shadow_models.append( self.get_unlearned_model(i) )
        # exit()

    def update_atk_summary(self, name, target_input, target_label, idx):
        if (not name in self.summary):
            self.summary[name] = dict()
        w_in, w_ex = [], []
        for i in self.include:
            model = self.unlearned_shadow_models[i]
            with torch.no_grad():
                target_output = model(target_input)
            w_in.append(self.w(target_output, target_label))
        for i in self.exclude:
            model = self.shadow_models[i]
            with torch.no_grad():
                target_output = model(target_input)
            w_ex.append(self.w(target_output, target_label))
        self.summary[name][idx] = {
            "target_input"      : target_input,
            "target_label"      : target_label,
            "w_in"      : w_in,
            "w_ex"      : w_ex,
        }
        return None
    
    def get_ternary_results(self, **kwargs):
        p = {}
        print("Calculating Ternary Results!")
        
        for name in ["unlearn", "retain", "test"]:
            p[name] = {}
            for i in self.summary[name]:
                with torch.no_grad():
                    target_output = self.target_model(self.summary[name][i]["target_input"])
                target_w = self.w(target_output, self.summary[name][i]["target_label"])
                if (len(self.summary[name][i]["w_in"]) == 0) or (len(self.summary[name][i]["w_ex"]) == 0):
                    p[name][i] = np.log(1)
                else:
                    p[name][i] = np.log(pr(target_w, self.summary[name][i]["w_in"]) / (pr(target_w, self.summary[name][i]["w_ex"]) + 1e-9))

        ths = np.unique([p["test"][i] for i in self.summary["test"]])
        ternary_points = []
        
        for th in tqdm(ths):
            classifications = {"unlearn": 0, "retain": 0, "test": 0}
            total_samples = 0
            
            for name in ["unlearn", "retain", "test"]:
                for i in self.summary[name]:
                    # High likelihood ratio indicates membership in unlearn set
                    if p[name][i] > th:
                        classifications["unlearn"] += 1
                    else:
                        # Low likelihood ratio indicates retain or test
                        if name == "retain":
                            classifications["retain"] += 1
                        else:
                            classifications["test"] += 1
                    total_samples += 1
            
            # Convert to proportions for ternary plot
            if total_samples > 0:
                ternary_point = [
                    classifications["unlearn"] / total_samples,
                    classifications["retain"] / total_samples,
                    classifications["test"] / total_samples
                ]
                ternary_points.append(ternary_point)
        
        return np.array(ternary_points), ths

def pr(x, obs):
    mean, std = norm.fit(obs)
    return norm.pdf(x, mean, std)