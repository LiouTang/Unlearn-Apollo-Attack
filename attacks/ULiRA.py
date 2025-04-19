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
    def __init__(self, dataset, shadow_models, args, idxs, shadow_col, unlearn_args):
        super().__init__(dataset, shadow_models, args, idxs, shadow_col, unlearn_args)
        self.unlearned_shadow_models = nn.ModuleList()
        for i in range(self.args.num_shadow):
            self.unlearned_shadow_models.append( self.get_unlearned_model(i) )
        # exit()

    def update_atk_summary(self, name, target_input, target_label, idx):
        if (not name in self.summary):
            self.summary[name] = dict()
        logit_in, logit_ex = [], []
        for i in self.include:
            model = self.unlearned_shadow_models[i]
            with torch.no_grad():
                output = model(target_input)
            logit_in.append(output[0, target_label.item()].item())
        for i in self.exclude:
            model = self.shadow_models[i]
            with torch.no_grad():
                output = model(target_input)
            logit_ex.append(output[0, target_label.item()].item())
        self.summary[name][idx] = {
            "target_input"      : target_input,
            "target_label"      : target_label,
            "logit_in"      : logit_in,
            "logit_ex"      : logit_ex,
        }
        return None
    
    def get_results(self, target_model, **kwargs):
        tp, fp, fn, tn = [], [], [], []

        for th in tqdm(np.arange(0, 10, 1e-2)):
            _tp, _fp, _fn, _tn = 0, 0, 0, 0
            for name in ["unlearn", "valid"]:
                for i in self.summary[name]:
                    with torch.no_grad():
                        target_output = target_model(self.summary[name][i]["target_input"])
                    target_logit = target_output[0, self.summary[name][i]["target_label"]].item()
                    if (len(self.summary[name][i]["logit_in"]) == 0) or (len(self.summary[name][i]["logit_ex"]) == 0):
                        p = 1
                    else:
                        p = pr(target_logit, self.summary[name][i]["logit_in"]) / pr(target_logit, self.summary[name][i]["logit_ex"])

                    if (name == "unlearn"):
                        _tp += int(p > th)
                        _fn += int(p <= th)
                    else:
                        _fp += int(p > th)
                        _tn += int(p <= th)
            tp.append(_tp)
            fp.append(_fp)
            fn.append(_fn)
            tn.append(_tn)
        return np.array(tp), np.array(fp), np.array(fn), np.array(tn)

def pr(x, obs):
    mean, std = norm.fit(obs)
    return norm.pdf(x, mean, std)