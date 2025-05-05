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
            w_in.append(w(target_output, target_label))
        for i in self.exclude:
            model = self.shadow_models[i]
            with torch.no_grad():
                target_output = model(target_input)
            w_ex.append(w(target_output, target_label))
        self.summary[name][idx] = {
            "target_input"      : target_input,
            "target_label"      : target_label,
            "w_in"      : w_in,
            "w_ex"      : w_ex,
        }
        return None
    
    def get_results(self, **kwargs):
        tp, fp, fn, tn = [], [], [], []
        p = {}
        ths = np.arange(-5, 5, 1e-2)

        print("Calculating Results!")
        for name in ["unlearn", "valid"]:
            p[name] = {}
            for i in self.summary[name]:
                with torch.no_grad():
                    target_output = self.target_model(self.summary[name][i]["target_input"])
                target_w = w(target_output, self.summary[name][i]["target_label"])
                if (len(self.summary[name][i]["w_in"]) == 0) or (len(self.summary[name][i]["w_ex"]) == 0):
                    p[name][i] = 1
                else:
                    p[name][i] = pr(target_w, self.summary[name][i]["w_in"]) / (pr(target_w, self.summary[name][i]["w_ex"]) + 1e-6)

        for th in tqdm(ths):
            _tp, _fp, _fn, _tn = 0, 0, 0, 0
            for name in ["unlearn", "valid"]:
                for i in self.summary[name]:
                    if (name == "unlearn"):
                        _tp += int(p[name][i] > th)
                        _fn += int(p[name][i] <= th)
                    else:
                        _fp += int(p[name][i] > th)
                        _tn += int(p[name][i] <= th)
            tp.append(_tp)
            fp.append(_fp)
            fn.append(_fn)
            tn.append(_tn)
        return np.array(tp), np.array(fp), np.array(fn), np.array(tn), ths

def w(output, label):
    # with torch.no_grad():
    #     w = F.softmax(output, dim=1)[0, label.item()].item()
    # return np.log(w / (1 - w))
    return output[0, label.item()].item()

def pr(x, obs):
    mean, std = norm.fit(obs)
    return norm.pdf(x, mean, std)