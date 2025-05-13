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

from sklearn.svm import SVC

from .attack_framework import Attack_Framework
from models import create_model
import unlearn

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class UMIA(Attack_Framework):
    def __init__(self, target_model, dataset, shadow_models, args, idxs, shadow_col, unlearn_args):
        super().__init__(target_model, dataset, shadow_models, args, idxs, shadow_col, unlearn_args)
    def set_include_exclude(self, target_idx):
        pass

    def train_surr(self, surr_idxs, surr_loaders):
        print(">>> U-MIA training")
        X_surr, Y_surr = None, []
        for name, loader in surr_loaders.items():
            for i, (input, label) in enumerate(pbar := tqdm(loader)):
                input, label = input.to(DEVICE), label.to(DEVICE)
                with torch.no_grad():
                    output = self.target_model(input)
                    X_surr = cat(X_surr, output)
                Y_surr.append(int(name == "unlearn"))
        X_surr = X_surr.cpu().numpy()
        Y_surr = np.array(Y_surr)

        self.clf = SVC(C=3, gamma="auto", kernel="rbf", probability=True)
        self.clf.fit(X_surr, Y_surr)

    def update_atk_summary(self, name, target_input, target_label, idx):
        if (not name in self.summary):
            self.summary[name] = dict()

        with torch.no_grad():
            target_output = self.target_model(target_input)

        self.summary[name][idx] = {
            "target_input"  : target_input,
            "target_label"  : target_label,
            "p"             : self.clf.predict_log_proba(target_output.cpu().numpy())
        }
        return None

    def get_roc(self, **kwargs):
        tp, fp, fn, tn = [], [], [], []
        p = {}

        print("Calculating Results!")
        for name in ["unlearn", "valid"]:
            p[name] = {}
            for i in self.summary[name]:
                p[name][i] = softmax(self.summary[name][i]["p"])

        ths = np.unique([p["valid"][i] for i in self.summary[name]])
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

def cat(A, B) -> torch.Tensor:
    if (A == None):
        return B
    else:
        return torch.cat([A, B], dim=0)

def softmax(output):
    return np.exp(output[0, 1]) / np.sum(np.exp(output))