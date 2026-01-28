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

    def get_ternary_results(self, **kwargs):
        p = {}
        print("Calculating UMIA Ternary Results!")
        
        for name in ["unlearn", "retain", "test"]:
            p[name] = {}
            for i in self.summary[name]:
                p[name][i] = softmax(self.summary[name][i]["p"])

        all_probs = []
        for name in ["unlearn", "retain", "test"]:
            for i in self.summary[name]:
                all_probs.append(p[name][i])
        ths = np.unique(all_probs)
        ternary_points = []
        
        for th in tqdm(ths):
            classifications = {"unlearn": 0, "retain": 0, "test": 0}
            total_samples = 0
            
            for name in ["unlearn", "retain", "test"]:
                for i in self.summary[name]:
                    likelihood_ratio = p[name][i]
                    if likelihood_ratio > th:
                        classifications["unlearn"] += 1
                    else:
                        if name == "test":
                            classifications["test"] += 1
                        else:
                            classifications["retain"] += 1
                    total_samples += 1

            if total_samples > 0:
                ternary_point = [
                    classifications["unlearn"] / total_samples,
                    classifications["retain"] / total_samples,
                    classifications["test"] / total_samples
                ]
                ternary_points.append(ternary_point)
        
        return np.array(ternary_points), ths

def cat(A, B) -> torch.Tensor:
    if (A == None):
        return B
    else:
        return torch.cat([A, B], dim=0)

def softmax(output):
    return np.exp(output[0, 1]) / np.sum(np.exp(output))