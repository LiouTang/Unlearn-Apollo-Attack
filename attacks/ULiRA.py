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
        exit()

    def get_unlearned_model(self, i: int):
        unlearned_model = create_model(model_name=self.args.shadow_model, num_classes=self.args.num_classes)
        save_path = os.path.join(self.args.shadow_path, f"{self.unlearn_args.size_train}-{self.unlearn_args.unlearn}-{self.args.N}")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        weights_path = os.path.join(save_path, f"{i}.pth.tar")

        if os.path.exists(weights_path):
            unlearned_model.load_state_dict(torch.load(weights_path, map_location=DEVICE, weights_only=True))
            unlearned_model.to(DEVICE)
            unlearned_model.eval()
            return unlearned_model
        else:
            forget_idx = np.array(list( set(self.idxs["unlearn"]).intersection(self.shadow_col[i]) ))
            retain_idx = np.array(list( set(self.idxs["unlearn"]).difference(self.shadow_col[i])   ))
            print(">>>", forget_idx[:5], retain_idx[:5])

            forget_set = self.dataset.get_subset(self.dataset.train_dataset, forget_idx)
            retain_set = self.dataset.get_subset(self.dataset.train_dataset, retain_idx)

            forget_loader = DataLoader(forget_set, batch_size=self.unlearn_args.batch_size, shuffle=True, num_workers=4)
            retain_loader = DataLoader(retain_set, batch_size=self.unlearn_args.batch_size, shuffle=True, num_workers=4)

            unlearn_dataloaders = OrderedDict(
                forget_train = forget_loader, retain_train = retain_loader,
                forget_valid = None, retain_valid = None,
            )

            unlearn_method = unlearn.create_unlearn_method(self.unlearn_args.unlearn)(self.shadow_models[i], self.CE, weights_path, self.unlearn_args)
            unlearn_method.prepare_unlearn(unlearn_dataloaders)
            unlearned_model = unlearn_method.get_unlearned_model()
            torch.save(unlearned_model.state_dict(), weights_path)

            return unlearned_model

    def update_atk_summary(self, name, target_input, target_label, idx):
        if (not name in self.summary):
            self.summary[name] = dict()
        logit_in, logit_ex = [], []
        for i in self.include:
            model = self.get_unlearned_model(i)
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
    
    def get_results(self, target_model):
        tp, fp, fn, tn = [], [], [], []

        for th in np.arange(0, 1, 1e-3):
            _tp, _fp, _fn, _tn = 0, 0, 0, 0
            for name in ["unlearn", "retain", "test"]:
                for i in self.summary[name]:
                    with torch.no_grad():
                        target_output = target_model(self.summary[name][i])
                    target_logit = target_output[0, self.summary[name][i]["target_label"]]
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
        return tp, fp, fn, tn

def pr(x, obs):
    mean, std = norm.fit(obs)
    return norm.pdf(x, mean, std)