import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from functorch import make_functional_with_buffers, vmap

from .attack_framework import Attack_Framework

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Apollo(Attack_Framework):
    def __init__(self, target_model, dataset, shadow_models, args, idxs, shadow_col, unlearn_args):
        super().__init__(target_model, dataset, shadow_models, args, idxs, shadow_col, unlearn_args)
        self.types = ["Under", "Over"]
        self.unlearned_shadow_models = nn.ModuleList()
        for i in range(self.args.num_shadow):
            self.unlearned_shadow_models.append( self.get_unlearned_model(i) )
        self.eps = self.args.eps
    
    def set_include_exclude(self, target_idx):
        super().set_include_exclude(target_idx)
        self.func_un, self.params_un, self.buffers_un = batched_models_(
            [self.unlearned_shadow_models[i] for i in self.include]
        )
        self.func_rt, self.params_rt, self.buffers_rt = batched_models_(
            [self.shadow_models[i] for i in self.exclude]
        )


    def get_near_miss_label(self, target_input, target_label):
        min_loss_label = None
        min_loss = None
        for label in range(self.args.num_classes):
            if (label == target_label.item()):
                continue

            sum_loss = 0
            for i in self.exclude:
                adv_output = self.shadow_models[i](target_input)
                loss = F.cross_entropy(adv_output, torch.Tensor([label]).to(torch.int64).to(DEVICE))
                sum_loss += loss.item()

            if (min_loss == None):
                min_loss, min_loss_label = sum_loss, label
            elif (sum_loss < min_loss):
                min_loss, min_loss_label = sum_loss, label
        return torch.Tensor([min_loss_label]).to(torch.int64).to(DEVICE)

    def loss(self, input, label_un, label_rt):
        loss_un, loss_rt = 0.0, 0.0
        for i in self.include:
            output = self.unlearned_shadow_models[i](input)
            loss_un += F.cross_entropy(output, label_un)
        for i in self.exclude:
            output = self.shadow_models[i](input)
            loss_rt += F.cross_entropy(output, label_rt)
        return  self.args.w[1] * loss_un / len(self.include) + \
                self.args.w[2] * loss_rt / len(self.include)

    def batched_loss(self, input, label_un, label_rt):
        loss_un = batched_loss_(input, label_un, self.func_un, self.params_un, self.buffers_un)
        loss_rt = batched_loss_(input, label_rt, self.func_rt, self.params_rt, self.buffers_rt)
        return  self.args.w[1] * loss_un / len(self.include) + \
                self.args.w[2] * loss_rt / len(self.include)

    def Under_Un_Adv(self, target_input, target_label):
        # Under-Unlearning: Given target (x, y), adv. input x',
        # original model (x in train set) x' --> y
        # unlearned model (x in unlearned set) x' --> y
        # retrained model (x not in unlearned set) x' --> y'
        adv_input = target_input.detach().clone().to(DEVICE)
        adv_input.requires_grad = True
        adv_label = self.get_near_miss_label(target_input, target_label)
        optimizer = torch.optim.SGD([adv_input], lr=self.args.atk_lr)

        conf, pred = [], []
        for epoch in range(self.args.atk_epochs):
            optimizer.zero_grad()
            loss = self.batched_loss(adv_input, target_label, adv_label)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                projected = proj(target_input, adv_input.data, self.eps * (epoch + 1) / (self.args.atk_epochs))
                adv_input.data.copy_(projected)
                adv_input.data.clamp_(0.0, 1.0)

            with torch.no_grad():
                adv_output = self.target_model(adv_input)
            pred.append(adv_output.max(1)[1].item())

            sum = 0
            for i in self.exclude:
                with torch.no_grad():
                    output = self.shadow_models[i](adv_input)
                sum += F.softmax(output, dim=1)[0, target_label].item()
            conf.append(sum / len(self.exclude))
        return conf, pred

    def Over_Un_Adv(self, target_input, target_label):
        # Over-Unlearning: Given target (x, y), adv. input x',
        # original model (x in train set) x' --> y
        # unlearned model (x in unlearned set) x' --> y'
        # retrained model (x not in unlearned set) x' --> y
        adv_input = target_input.detach().clone().to(DEVICE)
        adv_input.requires_grad = True
        adv_label = target_label.to(torch.int64).to(DEVICE)
        optimizer = torch.optim.SGD([adv_input], lr=self.args.atk_lr)

        conf, pred = [], []
        for epoch in range(self.args.atk_epochs):
            optimizer.zero_grad()
            loss = self.batched_loss(adv_input, adv_label, target_label)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                projected = proj(target_input, adv_input.data, self.eps * (epoch + 1) / (self.args.atk_epochs))
                adv_input.data.copy_(projected)
                adv_input.data.clamp_(0.0, 1.0)

            with torch.no_grad():
                adv_output = self.target_model(adv_input)
            pred.append(adv_output.max(1)[1].item())

            sum = 0
            for i in self.exclude:
                with torch.no_grad():
                    output = self.shadow_models[i](adv_input)
                sum += F.softmax(output, dim=1)[0, target_label].item()
            conf.append(sum / len(self.exclude))
        return conf, pred

    def update_atk_summary(self, name, target_input, target_label, idx):
        if (not name in self.summary):
            self.summary[name] = dict()
        un_conf, un_pred = self.Under_Un_Adv(target_input, target_label)
        ov_conf, ov_pred = self.Over_Un_Adv(target_input, target_label)
        self.summary[name][idx] = {
            "target_input"  : target_input,
            "target_label"  : target_label,
            "un_conf"       : un_conf,
            "un_pred"       : un_pred,
            "ov_conf"       : ov_conf,
            "ov_pred"       : ov_pred,
        }
        return None

    def get_results(self, **kwargs):
        return eval(f"self.get_results_{kwargs['type']}")()

    def get_results_Under(self):
        tp, fp, fn, tn = [], [], [], []
        gt, conf, pred = {}, {}, {}
        ths = np.arange(0, 10, 1e-2)

        print("Calculating Results!")
        for name in ["unlearn", "valid"]:
            gt[name]   = torch.cat([self.summary[name][i]["target_label"] for i in self.summary[name]], dim=0).cpu().numpy()
            conf[name] = np.array([self.summary[name][i]["un_conf"] for i in self.summary[name]])
            pred[name] = np.array([self.summary[name][i]["un_pred"] for i in self.summary[name]])

        for th in tqdm(ths):
            _tp, _fp, _fn, _tn = 0, 0, 0, 0
            for name in ["unlearn", "valid"]:
                idx = np.where((conf[name] < th), np.arange(self.args.atk_epochs), self.args.atk_epochs).min(axis=1)
                idx[idx == self.args.atk_epochs] = (self.args.atk_epochs - 1)
                pred_th = pred[name][np.arange(self.args.N), idx]

                if (name == "unlearn"):
                    _tp += np.sum(pred_th == gt[name])
                    _fn += np.sum(pred_th != gt[name])
                else:
                    _fp += np.sum(pred_th == gt[name])
                    _tn += np.sum(pred_th != gt[name])
            tp.append(_tp)
            fp.append(_fp)
            fn.append(_fn)
            tn.append(_tn)
        return np.array(tp), np.array(fp), np.array(fn), np.array(tn), ths

    def get_results_Over(self):
        tp, fp, fn, tn = [], [], [], []
        gt, conf, pred = {}, {}, {}
        ths = np.arange(0, 10, 1e-2)

        print("Calculating Results!")
        for name in ["unlearn", "valid"]:
            gt[name]   = torch.cat([self.summary[name][i]["target_label"] for i in self.summary[name]], dim=0).cpu().numpy()
            conf[name] = np.array([self.summary[name][i]["ov_conf"] for i in self.summary[name]])
            pred[name] = np.array([self.summary[name][i]["ov_pred"] for i in self.summary[name]])

        for th in tqdm(ths):
            _tp, _fp, _fn, _tn = 0, 0, 0, 0
            for name in ["unlearn", "valid"]:
                idx = np.where((conf[name] < th), np.arange(self.args.atk_epochs), self.args.atk_epochs).min(axis=1)
                idx[idx == self.args.atk_epochs] = (self.args.atk_epochs - 1)
                pred_th = pred[name][np.arange(self.args.N), idx]

                if (name == "unlearn"):
                    _tp += np.sum(pred_th != gt[name])
                    _fn += np.sum(pred_th == gt[name])
                else:
                    _fp += np.sum(pred_th != gt[name])
                    _tn += np.sum(pred_th == gt[name])
            tp.append(_tp)
            fp.append(_fp)
            fn.append(_fn)
            tn.append(_tn)
        return np.array(tp), np.array(fp), np.array(fn), np.array(tn), ths


def batched_models_(models_list):
    func, _, _ = make_functional_with_buffers(models_list[0])
    params_list  = [tuple(m.parameters()) for m in models_list]
    buffers_list = [tuple(m.buffers())    for m in models_list]

    batched_params  = tuple(torch.stack(p, dim=0) for p in zip(*params_list))
    batched_buffers = tuple(torch.stack(b, dim=0) for b in zip(*buffers_list))
    return func, batched_params, batched_buffers
    
def batched_loss_(input, label, func, params, buffers):
    outputs = vmap(lambda p, b, x: func(p, b, x), in_dims=(0, 0, None))(params, buffers, input)
    flat = outputs.reshape(-1, outputs.size(-1))
    label_rep = label.repeat(outputs.size(0))
    return F.cross_entropy(flat, label_rep, reduction="mean")

def normalize(tensor: torch.Tensor):
    norm = torch.norm(tensor, p=2, dim=-1, keepdim=True)
    return tensor / (norm + 1e-9), norm

def proj(A: torch.Tensor, B: torch.Tensor, r: float):
    with torch.no_grad():
        d = (B - A).view(-1).norm(p=2).item()
        scale = min(1.0, r / (d + 1e-9))
        return A + (B - A) * scale