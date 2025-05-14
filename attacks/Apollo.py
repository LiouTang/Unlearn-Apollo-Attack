import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.func import stack_module_state, functional_call

from .attack_framework import Attack_Framework

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Apollo(Attack_Framework):
    def __init__(self, target_model, dataset, shadow_models, args, idxs, shadow_col, unlearn_args):
        super().__init__(target_model, dataset, shadow_models, args, idxs, shadow_col, unlearn_args)
        self.types = ["Under", "Over"]
        self.unlearned_shadow_models = nn.ModuleList()
        for i in range(self.args.num_shadow):
            self.unlearned_shadow_models.append(self.get_unlearned_model(i))

    def set_include_exclude(self, target_idx):
        super().set_include_exclude(target_idx)
        if (len(self.include)):
            self.temp_un, self.params_un, self.buffers_un = batched_models_(
                [self.unlearned_shadow_models[i] for i in self.include]
            )
        if (len(self.exclude)):
            self.temp_rt, self.params_rt, self.buffers_rt = batched_models_(
                [self.shadow_models[i] for i in self.exclude]
            )


    def batched_loss(self, input, label):
        loss_un, loss_rt = 0., 0.
        if (len(self.include)):
            loss_un = batched_loss_(input, label, self.temp_un, self.params_un, self.buffers_un)
        if (len(self.exclude)):            
            loss_rt = batched_loss_(input, label, self.temp_rt, self.params_rt, self.buffers_rt)
        return  self.args.w[0] * loss_un - self.args.w[1] * loss_rt
    def batched_loss_Under(self, input, label):
        return  self.batched_loss(input, label)
    def batched_loss_Over(self, input, label):
        return  -self.batched_loss(input, label)

    def Un_Adv(self, target_input, target_label, loss_func):
        # Under-Unlearning: Given target (x, y), adv. input x',
        # original model (x in train set) x' --> y
        # unlearned model (x in unlearned set) x' --> y
        # retrained model (x not in unlearned set) x' --> y'
        adv_input = target_input.detach().clone().to(DEVICE)
        adv_input.requires_grad = True
        # adv_label = self.get_near_miss_label(target_input, target_label)
        optimizer = torch.optim.SGD([adv_input], lr=self.args.atk_lr)

        conf, pred = [], []
        for epoch in range(self.args.atk_epochs):
            optimizer.zero_grad()
            loss = loss_func(adv_input, target_label)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                projected = proj(target_input, adv_input.data, self.args.eps * (epoch + 1) / (self.args.atk_epochs))
                adv_input.data.copy_(projected)
                adv_input.data.clamp_(0.0, 1.0)

            with torch.no_grad():
                projected = proj_out(target_input, adv_input.data, self.args.eps * (epoch) / (self.args.atk_epochs))
                adv_input.data.copy_(projected)
                adv_input.data.clamp_(0.0, 1.0)

            with torch.no_grad():
                adv_output = self.target_model(adv_input)
            pred.append(adv_output.max(1)[1].item())

            sum = 0
            with torch.no_grad():
                for i in self.exclude:
                    output = self.shadow_models[i](adv_input)
                    sum += self.w(output, target_label)
            if (len(self.exclude)):
                conf.append(sum / len(self.exclude))
            else:
                conf.append(0.)
        return conf, pred

    def update_atk_summary(self, name, target_input, target_label, idx):
        if (not name in self.summary):
            self.summary[name] = dict()
        un_conf, un_pred = self.Un_Adv(target_input, target_label, self.batched_loss_Under)
        ov_conf, ov_pred = self.Un_Adv(target_input, target_label, self.batched_loss_Over)
        self.summary[name][idx] = {
            "target_input"  : target_input,
            "target_label"  : target_label,
            "un_conf"       : un_conf,
            "un_pred"       : un_pred,
            "ov_conf"       : ov_conf,
            "ov_pred"       : ov_pred,
        }
        return None

    def get_roc(self, **kwargs):
        prefix      = "un" if kwargs["type"] == "Under" else "ov"
        compare     = (lambda c, th: c < th) if kwargs["type"] == "Under" else (lambda c, th: c > th)
        is_positive = (lambda p, g: p == g)  if kwargs["type"] == "Under" else (lambda p, g: p != g)

        tp, fp, fn, tn = [], [], [], []
        gt, conf, pred = {}, {}, {}

        print("Calculating Results!")
        for name in ["unlearn", "valid"]:
            gt[name] =  torch.cat([self.summary[name][i]["target_label"]   for i in self.summary[name]], dim=0).cpu().numpy()
            conf[name] = np.array([self.summary[name][i][f"{prefix}_conf"] for i in self.summary[name]])
            pred[name] = np.array([self.summary[name][i][f"{prefix}_pred"] for i in self.summary[name]])

        ths = np.unique(conf["valid"])
        for th in tqdm(ths):
            _tp, _fp, _fn, _tn = 0, 0, 0, 0
            for name in ["unlearn", "valid"]:
                idx = np.where(compare(conf[name], th), np.arange(self.args.atk_epochs), self.args.atk_epochs).min(axis=1)
                idx[idx == self.args.atk_epochs] = (self.args.atk_epochs - 1)
                pred_th = pred[name][np.arange(self.args.N), idx]
                pos = is_positive(pred_th, gt[name])

                if (name == "unlearn"):
                    _tp += np.sum(pos)
                    _fn += np.sum(~pos)
                else:
                    _fp += np.sum(pos)
                    _tn += np.sum(~pos)
            tp.append(_tp)
            fp.append(_fp)
            fn.append(_fn)
            tn.append(_tn)
        return np.array(tp), np.array(fp), np.array(fn), np.array(tn), ths

class Apollo_Offline(Apollo):
    def __init__(self, target_model, dataset, shadow_models, args, idxs, shadow_col, unlearn_args):
        Attack_Framework.__init__(self, target_model, dataset, shadow_models, args, idxs, shadow_col, unlearn_args)
        self.types = ["Under", "Over"]

    def set_include_exclude(self, target_idx):
        Attack_Framework.set_include_exclude(self, target_idx)
        self.temp, self.params, self.buffers = batched_models_(self.shadow_models)

    def batched_loss_Under(self, input, label):
        outputs = torch.vmap(
            lambda ps, bs, xx: functional_call(self.temp, ps, (xx,), {}, tie_weights=True, strict=False),
            in_dims=(0, 0, None)
        )(self.params, self.buffers, input)

        flat = outputs.reshape(-1, outputs.size(-1))
        label_rep = label.repeat(outputs.size(0))

        loss_rt = F.cross_entropy(flat, label_rep)
        top2_vals, _ = outputs.topk(2, dim=-1)
        loss_db = top2_vals[0, :, 0] - top2_vals[0, :, 1]
        return self.args.w[0] * loss_db - self.args.w[1] * loss_rt
    def batched_loss_Over(self, input, label):
        outputs = torch.vmap(
            lambda ps, bs, xx: functional_call(self.temp, ps, (xx,), {}, tie_weights=True, strict=False),
            in_dims=(0, 0, None)
        )(self.params, self.buffers, input)

        flat = outputs.reshape(-1, outputs.size(-1))
        label_rep = label.repeat(outputs.size(0))

        loss_rt = F.cross_entropy(flat, label_rep)
        top2_vals, _ = outputs.topk(2, dim=-1)
        loss_db = top2_vals[0, :, 0] - top2_vals[0, :, 1]
        return self.args.w[0] * loss_db + self.args.w[1] * loss_rt


def batched_models_(models_list):
    temp = models_list[0]
    params, buffers = stack_module_state(models_list)
    return temp, params, buffers

def batched_loss_(input, label, temp, params, buffers):
    # outputs: (N, batch, classes)
    outputs = torch.vmap(
        lambda ps, bs, xx: functional_call(temp, ps, (xx,), {}, tie_weights=True, strict=False),
        in_dims=(0, 0, None)
    )(params, buffers, input)

    flat = outputs.reshape(-1, outputs.size(-1))
    label_rep = label.repeat(outputs.size(0))
    return F.cross_entropy(flat, label_rep)

def normalize(tensor: torch.Tensor):
    norm = torch.norm(tensor, p=2, dim=-1, keepdim=True)
    return tensor / (norm + 1e-9), norm

def proj(A: torch.Tensor, B: torch.Tensor, r: float):
    with torch.no_grad():
        d = (B - A).view(-1).norm(p=2).item()
        scale = min(1.0, r / (d + 1e-9))
        return A + (B - A) * scale

def proj_out(A: torch.Tensor, B: torch.Tensor, r: float):
    with torch.no_grad():
        d = (B - A).view(-1).norm(p=2).item()
        scale = max(1.0, r / (d + 1e-9))
        return A + (B - A) * scale