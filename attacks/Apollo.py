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
        self.types = ["Unified"]  # Single unified attack type
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

    def Un_Adv(self, target_input: torch.Tensor, target_label: torch.Tensor, loss_func):
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
                projected = proj(target_input, adv_input.data, self.args.eps * (epoch + 1), "in")
                projected = proj(target_input, projected,      self.args.eps * (epoch), "out")
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

    def get_ternary_results(self, **kwargs):
        print("Calculating Apollo Ternary Results!")
        
        gt, under_conf, over_conf, under_pred, over_pred = {}, {}, {}, {}, {}
        
        for name in ["unlearn", "retain", "test"]:
            gt[name] = torch.cat([self.summary[name][i]["target_label"] for i in self.summary[name]], dim=0).cpu().numpy()
            under_conf[name] = np.array([self.summary[name][i]["un_conf"] for i in self.summary[name]])
            over_conf[name] = np.array([self.summary[name][i]["ov_conf"] for i in self.summary[name]])
            under_pred[name] = np.array([self.summary[name][i]["un_pred"] for i in self.summary[name]])
            over_pred[name] = np.array([self.summary[name][i]["ov_pred"] for i in self.summary[name]])

        # Generate threshold combinations
        under_ths = np.unique(np.concatenate([under_conf["unlearn"], under_conf["retain"], under_conf["test"]]))
        over_ths = np.unique(np.concatenate([over_conf["unlearn"], over_conf["retain"], over_conf["test"]]))
        
        ternary_points = []
        threshold_pairs = []
        
        # Sample threshold combinations for efficiency
        n_samples = min(50, len(under_ths) * len(over_ths))  # Limit combinations
        under_sample = np.linspace(0, len(under_ths)-1, int(np.sqrt(n_samples)), dtype=int)
        over_sample = np.linspace(0, len(over_ths)-1, int(np.sqrt(n_samples)), dtype=int)
        
        for u_idx in tqdm(under_sample):
            for o_idx in over_sample:
                under_th = under_ths[u_idx]
                over_th = over_ths[o_idx]
                
                # Classification counts
                classifications = {"unlearn": 0, "retain": 0, "test": 0}
                total_samples = 0
                
                for name in ["unlearn", "retain", "test"]:
                    for i in range(self.args.N):
                        if i >= len(under_conf[name]) or i >= len(over_conf[name]):
                            continue
                            
                        # Get confidence values at optimal epochs
                        under_idx = np.where(under_conf[name][i] < under_th, 
                                           np.arange(self.args.atk_epochs), self.args.atk_epochs).min()
                        over_idx = np.where(over_conf[name][i] > over_th, 
                                          np.arange(self.args.atk_epochs), self.args.atk_epochs).min()
                        
                        under_idx = min(under_idx, self.args.atk_epochs - 1)
                        over_idx = min(over_idx, self.args.atk_epochs - 1)
                        
                        under_triggered = under_conf[name][i][under_idx] < under_th
                        over_triggered = over_conf[name][i][over_idx] > over_th
                        
                        under_pred_val = under_pred[name][i][under_idx]
                        over_pred_val = over_pred[name][i][over_idx]
                        true_label = gt[name][i]
                        
                        # Mutual exclusivity: prioritize by confidence strength
                        under_strength = abs(under_conf[name][i][under_idx] - under_th) if under_triggered else 0
                        over_strength = abs(over_conf[name][i][over_idx] - over_th) if over_triggered else 0
                        
                        if under_triggered and over_triggered:
                            # Both triggered - choose stronger signal
                            primary_under = under_strength > over_strength
                        else:
                            primary_under = under_triggered
                            
                        # Classification logic based on attack outcome
                        if primary_under:
                            # Under-unlearning detection: wrong prediction indicates forgetting
                            is_forgotten = (under_pred_val != true_label)
                            if name == "unlearn":
                                if is_forgotten:
                                    classifications["unlearn"] += 1  # Properly forgotten
                                else:
                                    classifications["retain"] += 1   # Under-unlearned (still remembered)
                            elif name == "retain":
                                if is_forgotten:
                                    classifications["unlearn"] += 1  # Over-unlearned (wrongly forgotten)
                                else:
                                    classifications["retain"] += 1   # Properly retained
                            else:  # test
                                classifications["test"] += 1
                        else:
                            # Over-unlearning detection: correct prediction on unlearn = under-unlearning
                            is_correct = (over_pred_val == true_label)
                            if name == "unlearn":
                                if is_correct:
                                    classifications["retain"] += 1   # Under-unlearned (still correct)
                                else:
                                    classifications["unlearn"] += 1  # Properly forgotten
                            elif name == "retain":
                                if is_correct:
                                    classifications["retain"] += 1   # Properly retained
                                else:
                                    classifications["unlearn"] += 1  # Over-unlearned (wrongly forgotten)
                            else:  # test
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
                    threshold_pairs.append((under_th, over_th))
        
        return np.array(ternary_points), np.array(threshold_pairs)

class Apollo_Offline(Apollo):
    def __init__(self, target_model, dataset, shadow_models, args, idxs, shadow_col, unlearn_args):
        Attack_Framework.__init__(self, target_model, dataset, shadow_models, args, idxs, shadow_col, unlearn_args)
        self.types = ["Unified"]  # Single unified attack type

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

    # Inherit the unified get_ternary_results from parent Apollo class


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

def proj(A: torch.Tensor, B: torch.Tensor, r: float, type):
    with torch.no_grad():
        d = (B - A).view(-1).norm(p=2).item()
        if (type == "in"):
            scale = min(1.0, r / (d + 1e-9))
        else:
            scale = max(1.0, r / (d + 1e-9))
        return A + (B - A) * scale