import os
import argparse
import time
import numpy as np
import pandas as pd
from collections import OrderedDict
import pickle as pkl
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import utils
from models import create_model
from dataset import create_dataset

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Attack_One_shot_Unlearn():
    def __init__(self, shadow_models, args):
        self.shadow_models = shadow_models
        self.include, self.exclude = [], []
        self.args = args

        self.CE = nn.CrossEntropyLoss()

    def set_include_exclude(self, include, exclude):
        self.include, self.exclude = include, exclude

    def get_near_miss_label(self, target_input, target_label):
        min_loss_label = None
        min_loss = None
        for label in range(self.args.num_classes):
            if (label == target_label.item()):
                continue

            sum_loss = 0
            for i in range(len(self.exclude)):
                adv_output = self.shadow_models[i](target_input)
                loss = self.CE(adv_output, torch.Tensor([label]).to(torch.int64).to(DEVICE))
                sum_loss += loss.item()

            if (min_loss == None):
                min_loss, min_loss_label = sum_loss, label
            elif (sum_loss < min_loss):
                min_loss, min_loss_label = sum_loss, label
        return torch.Tensor([min_loss_label]).to(torch.int64).to(DEVICE)

    def get_labels(self, input):
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

    def Under_Un_Adv(self, target_input, target_label):
        # Under-Unlearning: Given target (x, y), adv. input x',
        # unlearned model (x not in unlearned set) x' --> y
        # retrained model (x not in unlearned set) x' --> y' --> Find the nearest (x', y')
        adv_input = target_input.detach().clone().to(DEVICE)
        adv_input.requires_grad = True
        adv_label = self.get_near_miss_label(target_input, target_label)
        optimizer = torch.optim.SGD([adv_input], lr=self.args.lr)

        for epoch in range(self.args.atk_epochs):
            loss = 0.0
            optimizer.zero_grad()
            loss_exclude, loss_d, loss_db = 0.0, 0.0, 0.0

            # Only consider exclude
            for i in self.exclude:
                adv_output = self.shadow_models[i](adv_input)
                loss_exclude += F.cross_entropy(adv_output, adv_label)
                loss_db += adv_output.topk(2)[1][0, 0] - adv_output.topk(2)[1][0, 1]
            
            loss_d = F.mse_loss(adv_input, target_input)
            loss = self.args.w[0] * loss_exclude / len(self.exclude) + \
                   self.args.w[1] * loss_d + \
                   self.args.w[2] * loss_db / len(self.exclude)
            loss.backward()
            optimizer.step()
            torch.clamp(adv_input, min=0, max=1) # Image data
            if (loss.item() < .2):
                break

        if (self.args.debug):
            include_labels, exclude_labels = self.get_labels(adv_input)
            print(self.include, self.exclude)
            print(">>>>>>>>>>", target_label, adv_label)
            print("include_labels: ", include_labels)
            print("exclude_labels: ", exclude_labels)
        return adv_input.clone().detach()


def main():
    parser = argparse.ArgumentParser(description='Attack Config')
    parser.add_argument('--data_dir',       type=str,   default='./data',   help='path to dataset')
    parser.add_argument('--dataset',        type=str,   default='',         help='dataset type (default: ImageFolder/ImageTar if empty)')

    parser.add_argument('--model',          type=str,   default='ResNet18', help='model architechture (default: "ResNet18"')
    parser.add_argument('--num_classes',    type=int,   default=None,       help='number of label classes (Model default if None)')
    parser.add_argument('--input_size',     type=int,   default=None,       nargs=3, help='Input all image dimensions (d h w, e.g. --input_size 3 224 224)')
    parser.add_argument('--batch_size',     type=int,   default=128,        help='input batch size for training (default: 128)')
    parser.add_argument('--path',           type=str,   default='',         help='Initialize model from this path (default: none)')

    parser.add_argument('--num_shadow',     type=int,   default=16,         help='number of shadow models (default: 16)')
    parser.add_argument('--shadow_model',   type=str,   default='ResNet18', help='shadow model architechture (default: "ResNet18"')
    parser.add_argument('--shadow_path',    type=str,   default='',         help='Initialize shadow models from this path (default: none)')

    parser.add_argument('--atk_lr',         type=float, default=None,       help='learning rate, overrides lr-base if set (default: None)')
    parser.add_argument('--atk_epochs',     type=int,   default=30,         help='amx number of epochs for attack (default: 30)')
    parser.add_argument('--weight',         type=float, default=None,       nargs=3, help='Adv. loss function weights')
    # parser.add_argument('--eps',            type=float, default=1e-3,       help='epsilon for clipping')
    parser.add_argument('--debug',                      action="store_true")

    parser.add_argument('--seed',           type=int,   default=42,         help='random seed (default: 42)')
    args = parser.parse_args()


    utils.random_seed(args.seed)

    # dataloaders
    dataset = create_dataset(dataset_name=args.dataset, setting="Partial", root=args.data_dir, img_size=args.input_size[-1])
    with open(os.path.join(args.path, "data_split.pkl"), "rb") as f:
        data_split = pkl.load(f)
    forget_set = dataset.get_subset(dataset.train_dataset, dataset.unlearn_idx)
    retain_set = dataset.get_subset(dataset.train_dataset, dataset.retain_idx)
    test_set = dataset.valid_dataset

    forget_loader = DataLoader(forget_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    retain_loader = DataLoader(retain_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    idxs = OrderedDict(unlearn = dataset.unlearn_idx, retain = dataset.retain_idx)
    unlearn_loaders = OrderedDict(unlearn = forget_loader, retain = retain_loader, test = test_loader)

    # get network
    target_model = create_model(model_name=args.model, num_classes=args.num_classes)
    target_model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))
    target_model.to(DEVICE)
    target_model.eval()

    shadow_models = nn.ModuleList()
    for i in range(args.num_s):
        weights_path = os.path.join(args.shadow_path, f"{i}.pth.tar")
        model = create_model(model_name=args.shadow_model, num_classes=args.num_classes)

        model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
    with open(os.path.join(args.shadow_path, "data_split.pkl"), "wb") as f:
        data_split = pkl.load(f)
    # shadow_sets = data_split["shadow_idx_collection"]
    print("Models Loaded")

    atk = Attack_One_shot_Unlearn(shadow_models=shadow_models, args=args)
    for name, loader in unlearn_loaders.items():
        cnt_under, cnt_over = 0, 0
        for i, (target_input, target_label) in enumerate(pbar := tqdm(loader)):
            include, exclude = [], []
            if (name != "test"):
                for j in range(args.num_shadow):
                    if (idxs[name][i] in set(data_split["shadow_idx_collection"][j])):
                        include.append(j)
                    else:
                        exclude.append(j)
            else:
                include, exclude = [], [j for j in range(args.num_shadow)]
            atk.set_include_exclude(include, exclude)

            target_input, target_label = target_input.to(DEVICE), target_label.to(DEVICE)
            with torch.no_grad():
                output = target_model(target_input)
            pred = output.max(1)[1]

            adv_input_under = atk.Under_Un_Adv(target_input, target_label.to(torch.int64))
            with torch.no_grad():
                adv_output = target_model(adv_input_under)
            adv_pred = adv_output.max(1)[1]
            if (adv_pred.item() == target_label.item()):
                cnt_under += 1


if __name__ == '__main__':
    main()