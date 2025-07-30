import os
import argparse
import time

import numpy as np
from collections import OrderedDict
import pickle as pkl
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import utils
import attacks
from models import create_model
from dataset import create_dataset

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def plot_results(tp, fp, fn, tn, ths, title, path):
    if not os.path.exists(path):
        os.makedirs(path)
    sort = np.argsort(fp)
    tp, fp, fn, tn = tp[sort], fp[sort], fn[sort], tn[sort]
    # print(tp, fp, fn, tn)
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)

    plt.figure(figsize=(8, 6))
    plt.scatter(fpr, tpr, c=ths, label='ROC (step curve)')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
    plt.colorbar()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid(True)
    # plt.legend()
    plt.title(title)
    plt.savefig(os.path.join(path, title + ".pdf"))
    return

def setminus(A, B):
    return np.array(list(set(A).difference(set(B))))

def get_idx_loaders(split, dataset):
    unlearn_set = dataset.get_subset(split["unlearn"])
    valid_set   = dataset.get_subset(split["valid"])

    unlearn_loader = DataLoader(unlearn_set, batch_size=1, shuffle=False, num_workers=4)
    valid_loader   = DataLoader(valid_set,   batch_size=1, shuffle=False, num_workers=4)

    idxs    = OrderedDict(unlearn = split["unlearn"], valid = split["valid"])
    loaders = OrderedDict(unlearn = unlearn_loader,   valid = valid_loader)
    return idxs, loaders

def main():
    parser = argparse.ArgumentParser(description='Attack Config')
    parser.add_argument('--data_dir',       type=str,   default='./data',   help='path to dataset')
    parser.add_argument('--dataset',        type=str,   default='',         help='dataset type (default: ImageFolder/ImageTar if empty)')

    parser.add_argument('--model',          type=str,   default='ResNet18', help='model architechture (default: "ResNet18"')
    parser.add_argument('--num_classes',    type=int,   default=None,       help='number of label classes (Model default if None)')
    parser.add_argument('--input_size',     type=int,   default=None,       nargs=3, help='Input all image dimensions (d h w, e.g. --input_size 3 224 224)')
    parser.add_argument('--target_path',    type=str,   default='',         help='Initialize target (unlearned) model from this path (default: none)')

    parser.add_argument('--num_shadow',     type=int,   default=16,         help='number of shadow models (default: 16)')
    parser.add_argument('--shadow_model',   type=str,   default='ResNet18', help='shadow model architechture (default: "ResNet18"')
    parser.add_argument('--shadow_path',    type=str,   default='',         help='Initialize shadow models from this path (default: none)')

    parser.add_argument('--N',              type=int,   default=200,        help='number of samples to attack')
    parser.add_argument('--atk',            type=str,   default='Apollo',   help='Attack Name')
    parser.add_argument('--atk_lr',         type=float, default=1e-1,       help='Attack learning rate')
    parser.add_argument('--atk_epochs',     type=int,   default=30,         help='number of epochs for attack (default: 30)')
    parser.add_argument('--w',              type=float, default=None,       nargs=2, help='Adv. loss function weights')
    parser.add_argument('--eps',            type=float, default=10,         help='epsilon for bound')

    parser.add_argument('--save_to',        type=str,   required=True,      help='save results to this path')
    parser.add_argument('--debug',                      action="store_true")

    parser.add_argument('--seed',           type=int,   default=42,         help='random seed (default: 42)')
    args = parser.parse_args()


    utils.random_seed(args.seed)

    # Dataloaders
    dataset = create_dataset(dataset_name=args.dataset, setting="Partial", root=args.data_dir, img_size=args.input_size[-1])
    with open(os.path.join(args.target_path, "data_split.pkl"), "rb") as f:
        target_split_orig = pkl.load(f)

    print(target_split_orig["unlearn"][:10], target_split_orig["retain"][:10])
    target_split = {}
    target_split["unlearn"] = np.random.choice(target_split_orig["unlearn"], args.N, replace=False)
    target_split["valid"]   = np.random.choice(target_split_orig["valid"],   args.N, replace=False)
    target_idxs, target_loaders = get_idx_loaders(target_split, dataset)

    # U-MIA Functionality
    if (args.atk == "UMIA"):
        surr_split = {}
        surr_split["unlearn"] = setminus(target_split_orig["unlearn"], target_split["unlearn"])
        surr_split["valid"]   = setminus(target_split_orig["valid"],   target_split["valid"])
        surr_split["unlearn"] = np.random.choice(surr_split["unlearn"], args.N, replace=False)
        surr_split["valid"]   = np.random.choice(surr_split["valid"],   args.N, replace=False)
        surr_idxs, surr_loaders = get_idx_loaders(surr_split, dataset)

    # Target
    target_model = create_model(model_name=args.model, num_classes=args.num_classes)
    target_model.load_state_dict(torch.load(os.path.join(args.target_path, "unlearn.pth.tar"), map_location=DEVICE, weights_only=True))
    target_model.to(DEVICE)
    target_model.eval()

    with open(os.path.join(args.target_path, "unlearn_args.pkl"), "rb") as f:
        unlearn_args = pkl.load(f)
    print("Unlearn Arguments Loaded:", unlearn_args)

    # Shadow models
    shadow_models = nn.ModuleList()
    for i in range(args.num_shadow):
        weights_path = os.path.join(args.shadow_path, f"{i}.pth.tar")
        model = create_model(model_name=args.shadow_model, num_classes=args.num_classes)
        model.load_state_dict(torch.load(weights_path, map_location=DEVICE, weights_only=True))
        model.to(DEVICE)
        model.eval()
        shadow_models.append(model)

    with open(os.path.join(args.shadow_path, "data_split.pkl"), "rb") as f:
        data_split = pkl.load(f)
    # print(data_split.items())
    print("Models Loaded")

    # Attack!
    Atk = attacks.get_attack(
        name=args.atk,
        target_model=target_model,
        dataset=dataset,
        shadow_models=shadow_models,
        args=args,
        idxs=target_idxs,
        shadow_col=data_split["shadow_col"],
        unlearn_args=unlearn_args,
    )
    if (args.atk == "UMIA"):
        Atk.train_surr(surr_idxs, surr_loaders)
    time_col = []
    for name, loader in target_loaders.items():
        print(name)
        for i, (input, label) in enumerate(pbar := tqdm(loader)):
            Atk.set_include_exclude(target_idx=target_idxs[name][i])
            start_time = time.time()
            input, label = input.to(DEVICE), label.to(torch.int64).to(DEVICE)
            end_time = time.time()
            time_col.append(end_time - start_time)
            Atk.update_atk_summary(name, input, label, target_idxs[name][i])

    print("Time Used:", np.mean(time_col), np.std(time_col))

    # Save Summary
    base_path = os.path.join(
        args.save_to, f"{unlearn_args.model}-{unlearn_args.dataset}",
        f"perc-{unlearn_args.forget_perc}-class-{unlearn_args.forget_class}"
    )
    summary_path = os.path.join(base_path, "summary")
    if (not os.path.exists(summary_path)):
        os.makedirs(summary_path)
    with open(os.path.join(summary_path, f"{args.atk}-{unlearn_args.unlearn}.pkl"), "wb") as f:
        pkl.dump(Atk.get_atk_summary(), f)

    # Interpret results
    roc_path = os.path.join(base_path, "roc")
    if (not os.path.exists(roc_path)):
        os.makedirs(roc_path)
    for type in Atk.types:
        tp, fp, fn, tn, ths = Atk.get_roc(type=type)

        roc = {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "ths": ths}
        with open(os.path.join(roc_path, f"{args.atk}-{unlearn_args.unlearn}-{type}.pkl"), "wb") as f:
            pkl.dump(roc, f)

        plot_results(
            tp, fp, fn, tn, ths,
            f"{args.atk}-{unlearn_args.unlearn}-{type}",
            os.path.join(base_path, "figs")
        )

if __name__ == '__main__':
    main()