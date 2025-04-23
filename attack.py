import os
import argparse
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

def plot_results(tp, fp, fn, tn, ths, title):
    if not os.path.exists("./Figs/"):
        os.makedirs("./Figs/")
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
    plt.savefig("./Figs/" + title + ".pdf")
    return

def main():
    parser = argparse.ArgumentParser(description='Attack Config')
    parser.add_argument('--data_dir',       type=str,   default='./data',   help='path to dataset')
    parser.add_argument('--dataset',        type=str,   default='',         help='dataset type (default: ImageFolder/ImageTar if empty)')

    parser.add_argument('--model',          type=str,   default='ResNet18', help='model architechture (default: "ResNet18"')
    parser.add_argument('--num_classes',    type=int,   default=None,       help='number of label classes (Model default if None)')
    parser.add_argument('--input_size',     type=int,   default=None,       nargs=3, help='Input all image dimensions (d h w, e.g. --input_size 3 224 224)')
    # parser.add_argument('--batch_size',     type=int,   default=128,        help='input batch size for training (default: 128)')
    parser.add_argument('--target_path',    type=str,   default='',         help='Initialize model from this path (default: none)')

    parser.add_argument('--num_shadow',     type=int,   default=16,         help='number of shadow models (default: 16)')
    parser.add_argument('--shadow_model',   type=str,   default='ResNet18', help='shadow model architechture (default: "ResNet18"')
    parser.add_argument('--shadow_path',    type=str,   default='',         help='Initialize shadow models from this path (default: none)')

    parser.add_argument('--N',              type=int,   default=200,        help='number of samples to attack')
    parser.add_argument('--atk',            type=str,   default="Apollo",   help='Attack Name')
    parser.add_argument('--atk_lr',         type=float, default=1e-3,       help='Attack learning rate')
    parser.add_argument('--atk_epochs',     type=int,   default=30,         help='number of epochs for attack (default: 30)')
    parser.add_argument('--w',              type=float, default=None,       nargs=3, help='Adv. loss function weights')
    # parser.add_argument('--eps',            type=float, default=1e-3,       help='epsilon for clipping')
    parser.add_argument('--debug',                      action="store_true")

    parser.add_argument('--seed',           type=int,   default=42,         help='random seed (default: 42)')
    args = parser.parse_args()


    utils.random_seed(args.seed)

    # Dataloaders
    dataset = create_dataset(dataset_name=args.dataset, setting="Partial", root=args.data_dir, img_size=args.input_size[-1])
    with open(os.path.join(args.target_path, "data_split.pkl"), "rb") as f:
        data_split = pkl.load(f)

    print(data_split["unlearn"][:10], data_split["retain"][:10])
    data_split["unlearn"] = np.random.choice(data_split["unlearn"], args.N, replace=False)
    data_split["retain"]  = np.random.choice(data_split["retain"],  args.N, replace=False)
    data_split["valid"]   = np.random.choice(data_split["valid"],   args.N, replace=False)
    forget_set = dataset.get_subset(data_split["unlearn"])
    # retain_set = dataset.get_subset(data_split["retain"])
    valid_set  = dataset.get_subset(data_split["valid"])

    forget_loader = DataLoader(forget_set, batch_size=1, shuffle=False, num_workers=4)
    # retain_loader = DataLoader(retain_set, batch_size=1, shuffle=False, num_workers=4)
    valid_loader  = DataLoader(valid_set,   batch_size=1, shuffle=False, num_workers=4)

    idxs            = OrderedDict(unlearn = data_split["unlearn"],  valid = data_split["valid"]) # retain = data_split["retain"],
    unlearn_loaders = OrderedDict(unlearn = forget_loader,          valid = valid_loader)        # retain = retain_loader,

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
    print(data_split.items())
    print("Models Loaded")

    # Attack!
    Atk = attacks.get_attack(
        dataset=dataset,
        name=args.atk,
        shadow_models=shadow_models,
        args=args,
        idxs=idxs,
        shadow_col=data_split["shadow_col"],
        unlearn_args=unlearn_args
    )
    for name, loader in unlearn_loaders.items():
        print(name)
        for i, (target_input, target_label) in enumerate(pbar := tqdm(loader)):
            Atk.set_include_exclude(target_idx=idxs[name][i])

            # Origninal Prediction
            target_input, target_label = target_input.to(DEVICE), target_label.to(DEVICE)
            with torch.no_grad():
                output = target_model(target_input)
            pred = output.max(1)[1]

            Atk.update_atk_summary(name, target_input, target_label, idxs[name][i])
            if (args.debug):
                return
        # summary = Atk.get_atk_summary()

    # Interpret results
    if (not os.path.exists("./Results/")):
        os.makedirs("./Results")
    for type in Atk.types:
        tp, fp, fn, tn, ths = Atk.get_results(target_model, type=type)
        results = {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "ths": ths}
        with open(f"./Results/{args.atk}-{unlearn_args.unlearn}-{type}.pkl", "wb") as f:
            pkl.dump(results, f)
        # plot_results(tp, fp, fn, tn, ths, f"{args.atk}-{unlearn_args.unlearn}-{type}")

if __name__ == '__main__':
    main()