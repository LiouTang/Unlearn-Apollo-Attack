import os
import argparse
import time
import numpy as np
import pandas as pd
from collections import OrderedDict
import pickle as pkl

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import utils
from models import create_model
from dataset import create_dataset
import unlearn 
from trainer import validate
from evaluation import get_membership_attack_prob, get_js_divergence, get_SVC_MIA

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser(description='Unlearn Config')
    parser.add_argument('--data_dir',       type=str,   default='./data',   help='path to dataset')
    parser.add_argument('--dataset',        type=str,   default='',         help='dataset type (default: ImageFolder/ImageTar if empty)')
    parser.add_argument('--size_train',     type=int,   default=2500,       help='train set size')

    parser.add_argument('--model',          type=str,   default='ResNet18', help='Name of model to train (default: "resnet50"')
    parser.add_argument('--num_classes',    type=int,   default=None,       help='number of label classes (Model default if None)')
    parser.add_argument('--input_size',     type=int,   default=None,       nargs=3, help='Input all image dimensions (d h w, e.g. --input_size 3 224 224)')
    parser.add_argument('--batch_size',     type=int,   default=128,        help='input batch size for training (default: 128)')
    parser.add_argument('--checkpoint',     type=str,   default='',         help='Initialize model from this checkpoint (default: none)')

    parser.add_argument('--seed',           type=int,   default=42,         help='random seed (default: 42)')

    parser.add_argument("--unlearn",        type=str,   required=True,      help="select unlearning method from choice set")
    parser.add_argument("--forget_perc",    type=float, default=None,       help="forget random subset percentage")
    parser.add_argument("--forget_class",   type=int,   default=None,       help="forget class")
    args = parser.parse_args()


    utils.random_seed(args.seed)

    save_path = os.path.join("./save",
                             f"{args.model}-{args.dataset}-{str(args.size_train)}",
                             f"perc-{str(args.forget_perc)}-class-{str(args.forget_class)}",
                             args.unlearn)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # dataloaders
    dataset = create_dataset(dataset_name=args.dataset, setting="Partial", root=args.data_dir, img_size=args.input_size[-1])
    dataset.set_train_shadow_idx(size_train=args.size_train)
    dataset.set_unlearn_idx(un_perc=args.forget_perc, un_class=args.forget_class)

    forget_trainset = dataset.get_subset(dataset.train_dataset, dataset.unlearn_idx)
    retain_trainset = dataset.get_subset(dataset.train_dataset, dataset.retain_idx)
    testset = dataset.valid_dataset

    forget_trainloader = DataLoader(forget_trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    retain_trainloader = DataLoader(retain_trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # print(np.unique(forget_trainset.targets), np.unique(retain_trainset.targets))
    unlearn_dataloaders = OrderedDict(
        forget_train=forget_trainloader,
        retain_train=retain_trainloader,
        forget_valid=None,
        retain_valid=testloader
    )
    unlearn_sets = OrderedDict(
        forget_train=forget_trainset,
        retain_train=retain_trainset,
        retain_valid=testset
    )

    # get network
    model = create_model(model_name=args.model, num_classes=args.num_classes)
    model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))
    model.to(DEVICE)
    loss_function = nn.CrossEntropyLoss()

    # start unlearning
    start = time.time()
    unlearn_method = unlearn.create_unlearn_method(args.unlearn)(model, loss_function, save_path, args)
    unlearn_method.prepare_unlearn(unlearn_dataloaders)
    unlearned_model = unlearn_method.get_unlearned_model()
    end = time.time()

    # evaluate
    eval_result = {'method': args.unlearn, 'seed': args.seed}
    for name, loader in unlearn_dataloaders.items():
        if loader is None: continue 
        eval_metrics = validate(loader, unlearned_model, loss_function, name)
        for metr, v in eval_metrics.items():
            eval_result[f"{name} {metr}"] = v
    for mia_metric in ["entropy"]:
        eval_result[f"{mia_metric} mia"] = get_membership_attack_prob(retain_trainloader, forget_trainloader, testloader, unlearned_model, mia_metric)

    eval_result["time"] = end - start
    eval_result["params"] = unlearn_method.get_params()
    print(eval_result)

    # save results
    torch.save(unlearned_model.state_dict(), os.path.join(save_path, "unlearn.pth.tar"))
    data_split = {
        "train":    dataset.train_idx,
        "unlearn":  dataset.unlearn_idx,
        "retain":   dataset.retain_idx,
    }
    with open(os.path.join(save_path, "data_split.pkl"), "wb") as f:
        pkl.dump(data_split, f)
    with open(os.path.join(save_path, "eval_result.pkl"), "wb") as f:
        pkl.dump(eval_result, f)


if __name__ == '__main__':
    main()