import os
import sys
import random 
import argparse
import time
import numpy as np
import pandas as pd 
from datetime import datetime
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import utils
from models import create_model
from dataset import create_dataset, dataset_convert_to_valid
import unlearn 
from trainer import validate
from evaluation import get_membership_attack_prob, get_js_divergence, get_SVC_MIA

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser(description='Unlearn Config')
    parser.add_argument('--data_dir', metavar='DIR', default='./data',
                        help='path to dataset')
    parser.add_argument('--dataset', metavar='NAME', default='',
                        help='dataset type (default: ImageFolder/ImageTar if empty)')

    parser.add_argument('--model', default='ResNet18', type=str, metavar='MODEL',
                        help='Name of model to train (default: "resnet50"')
    parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                        help='Initialize model from this checkpoint (default: none)')
    parser.add_argument('--num_classes', type=int, default=None, metavar='N',
                        help='number of label classes (Model default if None)')
    parser.add_argument('--input_size', default=None, nargs=3, type=int, metavar='N',
                        help='Input all image dimensions (d h w, e.g. --input_size 3 224 224), uses model default if empty')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')

    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')

    parser.add_argument("--unlearn", type=str, required=True, nargs="?",
                        help="select unlearning method from choice set")
    parser.add_argument("--forget_perc", type=float, nargs="?", default=None,
                        help="forget random subset percentage")
    parser.add_argument("--forget_class", type=str, nargs="?", default=None,
                        help="forget class")
    args = parser.parse_args()


    utils.random_seed(args.seed)
    if (args.forget_perc != None):
        save_path = os.path.join("./save", args.model, args.dataset, "un_perc_" + str(args.forget_perc), args.unlearn)
    elif (args.forget_class != None):
        save_path = os.path.join("./save", args.model, args.dataset, "un_" + args.forget_class.replace(",", "_"), args.unlearn)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # dataloaders
    if (args.forget_perc != None):
        dataset = create_dataset(dataset_name=args.dataset, setting="RandomUnlearn", root=args.data_dir, img_size=args.input_size[-1])
    elif (args.forget_class != None):
        dataset = create_dataset(dataset_name=args.dataset, setting="FullClassUnlearn", root=args.data_dir, img_size=args.input_size[-1])
    _, _, testset = dataset.get_datasets()
    transform_valid = testset.transform
    # trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    if (args.forget_perc != None):
        forget_trainset, retain_trainset = dataset.random_split(args.forget_perc)
    elif (args.forget_class != None):
        fc = list(map(int, args.forget_class.split(",")))
        print(fc)
        forget_trainset, retain_trainset = dataset.full_class_split(fc)
    
    print(len(forget_trainset), len(retain_trainset))
    # print(np.unique(forget_trainset.targets), np.unique(retain_trainset.targets))
    forget_trainloader = DataLoader(forget_trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    retain_trainloader = DataLoader(retain_trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
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
        dataset_convert_to_valid(loader, transform_valid)
        eval_metrics = validate(loader, unlearned_model, loss_function, name)
        for metr, v in eval_metrics.items():
            eval_result[f"{name} {metr}"] = v
    for mia_metric in ["entropy"]:
        eval_result[f"{mia_metric} mia"] = get_membership_attack_prob(retain_trainloader, forget_trainloader, testloader, unlearned_model, mia_metric)

    # JS divergence and KL divergence
    # if retrained_model:
    #     eval_result['js_div'], eval_result['kl_div'] = get_js_divergence(forget_trainloader, unlearned_model, retrained_model)
    # else:
    #     eval_result['js_div'], eval_result['kl_div'] = None, None
    eval_result["time"] = end - start
    eval_result["params"] = unlearn_method.get_params()

    print(eval_result)
    torch.save(
        {
            "sets": unlearn_sets,
            "state_dict": unlearned_model.state_dict(),
            "eval_result": eval_result
        },
        os.path.join(save_path, "unlearn.pth.tar")
    )
    # print("|\t".join([str(k) for k in eval_result.keys()]))
    # print("|\t".join([str(v) for v in eval_result.values()]))

    file_path = os.path.join(save_path, f'results.csv')
    new_row = pd.DataFrame({k:[v] for k, v in eval_result.items()})
    new_row.to_csv(file_path, mode='a', index=False, header=not os.path.exists(file_path))


if __name__ == '__main__':
    main()