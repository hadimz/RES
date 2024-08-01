import argparse
import cv2
import os
import numpy as np
import json
from os import walk
import pandas as pd
import csv
import torch
import torch.nn as nn
import torch.nn.parallel
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import random
import math
import shutil
import time

from metrics import *
from utils import *
from utils_train import *


args = get_args()
path_to_attn, path_to_attn_resized = load_path_to_attentions()
# resize attention label from 224x224 to 14x14
# path_to_attn_resized = resize_attention_label(path_to_attn)


""" python grad_cam.py <path_to_image>
1. Loads an image with opencv.
2. Preprocesses it for VGG19 and converts to a pytorch variable.
3. Makes a forward pass to find the category index with the highest score,
and computes intermediate activations.
Makes the visualization. """
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# Data loading code
if args.fw_sample > 0:
    print('Performing few sample learning with # of samples = ', args.fw_sample)

    if args.data_dir == 'gender_data':
        sample_selection_with_explanations_gender(args.fw_sample, path_to_attn)
    elif args.data_dir == 'places':
        sample_selection_with_explanations_places(args.fw_sample, path_to_attn)
    elif args.data_dir == 'sixray':
        sample_selection_with_explanations_sixray(args.fw_sample, path_to_attn)
    else:
        print('Error: Unrecognized dataset:', args.data)

    traindir = os.path.join(args.data_dir, 'fw_' + str(args.fw_sample) + '_seed_' + str(args.random_seed))
else:
    traindir = os.path.join(args.data_dir, 'train')
valdir = os.path.join(args.data_dir, 'val')
testdir = os.path.join(args.data_dir, 'test')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

if args.trainWithMap:
    train_with_map_loader = torch.utils.data.DataLoader(
        ImageFolderWithMapsAndWeights(traindir, transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.train_batch, shuffle=True,
        pin_memory=True)
else:
    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.train_batch, shuffle=True,
        pin_memory=True)

train_for_eval_loader = torch.utils.data.DataLoader(
    ImageFolderWithPaths(traindir, transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=args.test_batch, shuffle=False,
    pin_memory=True)

val_loader = torch.utils.data.DataLoader(
    ImageFolderWithPaths(valdir, transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=args.test_batch, shuffle=False,
    pin_memory=True)

test_loader = torch.utils.data.DataLoader(
    ImageFolderWithPaths(testdir, transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=args.test_batch, shuffle=False,
    pin_memory=True)

# check target_layer_names when change resnet18 to resnet50 (18 should be [1] 50 should be [2])
model = models.resnet50(pretrained=True)
# replace the original output layer from 1000 classes to 2 class for man and woman task
model.fc = nn.Linear(2048, 2)

if args.transforms == 'S1':
    # shallow imputation without X
    model.imp = nn.Conv2d(1, 1, 64, stride=32, padding=16)
elif args.transforms == 'S2':
    # shallow imputation with X as additional input
    model.imp = nn.Conv2d(4, 1, 64, stride=32, padding=16)
elif args.transforms == 'D1':
    # deep imputation without X
    model.imp_conv1 = nn.Conv2d(1, 1, 7, stride=2, padding=3)
    model.imp_conv2 = nn.Conv2d(1, 1, 3, stride=2, padding=1)
    model.imp_conv3 = nn.Conv2d(1, 1, 3, stride=2, padding=1)
    model.imp_conv4 = nn.Conv2d(1, 1, 3, stride=2, padding=1)
    model.imp_conv5 = nn.Conv2d(1, 1, 3, stride=2, padding=1)
elif args.transforms == 'D2':
    # deep imputation with X as residual input
    model.imp_conv1 = nn.Conv2d(4, 4, 7, stride=2, padding=3)
    model.imp_conv2 = nn.Conv2d(4, 4, 3, stride=2, padding=1)
    model.imp_conv3 = nn.Conv2d(4, 4, 3, stride=2, padding=1)
    model.imp_conv4 = nn.Conv2d(4, 4, 3, stride=2, padding=1)
    model.imp_conv5 = nn.Conv2d(4, 1, 3, stride=2, padding=1)
model.a = nn.Parameter(torch.tensor([args.a]))

if args.use_cuda:
    model.cuda()

if args.evaluate_all:
    for (dirpath, dirnames, model_fns) in walk(args.model_dir):
        for model_fn in model_fns:
            model = torch.load(os.path.join(dirpath, model_fn))
            test_acc, test_iou, test_precision, test_recall, test_f1 = model_test(model, test_loader,
                                                                                            output_attention=True,
                                                                                            output_iou=True)
            print('Model:', model_fn, ', Acc:', test_acc, ', IOU:', test_iou,
                    ', P:', test_precision, ', R:', test_recall, ', F1:', test_f1)

elif args.evaluate:
    model = torch.load(os.path.join(args.model_dir, args.model_name))

    # evaluate model on test set (set output_attention=true if you want to save the model generated attention)
    test_acc, test_iou, test_precision, test_recall, test_f1 = model_test(model, test_loader, output_attention=True, output_iou=True)
    print('Finish Testing. Test Acc:', test_acc, ', Test IOU:', test_iou, ', Test Precision:', test_precision, ', Test Recall:', test_recall, ', Test F1:', test_f1)
else:
    if args.trainWithMap:
        # for ours
        print('Init training with explanation supervision..')
        best_val_acc = model_train_with_map(model, train_with_map_loader, val_loader, transforms=args.transforms, area=args.area, eta=args.eta)
        print('Finish Training. Best Validation acc:', best_val_acc)
    else:
        # for baseline without explanation supervision
        print('Init training for baseline..')
        best_val_acc = model_train(model, train_loader, val_loader)
        print('Finish Training. Best Validation acc:', best_val_acc)
