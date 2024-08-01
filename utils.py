import numpy as np
import pandas as pd
import torch
import os
from os import walk
import cv2
import torchvision.datasets as datasets
import argparse
import json


def preprocess_image(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = preprocessed_img.requires_grad_(True)
    return input

def show_cam_on_image(img,mask,path,file_name):
    save_path = os.path.join(path, file_name)
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    cam = heatmap + img
    cam = cam / np.max(cam)
    cv2.imwrite(save_path, np.uint8(255 * cam))

def resize_attention_label(path_to_attn, width=7, height=7):
    path_to_attn_resized = {}
    for img_path, img_att in path_to_attn.items():
        att_map = np.uint8(img_att) * 255
        img_att_resized = cv2.resize(att_map, (width, height), interpolation=cv2.INTER_NEAREST)

        path_to_attn_resized[img_path] = np.float32(img_att_resized / 255)
    return path_to_attn_resized

def load_path_to_attentions(args):
    # csv file format: img_idx,attention,img_check,matrix_resize
    path_to_attn = {}
    path_to_pos_attn = {}
    path_to_neg_attn = {}
    source_path = os.path.join(args.data_dir, 'attention_label')
    fns = next(walk(source_path), (None, None, []))[2]

    for fn in fns:
        df = pd.read_csv(os.path.join(source_path, fn))
        

        if 'counterfactual' in fn:
            # negative attention labels
            for index, row in df.iterrows():
                if row['img_check'] == 'good':
                    if row['img_idx'][-4:] != '.jpg':
                        img_fn = row['img_idx'] + '.jpg'
                    else:
                        img_fn = row['img_idx']
                    path_to_neg_attn[img_fn] = np.array(json.loads(row['attention']))
        else:
            # positive attention labels
            for index, row in df.iterrows():
                if row['img_check'] == 'good':
                    if row['img_idx'][-4:] != '.jpg':
                        img_fn = row['img_idx'] + '.jpg'
                    else:
                        img_fn = row['img_idx']
                    path_to_pos_attn[img_fn] = np.array(json.loads(row['attention']))
                    path_to_attn[img_fn] = ''
    
    for img_fn in path_to_attn.keys():
        if img_fn in path_to_pos_attn:
            if img_fn in path_to_neg_attn:
                path_to_attn[img_fn] = path_to_pos_attn[img_fn] - path_to_neg_attn[img_fn]
            else:
                path_to_attn[img_fn] = path_to_pos_attn[img_fn]

    # resized
    path_to_attn_resized = {}
    path_to_pos_attn = resize_attention_label(path_to_pos_attn)
    path_to_neg_attn = resize_attention_label(path_to_neg_attn)

    for img_fn in path_to_attn.keys():
        if img_fn in path_to_pos_attn:
            if img_fn in path_to_neg_attn:
                path_to_attn_resized[img_fn] = path_to_pos_attn[img_fn] - path_to_neg_attn[img_fn]
            else:
                path_to_attn_resized[img_fn] = path_to_pos_attn[img_fn]

    return path_to_attn, path_to_attn_resized

class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]

        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


class ImageFolderWithMaps(datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithMaps, self).__getitem__(index)
        path = self.imgs[index][0]

        head, tail = os.path.split(path)

        # Add pred_weight & att_weight here
        if tail in path_to_attn_resized:
            true_attention_map = path_to_attn_resized[tail]
        else:
            true_attention_map = np.zeros((7, 7), dtype=np.float32)

        tuple_with_map = (original_tuple + (true_attention_map,))
        return tuple_with_map


class ImageFolderWithMapsAndWeights(datasets.ImageFolder):

    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithMapsAndWeights, self).__getitem__(index)
        path = self.imgs[index][0]

        head, tail = os.path.split(path)

        # default weights
        pred_weight = 1
        att_weight = 0
        # Add pred_weight & att_weight here
        if tail in path_to_attn_resized:
            true_attention_map_org = path_to_attn[tail]
            true_attention_map = path_to_attn_resized[tail]
            pred_weight = 1
            att_weight = 1

        else:
            true_attention_map_org = None
            true_attention_map = np.zeros((7, 7), dtype=np.float32)

        tuple_with_map_and_weights = (original_tuple + (true_attention_map, true_attention_map_org, pred_weight, att_weight))
        return tuple_with_map_and_weights


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default='./examples/both.png',
                        help='Input image path')
    parser.add_argument('--n_epoch', type=int, default=20,
                        help='Number of epoch to run')
    parser.add_argument('--data_dir', default='gender_data', type=str)
    parser.add_argument('--model_dir', type=str, default='./model_save/',
                        help='The address for storing the models and optimization results.')
    parser.add_argument('--model_name', type=str, default='model_out',
                        help='The model filename that will be used for evaluation or phase 2 fine-tuning.')
    parser.add_argument('--train-batch', default=256, type=int, metavar='N',
                        help='train batchsize (default: 256)')
    parser.add_argument('--test-batch', default=200, type=int, metavar='N',
                        help='test batchsize (default: 200)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on test set')
    parser.add_argument('-ea', '--evaluate_all', dest='evaluate_all', action='store_true',
                        help='evaluate all models stored in model_dir')
    parser.add_argument('--trainWithMap', dest='trainWithMap', action='store_true',
                        help='train with edited attention map')
    parser.add_argument('--attention_weight', default=1.0, type=float,
                        help='Scale factor that balance between task loss and attention loss')
    parser.add_argument('--transforms', type=str, default=None,
                        help='The transform method to prcoess the human label, choices [None, gaussian, S1, S2, D1, D2]')
    parser.add_argument('--area', dest='area', action='store_true',
                        help='If only apply explanation loss to human labeled regions')
    parser.add_argument('--a', default=0.75, type=float,
                        help='Threshold  for function U')
    parser.add_argument('--eta', default=0.0, type=float,
                        help='Slack factor for robust attention loss')
    parser.add_argument('--fw_sample', default=0, type=int,
                        help='if non-zero, randomly sample instances to construct the dataset and perform learning')
    parser.add_argument('--random_seed', default=0, type=int, metavar='N',
                        help='random seed for sampling the dataset')
    parser.add_argument('--reg', default=0.0, type=float,
                        help='The scale factor for L2 regularization for deep imputation')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args
