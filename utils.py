import numpy as np
import pandas as pd
import torch
import os
from os import walk
import cv2
import torchvision.datasets as datasets
import argparse
import json

class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x

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
    def __init__(self, path_to_attn_resized):
        self.path_to_attn_resized = path_to_attn_resized
    def __getitem__(self, index):
        path_to_attn_resized = self.path_to_attn_resized
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
    def __init__(self, traindir, transforms, path_to_attn, path_to_attn_resized):
        super().__init__(traindir, transforms)
        self.path_to_attn = path_to_attn
        self.path_to_attn_resized = path_to_attn_resized

    def __getitem__(self, index):
        path_to_attn = self.path_to_attn
        path_to_attn_resized = self.path_to_attn_resized

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
        
        if true_attention_map_org is None:
            true_attention_map_org = np.ones((224, 224))

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
    parser.add_argument('--mine', default=False, type=bool,
                        help='parameter for using my approach')


    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args

class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        for name, module in self.model._modules.items():
            if module == self.feature_module:
                target_activations, x = self.feature_extractor(x)
            elif "avgpool" in name.lower():
                x = module(x)
                x = x.view(x.size(0), -1)
            elif "imp" in name.lower():
                continue
            else:
                x = module(x)

        return target_activations, x

class GradCam:
    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        self.model = model
        self.feature_module = feature_module
        # self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def get_attention_map(self, input, index=None, norm = None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1]

        target = features[-1].squeeze()

        weights = torch.mean(grads_val, axis=(2, 3)).squeeze()

        if self.cuda:
            cam = torch.zeros(target.shape[1:]).cuda()
        else:
            cam = torch.zeros(target.shape[1:])

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        if norm == 'ReLU':
            cam = torch.relu(cam)
            cam = cam / (torch.max(cam) + 1e-6)
        elif norm == 'Sigmoid':
            cam = torch.sigmoid(cam)
        else:
            cam = cam - torch.min(cam)
            cam = cam / (torch.max(cam) + 1e-6)

        return cam, output

    def __call__(self, input, index=None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        # use when visualizing explanation
        cam = cam - np.min(cam)
        cam = cv2.resize(cam, input.shape[2:])
        cam = cam / (np.max(cam) + 1e-6)

        return cam




def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img * 255)

def BF_solver(X, Y):
    epsilon = 1e-4

    with torch.no_grad():
        x = torch.flatten(X)
        y = torch.flatten(Y)
        g_idx = (y<0).nonzero(as_tuple=True)[0]
        le_idx = (y>0).nonzero(as_tuple=True)[0]
        len_g = len(g_idx)
        len_le = len(le_idx)
        a = 0
        a_ct = 0.0
        for idx in g_idx:
            v = x[idx] + epsilon # to avoid miss the constraint itself
            v_ct = 0.0
            for c_idx in g_idx:
                v_ct += (v>x[c_idx]).float()/len_g
            for c_idx in le_idx:
                v_ct += (v<=x[c_idx]).float()/len_le
            if v_ct>a_ct:
                a = v
                a_ct = v_ct
                # print('New best solution found, a=', a, ', # of constraints matches:', a_ct)

        for idx in le_idx:
            v = x[idx]
            v_ct = 0.0
            for c_idx in g_idx:
                v_ct += (v>x[c_idx]).float()/len_g
            for c_idx in le_idx:
                v_ct += (v<=x[c_idx]).float()/len_le
            if v_ct>a_ct:
                a = v
                a_ct = v_ct
                # print('New best solution found, a=', a, ', # of constraints matches:', a_ct)

    # print('optimal solution for batch, a=', a)
    # print('final threshold a is assigned as:', am)

    return torch.tensor([a]).cuda()
