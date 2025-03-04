from torchviz import make_dot
import graphviz
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
from torchsummary import summary
import random
import math
import shutil
import time
from integrated_gradients import *

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

def load_path_to_attentions():
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



def resize_attention_label(path_to_attn, width=7, height=7):
    path_to_attn_resized = {}
    for img_path, img_att in path_to_attn.items():
        att_map = np.uint8(img_att) * 255
        img_att_resized = cv2.resize(att_map, (width, height), interpolation=cv2.INTER_NEAREST)

        path_to_attn_resized[img_path] = np.float32(img_att_resized / 255)
    return path_to_attn_resized

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

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def sample_selection_with_explanations_gender(n_smaple_with_label, path_to_attn, label_ratio = 1):
    n_smaple_without_label = int(n_smaple_with_label/label_ratio)-n_smaple_with_label

    path_to_attn_men = {}
    path_to_attn_women = {}
    source_dir_path = './gender_data/train'
    # before selection, let's create two pools for men and women separately to ensure our selection with be balanced
    for path in path_to_attn:
        if os.path.isfile(source_dir_path + '/men/' + path):
            path_to_attn_men[path] = path_to_attn[path]
        elif os.path.isfile(source_dir_path + '/women/' + path):
            path_to_attn_women[path] = path_to_attn[path]
        # else:
        #     print('Something wrong with this image:', path)

    print('Total number of explanation labels in train set - men:', len(path_to_attn_men))
    print('Total number of explanation labels in train set - women:', len(path_to_attn_women))
    random.seed(args.random_seed)
    sample_paths_men = random.sample(list(path_to_attn_men), n_smaple_with_label)
    sample_paths_women = random.sample(list(path_to_attn_women), n_smaple_with_label)

    path_to_attn_fw = {}
    for path in sample_paths_men:
        path_to_attn_fw[path]= path_to_attn[path]
    for path in sample_paths_women:
        path_to_attn_fw[path]= path_to_attn[path]

    path_to_attn = path_to_attn_fw
    # create and copy the select
    fw_dir_path = './gender_data/' + 'fw_' + str(args.fw_sample) + '_seed_' + str(args.random_seed)

    if not os.path.isdir(fw_dir_path):
        os.mkdir(fw_dir_path)
        os.mkdir(fw_dir_path + '/men')
        os.mkdir(fw_dir_path + '/women')
        # copy the selected images from source to new folder
        for path in path_to_attn:
            # print(path)
            if os.path.isfile(source_dir_path + '/men/' + path):
                src = source_dir_path + '/men/' + path
                dst = fw_dir_path + '/men/' + path
                shutil.copyfile(src, dst)
            elif os.path.isfile(source_dir_path + '/women/' + path):
                src = source_dir_path + '/women/' + path
                dst = fw_dir_path + '/women/' + path
                shutil.copyfile(src, dst)
            else:
                print('Something wrong with this image:', fw_dir_path + '/' + path)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super(TinyModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, stride=4)
        self.activation1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Sequential(torch.nn.Conv2d(16, 4, 3, stride=8))
        self.activation2 = torch.nn.ReLU()
        self.avgpool = torch.nn.AvgPool2d(7)
        self.linear1 = torch.nn.Linear(4, 4)
        self.linear2 = torch.nn.Linear(4, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation1(x)
        x = self.conv2(x)
        x = self.activation2(x)
        x = self.avgpool(x)
        print(f'X has shape {x.shape}')
        x = self.linear1(x.squeeze())
        x = self.linear2(x)
        return x

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

if __name__ == '__main__':
    # check target_layer_names when change resnet18 to resnet50 (18 should be [1] 50 should be [2])
    # model = models.resnet18(pretrained=True)
    # summary(model, (3, 224, 224))
    # replace the original output layer from 1000 classes to 2 class for man and woman task
    # model.fc = nn.Linear(512, 2)
    # print("resnet model layers!")
    # for name, module in model._modules.items():
    #     print(f'{module} is called {name}!')
    
    model = TinyModel().cuda()
    print("TinyModel layers!")
    for name, module in model._modules.items():
        print(f'{module} is called {name}!')
    args = get_args()
    # Data loading code
    path_to_attn, path_to_attn_resized = load_path_to_attentions()
    if args.fw_sample > 0:
        print('Performing few sample learning with # of samples = ', args.fw_sample)

        if args.data_dir == 'gender_data':
            sample_selection_with_explanations_gender(args.fw_sample, path_to_attn)

        else:
            print('Error: Unrecognized dataset:', args.data)

        traindir = os.path.join(args.data_dir, 'fw_' + str(args.fw_sample) + '_seed_' + str(args.random_seed))
    else:
        traindir = os.path.join(args.data_dir, 'train')

    eta = torch.tensor([args.eta]).cuda()
    reg_criterion = nn.MSELoss()
    BCE_criterion = nn.BCELoss()
    task_criterion = nn.CrossEntropyLoss(reduction='none')
    attention_criterion = nn.L1Loss(reduction='none')

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    best_val_acc = 0

    

    # switch to train mode
    model.train()

    st = time.time()
    train_losses = []
    if args.use_cuda:
        torch.cuda.empty_cache()

    outputs_all = []
    targets_all = []

    train_loader = torch.utils.data.DataLoader(
                ImageFolderWithMapsAndWeights(traindir, transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])),
                batch_size=args.train_batch, shuffle=True,
                pin_memory=True)

    for batch_idx, (inputs, targets, target_maps, target_maps_org, pred_weight, att_weight) in enumerate(train_loader):
        attention_loss = 0
        if args.use_cuda:
            inputs, targets, target_maps, target_maps_org, pred_weight, att_weight = inputs.cuda(), targets.cuda(
                non_blocking=True), target_maps.cuda(), target_maps_org.cuda(), pred_weight.cuda(), att_weight.cuda()
        att_maps = []
        att_map_labels = []
        att_map_labels_trans = []
        att_weights = []
        outputs = model(inputs)
        dot = make_dot(outputs[0].mean(), params=dict(model.named_parameters()))
        file_name = "tinymodel4_graph.dot"
        with open(file_name, "w") as file:
            file.write(dot.source)

        for input, target, target_map, target_map_org, valid_weight in zip(inputs, targets, target_maps, target_maps_org, att_weight):
            # only train on img with attention labels
            if valid_weight > 0.0:
                # get attention maps from grad-CAM
                att_map, temp_batch = integrated_gradients(model, input.to('cuda:0'), target)
                print(f'single attention map shape: {att_map.shape}')
                att_maps.append(att_map)

                if transforms == 'Gaussian':
                    # here we only work on positive labels for D loss
                    target_map_pos = np.maximum(target_map.cpu().numpy(), 0)
                    target_map_trans = cv2.GaussianBlur(target_map_pos, (3, 3), 0)
                    target_map_trans = target_map_trans / (np.max(target_map_trans)+1e-6)
                    att_map_labels_trans.append(torch.from_numpy(target_map_trans).cuda())
                elif transforms == 'S1':
                    target_map_pos_org = torch.unsqueeze((target_map_org>0).float(),0)
                    input_imp = target_map_pos_org

                    target_map_trans = model.imp(torch.unsqueeze(input_imp, 0))
                    temp = torch.squeeze(target_map_trans)
                    temp = temp - torch.min(temp)
                    temp = temp / (torch.max(temp) + 1e-6)
                    att_map_labels_trans.append(temp)
                elif transforms == 'S2':
                    target_map_pos_org = torch.unsqueeze((target_map_org>0).float(),0)
                    # with both input X and the human mask F (input 3x224x224, we need the raw target map in 1x224x224)
                    input_imp = torch.cat((target_map_pos_org, input), 0)

                    target_map_trans = model.imp(torch.unsqueeze(input_imp, 0))
                    temp = torch.squeeze(target_map_trans)
                    temp = temp - torch.min(temp)
                    temp = temp / (torch.max(temp) + 1e-6)
                    att_map_labels_trans.append(temp)
                elif transforms == 'D1':
                    target_map_pos_org = torch.unsqueeze((target_map_org>0).float(),0)
                    input_imp = target_map_pos_org

                    H1 = torch.relu(model.imp_conv1(torch.unsqueeze(input_imp, 0)))
                    H2 = torch.relu(model.imp_conv2(H1))
                    H3 = torch.relu(model.imp_conv3(H2))
                    H4 = torch.relu(model.imp_conv4(H3))
                    H5 = model.imp_conv5(H4)

                    temp = torch.squeeze(H5)
                    temp = temp - torch.min(temp)
                    temp = temp / (torch.max(temp) + 1e-6)
                    att_map_labels_trans.append(temp)
                elif transforms == 'D2':
                    target_map_pos_org = torch.unsqueeze((target_map_org>0).float(),0)
                    # 4x224x224
                    input_imp = torch.cat((target_map_pos_org, input), 0)

                    H1 = torch.relu(model.imp_conv1(torch.unsqueeze(input_imp, 0)))
                    H2 = torch.relu(model.imp_conv2(H1))
                    H3 = torch.relu(model.imp_conv3(H2))
                    H4 = torch.relu(model.imp_conv4(H3))
                    H5 = model.imp_conv5(H4)

                    temp = torch.squeeze(H5)
                    temp = temp - torch.min(temp)
                    temp = temp / (torch.max(temp) + 1e-6)
                    att_map_labels_trans.append(temp)

                att_map_labels.append(target_map_org)
                att_weights.append(valid_weight)
            break
            

        # compute task loss
        task_loss = task_criterion(outputs, targets)
        task_loss = torch.mean(pred_weight * task_loss)

        # compute exp loss
        if att_maps:
            att_maps = torch.stack(att_maps)
            att_map_labels = torch.stack(att_map_labels)

            if transforms == 'S1' or transforms == 'S2' or transforms == 'D1' or transforms == 'D2' or transforms == 'Gaussian':
                # hard threshold solver for a
                a = BF_solver(att_maps, att_map_labels)
                # alternatively, we can use tanh as surrogate loss to make att_maps trainable
                temp1 = torch.tanh(5*(att_maps - a))
                temp_loss = attention_criterion(temp1, att_map_labels)

                # normalize by effective areas
                temp_size = (att_map_labels != 0).float()
                eff_loss = torch.sum(temp_loss * temp_size) / torch.sum(temp_size)
                attention_loss += torch.relu(torch.mean(eff_loss) - eta)
            else:
                a = 0

            if transforms == 'S1' or transforms == 'S2' or transforms == 'D1' or transforms == 'D2':
                att_map_labels_trans = torch.stack(att_map_labels_trans)
                tempD = attention_criterion(att_maps, att_map_labels_trans)
                # regularization (currently prefer not use it)
                reg_loss = reg_criterion(att_map_labels_trans, att_map_labels * (att_map_labels > 0).float())
                attention_loss += args.reg * reg_loss
            elif transforms == 'Gaussian':
                att_map_labels_trans = torch.stack(att_map_labels_trans)
                tempD = attention_criterion(att_maps, att_map_labels_trans)
            elif transforms == 'HAICS':
                tempD = BCE_criterion(att_maps, att_map_labels * (att_map_labels > 0).float()) * (att_map_labels != 0).float()
            else: # GRADIA
                print(f'attention map shape: {att_maps.shape}')
                print(f'attention label shape: {att_map_labels.shape}')
                tempD = attention_criterion(att_maps, att_map_labels * (att_map_labels > 0).float())

            attention_loss += torch.mean(tempD)
            loss = task_loss + attention_loss
        else:
            loss = task_loss

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses += [loss.cpu().detach().tolist()]

        outputs_all += [outputs]
        targets_all += [targets]

        print('Batch_idx :', batch_idx, ', task_loss', task_loss, ', attention_loss', attention_loss, ', a:', a)
        # print('Batch_idx :', batch_idx, ', task_loss:', task_loss, ', attention_loss', 0.3*attention_loss, ', pos_loss:', torch.mean(pos_loss), ', neg_loss:', torch.mean(neg_loss), ', a:', a)
        break

    et = time.time()
    train_time = et - st

    train_acc = accuracy(torch.cat(outputs_all), torch.cat(targets_all))[0].cpu().detach()

    print('Epoch:', 1, ', Train Time:', train_time, ', Train Loss:', np.average(train_losses), ', Train Acc:', train_acc)

    print(f'Attention maps shape: {att_maps.shape}')
    # Visualize model computation graph
    params = dict(model.named_parameters())
    params['temp inputs'] = temp_batch
    dot = make_dot(loss.mean(), params=params)
    file_name = "IG5_guided_loss_graph_with_inputs.dot"
    with open(file_name, "w") as file:
        file.write(dot.source)