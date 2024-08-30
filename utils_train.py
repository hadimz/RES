import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as ff
import random
import os
import shutil
import time
import cv2
from utils import *
from my_utils import *
from metrics import *

def sample_selection_with_explanations_gender(n_smaple_with_label, path_to_attn, args, label_ratio = 1):
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
            # else:
                # print('Something wrong with this image:', fw_dir_path + '/' + path)

def sample_selection_with_explanations_places(n_smaple_with_label, path_to_attn, args, label_ratio = 1):
    n_smaple_without_label = int(n_smaple_with_label/label_ratio)-n_smaple_with_label

    path_to_attn_nature = {}
    path_to_attn_urban = {}
    source_dir_path = './places/train'
    # before selection, let's create two pools for nature and urban separately to ensure our selection with be balanced
    for path in path_to_attn:
        if os.path.isfile(source_dir_path + '/nature/' + path):
            path_to_attn_nature[path] = path_to_attn[path]
        elif os.path.isfile(source_dir_path + '/urban/' + path):
            path_to_attn_urban[path] = path_to_attn[path]
        else:
            print('Something wrong with this image:', path)
    print('Total number of explanation labels in train set - nature:', len(path_to_attn_nature))
    print('Total number of explanation labels in train set - urban:', len(path_to_attn_urban))

    random.seed(args.random_seed)
    sample_paths_nature = random.sample(list(path_to_attn_nature), n_smaple_with_label)
    sample_paths_urban = random.sample(list(path_to_attn_urban), n_smaple_with_label)

    path_to_attn_fw = {}
    for path in sample_paths_nature:
        path_to_attn_fw[path]= path_to_attn[path]
    for path in sample_paths_urban:
        path_to_attn_fw[path]= path_to_attn[path]

    path_to_attn = path_to_attn_fw
    # create and copy the select
    fw_dir_path = './places/' + 'fw_' + str(args.fw_sample) + '_seed_' + str(args.random_seed)

    if not os.path.isdir(fw_dir_path):
        os.mkdir(fw_dir_path)
        os.mkdir(fw_dir_path + '/nature')
        os.mkdir(fw_dir_path + '/urban')
        # copy the selected images from source to new folder
        for path in path_to_attn:
            # print(path)
            if os.path.isfile(source_dir_path + '/nature/' + path):
                src = source_dir_path + '/nature/' + path
                dst = fw_dir_path + '/nature/' + path
                shutil.copyfile(src, dst)
            elif os.path.isfile(source_dir_path + '/urban/' + path):
                src = source_dir_path + '/urban/' + path
                dst = fw_dir_path + '/urban/' + path
                shutil.copyfile(src, dst)
            else:
                print('Something wrong with this image:', fw_dir_path + '/' + path)

def sample_selection_with_explanations_sixray(n_smaple_with_label, path_to_attn, args, label_ratio = 1):
    n_smaple_without_label = int(n_smaple_with_label/label_ratio)-n_smaple_with_label

    path_to_attn_neg = {}
    path_to_attn_pos = {}
    source_dir_path = './sixray/train'
    # before selection, let's create two pools for positive and negative separately to ensure our selection with be balanced
    for path in path_to_attn:
        # if os.path.isfile(source_dir_path + '/neg/' + path):
        #     path_to_attn_neg[path] = path_to_attn[path]
        #     print(source_dir_path + '/neg/' + path)
        if os.path.isfile(source_dir_path + '/pos/' + path):
            path_to_attn_pos[path] = path_to_attn[path]
        else:
            print('Something wrong with this image:', path)
    
    neg_filenames = next(os.walk('./sixray/train/neg/'), (None, None, []))[2]
    for path in neg_filenames:
        if os.path.isfile(source_dir_path + '/neg/' + path):
            path_to_attn[path] = np.ones(224)
            path_to_attn_neg[path] = path_to_attn[path]

    print('Total number of explanation labels in train set - negative:', len(path_to_attn_neg))
    print('Total number of explanation labels in train set - positive:', len(path_to_attn_pos))
    random.seed(args.random_seed)
    sample_paths_pos = random.sample(list(path_to_attn_pos), n_smaple_with_label)
    sample_paths_neg = random.sample(list(path_to_attn_neg), n_smaple_with_label)

    path_to_attn_fw = {}
    for path in sample_paths_pos:
        path_to_attn_fw[path]= path_to_attn[path]
    for path in sample_paths_neg:
        path_to_attn_fw[path]= path_to_attn[path]

    path_to_attn = path_to_attn_fw
    # create and copy the select
    fw_dir_path = './sixray/' + 'fw_' + str(args.fw_sample) + '_seed_' + str(args.random_seed)

    if not os.path.isdir(fw_dir_path):
        os.mkdir(fw_dir_path)
        os.mkdir(fw_dir_path + '/neg')
        os.mkdir(fw_dir_path + '/pos')
        # copy the selected images from source to new folder
        for path in path_to_attn:
            # print(path)
            if os.path.isfile(source_dir_path + '/neg/' + path):
                src = source_dir_path + '/neg/' + path
                dst = fw_dir_path + '/neg/' + path
                shutil.copyfile(src, dst)
            elif os.path.isfile(source_dir_path + '/pos/' + path):
                src = source_dir_path + '/pos/' + path
                dst = fw_dir_path + '/pos/' + path
                shutil.copyfile(src, dst)
            else:
                print('Something wrong with this image:', fw_dir_path + '/' + path)

def model_test(model, test_loader, args, path_to_attn, output_attention=False, output_iou=False):
    # model.eval()
    iou = AverageMeter()
    exp_precision = AverageMeter()
    exp_recall = AverageMeter()
    exp_f1 = AverageMeter()
    ious = {}
    st = time.time()
    mine_all = []
    outputs_all = []
    targets_all = []
    img_fns = []

    grad_cam = GradCam(model=model, feature_module=model.layer4, \
                       target_layer_names=["2"], use_cuda=args.use_cuda)
    y_label = np.array([])
    y_predict = np.array([])
    misclassified = np.array([])

    for batch_idx, (inputs, targets, paths) in enumerate(test_loader):
        y_label = np.append(y_label, targets)
        misclassified = np.append(misclassified, paths)

        inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)

        # compute output
        with torch.no_grad():
            logits = model(inputs)
            outputs = torch.nn.functional.softmax(logits, dim=1)
            _, predicted = torch.max(outputs, 1)
            predicted = predicted.cpu()
            y_predict = np.append(y_predict, predicted)

        if output_attention:
            for img_path in paths:
                _, img_fn = os.path.split(img_path)

                img_fns.append(img_fn)

                img = cv2.imread(img_path, 1)
                img = np.float32(cv2.resize(img, (224, 224))) / 255
                input = preprocess_image(img)
                mask = grad_cam(input)
                show_cam_on_image(img, mask, 'attention', img_path)

                if output_iou and img_fn in path_to_attn:
                    
                    item_att_binary = (mask > 0.5)
                    target_att = path_to_attn[img_fn]
                    target_att_binary = (target_att > 0)
                    single_iou = compute_iou(item_att_binary, target_att_binary)
                    iou.update(single_iou.item(), 1)

                    p, r, f1 = compute_exp_score(item_att_binary, target_att)
                    exp_precision.update(p.item(), 1)
                    exp_recall.update(r.item(), 1)
                    exp_f1.update(f1.item(), 1)

                    ious[img_fn] = single_iou.item()
                    
                    mine_all += [torch.sum((1-item_att_binary)*ff.resize(mask, size=(224, 224)))/torch.sum(ff.resize(mask, size=(224, 224)))]

        outputs_all += [outputs]
        targets_all += [targets]

    et = time.time()
    test_time = et - st

    test_acc = accuracy(torch.cat(outputs_all, dim=0), torch.cat(targets_all))[0].cpu().detach()
    my_metric = torch.mean(mine_all.cpu().detach())

    return test_acc, iou.avg, exp_precision.avg, exp_recall.avg, exp_f1.avg

def model_train_with_map(model, train_loader, val_loader, args, path_to_attn, transforms = None, area = False, eta = 0.0):
    eta = torch.tensor([eta]).cuda()
    reg_criterion = nn.MSELoss()
    # reg_criterion = nn.L1Loss()
    BCE_criterion = nn.BCELoss()
    task_criterion = nn.CrossEntropyLoss(reduction='none')
    attention_criterion = nn.L1Loss(reduction='none')

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    best_val_acc = 0

    # load grad_cam module
    grad_cam = GradCam(model=model, feature_module=model.layer4, target_layer_names=["2"], use_cuda=args.use_cuda)
    for epoch in np.arange(args.n_epoch) + 1:
        # switch to train mode
        model.train()

        st = time.time()
        train_losses = []
        if args.use_cuda:
            torch.cuda.empty_cache()

        outputs_all = []
        targets_all = []

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

            for input, target, target_map, target_map_org, valid_weight in zip(inputs, targets, target_maps, target_maps_org, att_weight):
                # only train on img with attention labels
                if valid_weight > 0.0:
                    # get attention maps from grad-CAM
                    att_map, _ = grad_cam.get_attention_map(torch.unsqueeze(input, 0), target, norm = None)
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

                    att_map_labels.append(target_map)
                    att_weights.append(valid_weight)

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

        et = time.time()
        train_time = et - st

        train_acc = accuracy(torch.cat(outputs_all), torch.cat(targets_all))[0].cpu().detach()

        '''
            Valid
        '''
        print('start validation')
        model.eval()
        st = time.time()
        outputs_all = []
        targets_all = []

        iou = AverageMeter()
        for batch_idx, (inputs, targets, paths) in enumerate(val_loader):
            if args.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            with torch.no_grad():
                outputs = model(inputs)

            for img_path in paths:
                _, img_fn = os.path.split(img_path)
                img = cv2.imread(img_path, 1)
                img = np.float32(cv2.resize(img, (224, 224))) / 255
                input = preprocess_image(img)
                mask = grad_cam(input)

                if img_fn in path_to_attn:
                    item_att_binary = (mask > 0.5)
                    target_att = path_to_attn[img_fn]
                    target_att_binary = (target_att > 0)
                    single_iou = compute_iou(item_att_binary, target_att_binary)
                    iou.update(single_iou.item(), 1)

            outputs_all += [outputs]
            targets_all += [targets]

        et = time.time()
        test_time = et - st

        val_acc = accuracy(torch.cat(outputs_all), torch.cat(targets_all))[0].cpu().detach()
        val_iou = iou.avg
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model, os.path.join(args.model_dir, args.model_name))
            print('UPDATE!!!')

        print('Epoch:', epoch, ', Train Time:', train_time, ', Train Loss:', np.average(train_losses), ', Train Acc:', train_acc, 'Val Acc:', val_acc, 'Val IOU:', val_iou)

    return best_val_acc

def model_train(model, train_loader, val_loader, args):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    best_val_acc = 0

    for epoch in np.arange(args.n_epoch) + 1:
        # switch to train mode
        model.train()

        st = time.time()
        train_losses = []
        if args.use_cuda:
            torch.cuda.empty_cache()

        outputs_all = []
        targets_all = []

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            if args.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)

            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses += [loss.cpu().detach().tolist()]

            outputs_all += [outputs]
            targets_all += [targets]

            print('Batch_idx :', batch_idx, ', loss', loss)

        et = time.time()
        train_time = et - st

        train_acc = accuracy(torch.cat(outputs_all), torch.cat(targets_all))[0].cpu().detach()

        '''
            Valid
        '''
        print('start validation')
        model.eval()
        st = time.time()
        outputs_all = []
        targets_all = []
        with torch.no_grad():
            for batch_idx, (inputs, targets, paths) in enumerate(val_loader):
                if args.use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)

                # compute output
                outputs = model(inputs)

                outputs_all += [outputs]
                targets_all += [targets]

        et = time.time()
        val_time = et - st
        val_acc = accuracy(torch.cat(outputs_all), torch.cat(targets_all))[0].cpu().detach()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model, os.path.join(args.model_dir, args.model_name))
            print('UPDATE!!!')

        print('Epoch:', epoch, ', Train Time:', train_time, ', Train Loss:', np.average(train_losses), ', Train Acc:', train_acc, 'Val Acc:', val_acc)

    return best_val_acc

def model_train_mine(model, train_loader, val_loader, args, path_to_attn, eta = 1.0):
    eta = torch.tensor([eta]).cuda()
    task_criterion = nn.CrossEntropyLoss(reduction='none')
    attention_criterion = nn.L1Loss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    best_val_acc = 0

    # load grad_cam module
    grad_cam = GradCam_mine(model=model, feature_module=model.layer4, target_layer_names=["2"], use_cuda=args.use_cuda)
    for epoch in np.arange(args.n_epoch) + 1:
        # switch to train mode
        model.train()

        st = time.time()
        train_losses = []
        if args.use_cuda:
            torch.cuda.empty_cache()

        outputs_all = []
        targets_all = []

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

            for i, (input, target, target_map, target_map_org, valid_weight) in enumerate(zip(inputs, targets, target_maps, target_maps_org, att_weight)):
                # only train on img with attention labels
                if valid_weight > 0.0:  
                    att_map, _ = grad_cam.get_attention_map(torch.unsqueeze(input, 0), target, norm = None)
                    att_maps.append(att_map)
                    att_map_labels.append((target_map_org==1).float())
                    att_weights.append(valid_weight)
            # compute task loss
            task_loss = task_criterion(outputs, targets)
            task_loss = torch.mean(pred_weight * task_loss)

            # compute exp loss
            if att_maps:
                att_maps = torch.stack(att_maps)
                att_map_labels = torch.stack(att_map_labels)
                
                masks_binary = torch.where(att_map_labels==0, 1, 0)
                import matplotlib.pyplot as plt
                plt.subplot(1,3,1)
                plt.imshow(deprocess_image(input.cpu().detach().squeeze().moveaxis(0,-1).numpy()), cmap='gray')
                plt.subplot(1,3,2)
                plt.imshow(att_map_labels[-1].cpu().detach().squeeze(), cmap='jet')
                plt.subplot(1,3,3)
                plt.imshow(deprocess_image(input.cpu().detach().squeeze().moveaxis(0,-1).numpy()), cmap='gray')
                plt.imshow(att_map_labels[-1].cpu().detach().squeeze(), cmap='jet', alpha=0.5)
                plt.savefig(f'{args.data_dir}_mask_{batch_idx}.png')
                plt.close('all')
                tempD = torch.sum(masks_binary*ff.resize(att_maps, size=(224, 224)))/torch.sum(ff.resize(att_maps, size=(224, 224)))

                attention_loss += torch.mean(tempD)
                loss = task_loss + eta*attention_loss
            else:
                loss = task_loss

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses += [loss.cpu().detach().tolist()]

            outputs_all += [outputs]
            targets_all += [targets]

            # print('Batch_idx :', batch_idx, ', task_loss', task_loss, ', attention_loss', attention_loss, ', a:', a)
            # print('Batch_idx :', batch_idx, ', task_loss:', task_loss, ', attention_loss', 0.3*attention_loss, ', pos_loss:', torch.mean(pos_loss), ', neg_loss:', torch.mean(neg_loss), ', a:', a)

        et = time.time()
        train_time = et - st

        train_acc = accuracy(torch.cat(outputs_all), torch.cat(targets_all))[0].cpu().detach()

        '''
            Valid
        '''
        print('start validation')
        model.eval()
        st = time.time()
        outputs_all = []
        targets_all = []

        iou = AverageMeter()
        for batch_idx, (inputs, targets, paths) in enumerate(val_loader):
            if args.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            with torch.no_grad():
                outputs = model(inputs)

            for img_path in paths:
                _, img_fn = os.path.split(img_path)
                img = cv2.imread(img_path, 1)
                img = np.float32(cv2.resize(img, (224, 224))) / 255
                input = preprocess_image(img)
                mask = grad_cam(input)

                if img_fn in path_to_attn:
                    item_att_binary = (mask > 0.5)
                    target_att = path_to_attn[img_fn]
                    target_att_binary = (target_att > 0)
                    single_iou = compute_iou(item_att_binary, target_att_binary)
                    iou.update(single_iou.item(), 1)

            outputs_all += [outputs]
            targets_all += [targets]

        et = time.time()
        test_time = et - st

        val_acc = accuracy(torch.cat(outputs_all), torch.cat(targets_all))[0].cpu().detach()
        val_iou = iou.avg
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model, os.path.join(args.model_dir, args.model_name))
            print('UPDATE!!!')

        print('Epoch:', epoch, ', Train Time:', train_time, ', Train Loss:', np.average(train_losses), ', Train Acc:', train_acc, 'Val Acc:', val_acc, 'Val IOU:', val_iou)

    return best_val_acc
