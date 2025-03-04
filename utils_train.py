import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as ff
import torchvision
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

# def sample_selection_with_explanations_sixray(n_smaple_with_label, path_to_attn, args, label_ratio = 1):
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

def model_train_mine(model, train_loader, val_loader, args, path_to_attn, eta = 1.0, num_random_entries = 10):
    eta = torch.tensor([eta]).cuda()
    task_criterion = nn.CrossEntropyLoss(reduction='none')
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

        for batch_idx, (inputs, targets, target_maps, target_maps_org, pred_weight, att_weight, _) in enumerate(train_loader):
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
                # generator = torch.Generator()
                # for mask_id, mask in enumerate(masks_binary):
                #     generator.manual_seed(42)
                #     ones_positions = torch.nonzero(mask == 1, as_tuple=False)
                    
                #     # Convert the positions to a list of tuples
                #     ones_positions = [tuple(pos.tolist()) for pos in ones_positions]

                #     # Define how many random entries you want to select
                #     # num_random_entries = 10
                    
                #     # Randomly select a subset of the positions
                #     random.seed(42)
                #     random_entries = random.sample(ones_positions, num_random_entries)

                #     weak_mask = torch.zeros_like(mask)
                    
                #     for entry in random_entries:
                #         weak_mask[entry] = 1.

                #     # import matplotlib.pyplot as plt
                #     # plt.imshow(weak_mask.detach().cpu())
                #     # plt.savefig('weak_mask.png')
                #     masks_binary[mask_id] = weak_mask.clone()
                    
                # import matplotlib.pyplot as plt
                # plt.subplot(1,3,1)
                # plt.imshow(deprocess_image(input.cpu().detach().squeeze().moveaxis(0,-1).numpy()), cmap='gray')
                # plt.subplot(1,3,2)
                # plt.imshow(att_map_labels[-1].cpu().detach().squeeze(), cmap='jet')
                # plt.subplot(1,3,3)
                # plt.imshow(deprocess_image(input.cpu().detach().squeeze().moveaxis(0,-1).numpy()), cmap='gray')
                # plt.imshow(att_map_labels[-1].cpu().detach().squeeze(), cmap='jet', alpha=0.5)
                # plt.savefig(f'{args.data_dir}_mask_{batch_idx}.png')
                # plt.close('all')
                from guidance_loss import exp_loss
                tempD = exp_loss(masks_binary, att_maps)
                # tempD = torch.sum(masks_binary*ff.resize(att_maps, size=(224, 224)))/torch.sum(ff.resize(att_maps, size=(224, 224)))
                attention_loss += torch.mean(tempD)
                loss = task_loss + eta*attention_loss
                # print(f'on epoch {epoch}, training with lambda {eta} with {num_random_entries} guiding points!')
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

def model_train_mine_act(model, train_loader, val_loader, args, path_to_attn, eta = 1.0, num_random_entries = 10):
    eta = torch.tensor([eta]).cuda()
    task_criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    best_val_acc = 0

    # load grad_cam module
    grad_cam_act = GradCam_mine_act(model=model, feature_module=model.layer4, target_layer_names=["2"], use_cuda=args.use_cuda)
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

        for batch_idx, (inputs, targets, target_maps, target_maps_org, pred_weight, att_weight, _) in enumerate(train_loader):
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
                    att_map, _ = grad_cam_act.get_attention_map(torch.unsqueeze(input, 0), target, norm = None)
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
            #     generator = torch.Generator()
            #     for mask_id, mask in enumerate(masks_binary):
            #         generator.manual_seed(42)
            #         ones_positions = torch.nonzero(mask == 1, as_tuple=False)
                    
            #         # Convert the positions to a list of tuples
            #         ones_positions = [tuple(pos.tolist()) for pos in ones_positions]

            #         # Define how many random entries you want to select
            #         # num_random_entries = 10
                    
            #         # Randomly select a subset of the positions
            #         random.seed(42)
            #         random_entries = random.sample(ones_positions, num_random_entries)

            #         weak_mask = torch.zeros_like(mask)
                    
            #         for entry in random_entries:
            #             weak_mask[entry] = 1.

            #         # import matplotlib.pyplot as plt
            #         # plt.imshow(weak_mask.detach().cpu())
            #         # plt.savefig('weak_mask.png')
            #         masks_binary[mask_id] = weak_mask.clone()
                    
                # import matplotlib.pyplot as plt
                # plt.subplot(1,3,1)
                # plt.imshow(deprocess_image(input.cpu().detach().squeeze().moveaxis(0,-1).numpy()), cmap='gray')
                # plt.subplot(1,3,2)
                # plt.imshow(att_map_labels[-1].cpu().detach().squeeze(), cmap='jet')
                # plt.subplot(1,3,3)
                # plt.imshow(deprocess_image(input.cpu().detach().squeeze().moveaxis(0,-1).numpy()), cmap='gray')
                # plt.imshow(att_map_labels[-1].cpu().detach().squeeze(), cmap='jet', alpha=0.5)
                # plt.savefig(f'{args.data_dir}_mask_{batch_idx}.png')
                # plt.close('all')
                from guidance_loss import exp_loss
                tempD = exp_loss(masks_binary, att_maps)
                # tempD = torch.sum(masks_binary*ff.resize(att_maps, size=(224, 224)))/torch.sum(ff.resize(att_maps, size=(224, 224)))
                attention_loss += torch.mean(tempD)
                loss = task_loss + eta*attention_loss
                # print(f'on epoch {epoch}, training with lambda {eta} with {num_random_entries} guiding points!')
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


def model_train_informed(model, train_loader, val_loader, args, path_to_attn, max_num_random_entries = 10):
    print('training with informed guiding points!')
    task_criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    best_val_acc = 0

    # load grad_cam module
    grad_cam = GradCam_mine(model=model, feature_module=model.layer4, target_layer_names=["2"], use_cuda=args.use_cuda)
    guiding_points = {}
    for epoch in np.arange(args.n_epoch) + 1:
        # switch to train mode
        model.train()

        st = time.time()
        train_losses = []
        if args.use_cuda:
            torch.cuda.empty_cache()

        outputs_all = []
        targets_all = []

        for batch_idx, (inputs, targets, target_maps, target_maps_org, pred_weight, att_weight, image_paths) in enumerate(train_loader):

            attention_loss = 0
            if args.use_cuda:
                inputs, targets, target_maps, target_maps_org, pred_weight, att_weight = inputs.cuda(), targets.cuda(
                    non_blocking=True), target_maps.cuda(), target_maps_org.cuda(), pred_weight.cuda(), att_weight.cuda()
            att_maps = []
            att_map_labels = []
            att_map_labels_trans = []
            att_weights = []
            outputs = model(inputs)

            image_ids = []
            for i, (input, target, target_map, target_map_org, valid_weight, image_path) in enumerate(zip(inputs, targets, target_maps, target_maps_org, att_weight, image_paths)):
                # only train on img with attention labels
                if valid_weight > 0.0:  
                    att_map, _ = grad_cam.get_attention_map(torch.unsqueeze(input, 0), target, norm = None)
                    att_maps.append(att_map)
                    att_map_labels.append((target_map_org==1).float())
                    att_weights.append(valid_weight)
                    # find the indices of the largest value in the attention map to be used as a guiding point
                    image_ids.append(image_path)
                        

            # compute task loss
            task_loss = task_criterion(outputs, targets)
            task_loss = torch.mean(pred_weight * task_loss)

            # compute exp loss
            if att_maps:
                att_maps = torch.stack(att_maps)
                att_map_labels = torch.stack(att_map_labels)
                
                masks_binary = torch.where(att_map_labels==0, 1, 0)
                generator = torch.Generator()
                for mask_id, mask in enumerate(masks_binary):
                    generator.manual_seed(42)
                    # ones_positions = torch.nonzero(mask == 1, as_tuple=False)
                    
                    # Convert the positions to a list of tuples
                    # ones_positions = [tuple(pos.tolist()) for pos in ones_positions]
                    
                    max_index = torch.argmax(ff.resize(att_maps[mask_id:mask_id+1], size=(224, 224), interpolation=torchvision.transforms.InterpolationMode.BICUBIC).squeeze()*mask)
                    x_index = max_index // 224  # Row index
                    y_index = max_index % 224   # Column index                    
                    if image_path in guiding_points:
                        guiding_points[image_paths[mask_id]].append((x_index,y_index))
                    else:
                        guiding_points[image_paths[mask_id]] = [(x_index,y_index)]
                    weak_mask = torch.zeros_like(mask)
                    
                    for entry in guiding_points[image_paths[mask_id]]:
                        # for i in range(torch.max(torch.tensor(0),entry[0]-1), torch.min(torch.tensor(223),entry[0]+2)):
                        #     for j in range(torch.max(torch.tensor(0),entry[1]-1), torch.min(torch.tensor(223),entry[1]+2)):
                        #         weak_mask[i,j] = 1.
                        weak_mask[entry] = 1.

                    # import matplotlib.pyplot as plt
                    # plt.figure(figsize=(20,5))
                    # plt.subplot(1,4,1)
                    # plt.imshow(inputs[mask_id].detach().cpu().moveaxis(0,-1))
                    # plt.subplot(1,4,2)
                    # plt.imshow(ff.resize(att_maps[mask_id:mask_id+1], size=(224, 224)).squeeze().detach().cpu(), cmap='gray')
                    # plt.subplot(1,4,3)
                    # plt.imshow(mask.detach().cpu(), cmap='gray')
                    # plt.subplot(1,4,4)
                    # plt.imshow(weak_mask.detach().detach().cpu())
                    # plt.savefig(f'weak_mask_epoch_{epoch}.png')

                    masks_binary[mask_id] = weak_mask.clone()
                
                # import matplotlib.pyplot as plt
                # plt.subplot(1,3,1)
                # plt.imshow(deprocess_image(input.cpu().detach().squeeze().moveaxis(0,-1).numpy()), cmap='gray')
                # plt.subplot(1,3,2)
                # plt.imshow(att_map_labels[-1].cpu().detach().squeeze(), cmap='jet')
                # plt.subplot(1,3,3)
                # plt.imshow(deprocess_image(input.cpu().detach().squeeze().moveaxis(0,-1).numpy()), cmap='gray')
                # plt.imshow(att_map_labels[-1].cpu().detach().squeeze(), cmap='jet', alpha=0.5)
                # plt.savefig(f'{args.data_dir}_mask_{batch_idx}.png')
                # plt.close('all')
                tempD = torch.sum(masks_binary*ff.resize(att_maps, size=(224, 224)))/torch.sum(ff.resize(att_maps, size=(224, 224)))
                attention_loss += torch.mean(tempD)
                eta = torch.tensor([20*224.*224./epoch]).cuda()
                loss = task_loss + eta*attention_loss
                # print(f'on epoch {epoch}, training with lambda {eta} with {num_random_entries} guiding points!')
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
            print(f'Batch_idx : {batch_idx}, task_loss: {task_loss.item()}, exp_loss: {attention_loss.item()}, weighted exp_loss: {(eta*attention_loss).item()}, total loss: {loss.item()}')
        import matplotlib.pyplot as plt
        plt.figure(figsize=(20,5))
        plt.subplot(1,4,1)
        plt.imshow(deprocess_image(inputs[-1].cpu().detach().squeeze().moveaxis(0,-1).numpy()))
        plt.subplot(1,4,2)
        plt.imshow(ff.resize(att_maps[-1:], size=(224, 224)).squeeze().detach().cpu())
        plt.subplot(1,4,3)
        plt.imshow(att_map_labels[-1].detach().cpu())
        plt.subplot(1,4,4)
        plt.imshow(att_map_labels[-1].detach().cpu())
        plt.imshow(masks_binary[-1].detach().cpu(), cmap='gray', alpha=0.8)
        plt.savefig(f'seed_{args.random_seed}_weak_mask_epoch_{epoch}.png')
        plt.close('all')

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

def model_train_vit(model, train_loader, val_loader, args):
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


def model_train_mine_L2(model, train_loader, val_loader, args, path_to_attn, eta = 1.0, num_random_entries = 10):
    eta = torch.tensor([eta]).cuda()
    task_criterion = nn.CrossEntropyLoss(reduction='none')
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

        for batch_idx, (inputs, targets, target_maps, target_maps_org, pred_weight, att_weight, _) in enumerate(train_loader):
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
                
                masks_binary = torch.where(att_map_labels==0, 1, 0).type(torch.float32)

                tempD = torch.sum((masks_binary*ff.resize(att_maps, size=(224, 224)))**2)/torch.sum(ff.resize(att_maps, size=(224, 224)))
                
                attention_loss += torch.mean(tempD)
                loss = task_loss + eta*attention_loss
                # print(f'on epoch {epoch}, training with lambda {eta} with {num_random_entries} guiding points!')
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



def model_train_mine_L1(model, train_loader, val_loader, args, path_to_attn, eta = 1.0, num_random_entries = 10):
    eta = torch.tensor([eta]).cuda()
    task_criterion = nn.CrossEntropyLoss(reduction='none')
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

        for batch_idx, (inputs, targets, target_maps, target_maps_org, pred_weight, att_weight, _) in enumerate(train_loader):
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
                
                masks_binary = torch.where(att_map_labels==0, 1, 0).type(torch.float32)

                tempD = torch.sum((masks_binary*ff.resize(att_maps, size=(224, 224))))/torch.sum(ff.resize(att_maps, size=(224, 224)))
                
                attention_loss += torch.mean(tempD)
                loss = task_loss + eta*attention_loss
                # print(f'on epoch {epoch}, training with lambda {eta} with {num_random_entries} guiding points!')
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


# def model_train_mine_KLD(model, train_loader, val_loader, args, path_to_attn, eta = 1.0, num_random_entries = 10):
#     eta = torch.tensor([eta]).cuda()
#     task_criterion = nn.CrossEntropyLoss(reduction='none')
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

#     best_val_acc = 0

#     # load grad_cam module
#     grad_cam = GradCam_mine(model=model, feature_module=model.layer4, target_layer_names=["2"], use_cuda=args.use_cuda)
#     for epoch in np.arange(args.n_epoch) + 1:
#         # switch to train mode
#         model.train()

#         st = time.time()
#         train_losses = []
#         if args.use_cuda:
#             torch.cuda.empty_cache()

#         outputs_all = []
#         targets_all = []

#         for batch_idx, (inputs, targets, target_maps, target_maps_org, pred_weight, att_weight, _) in enumerate(train_loader):
#             attention_loss = 0
#             if args.use_cuda:
#                 inputs, targets, target_maps, target_maps_org, pred_weight, att_weight = inputs.cuda(), targets.cuda(
#                     non_blocking=True), target_maps.cuda(), target_maps_org.cuda(), pred_weight.cuda(), att_weight.cuda()
#             att_maps = []
#             att_map_labels = []
#             att_map_labels_trans = []
#             att_weights = []
#             outputs = model(inputs)

#             for i, (input, target, target_map, target_map_org, valid_weight) in enumerate(zip(inputs, targets, target_maps, target_maps_org, att_weight)):
#                 # only train on img with attention labels
#                 if valid_weight > 0.0:  
#                     att_map, _ = grad_cam.get_attention_map(torch.unsqueeze(input, 0), target, norm = None)
#                     att_maps.append(att_map)
#                     att_map_labels.append((target_map_org==1).float())
#                     att_weights.append(valid_weight)
#             # compute task loss
#             task_loss = task_criterion(outputs, targets)
#             task_loss = torch.mean(pred_weight * task_loss)
            
#             # compute exp loss
#             if att_maps:
#                 att_maps = torch.stack(att_maps)
#                 att_map_labels = torch.stack(att_map_labels)
#                 masks_binary = torch.where(att_map_labels==0, 1, 0).type(torch.float32)
#                 saliency_maps_resized = ff.resize(att_maps, masks_binary.shape[2:])
            
#                 if args.normalize:
#                     saliency_maps_resized_normalized = saliency_maps_resized/torch.sum(saliency_maps_resized, dim=(1,2), keepdim=True)
#                     tempD = - masks_binary*torch.log(1.e-8 + (1-saliency_maps_resized_normalized)/(masks_binary + 1.e-8))
#                 else:
#                     tempD = - masks_binary*torch.log(1.e-8 + (1-saliency_maps_resized)/(masks_binary + 1.e-8))
#                 print(f'tempD requires grad: {tempD.requires_grad}')
#                 attention_loss += torch.mean(tempD)


#                 import matplotlib.pyplot as plt
#                 plt.subplot(1,4,1)
#                 plt.imshow(masks_binary[0].cpu().detach(), cmap='gray')
#                 plt.subplot(1,4,2)
#                 plt.imshow(inputs[0].cpu().detach().moveaxis(0,-1))
#                 plt.subplot(1,4,3)
#                 plt.imshow(saliency_maps_resized[0].cpu().detach(), cmap='gray')
#                 plt.subplot(1,4,4)
#                 plt.imshow(tempD[0].detach().cpu(), cmap='gray')
#                 plt.savefig('plt.png')

                
#                 loss = task_loss + eta*attention_loss
#             else:
#                 loss = task_loss

#             # compute gradient and do SGD step
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             train_losses += [loss.cpu().detach().tolist()]

#             outputs_all += [outputs]
#             targets_all += [targets]

#             # print('Batch_idx :', batch_idx, ', task_loss', task_loss, ', attention_loss', attention_loss, ', a:', a)
#             # print('Batch_idx :', batch_idx, ', task_loss:', task_loss, ', attention_loss', 0.3*attention_loss, ', pos_loss:', torch.mean(pos_loss), ', neg_loss:', torch.mean(neg_loss), ', a:', a)

#         et = time.time()
#         train_time = et - st

#         train_acc = accuracy(torch.cat(outputs_all), torch.cat(targets_all))[0].cpu().detach()

#         '''
#             Valid
#         '''
#         print('start validation')
#         model.eval()
#         st = time.time()
#         outputs_all = []
#         targets_all = []

#         iou = AverageMeter()
#         for batch_idx, (inputs, targets, paths) in enumerate(val_loader):
#             if args.use_cuda:
#                 inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
#             with torch.no_grad():
#                 outputs = model(inputs)

#             for img_path in paths:
#                 _, img_fn = os.path.split(img_path)
#                 img = cv2.imread(img_path, 1)
#                 img = np.float32(cv2.resize(img, (224, 224))) / 255
#                 input = preprocess_image(img)
#                 mask = grad_cam(input)

#                 if img_fn in path_to_attn:
#                     item_att_binary = (mask > 0.5)
#                     target_att = path_to_attn[img_fn]
#                     target_att_binary = (target_att > 0)
#                     single_iou = compute_iou(item_att_binary, target_att_binary)
#                     iou.update(single_iou.item(), 1)

#             outputs_all += [outputs]
#             targets_all += [targets]

#         et = time.time()
#         test_time = et - st

#         val_acc = accuracy(torch.cat(outputs_all), torch.cat(targets_all))[0].cpu().detach()
#         val_iou = iou.avg
#         if val_acc > best_val_acc:
#             best_val_acc = val_acc
#             torch.save(model, os.path.join(args.model_dir, args.model_name))
#             print('UPDATE!!!')

#         print('Epoch:', epoch, ', Train Time:', train_time, ', Train Loss:', np.average(train_losses), ', Train Acc:', train_acc, 'Val Acc:', val_acc, 'Val IOU:', val_iou)

#     return best_val_acc


def model_train_mine_KLD(model, train_loader, val_loader, args, path_to_attn, eta = 1.0, num_random_entries = 10):
    eta = torch.tensor([eta]).cuda()
    task_criterion = nn.CrossEntropyLoss(reduction='none')
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

        for batch_idx, (inputs, targets, target_maps, target_maps_org, pred_weight, att_weight, _) in enumerate(train_loader):
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
                
                masks_binary = torch.where(att_map_labels==0, 1, 0).type(torch.float32)
                saliency_maps_resized = ff.resize(att_maps, size=(224, 224))


                if args.normalize:
                    saliency_maps_resized_normalized = saliency_maps_resized/torch.sum(saliency_maps_resized, dim=(1,2), keepdim=True)
                    tempD = - torch.sum(masks_binary*torch.log(1.e-8 + (1-saliency_maps_resized_normalized))/(masks_binary + 1.e-8))/torch.sum(saliency_maps_resized_normalized)
                else:
                    tempD = - torch.sum(masks_binary*torch.log(1.e-8 + (1-saliency_maps_resized)/(masks_binary + 1.e-8)))/torch.sum(saliency_maps_resized)
                
                

                # import matplotlib.pyplot as plt
                # plt.subplot(1,5,1)
                # plt.imshow(masks_binary[0].detach().cpu(), cmap='gray')
                # plt.subplot(1,5,2)
                # plt.imshow(inputs[0].detach().cpu().moveaxis(0,-1))
                # plt.subplot(1,5,3)
                # plt.imshow(saliency_maps_resized[0].detach().cpu(), cmap='gray')
                # plt.subplot(1,5,4)
                # plt.imshow(tempD[0].detach().cpu(), cmap='gray')
                # plt.subplot(1,5,5)
                # plt.imshow((1. + saliency_maps_resized*masks_binary)[0].detach().cpu(), cmap='gray')
                # plt.savefig('plt.png')

                attention_loss += torch.mean(tempD)  
                
                loss = task_loss + eta*attention_loss
                # print(f'on epoch {epoch}, training with lambda {eta} with {num_random_entries} guiding points!')
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