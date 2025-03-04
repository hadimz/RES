import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as ff
import torchvision
import os
import time
from utils import *
from my_utils import *
from metrics import *
from integrated_gradients import *


def model_train_mine_IG(model, train_loader, val_loader, args, path_to_attn, eta = 1.0):
    eta = torch.tensor([eta]).cuda()
    task_criterion = nn.CrossEntropyLoss(reduction='none')
    attention_criterion = nn.L1Loss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    best_val_acc = 0

    # load grad_cam module
    # exp_method = integrated_gradients()
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
                    att_map = integrated_gradients(model, input.to('cuda:0'), target)
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
                mask = integrated_gradients(model, input.squeeze().to('cuda:0'))

                if img_fn in path_to_attn:
                    item_att_binary = (mask > 0.5).cpu().numpy()
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