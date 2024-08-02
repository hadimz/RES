import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.optim import SGD
import Network
from torch.utils.data import TensorDataset, DataLoader

from gradCAM import class_activation_map

from torch.utils.tensorboard import SummaryWriter

writer_train = SummaryWriter(log_dir='results_guided_10/logs/train')
writer_valid = SummaryWriter(log_dir='results_guided_10/logs/valid')

device = 'cuda:0'

model = Network.WeakFinder(in_channels=1, 
                           out_channels=10,
                           size=224,
                           grad=True).to(device)


train_data = np.load('mnist_train_224.npz')
train_images, train_masks, train_labels = train_data['images'], train_data['masks'], [label[0] for label in train_data['labels']]

train_set = TensorDataset(torch.tensor(train_images).unsqueeze(1).float(),
                          torch.tensor(train_masks).unsqueeze(1).float(),
                          torch.tensor(train_labels).type(torch.LongTensor))
train_loader = DataLoader(train_set, batch_size=16)

test_data = np.load('mnist_test_224.npz')
test_images, test_masks, test_labels = test_data['images'], test_data['masks'], [label[0] for label in test_data['labels']]
test_set = TensorDataset(torch.tensor(test_images).unsqueeze(1).float(),
                         torch.tensor(test_masks).unsqueeze(1).float(),
                         torch.tensor(test_labels).type(torch.LongTensor))
test_loader = DataLoader(test_set, batch_size=16)


optimizer = SGD(model.parameters(), lr=0.001)
CELoss = torch.nn.CrossEntropyLoss()
log_interval = 30
train_losses = []
train_counter = []


for epoch in range(100):
  running_tloss = 0.0
  tpreds = []
  t_targets = []
  tlen = 0
  t_acc = 0

  # main training loop. Trains for 1 epoch
  model.train()
  for batch_idx, (data, masks, target) in enumerate(train_loader):
    # **************************************************************************
    gradcams, _, _ = class_activation_map(model, data.to(device), labels=target, cuda=True, size=224)
    temp = masks - target.view([-1, 1, 1, 1]) - 1
    masks_binary = torch.where(temp==0, 0, 1)
    loss_cam = torch.mean(masks_binary.to(device)*gradcams)
    # **************************************************************************
    optimizer.zero_grad()
    outputs = model(data.to(device))
    loss = CELoss(outputs, target.to(device)) + 10.*loss_cam
    loss.backward()
    optimizer.step()
    del gradcams
    _, preds = torch.max(F.softmax(outputs.detach(), dim=1), 1)
    t_acc = t_acc + torch.sum(preds == target.to(device))

    tlen = tlen + len(target)
    running_tloss = running_tloss + loss.detach()
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))
      train_losses.append(loss.item())
      train_counter.append((batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
      torch.save(model.state_dict(), f'results_guided_10/model_epoch_{epoch:03}.pth')
      torch.save(optimizer.state_dict(), f'results_guided_10/optimizer_epoch_{epoch:03}.pth')
      
      # **************************************************************************
      gradcams, _, _ = class_activation_map(model, data.to(device), labels=target, cuda=True, size=224)
      plt.subplot(1,3,1)
      plt.imshow(data[0].squeeze().cpu())
      plt.subplot(1,3,2)
      plt.imshow(gradcams[0].squeeze().detach().cpu())
      plt.subplot(1,3,3)
      plt.imshow(data[0].squeeze().cpu())
      plt.imshow(gradcams[0].squeeze().detach().cpu(), alpha=0.7)
      plt.show()
      plt.savefig(f'results_guided_10/figs/epoch_{epoch:03}_batch_{batch_idx:04}.jpg')
      del gradcams
      # **************************************************************************

  
  # evaluate the model after an epoch of training
  running_vloss = 0.0
  vpreds = []
  vtargets = []
  vlen = 0
  v_acc = 0
  with torch.no_grad():
    model.eval()
    for i, (vdata, vmasks, vtarget) in enumerate(test_loader):
        outputs = model(vdata.to(device))
        vloss = CELoss(outputs, vtarget.to(device).to(torch.int64))
        running_vloss += vloss

        _, vpreds = torch.max(F.softmax(outputs, dim=1), 1)
        v_acc = v_acc + torch.sum((vpreds.cpu() == vtarget))
        vlen = vlen + len(vtarget)
  v_acc = v_acc/vlen
  t_acc = t_acc/tlen
  print(f'training accuracy: {t_acc}, \n training loss: {running_tloss/len(train_loader)}')
  print(f'validation accuracy: {v_acc}, \n validation loss: {running_vloss/len(test_loader)}')
  
  writer_train.add_scalar("Loss", running_tloss/len(train_loader), epoch)
  writer_train.add_scalar("Accuracy", t_acc, epoch)
  writer_valid.add_scalar("Loss", running_vloss/len(test_loader), epoch)
  writer_valid.add_scalar("Accuracy", v_acc, epoch)
  writer_train.flush()
  writer_valid.flush()

