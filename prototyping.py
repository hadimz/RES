import torch
import matplotlib.pyplot as plt

x = torch.load('x_adv.pt')
print(x['apgd-ce'].shape)
# plt.imshow(x['apgd-ce'][0].moveaxis(0,-1))
