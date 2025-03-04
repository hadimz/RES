import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import matplotlib.pyplot as plt

def compute_gradients(model, x, class_id=None, train=True):
    # Enable gradient tracking
    if not x.requires_grad:
        x.requires_grad = True
    
    # Calculate model outputs
    output = model(x).sum(dim=0, keepdims=True)

    # Call autograd directly to calculate the gradients and contstruct the gradient computation graph
    if train:
        grads = torch.autograd.grad(output[:, class_id], x, create_graph=True)
    else:
        grads = torch.autograd.grad(output[:, class_id], x)

    return grads[0]

def integrated_gradients(model, input, class_id=None, device='cuda:0', baseline=None, steps=5, train=True):
    # Determine the targets
    if class_id is None:
        class_id = torch.argmax(model(input.unsqueeze(0).to(device)).softmax(-1), keepdims=True)
    # Define the baselines
    if baseline is None:
        baseline = torch.zeros([steps, input.shape[0], input.shape[1], input.shape[2]]).to(device)
    
    # Define the intermediary samples
    x_int = torch.linspace(0, 1, steps).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(device)
    
    # Construct the batch of intermediary samples used for integrating gradients
    temp_batch = x_int*(input-baseline)
    temp_batch += baseline
    
    # Calculate gradients
    integrated_grads = compute_gradients(model, temp_batch, class_id=class_id, train=train)

    # Sum the gradients proportionally to their distance from the sample
    integrated_grads = (integrated_grads*x_int).sum(dim=0)/steps

    # Normalize the explanations to [0, 1] range (not part of the original IG)
    integrated_grads[integrated_grads<0.] = 0. # Clamp instead
    integrated_grads = integrated_grads/integrated_grads.view(-1).max()

    return integrated_grads.sum(dim=0), temp_batch
