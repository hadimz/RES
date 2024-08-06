import matplotlib.pyplot as plt
import torch
from tqdm import *

def class_activation_map(network, X, labels, cuda=False, size=224):
    """
    Creates the class activation maps for every layer of the network given an
    input image

    :param network: The network instance
    :param X: The data to evaluate
    :param cuda: Wheter to use cuda
    :param size: The size of the input images
    """

    grad_eye = torch.eye(2, requires_grad=True)
    if cuda:
        grad_eye = grad_eye.cuda()

    CAM, predictions = [], []
    activations, gradients = [], []
    for id, _X in enumerate(X):
        pred = network.forward(_X.unsqueeze(0))
        local_maps_class = []
        i = labels[id]
        # for i in range(pred.shape[0]):
        network.zero_grad()
        pred.backward(grad_eye[i], retain_graph=True)
        local_maps = []
        for key in network.outputs.keys():

            A_k = network.outputs[key]
            grad = network.grads[key]

            alpha_k = grad.mean(axis=(2, 3))

            gradients.append(grad.detach().cpu())

            local_map = torch.sum(A_k * alpha_k.unsqueeze(-1).unsqueeze(-1), axis=1)
            

            if key in ["2"]:
                local_map = torch.maximum(local_map, torch.tensor(0.))
            minimum, _ = local_map.min(dim=2)
            minimum, _ = minimum.min(dim=1)
            maximum, _ = local_map.max(dim=2)
            maximum, _ = maximum.max(dim=1)
            # maximum[maximum == 0] = 1
            local_map = (local_map - minimum.view(-1,1,1)) / (maximum - minimum + 1e-12).view(-1,1,1)

            # ********************************
            A_k_log = torch.sum(A_k.clone(), axis=1)
            if key in ["2"]:
                A_k_log = torch.maximum(A_k_log, torch.tensor(0.))
            minimum_k, _ = A_k_log.min(dim=2)
            minimum_k, _ = minimum_k.min(dim=1)
            maximum_k, _ = A_k_log.max(dim=2)
            maximum_k, _ = maximum_k.max(dim=1)
            # maximum[maximum == 0] = 1
            A_k_log = (A_k_log - minimum_k.view(-1,1,1)) / (maximum_k - minimum_k + 1e-12).view(-1,1,1)
            A_k_log = torch.nn.functional.interpolate(A_k_log.unsqueeze(0), size=size, mode='bilinear')
            activations.append(torch.sum(A_k_log, axis=1).detach().cpu())
            # ********************************
            

            s = local_map.shape
            upsampled = torch.nn.functional.interpolate(local_map.unsqueeze(0), size=size, mode='bilinear')
            local_maps.append(upsampled)
                
            local_maps = torch.stack(local_maps).moveaxis(0,1)
            local_maps_class.append(local_maps.squeeze())                
            

        CAM.append(torch.stack(local_maps_class ,dim=0))
        predictions.append((torch.sigmoid(pred) > 0.5))

    return torch.stack(CAM, dim=0), torch.stack(predictions, dim=0), activations, gradients
