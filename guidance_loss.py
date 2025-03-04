import torch
import torchvision.transforms.functional as ff
from torch.nn import KLDivLoss

def exp_loss(binary_masks, saliency_maps):
    """ Guidance/Explanation loss calculated based on the normalized agreement between the
     saliency maps and binary masks """
    saliency_maps_resized = ff.resize(saliency_maps, binary_masks.shape[2:])
    loss = torch.sum(binary_masks*saliency_maps_resized)/torch.sum(saliency_maps_resized)
    return loss

import torch

def kl_divergence(P, Q):
    """
    Calculate the Kullback-Leibler (KL) Divergence between two probability distributions P and Q.
    
    Args:
        P (torch.Tensor): The first probability distribution tensor.
        Q (torch.Tensor): The second probability distribution tensor.
        
    Returns:
        torch.Tensor: The KL-Divergence between P and Q.
    """
    # Ensure the tensors are of the same shape and type
    if P.shape != Q.shape:
        raise ValueError("Tensors P and Q must have the same shape.")
    if P.dtype != Q.dtype:
        print(P.dtype, Q.dtype)
        raise ValueError("Tensors P and Q must have the same data type.")
    
    # Avoid division by zero or log(0) which is undefined in math
    epsilon = 1e-8
    
    # Compute the KL-Divergence
    kl_div = P * (torch.log(P + epsilon) - torch.log(Q + epsilon))
    kl_div = kl_div.mean()
    
    return kl_div