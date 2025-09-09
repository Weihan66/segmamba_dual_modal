# utils.py
import torch
import numpy as np
import os
from pathlib import Path

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)

def load_checkpoint(model, optimizer, path, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optim_state"])
    epoch = checkpoint.get("epoch", 0)
    return model, optimizer, epoch

def dice_coeff(pred, target, eps=1e-6):
    # pred, target: torch tensors in [B,1,D,H,W]
    pred = (pred > 0.5).float()
    target = target.float()
    intersection = (pred * target).sum(dim=[1,2,3,4])
    union = pred.sum(dim=[1,2,3,4]) + target.sum(dim=[1,2,3,4])
    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean().item()

def soft_dice_loss(pred, target, eps=1e-6):
    # pred: logits or probs [B,1,D,H,W]
    pred = pred.view(pred.shape[0], -1)
    target = target.view(target.shape[0], -1)
    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)
    loss = 1 - (2 * intersection + eps) / (union + eps)
    return loss.mean()

def bce_dice_loss(probs, target):
    bce = torch.nn.functional.binary_cross_entropy(probs, target)
    sdice = soft_dice_loss(probs, target)
    return bce + sdice
