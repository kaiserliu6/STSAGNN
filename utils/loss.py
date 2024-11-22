import torch
import numpy as np

def masked_mae_torch(preds, labels, threshold):
    if threshold is None:
        mask = 1
    else:
        mask = labels > threshold
        mask = mask.float()
        mask /= mask.mean()
    loss = torch.abs(preds - labels)
    loss = loss * mask
    return loss.mean()

def masked_mape_torch(preds, labels, threshold):
    if threshold is None:
        mask = labels > 0
    else:
        mask = labels > threshold
    mask = mask.float()
    mask /= mask.mean()
    loss = torch.abs(preds - labels) / labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return loss.mean()

def masked_smape_torch(preds, labels, threshold):
    if threshold is None:
        mask = labels > 0
    else:
        mask = labels > threshold
    mask = mask.float()
    mask /= mask.mean()
    loss = torch.abs(preds - labels) / (preds + labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return loss.mean()

def masked_mse_torch(preds, labels, threshold):
    if threshold is None:
        mask = 1
    else:
        mask = labels > threshold
        mask = mask.float()
        mask /= mask.mean()
    loss = torch.pow(preds - labels, 2)
    loss = loss * mask
    return loss.mean()

def masked_rmse_torch(preds, labels, threshold):
    return masked_mse_torch(preds, labels, threshold).sqrt()