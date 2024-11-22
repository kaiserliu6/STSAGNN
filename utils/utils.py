import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class Unfold(nn.Module):
    def __init__(self, lenth, stride, dialation=1, padding=0):
        super().__init__()

        self.lenth = lenth
        self.stride = stride
        self.dialation = dialation
        self.padding = padding

    def forward(self, x):
        if self.padding > 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.padding), 'replicate')
        if self.dialation > 1:
            x = x.unfold(1, (self.lenth - 1) * self.dialation, self.stride)
            x = x[:, : self.lenth * self.dialation: self.dialation]
        else:
            x = x.unfold(1, self.lenth, self.stride)
        return x

def ensure_dir(dir_path):
    """Make sure the directory exists, if it does not exist, create it.

    Args:
        dir_path (str): directory path
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path