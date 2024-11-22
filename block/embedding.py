import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TemporalEmbedding(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        # self.emb = nn.Embedding(55, emb_dim)
        self.emb1 = nn.Embedding(48, emb_dim)
        self.emb2 = nn.Embedding(7, emb_dim)

    def forward(self, x):
        emb1 = self.emb1(x[...,1])
        # emb2 = self.emb2(x[...,0])
        emb2 = 0
        return emb1 + emb2

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class Patch_Embedding(nn.Module):
    def __init__(self, patch_len, stride, emb_dim, padding=True):
        super().__init__()

        self.patch_len = patch_len
        self.stride = stride
        self.emb_dim = emb_dim
        self.padding = padding
        self.lin = nn.Linear(patch_len, emb_dim)

    def forward(self, x):
        if self.padding:
            x = F.pad(x, (0, self.stride), 'replicate')
        x = x.unfold(-1, self.patch_len, self.stride)
        x = self.lin(x)
        return x