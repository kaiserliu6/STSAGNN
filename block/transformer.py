import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
import copy

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_head=8, d_ff=None, attention=None, dropout=0.1, activation="relu", dim=-2):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention or nn.MultiheadAttention(d_model, num_head, dropout, batch_first=True)
        # self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        # self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ffn = FFN(d_model, d_ff, dropout, activation, True)
        self.dim = dim

    def forward(self, x):
        x = x.transpose(-2, self.dim)
        shape = x.shape
        x = x.reshape(-1, shape[-2], shape[-1]) # assert attn input to be 3-d tensor B,N,D, origin could be B,T,N,D
        x_ = self.norm1(x)
        new_x, attn = self.attention(x_, x_, x_)
        # shape[-1] = new_x.shape[-1]
        # new_x = new_x.reshape(*shape)
        x = x + self.dropout(new_x)
        return self.ffn(x).unflatten(0, shape[:-2]).transpose(-2, self.dim)


class Encoder(nn.Module):
    def __init__(self, attn_layers, num_layer=1, conv_layers=None, norm_layer=None):
        super().__init__()
        self.attn_layers = nn.ModuleList([copy.deepcopy(attn_layers) for i in range(num_layer)])
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x):
        # x [B, L, D]
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                x = attn_layer(x)
                x = conv_layer(x)
            x = self.attn_layers[-1](x)
        else:
            for attn_layer in self.attn_layers:
                x = attn_layer(x)

        if self.norm is not None:
            x = self.norm(x)

        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_head=8, d_ff=None, attention=None, dropout=0.1, activation="relu", dim=-2):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention or nn.MultiheadAttention(d_model, num_head, dropout, batch_first=True)
        # self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        # self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ffn = FFN(d_model, d_ff, dropout, activation, True)
        self.dim = dim

    def forward(self, trg, src):
        """
        :param src: b,t,n,d
        :param trg: b,h,n,d
        :return: b,h,n,d
        """
        if src.ndim < trg.ndim:
            src = src.flatten(0, -2).unsqueeze(-2)
        else:
            src = src.transpose(-2, self.dim)
            src = src.flatten(0, -3)
        trg = trg.transpose(-2, self.dim)
        shape = trg.shape
        trg = trg.flatten(0, -3)
        new_x, attn = self.attention(self.norm(trg), self.norm(src), self.norm(src))
        out = trg + self.dropout(new_x)
        return self.ffn(out).unflatten(0, shape[:-2]).transpose(-2, self.dim)

class Decoder(nn.Module):
    def __init__(self, attn_layers, num_layer=1, conv_layers=None, norm_layer=None):
        super().__init__()
        self.attn_layers = nn.ModuleList([copy.deepcopy(attn_layers) for i in range(num_layer)])
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, trg, src):
        """
        this decoder DO NOT contain self attention
        :param src: b,t,n,d
        :param trg: b,h,n,d
        :return: b,h,n,d
        """
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                trg = attn_layer(trg, src)
                trg = conv_layer(trg)
            trg = self.attn_layers[-1](trg)
        else:
            for attn_layer in self.attn_layers:
                trg = attn_layer(trg, src)

        if self.norm is not None:
            trg = self.norm(trg)

        return trg
    
class FFN(nn.Module):
    def __init__(self, emb_dim, hid_dim=None, dropout=0.1, activation="gelu", norm_first=False):
        super().__init__()
        hid_dim = hid_dim or emb_dim * 4
        self.lin1 = nn.Linear(emb_dim, hid_dim)
        self.lin2 = nn.Linear(hid_dim, emb_dim)
        self.norm = nn.LayerNorm(emb_dim)
        self.dp = nn.Dropout(dropout)
        self.af = F.relu if activation == "relu" else F.gelu
        self.nf = norm_first

    def forward(self, x):
        if self.nf:
            x_ = self.dp(self.af(self.lin1(self.norm(x))))
            x_ = self.dp(self.lin2(x_))
            return x + x_
        else:
            x_ = self.dp(self.af(self.lin1(x)))
            x_ = self.dp(self.lin2(x_))
            return self.norm(x + x_)

class FullAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v):
        b, t, n, d = q.shape
