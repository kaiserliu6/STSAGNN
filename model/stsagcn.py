import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from model.model import Model_base, Norm
from block import FFN
from block import GRU, GRU_Decoder

class Model(Model_base):
    def __init__(self, config, data):
        conf = {
            'schdeuler': 'multisteplr',
            'lr_decay_ratio': 0.1,
            'milestones': [50, 150],
            'learning_rate': 0.01,
            'patch_len': 4,
            'stride': 4,
            'emb_dim': 64,
            'epoch': 500,
            'batch_size': 256,
            'need_time': True,
            'topk': 2,
            'need_od': True,
        }
        if config.num_nodes > 310:
            conf['batch_size'] = 128
        config.update(conf, 5)
        super().__init__(config)
        num_times = self.config.num_times or 48
        self.model = STSAGCN(config.num_nodes, self.dim_in, config.emb_dim, self.dim_out, config.patch_len, config.stride,
                             config.input_window, config.output_window, config.topk, num_times)
        self.norm = Norm(3)

    def forward(self, *data):
        x = self.norm.norm(data[0])
        p = data[1][..., [0]]
        output = self.model(x, p)
        output = self.norm.inorm(output)
        return output

    def get_data(self, data):
        x, y, xe, ye = data
        x, y, xe, ye = x.cuda().to(torch.float32), y.cuda().to(torch.float32), xe.cuda().to(torch.int), ye.cuda().to(torch.int)
        return (x, torch.cat((xe, ye), 1)), y

class STSAGCN(nn.Module):
    def __init__(self, num_nodes, dim_in, dim_hid, dim_out, patch_len, stride, win_in, win_out, topk, num_times):
        super().__init__()

        self.stide = stride
        self.patch_len = patch_len
        self.win_in = win_in
        self.win_out = win_out
        self.dim_out = dim_out
        if dim_in > 2:
            self.emb = PF_Emb(patch_len, stride, num_nodes, dim_hid, win_in)
        else:
            self.emb = DP_Emb(patch_len, stride, dim_hid, dim_in)
        self.st_emb = ST_Embedding(dim_hid, num_times, num_nodes)
        self.encoder = SGC(dim_hid, topk)
        self.gru1 = GRU(dim_hid, dim_hid, dim=1)
        self.gru2 = GRU_Decoder(dim_hid, dim_hid, dim_out, dim=1, add_extern_factor=True)
        self.ne = nn.Embedding(num_nodes, 10)

    def forward(self, x, p):
        """
        :param x: input od od-flow our in/out-flow, B,T,N,N or B,T,N,2
        :param p: time-of-the-day of input and output time B,(T+H)
        :return: output od od-flow our in/out-flow, B,T,N,N or B,T,N,2
        """
        # patch OD embedding
        emb = self.emb(x)
        emb = self.gru1(emb)

        # spatio-temporal embedding of input
        q = torch.cat((p[:, self.patch_len - 1:self.win_in:self.stide], p[:, [self.win_in]]), 1)
        p_emb = self.st_emb(q)

        # STSAGCN layer
        enc = emb
        stg = torch.einsum('bnd,btmd->bntm', p_emb[:, -1], p_emb)
        enc = self.encoder(enc, stg)

        # spatio-temporal embedding of output
        p_emb = self.st_emb(p[:, self.win_in:])

        # spatio-temporal enhanced gru decoder
        out = self.gru2(p_emb, enc)

        return out

class SGC(nn.Module):
    def __init__(self, emb_dim, topk):
        super().__init__()
        self.q = nn.Linear(emb_dim, emb_dim)
        self.k = nn.Linear(emb_dim, emb_dim)
        self.norm = nn.LayerNorm(emb_dim)
        self.scale = 1 / math.sqrt(emb_dim)
        self.dp = nn.Dropout(0.1)
        self.topk = topk
        self.ffn = FFN(emb_dim)

    def forward(self, x, stg):
        b, t, n, d = x.shape
        x_ = self.norm(x)
        y_ = x_[:, -1]
        dg = torch.einsum('bnd,btmd->bntm', self.q(y_), self.k(x_)) * self.scale
        sag = torch.sigmoid(dg) * stg * self.scale
        sag = sag.reshape(b, n, t * n)

        if self.topk > 0:
            if self.topk < 5:
                self.topk *= n
            top_values, top_indices = torch.topk(sag, self.topk, dim=-1)
            sag = torch.where(sag > top_values[..., [-1]], sag, float('-inf')) # This is faster than masked_select

        sag = self.dp(torch.softmax(sag, -1)).reshape(b, n, t, n)
        o = torch.einsum('bntm,btmd->bnd', sag, x_)
        return self.ffn(o + y_)

class ST_Embedding(nn.Module):
    def __init__(self, emb_dim, num_times, num_nodes):
        super().__init__()

        self.emb_s = nn.Embedding(num_times, num_nodes * emb_dim)
        self.emb_t = nn.Embedding(num_times, emb_dim)
        self.n = num_nodes
        self.d = emb_dim

    def forward(self, p):
        emb_s = self.emb_s(p).unflatten(-1, (self.n, self.d)).squeeze()
        emb_t = self.emb_t(p)
        return emb_s + emb_t

class PF_Emb(nn.Module):
    def __init__(self, patch_len, stride, num_nodes, emb_dim, time_len):
        super().__init__()

        self.patch = patch_len
        self.stride = stride
        self.layer_o = PF_Embedding_sub(patch_len, num_nodes, emb_dim // 2)
        self.layer_d = PF_Embedding_sub(patch_len, num_nodes, emb_dim // 2)

    def forward(self, x):
        x = F.pad(x, (0, 0, 0, 0, 0, self.stride), 'replicate')
        x = x.unfold(1, self.patch, self.stride).permute(0, 1, 4, 2, 3)
        x1 = self.layer_o(x)
        x2 = self.layer_d(x.transpose(-1, -2))
        x = torch.cat([x1, x2], -1)  # B,P,N,D
        return x

class PF_Embedding_sub(nn.Module):
    def __init__(self, patch_len, num_nodes, emb_dim):
        super().__init__()

        self.lin1 = nn.Linear(num_nodes, emb_dim)
        self.lin2 = nn.Linear(patch_len, 1)
        self.lin3 = nn.Linear(num_nodes, emb_dim)
        self.af = nn.Sigmoid()
        self.dp = nn.Dropout(0.1)
        self.gru = GRU(num_nodes, emb_dim, dim=2)

    def forward(self, x):
        b, p, l, n, _ = x.size()
        x1 = self.lin1(x)
        x2 = self.af(self.lin3(x))
        x = x1 * x2
        x = x.permute(0, 1, 3, 4, 2)
        x1 = self.lin2(x).squeeze(-1)
        return self.dp(x1)

class DP_Emb(nn.Module):
    def __init__(self, patch_len, stride, emb_dim, dim_in):
        super().__init__()

        self.patch = patch_len
        self.stride = stride
        self.lin = PF_Embedding_sub(patch_len, dim_in, emb_dim)
        self.af = nn.Sigmoid()
        self.dp = nn.Dropout(0.1)

    def forward(self, x):
        x = F.pad(x, (0, 0, 0, 0, 0, self.stride), 'replicate')
        x = x.unfold(1, self.patch, self.stride).permute(0, 1, 4, 2, 3)
        return self.lin(x)