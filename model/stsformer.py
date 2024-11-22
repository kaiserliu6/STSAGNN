import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from model.model import Model_base, Norm
from block import FFN, Encoder, EncoderLayer, Decoder, DecoderLayer
from block import GRU

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
        self.model = STSformer(config.num_nodes, self.dim_in, config.emb_dim, self.dim_out, config.patch_len, config.stride,
                             config.input_window, config.output_window, config.topk, num_times)
        self.norm = Norm(3)

    def forward(self, *data):
        x = self.norm.norm(data[0])
        p = data[1][..., [0]]
        output = self.model(x, p)
        output = self.norm.inorm(output)
        return output

    def io_forecast(self, *data):
        return self.forward(*data)

    def get_data(self, data):
        x, y, xe, ye = data
        x, y, xe, ye = x.cuda().to(torch.float32), y.cuda().to(torch.float32), xe.cuda().to(torch.int), ye.cuda().to(torch.int)
        return (x, torch.cat((xe, ye), 1)), y

class STSformer(nn.Module):
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
        self.t_emb = nn.Embedding(num_times, dim_hid)
        self.s_emb = nn.Embedding(num_times, dim_hid * num_nodes)
        self.encoder = nn.ModuleList([SGC(dim_hid, topk) for i in range(1)])
        self.tsm1 = Encoder(EncoderLayer(dim_hid, dim=1), num_layer=1)
        self.tsm2 = Decoder(DecoderLayer(dim_hid, dim=1), num_layer=1)
        self.ape1 = nn.Parameter(torch.randn((win_in - patch_len) // stride + 2, num_nodes, dim_hid))
        self.ape2 = nn.Parameter(torch.randn(win_out, num_nodes, dim_hid))
        self.pred = nn.Linear(dim_hid, dim_out)


    def forward(self, x, p):
        """
        :param x: input od od-flow our in/out-flow, B,T,N,N or B,T,N,2
        :param p: time-of-the-day of input and output time B,(T+H)
        :return: output od od-flow our in/out-flow, B,T,N,N or B,T,N,2
        """
        # gated patch embedding
        emb = self.emb(x)

        # spatio-temporal embedding of input
        b, t, n, d = emb.shape
        q = torch.cat((p[:, self.patch_len - 1:self.win_in:self.stide], p[:, [self.win_in]]), 1)
        p_emb = self.st_emb(q)

        # transformer encoder
        emb = self.tsm1(emb + self.ape1 + p_emb)

        # STSAGCN layer
        enc = emb
        stg = torch.einsum('btnd,bkmd->btnkm', p_emb, p_emb)
        for m in self.encoder:
            enc = m(enc, stg)

        # spatio-temporal embedding of output
        p_emb = self.st_emb(p[:, self.win_in:])

        # spatio-temporal enhanced gru decoder
        out = self.tsm2(p_emb, enc[:, [-1]])
        out = self.pred(out)

        return out

class SGC(nn.Module):
    def __init__(self, emb_dim, topk, q_dim=None):
        super().__init__()
        qdim = q_dim or emb_dim
        self.q = nn.Linear(qdim, emb_dim)
        self.k = nn.Linear(emb_dim, emb_dim)
        self.v = nn.Linear(emb_dim, emb_dim)
        self.norm = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(qdim)
        self.scale = 1 / math.sqrt(emb_dim)
        self.dp = nn.Dropout(0.1)
        self.topk = topk
        self.ffn = FFN(emb_dim)

    def forward(self, x, stg, trg=None):
        b, t, n, d = x.shape
        x_ = self.norm(x)
        if trg is not None:
            y_ = self.norm2(trg)
            h = y_.shape[1]
        else:
            y_ = x_
            h = t
        dg = torch.einsum('btnd,bkmd->btnkm', self.q(y_), self.k(x_)) * self.scale
        stg = (torch.sigmoid(dg)) * stg
        stg = stg.reshape(b, h * n, t * n)
        if self.topk > 0:
            if self.topk < 5:
                self.topk *= n
            top_values, top_indices = torch.topk(stg, self.topk, dim=-1)
            stg = torch.where(stg > top_values[..., [-1]], stg, float('-inf'))
        stg = self.dp(torch.softmax(stg, -1)).reshape(b, h, n, t, n)
        o = torch.einsum('btnkm,bkmd->btnd', stg, x_)
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

class OD_pred(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()

        self.lin1 = nn.Linear(emb_dim, 1)
        self.lin2 = nn.Linear(emb_dim, 1)

    def forward(self, x):
        o = self.lin1(x)
        d = self.lin2(x)
        return o + d.transpose(-1, -2)