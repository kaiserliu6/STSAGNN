import numpy as np
import torch
import torch.nn as nn

class Model_base(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        if config is not None:
            self.config = config
            self.task = config.task_name
            if self.task == 'OD':
                self.dim_in = config.num_nodes
                self.dim_out = config.num_nodes
            elif self.task == 'IO':
                if self.config.need_od:
                    self.dim_in = config.num_nodes
                else:
                    self.dim_in = 2
                self.dim_out = 2
            elif self.task == 'TF':
                self.dim_in = config.input_dim or 1
                self.dim_out = config.input_dim or 1
            self.norm = Norm(3, task=config.task_name)
            self.thre = config.metirc_threshold if not isinstance(config.metirc_threshold, list) else min(config.metirc_threshold)
        # 暂时添加，记得删除
        # self.dim_in = 1
        # self.dim_out = 1

    # def forward(self):
    #     pass
    """
    The model support 3 types of task for now, OD, IO, and TF
    'OD' is for Origin-Destination Flow Prediction
    'IO' is for In-flow and Out-flow Prediction, use a Dual Evaluator to predict in-flow and out-flow simultaneously
    'TF' is for normal Spatio-Temporal Forecating
    """
    def forward(self, *data):
        x = self.norm.norm(data[0])
        if data[1] is not None:
            p = data[1][..., [0]]
            output = self.model(x, p)
        else:
            output = self.model(x)
        output = self.norm.inorm(output)
        return output

    def od_forecast(self, *data):
        return self.forward(*data)

    def io_forecast(self, *data):
        if data[0].shape[-1] > 2:
            x = data[0]
            x = torch.stack([x.sum(-2), x.sum(-1)], -1)
            data = (x, *data[1:])
        return self.forward(*data)

    def tf_forecast(self, *data):
        return self.forward(*data)

    def calculate_loss(self, data):
        if not hasattr(self, 'loss'):
            self.loss = self.masked_mae
        x, y = self.get_data(data)
        output = self.forward(*x)
        loss = self.loss(output, y)
        return loss
        if self.task == 'OD':
            # assert self.config.input_dim > 2, 'This Dataset DO NOT Support OD-flow'
            output = self.od_forecast(*x)
            loss = self.loss(output, y)  # masked_mae_loss(y_pred, y_true)
        elif self.task == 'IO':
            output = self.io_forecast(*x)
            if y.shape[-1] > 2:
                y = torch.stack([y.sum(-2), y.sum(-1)], -1)
            loss = self.loss(output, y)  # masked_mae_loss(y_pred, y_true)
        elif self.task == 'TF':
            output = self.tf_forecast(*x)
            loss = self.loss(output, y)  # masked_mae_loss(y_pred, y_true)
        else:
            raise NotImplemented
        return loss

    def predict(self, data):
        x, y = self.get_data(data)
        output = self.forward(*x)
        return output
        if self.task == 'OD':
            output = self.od_forecast(*x)
        elif self.task == 'IO':
            output = self.io_forecast(*x)
        elif self.task == 'TF':
            output = self.tf_forecast(*x)
        else:
            raise NotImplemented
        return output

    #below are some loss func
    def masked_mse(self, x, y, mask = None):
        x = x.squeeze()
        y = y.squeeze()
        loss = (x - y).pow(2)
        mask = self.thre
        if mask is not None:
            m = (y > self.thre)
            m = m.float()
            m /= m.mean()
            loss = loss * m
        return loss.mean()

    def masked_mae(self, x, y, mask = None):
        x = x.squeeze()
        y = y.squeeze()
        loss = torch.abs(x - y)
        mask = self.thre
        if mask is not None:
            m = (y > self.thre)
            m = m.float()
            m /= m.mean()
            loss = loss * m
        return loss.mean()

    def masked_rmse(self, x, y, mask = None):
        return self.masked_mse(x, y, mask).sqrt()

    def last_info(self):
        pass

    def get_data(self, data):
        if self.config.need_time:
            x, y, xe, ye = data
            x, y, xe = x.cuda().to(torch.float32), y.cuda().to(torch.float32), xe.cuda().to(torch.int)
            return (x, xe), y
        else:
            if len(data) == 2:
                x, y = data
            else:
                x, y, *_ = data
            x, y = x.cuda().to(torch.float32), y.cuda().to(torch.float32)
            return (x, None), y

    def tsne_analyse(self, high_dim_tensor):
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=0)
        low_dim_data = tsne.fit_transform(high_dim_tensor.cpu().numpy())

        plt.scatter(low_dim_data[:, 0], low_dim_data[:, 1])
        plt.title('t-SNE of High-Dimensional Tensor')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.show()

class RNN_Model(Model_base):
    def __init__(self, config=None):
        super().__init__(config)
    """
    The main difference of this variant is it support RNN trainer, passing y to model
    """
    def forward(self, *data):
        x = self.norm.norm(data[0])
        if data[1] is not None:
            p = data[1][..., [0]]
            output = self.model(x, p)
        else:
            output = self.model(x)
        output = self.norm.inorm(output)
        return output

    def calculate_loss(self, data, idx=None):
        if not hasattr(self, 'loss'):
            self.loss = self.masked_mae
        x, y = self.get_data(data)
        if self.task == 'OD':
            x = (*x, y, idx)
            output = self.od_forecast(*x)
            loss = self.loss(output, y)  # masked_mae_loss(y_pred, y_true)
        elif self.task == 'IO':
            if y.shape[-1] > 2:
                y = torch.stack([y.sum(-2), y.sum(-1)], -1)
            x = (*x, y, idx)
            output = self.io_forecast(*x)
            loss = self.loss(output, y)  # masked_mae_loss(y_pred, y_true)
        elif self.task == 'TF':
            x = (*x, y, idx)
            output = self.tf_forecast(*x)
            loss = self.loss(output, y)  # masked_mae_loss(y_pred, y_true)
        else:
            raise NotImplemented
        return loss

class template(Model_base):
    def __init__(self, config, model):
        super().__init__()
        conf = {}
        config.update(conf, 5)
        self.task = config.task_name
        if self.task == 'OD':
            dim_in = config.num_nodes
            dim_out = config.num_nodes
        elif self.task == 'IO':
            dim_in = 2
            dim_out = 2
        elif self.task == 'TF':
            dim_in = config.input_dim or 1
            dim_out = config.input_dim or 1
        self.norm = Norm(3, task=config.task_name)
        self.thre = config.metirc_threshold if isinstance(config.metirc_threshold, int) else min(
            config.metirc_threshold)

    def forward(self, x, p=None):
        output = self.model(x, p)
        output = self.norm.inorm(output)
        return output


class GCN(nn.Module):
    def __init__(self, g, dim_list, norm = None, **kwargs):
        super().__init__()

        if norm is None:
            self.g = self.norm(g).cuda()
        else:
            if isinstance(norm, int):
                self.g = self.norm(g, norm).cuda()
            else:
                self.g = norm(g).cuda()
        # self.g = self.norm(g) if norm is None else norm(g, kwargs)
        self.layer = nn.ModuleList([nn.Linear(dim_list[i], dim_list[i+1]) for i in range(len(dim_list) - 1)])

    def forward(self, x):
        # expect x to be B,T,N,D
        for lin in self.layer:
            x = self.g @ x
            x = lin(x)
            x = torch.sigmoid(x)
        return x

    def norm(self, g, type=2):
        if type == 1:
            d = 1 / g.sum(dim=0, keepdim=True)
            g = g * d
        elif type == 2:
            d = 1 / g.sum(dim=0, keepdim=True).sqrt()
            g = d * g * d.T
        return g

class Norm(nn.Module):
    def __init__(self, type, instant=False, dim=0, task=None):
        super().__init__()

        method_list = ['none', 'min-max01', 'min-max11', 'z-score']
        if isinstance(type, int):
            self.type = method_list[type]
        else:
            self.type = type.lower()
            assert self.type in method_list, f'only support {method_list} normalization, use number instead is OK'
        self.instant = instant
        self.mode = task or 'OD'

    def norm(self, x):
        if self.instant:
            if self.type == 'z-score':
                self.mean = x.mean(0)
                self.std = x.std(0)
                return (x - self.mean) / self.std
            elif self.type == 'min-max01':
                self.max = x.max()
                self.min = x.min()
                return (x - self.min) / self.max
            elif self.type == 'min-max11':
                self.max = x.max()
                self.min = x.min()
                return ((x - self.min) / self.max) * 2 - 1
            else:
                return x
        else:
            if self.type == 'z-score':
                return (x - self.mean) / self.std
            elif self.type == 'min-max01':
                return (x - self.min) / self.max
            elif self.type == 'min-max11':
                return ((x - self.min) / self.max) * 2 - 1
            else:
                return x

    def inorm(self, x):
        if self.type == 'z-score':
            return x * self.std + self.mean
        elif self.type == 'min-max01':
            return x * self.max + self.min
        elif self.type == 'min-max11':
            return (x + 1) / 2 * self.max + self.min
        else:
            return x

    def initialize(self, data):
        if not self.instant:
            if self.mode == 'IO':
                if data.shape[-1] > 2:
                    data_dp = np.concatenate([data.sum(-2, keepdims=True), data.sum(-1, keepdims=True).transpose(0, 2, 1)], -1) #T, N, 2
                else:
                    data_dp = data
                if self.type == 'z-score':
                    self.mean = torch.tensor(data_dp.mean(), dtype=torch.float32).cuda()
                    self.std = torch.tensor(data_dp.std(), dtype=torch.float32).cuda()
                elif self.type == 'none':
                    pass
                else:
                    self.max = torch.tensor(data_dp.max(), dtype=torch.float32).cuda()
                    self.min = torch.tensor(data_dp.min(), dtype=torch.float32).cuda()
            else:
                if self.type == 'z-score':
                    self.mean = torch.tensor(data.mean(), dtype=torch.float32).cuda()
                    self.std = torch.tensor(data.std(), dtype=torch.float32).cuda()
                elif self.type == 'none':
                    pass
                else:
                    self.max = torch.tensor(data.max(), dtype=torch.float32).cuda()
                    self.min = torch.tensor(data.min(), dtype=torch.float32).cuda()
        return