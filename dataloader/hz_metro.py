import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import pickle

from dataloader.dataloader import Data_OD

class Data_hz_metro(Data_OD):
    def __init__(self, config, dataset_name):
        super().__init__()
        conf = {
            'num_nodes': 80,
            'input_dim': 2,
            'total_size': 25,
            'train_size': 16,
            'test_size': 7,
            'num_times': 73
        }
        config.update(conf, 7)
        self.config = config
        self.file_name = 'HZMetro/'

    def load_data(self, mode):
        file_name = self.config.data_path + self.file_name + mode + '.pkl'
        with open(file_name, 'rb') as f:
            data = pickle.load(f)
        return data

    def get_dataloader(self, norm=None):
        data = np.load(self.config.data_path + self.file_name + 'data.npy')
        time = np.load(self.config.data_path + self.file_name + 'time.npy')
        train = Data4dl(data[:self.config.train_size], time[:self.config.train_size], self.config.input_window,
                       self.config.output_window, self.config.need_time)
        val = Data4dl(data[self.config.train_size: -self.config.test_size],
                     time[self.config.train_size: -self.config.test_size], self.config.input_window,
                     self.config.output_window, self.config.need_time)
        test = Data4dl(data[-self.config.test_size:], time[-self.config.test_size:], self.config.input_window,
                      self.config.output_window, self.config.need_time)

        if norm is not None:
            norm.initialize(train.x)
        train = DataLoader(train, self.config.batch_size, shuffle=True)
        val = DataLoader(val, self.config.batch_size)
        test = DataLoader(test, self.config.batch_size)
        return train, val, test

class Data4dl(Dataset):
    def __init__(self, data, time, win_in, win_out, need_time):
        super().__init__()
        temp = torch.tensor(data).unfold(1, win_in + win_out, 1).flatten(0, 1).permute(0, 3, 1, 2) #D*(73-win_in-win_out+1), win_in+wi_out, N, 2
        self.x = temp[:, :win_in]
        self.y = temp[:, -win_out:]
        temp = torch.tensor(time).unfold(1, win_in + win_out, 1).flatten(0, 1).unsqueeze(-1)  # D*(73-win_in-win_out+1), win_in+wi_out
        self.xtime = temp[:, :win_in]
        self.ytime = temp[:, -win_out:]
        self.need_time = need_time

    def __getitem__(self, item):
        if self.need_time:
            return self.x[item], self.y[item], self.xtime[item], self.ytime[item]
        else:
            return self.x[item], self.y[item]

    def __len__(self):
        return self.x.shape[0]