import numpy as np
from logging import getLogger
from torch.utils.data import DataLoader, Dataset

class Data_OD():
    def __init__(self):
        self.logger = getLogger()
        self.process_od = False

    def load_data(self):
        #load data
        data = np.load(self.config.data_path + self.file_name + 'data.npy')

        #load temporal information (time-of-day, day-of-week)
        extern = np.load(self.config.data_path + self.file_name + 'time.npy')

        #reset the start and end index if needed
        if data.shape[0] > self.config.total_size:
            if self.config.start_date + self.config.total_size > data.shape[0]:
                raise ValueError
            data = data[self.config.start_date:self.config.total_size]
            extern = extern[self.config.start_date:self.config.total_size]

        #if task_mode is IO and need_od is False, then process the data to io data
        if self.process_od:
            data = np.stack([data.sum(-2), data.sum(-1)], -1)

        return data, extern

    def split_tvt(self, data, extern=None):
        total_time = data.shape[0]
        #split dataset based on previous setting
        if isinstance(self.config.train_size, int):
            num_train = self.config.train_size
            num_test = self.config.test_size
        elif isinstance(self.config.train_size, float):
            num_train = round(total_time * self.config.train_size)
            num_test = round(total_time * self.config.test_size)
        else:
            raise "train_size only support int or float"

        # period processing
        pw = self.config.period_window or 0
        if self.config.use_period:
            t = data[: num_train]
            v = data[num_train: -num_test]
            ts = data[-num_test - self.config.num_times*pw:]
        else:
            t = data[: num_train]
            v = data[num_train: -num_test]
            ts = data[-num_test:]

        #extern data
        if extern is None:
            te = ve = tse = None
        else:
            te = extern[: num_train]
            ve = extern[num_train: -num_test]
            tse = extern[-num_test - self.config.num_times*pw:]

        return [t, te], [v, ve], [ts, tse]

    def get_dataloader(self, norm=None):
        data = self.load_data()
        train, val, test = self.split_tvt(*data)
        if norm is not None:
            norm.initialize(train[0])
        train = self.create_dataloader(*train, True)
        val = self.create_dataloader(*val)
        test = self.create_dataloader(*test)
        return train, val, test

    def create_dataloader(self, data, extern, shuffle=False):
        period_len = self.config.period_window or 7
        period_size = self.config.period_len or [0, 1]
        data = Data4dl(self.config.input_window, self.config.output_window, data, extern, self.config.use_period, period_len, period_size, self.config.need_time, od_flag=self.process_od)
        dl = DataLoader(data, self.config.batch_size, shuffle=shuffle)
        return dl

    def get_whole_dataloader(self):
        data = self.load_data()
        dl = self.create_dataloader(*data)
        return dl

class Data4dl(Dataset):
    def __init__(self, input_len, output_len, data, extern=None, use_period=False, period_len=7, period_size=(0, 1), use_time=False, od_flag=False):
        super().__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.data = data
        if data.ndim == 2:
            self.data = data[:, :, np.newaxis]
        if self.data.shape[-1] > 2:
            self.data_y = np.stack([data.sum(-2), data.sum(-1)], -1)
        else:
            self.data_y = self.data
        self.extern = extern
        self.use_period = use_period
        self.use_time = use_time
        self.p_len = period_len
        self.p_size = period_size

        if self.use_period:
            self.process_period_data()
            self.len = self.data.shape[0] - self.output_len - max(self.input_len, -self.p_size[1]) + 1 - period_len * 48
        else:
            self.len = self.data.shape[0] - self.output_len - self.input_len + 1

    def __getitem__(self, item):
        if self.use_period:
            a = item + self.input_len
            b = item + self.p_len * 48
            c = b + self.input_len
            rd = [self.data[b: c], self.data[c: c + self.output_len], self.p_data[item: b: 48]]
            if self.use_time:
                rd.extend([self.extern[b: c], self.extern[c: c + self.output_len], self.extern[item: b: 48]])
            return tuple(rd)
        else:
            if self.extern is None:
                return self.data[item: item + self.input_len], self.data_y[item + self.input_len: item + self.input_len + self.output_len]
            else:
                return self.data[item: item + self.input_len], self.data_y[item + self.input_len: item + self.input_len + self.output_len]\
                    , self.extern[item: item + self.input_len], self.extern[item + self.input_len: item + self.input_len + self.output_len]

    def __len__(self):
        return self.len

    def process_period_data(self):
        s_p = max(self.input_len, -self.p_size[1]) + self.p_size[0]
        p_s = self.p_size[1] - self.p_size[0]
        self.p_data = np.stack([self.data[s_p + i: -p_s + i - 47] for i in range(p_s)], 0).transpose((1, 0, 2, 3)).squeeze()