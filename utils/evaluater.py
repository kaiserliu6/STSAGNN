from logging import getLogger

import numpy as np

from utils.loss import *


class Evaluator():
    def __init__(self, metric_list, metric_len, threshold, mode='average', valid=False):
        self.metrics = metric_list
        self.len = metric_len
        self.allowed_metrics = ['MAE', 'MSE', 'RMSE', 'MAPE', 'SMAPE']
        self.mode = mode
        self.valid = valid
        self.result = {}
        self._logger = getLogger()
        self.threshold = threshold
        self._check_config()
        self.initialize_metric()

        self.truth = []
        self.preds = []

    def _check_config(self):
        if not isinstance(self.metrics, list):
            raise TypeError('Metric type is not list')
        for metric in self.metrics:
            if 'mask' in metric:
                raise ValueError(f'the "mask" in metric {metric} is not needed and all metric is masked')
            if metric not in self.allowed_metrics:
                raise ValueError(f'the metric {metric} is not supported yet, only support {self.allowed_metrics}')
        if isinstance(self.len, int):
            self.len = [self.len]
        if isinstance(self.threshold, int) or isinstance(self.threshold, float):
            self.threshold = [self.threshold]
        self.len.sort()
        if self.threshold is not None:
            self.threshold.sort()
        if self.valid:
            self.len = [self.len[-1]]
            if self.threshold is not None:
                self.threshold = [self.threshold[0]]

    def collect(self, y_pred, y_true):
        """
        收集每一轮的预测值
        Args:
             'y_true': B, T, N, D or B, T, N, N
             'y_pred': B, T, N, D
        """
        assert y_pred.ndim == 4, 'Only support shape as B, T, N, D, unsqueeze first if D=1 and shape as B, T, N'
        if y_pred.shape[-1] != y_true.shape[-1]:
            y_true = torch.stack([y_true.sum(-2), y_true.sum(-1)], -1)
        assert y_true.shape == y_pred.shape, 'The shape of predict and truth is not same'
        self.truth.append(y_true)
        self.preds.append(y_pred)
        return

    def initialize_metric(self):
        for i in reversed(self.len):
            if self.threshold is None:
                for metric in self.metrics:
                    key = metric + '@' + str(i)
                    if key not in self.result:
                        self.result[key] = 0
                continue
            for j in self.threshold:
                for metric in self.metrics:
                    key = metric + '@' + str(i) + '>' + str(j)
                    if key not in self.result:
                        self.result[key] = 0

    def compute_metric(self, y_pred, y_true):
        if self.mode.lower() == 'average':  # 前i个时间步的平均loss
            for i in self.len:
                if self.threshold is None:
                    for metric in self.metrics:
                        key = metric + '@' + str(i)
                        self.result[key] = self.get_loss(y_pred[:, :i], y_true[:, :i], None, metric)
                    continue
                for j in self.threshold:
                    for metric in self.metrics:
                        key = metric + '@' + str(i) + '>' + str(j)
                        self.result[key] = self.get_loss(y_pred[:, :i], y_true[:, :i], j, metric)
        elif self.mode.lower() == 'single':  # 第i个时间步的loss
            for i in self.len:
                if self.threshold is None:
                    for metric in self.metrics:
                        key = metric + '@' + str(i)
                        self.result[key] = self.get_loss(y_pred[:, i-1], y_true[:, i-1], None, metric)
                    continue
                for j in self.threshold:
                    for metric in self.metrics:
                        key = metric + '@' + str(i) + '>' + str(j)
                        self.result[key] = self.get_loss(y_pred[:, i-1], y_true[:, i-1], j, metric)
        else:
            raise ValueError(
                'Error parameter evaluator_mode={}, please set `single` or `average`.'.format(self.mode))

    def get_loss(self, y_pred, y_true, thre, type):
        if type == 'MAE':
            return masked_mae_torch(y_pred, y_true, thre).item()
        elif type == 'MSE':
            return masked_mse_torch(y_pred, y_true, thre).item()
        elif type == 'RMSE':
            return masked_rmse_torch(y_pred, y_true, thre).item()
        elif type == 'MAPE':
            return masked_mape_torch(y_pred, y_true, thre).item()

    def evaluate(self):
        """
        返回之前收集到的所有的评估结果
        """
        truth = torch.cat(self.truth, 0)
        preds = torch.cat(self.preds, 0)
        self.compute_metric(preds, truth)
        self.compute_metric(preds, truth)
        return self.result

    def clear(self):
        """
        清除之前收集到的 batch 的评估信息，适用于每次评估开始时进行一次清空，排除之前的评估输入的影响。
        """
        self.result = {}
        self.initialize_metric()
        self.truth = []
        self.preds = []

    def save_result(self, path):
        truth = torch.cat(self.truth, 0).cpu().numpy()
        preds = torch.cat(self.preds, 0).cpu().numpy()
        np.savez(path, truth=truth, preds=preds)

class Evaluator_Dual(Evaluator):
    def __init__(self, metric_list, metric_len, threshold, mode='average', valid=False):
        super().__init__(metric_list, metric_len, threshold, mode=mode, valid=valid)
        if not valid:
            self.result = [{}, {}]  # 两种指标的结果
            self.initialize_metric_dual()
        self.valid = valid

    def initialize_metric_dual(self):
        for i in reversed(self.len):
            if self.threshold is None:
                for metric in self.metrics:
                    key = metric + '@' + str(i)
                    if key not in self.result:
                        self.result[0][key] = 0
                        self.result[1][key] = 0
                continue
            for j in self.threshold:
                for metric in self.metrics:
                    key = metric + '@' + str(i) + '>' + str(j)
                    if key not in self.result:
                        self.result[0][key] = 0
                        self.result[1][key] = 0

    def compute_metric_dual(self, y_pred, y_true):
        if self.mode.lower() == 'average':  # 前i个时间步的平均loss
            for i in self.len:
                if self.threshold is None:
                    for metric in self.metrics:
                        for k in range(2):
                            key = metric + '@' + str(i)
                            self.result[k][key] = self.get_loss(y_pred[:, :i, :, k], y_true[:, :i, :, k], None, metric)
                    continue
                for j in self.threshold:
                    for metric in self.metrics:
                        for k in range(2):
                            key = metric + '@' + str(i) + '>' + str(j)
                            self.result[k][key] = self.get_loss(y_pred[:, :i, :, k], y_true[:, :i, :, k], j, metric)
        elif self.mode.lower() == 'single':  # 第i个时间步的loss
            for i in self.len:
                if self.threshold is None:
                    for metric in self.metrics:
                        for k in range(2):
                            key = metric + '@' + str(i)
                            self.result[k][key] = self.get_loss(y_pred[:, i-1, :, k], y_true[:, i-1, :, k], None, metric)
                    continue
                for j in self.threshold:
                    for metric in self.metrics:
                        for k in range(2):
                            key = metric + '@' + str(i) + '>' + str(j)
                            self.result[k][key] = self.get_loss(y_pred[:, i-1, :, k], y_true[:, i-1, :, k], j, metric)
        else:
            raise ValueError(
                'Error parameter evaluator_mode={}, please set `single` or `average`.'.format(self.mode))

    def evaluate(self):
        """
        返回之前收集到的所有的评估结果
        """
        truth = torch.cat(self.truth, 0)
        preds = torch.cat(self.preds, 0)
        if self.valid:
            self.compute_metric(preds, truth)
            return self.result
        else:
            self.compute_metric_dual(preds, truth)
            return self.result


    def clear(self):
        """
        清除之前收集到的评估信息，适用于每次评估开始时进行一次清空，排除之前的评估输入的影响。
        """
        if self.valid:
            self.result = {}
            self.initialize_metric()
        else:
            self.result = [{}, {}]
            self.initialize_metric_dual()
        self.truth = []
        self.preds = []
