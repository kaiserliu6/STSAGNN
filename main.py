import argparse
import logging
from trainer import *
import os
import csv
import numpy as np

DEBUG=True
round_save = 4
"""
currently have 2 evaluator:
evaluator: result dict {'{metric}@{len}>{thre}':value}
dual_evaluator: result [{'{metric}@{len}>{thre}':value},{..}]
r, s, b for result, seed, run epoch, respectively
dir: csv save dir
result: shape as above
run_info: [s, b]
model_name: need if is_average=False, save in csv as index
is_average: whether the result is the average result
"""
class result_saver():
    def __init__(self, task, dir, metric=None, mlen=None, thre=None):
        self.task = task
        self.dir = dir
        if os.path.exists(self.dir):
            with open(self.dir, 'r') as csv_file:
                line = csv_file.readline().split(',')
                if self.task == 'IO':
                    self.head = line[2:-2]
                else:
                    self.head = line[1:-2]
        else:
            self.head = []
            assert metric is not None
            if isinstance(mlen, int):
                mlen = [mlen]
            if thre is None:
                for i in reversed(mlen):
                        for mtc in metric:
                            self.head.append(f'{mtc}@{i}')
            else:
                if isinstance(thre, int):
                    thre = [thre]
                for i in reversed(mlen):
                    for j in thre:
                        for mtc in metric:
                            self.head.append(f'{mtc}@{i}>{j}')
            with open(self.dir, 'w', newline='') as csv_file:
                writer = csv.writer(csv_file)
                if self.task == 'IO':
                    writer.writerow(['Model Name', 'Type'] + self.head + ['Seed', 'Epoch'])
                else:
                    writer.writerow(['Model Name'] + self.head + ['Seed', 'Epoch'])
        if self.task == 'IO':
            self.result = [[[] for i in self.head], [[] for i in self.head]]
        else:
            self.result = [[] for i in self.head]

    def save_result(self, result, run_info, model_name):
        self.write_csv(result, run_info, model_name)

    def write_csv(self, result, run_info, model_name):
        # type 2
        if self.task == 'IO':
            for i, k in enumerate(self.head):
                self.result[0][i].append(result[0][k])
                self.result[1][i].append(result[1][k])
            with open(self.dir, 'a', newline='') as csv_file:
                writer = csv.writer(csv_file)
                row1 = [model_name, 'in-flow'] + [round(result[0][i], round_save) for i in self.head] + run_info
                row2 = [model_name, 'out-flow'] + [round(result[1][i], round_save) for i in self.head] + run_info
                writer.writerow(row1)
                writer.writerow(row2)
        # type 1
        else:
            for i, k in enumerate(self.head):
                self.result[i].append(result[k])
            with open(self.dir, 'a', newline='') as csv_file:
                writer = csv.writer(csv_file)
                row = [model_name] + [round(result[i], round_save) for i in self.head] + run_info
                writer.writerow(row)

    def save_average(self):
        if self.task == 'IO':
            result = [[0 for i in self.head] for j in range(2)]
            for i, k in enumerate(self.head):
                result[0][i] = round(np.mean(self.result[0][i]), round_save)
                result[1][i] = round(np.mean(self.result[1][i]), round_save)
            with open(self.dir, 'a', newline='') as csv_file:
                writer = csv.writer(csv_file)
                row1 = ['Average', 'in-flow'] + [round(i, round_save) for i in result[0]] + ['','']
                row2 = ['Average', 'out-flow'] + [round(i, round_save) for i in result[1]] + ['','']
                writer.writerow(row1)
                writer.writerow(row2)
        else:
            result = [0 for i in self.head]
            for i, k in enumerate(self.head):
                result[i] = round(np.mean(self.result[i]), round_save)
            with open(self.dir, 'a', newline='') as csv_file:
                writer = csv.writer(csv_file)
                row = ['Average'] + [round(i, round_save) for i in result] + ['','']
                writer.writerow(row)

def get_trainer(model, dataset, config_file_list, cd):
    # if model in ['CCRNN', 'MegaCRN', 'HimNet']:
    #     return RNN_Trainer(model, dataset, config_file_list, cd)
    # else:
    #     return Trainer(model, dataset, config_file_list, cd)
    return Trainer(model, dataset, config_file_list, cd)


if __name__ == "__main__":
    model_list = ['GWNet', 'AGCRN', 'MegaCRN', 'STSAGNN', 'test3']
    model_list = ['PatchTST', 'iTransformer', 'GWNet', 'AGCRN', 'MegaCRN', 'STSAGNN', 'test3']
    # model_list = ['MegaCRN', 'STSAGNN', 'STSformer', 'test3']
    # model_list = ['MegaCRN', 'STSAGNN', 'test2', 'CycleNet']
    # model_list = ['STSAGNN', 'test3', 'STSformer']
    model_list = ['PatchTST', 'iTransformer', 'STSAGNN_V']
    # model_list = ['STSAGNN', 'test3']
    # model_list = ['MegaCRN', 'GWNet']
    # model_list = ['CCRNN', 'TGCRN']
    data_list = ['NYC_2023', 'NYC_bike']
# data_list = ['NYC_2023']
    # data_list = ['METR_LA', 'PEMS_BAY', 'pemsd3', 'pemsd4', 'pemsd7', 'pemsd8']
    # data_list = ['METR_LA', 'PEMS_BAY']
    model_name = model_list[2]
    data_name = data_list[1]
    # data_name = 'METR_LA'
    param_name = 'emb_dim'
    param_dict = [16, 32, 64, 128, 256]
    param_name = 'mode'
    param_dict = [0, 'SG', 'DG', 'AG']
    run_times = 1
    debug_flag = 3

    cd = {'output_window': 12, 'metric_len': [1, 6, 12], 'metirc_threshold': [0], 'task_name': 'TF'}
    cd = {'output_window': 12, 'metric_len': [1, 6, 12], 'metirc_threshold': None, 'task_name': 'IO'}
    cd = {'output_window': 12, 'metric_len': [12], 'metirc_threshold': None, 'task_name': 'IO'}
    # # cd = {'output_window': 12, 'metric_len': [1, 6, 12], 'metirc_threshold': None, 'task_name': 'IO', 'save_result':True}
    # cd = {'output_window': 12, 'metric_len': [1, 6, 12], 'metirc_threshold': [10], 'task_name': 'OD'}

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default=model_name, help='name of models, see model for more details')
    parser.add_argument("--dataset", "-d", type=str, default=data_name, help='name of datasets, see dataloader for more details')
    parser.add_argument("--config_files", type=str, default=None, help="config files, split with ','")
    parser.add_argument('--run_times', type=int, default=run_times, help='times of running each model')
    parser.add_argument("--debug_mode", type=int, default=debug_flag, help="running mode, model, data, hyper")
    parser.add_argument('--model_list', type=list, default=model_list, help='model list for debug mode 1')
    parser.add_argument('--data_list', type=list, default=data_list, help='data list for debug mode 2')
    # parser.add_argument('--param_name', type=list, default=param_name, help='parameter name for debug mode 3')
    # parser.add_argument('--param_dict', type=list, default=param_dict, help='searching value for debug mode 3')
    """
    running mode supported:
    0: one model, one dataset, config unchange
    1: models in model_list, one dataset, config unchange
    2: one model, datasets in data_list, config unchange
    3: one model, one dataset, different value of give parameter
    """

    args, _ = parser.parse_known_args()
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S')
    config_file_list = (args.config_files.strip().split(",") if args.config_files else None)

    mlen = cd['metric_len']
    thre = cd['metirc_threshold']
    task = cd['task_name']

    if args.debug_mode == 0:
        result = result_saver(task, f"{task}_{args.dataset}_{mlen}_{thre}.csv",
                              ['MAE', 'RMSE', 'MAPE'], mlen, thre)
        for j in range(args.run_times):
            model = get_trainer(args.model, args.dataset, config_file_list, cd)
            r, s, b = model.run(True)
            result.save_result(r, [s, b], args.model)
        result.save_average()
    elif args.debug_mode == 1:
        for i, m in enumerate(args.model_list):
            result = result_saver(task, f"{task}_{args.dataset}_{mlen}_{thre}.csv",
                                  ['MAE','RMSE','MAPE'], mlen, thre)
            for j in range(args.run_times):
                model = get_trainer(m, args.dataset, config_file_list, cd)
                r, s, b = model.run(True)
                result.save_result(r, [s, b], m)
            result.save_average()
    elif args.debug_mode == 2:
        for i, d in enumerate(args.data_list):
            result = result_saver(task, f"{task}_{d}_{mlen}_{thre}.csv",
                                  ['MAE', 'RMSE', 'MAPE'], mlen, thre)
            for j in range(args.run_times):
                model = get_trainer(args.model, d, config_file_list, cd)
                r, s, b = model.run(True)
                result.save_result(r, [s, b], args.model)
            result.save_average()
    elif args.debug_mode == 3:
        for i, p in enumerate(param_dict):
            result = result_saver(task, f"{task}_{args.dataset}_{mlen}_{thre}.csv",
                                  ['MAE', 'RMSE', 'MAPE'], mlen, thre)
            cd[param_name] = p
            for j in range(args.run_times):
                model = get_trainer(args.model, args.dataset, config_file_list, cd)
                r, s, b = model.run(True)
                result.save_result(r, [s, b], f'{param_name}={p}')
            result.save_average()
    elif args.debug_mode == 4:
        for i, d in enumerate(args.data_list):
            for i, m in enumerate(args.model_list):
                if d=='METR_LA' or (d=='PEMS_BAY' and m in ['PatchTST', 'iTransformer', 'CycleNet', 'GWNet', 'AGCRN', 'MegaCRN']):
                    continue
                result = result_saver(task, f"{task}_{d}_{mlen}_{thre}.csv",
                                      ['MAE', 'RMSE', 'MAPE'], mlen, thre)
                for j in range(args.run_times):
                    model = get_trainer(m, d, config_file_list, cd)
                    r, s, b = model.run(True)
                    result.save_result(r, [s, b], m)
                result.save_average()