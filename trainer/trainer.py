import os
import torch
import time
from datetime import datetime
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import importlib
from tqdm import tqdm
from matplotlib import pyplot as plt

from logging import getLogger

from utils import Config, Evaluator, ensure_dir, Evaluator_Dual
from dataloader import *

class Trainer():
    def __init__(self, model_name, dataset_name, config_file_list = None, config_dict=None, random_seed=None):
        # set random seed
        self.seed = random_seed or np.random.randint(10000)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        # initial config
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.config = Config(config_file_list=config_file_list, hyper_config=config_dict)
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        # get dataloader, data not loaded yet, load config in dataset first
        self.data = self.get_dataloader()

        # get model
        self.model = getattr(importlib.import_module('model'), model_name)(self.config, self.data).to(self.device)

        # set evaluator according to task, for OD and TF simple, for IO evaluate both in-flow and out-flow
        if self.config.task_name == 'IO' and self.config.dual_evaluator:
            self.evaluator = Evaluator_Dual(self.config.metric_list, self.config.metric_len, self.config.metirc_threshold)
            self.valider = Evaluator_Dual(self.config.metric_list, self.config.metric_len, self.config.metirc_threshold, valid=True)
        else:
            self.evaluator = Evaluator(self.config.metric_list, self.config.metric_len, self.config.metirc_threshold)
            self.valider = Evaluator(self.config.metric_list, self.config.metric_len, self.config.metirc_threshold, valid=self.config.short_valid)

        # initialize logging
        self.logger = getLogger()
        self._writer = SummaryWriter(ensure_dir('cache/summary_cache/'))
        time = datetime.now().strftime('%Y_%m_%d_%H')
        ensure_dir(f'cache/{dataset_name}')
        self.cache_dir = f'cache/{dataset_name}/{model_name}_{time}.temp'
        self.evaluate_res_dir = ensure_dir('result/')

        # load dataset
        if not self.config.need_od and self.config.task_name == 'IO':
            self.data.process_od = True
        if hasattr(self.model, 'norm'):
            self.train_dl, self.val_dl, self.test_dl = self.data.get_dataloader(self.model.norm)
        else:
            self.train_dl, self.val_dl, self.test_dl = self.data.get_dataloader()

        # intialize training component
        self.optimizer = self._build_optimizer()
        self.lr_scheduler = self._build_lr_scheduler()

        # report model imformation
        # for name, param in self.model.named_parameters():
        #     print(name, param.shape, param.requires_grad)
        self.logger.info(f'Total parameter numbers: {sum([param.nelement() for param in self.model.parameters()])}')

        self.task = self.config.task_name or 'OD'
        self.run_batch = 0

    def run(self, message=False):
        # self.load_model()
        self.train()
        self.load_model()
        result = self.evaluate()
        self.model.last_info()
        if message:
            return result, self.seed, self.run_batch
        else:
            return result

    def save_model(self):
        torch.save((self.model.state_dict(), self.optimizer.state_dict()), self.cache_dir)

    def load_model(self, dir=None):
        dir = dir or self.cache_dir
        model_state, optimizer_state = torch.load(dir)
        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(optimizer_state)

    def evaluate(self):
        self.logger.info('Start evaluating ...')
        with torch.no_grad():
            self.model.eval()
            self.evaluator.clear()
            for batch in self.test_dl:
                output = self.model.predict(batch)
                self.evaluator.collect(output, batch[1].to(self.device))
            test_result = self.evaluator.evaluate()
            if self.config.save_result:
                self.evaluator.save_result(self.config.save_dir + f'{self.model_name}_{self.dataset_name}.npz')
            self.logger.info(test_result)
            return test_result

    def train(self):
        self.logger.info('Start training ...')
        train_time = []
        eval_time = []

        for epoch_idx in range(self.config.epoch):
            start_time = time.time()
            train_loss = self._train_epoch(epoch_idx)
            t1 = time.time()
            train_time.append(t1 - start_time)
            t2 = time.time()
            val_loss = self._valid_epoch(epoch_idx)
            end_time = time.time()
            eval_time.append(end_time - t2)

            if self.lr_scheduler is not None:
                if self.config.schdeuler.lower() == 'reducelronplateau':
                    self.lr_scheduler.step(val_loss.values()[0])
                else:
                    self.lr_scheduler.step()

            if (epoch_idx % self.config.log_step) == 0:
                self.logger.info(f'Epoch {epoch_idx}/{self.config.epoch}: train_loss is {train_loss}, val result is {val_loss}, train time {t1 - start_time}, val time {end_time - t2}')

            if not hasattr(self, 'best_performance'):
                self.best_performance = val_loss
                self.bpt = val_loss
                self.not_improve = 0
            if epoch_idx % self.config.val_step == 0:
                better = 0
                all = 0
                if self.config.val_use_loss:
                    if val_loss < self.best_performance:
                        self.logger.info('Current model is better')
                        self.best_performance = val_loss
                        self.bpt = self._test_epoch(epoch_idx)
                        self.logger.info(f'Current Performance of test: {self.bpt}')
                        self.not_improve = 0
                        self.save_model()
                    else:
                        # self.logger.info(f'Best Performance of valid: {self.best_performance}')
                        # self.logger.info(f'Best Performance of test: {self.bpt}')
                        self.not_improve += self.config.val_step
                        self.logger.info('Not improved for %d epoch', self.not_improve)
                else:
                    for key, value in val_loss.items():
                        all += 1
                        if value < self.best_performance[key]:
                            better += 1
                    if better / all > 0.5:
                        self.logger.info('Current model is better')
                        self.best_performance = val_loss
                        self.bpt = self._test_epoch(epoch_idx)
                        self.logger.info(f'Current Performance of test: {self.bpt}')
                        self.not_improve = 0
                        self.save_model()
                    else:
                        # self.logger.info(f'Best Performance of valid: {self.best_performance}')
                        # self.logger.info(f'Best Performance of test: {self.bpt}')
                        self.not_improve += self.config.val_step
                        self.logger.info('Not improved for %d epoch', self.not_improve)
                if self.config.early_stop and self.config.early_stop < self.not_improve:
                    self.logger.warning(f'Early stop at epoch {epoch_idx}')
                    break
        self.logger.info(f'Trained totally {len(train_time)} epochs, average train time is {sum(train_time) / len(train_time)}s, average eval time is {sum(eval_time) / len(eval_time)}s')
        self.run_batch = len(train_time)
        return

    def _train_epoch(self, epoch_idx):
        self.model.train()
        losses = []
        for batch in tqdm(self.train_dl):
            loss = self.model.calculate_loss(batch)
            self.optimizer.zero_grad()
            loss.backward()
            losses.append(loss.item())
            if self.config.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
        mean_loss = np.mean(losses)
        self._writer.add_scalar('training loss', mean_loss, epoch_idx)
        return mean_loss

    def _valid_epoch(self, epoch_idx):
        with torch.no_grad():
            self.model.eval()
            if self.config.val_use_loss:
                total_loss = []
                for batch in self.val_dl:
                    loss = self.model.calculate_loss(batch)
                    total_loss.append(loss.item())
                valid_result = np.mean(total_loss)
            else:
                if hasattr(self, 'valider'):
                    self.valider.clear()
                    for batch in self.val_dl:
                        output = self.model.predict(batch)
                        self.valider.collect(output, batch[1].to(self.device))
                    valid_result = self.valider.evaluate()
            return valid_result

    def _test_epoch(self, epoch_idx):
        with torch.no_grad():
            self.model.eval()
            self.evaluator.clear()
            for batch in self.test_dl:
                output = self.model.predict(batch)
                self.evaluator.collect(output, batch[1].to(self.device))
            test_result = self.evaluator.evaluate()
            # self.logger.info(test_result)
            return test_result

    def get_dataloader(self):
        try:
            data = getattr(importlib.import_module('dataloader'), f'Data_{self.dataset_name.lower()}')(self.config, self.dataset_name)
        except:
            raise Exception(f'Data_{self.dataset_name.lower()} not exist, see dataloader for supported dataset')
        return data

    def _build_optimizer(self):
        """
        根据全局参数`optimizer`选择optimizer
        """
        optmizer_name = self.config.optimizer.lower()
        self.logger.info(f'You select {optmizer_name} optimizer.')
        if optmizer_name == 'adam':
            optimizer = torch.optim.Adam(self.model.model.parameters(), lr=self.config.learning_rate,
                                         eps=self.config.lr_epsilon, betas=self.config.lr_betas, weight_decay=self.config.weight_decay)
        elif optmizer_name == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config.learning_rate,
                                        momentum=self.config.lr_momentum, weight_decay=self.config.weight_decay)
        elif optmizer_name == 'adagrad':
            optimizer = torch.optim.Adagrad(self.model.parameters(), lr=self.config.learning_rate,
                                            eps=self.config.lr_epsilon, weight_decay=self.config.weight_decay)
        elif optmizer_name == 'rmsprop':
            optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.config.learning_rate,
                                            alpha=self.config.lr_alpha, eps=self.config.lr_epsilon,
                                            momentum=self.config.lr_momentum, weight_decay=self.config.weight_decay)
        elif optmizer_name == 'sparse_adam':
            optimizer = torch.optim.SparseAdam(self.model.parameters(), lr=self.config.learning_rate,
                                               eps=self.config.lr_epsilon, betas=self.config.lr_betas)
        else:
            self.logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate,
                                         eps=self.config.lr_epsilon, weight_decay=self.config.weight_decay)
        return optimizer

    def _build_lr_scheduler(self):
        """
        根据全局参数`lr_scheduler`选择对应的lr_scheduler
        """
        if self.config.lr_decay:
            scheduler_name = self.config.schdeuler.lower()
            self.logger.info(f'You select {scheduler_name} lr_scheduler.')
            if scheduler_name == 'multisteplr':
                lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    self.optimizer, milestones=self.config.milestones, gamma=self.config.lr_decay_ratio)
            elif scheduler_name == 'steplr':
                lr_scheduler = torch.optim.lr_scheduler.StepLR(
                    self.optimizer, step_size=self.config.step_size, gamma=self.config.lr_decay_ratio)
            elif scheduler_name == 'exponentiallr':
                lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    self.optimizer, gamma=self.config.lr_decay_ratio)
            elif scheduler_name == 'cosineannealinglr':
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=self.config.lr_T_max, eta_min=self.config.lr_eta_min)
            elif scheduler_name == 'lambdalr':
                lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                    self.optimizer, lr_lambda=self.config.lr_lambda)
            elif scheduler_name == 'reducelronplateau':
                lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, mode='min', patience=self.config.lr_patience,
                    factor=self.config.lr_decay_ratio, threshold=self.config.lr_threshold)
            else:
                self.logger.warning('Received unrecognized lr_scheduler, '
                                     'please check the parameter `lr_scheduler`.')
                lr_scheduler = None
        else:
            lr_scheduler = None
        return lr_scheduler