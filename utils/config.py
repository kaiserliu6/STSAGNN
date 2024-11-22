import sys
from logging import getLogger

class Config():
    def __init__(self, config_dict={}, config_file_list=None, hyper_config=None):
        self.config = {}
        self.config_prior = {}
        self.config.update(self._load_config_files(config_file_list))
        self.config.update(config_dict)
        self.config.update(self._load_cmd_line())

        for key in self.config:
            self.config[key] = self._para_format(self.config[key])
            self.config_prior[key] = 3
        self.default_conf()
        self.update(hyper_config, 9)
    # 3 is default priority
    # 5 is model deflaut priority
    # 7 is data default prioritu
    # 9 is hyper parameter prioritu
    def update(self, dict, prior=3):
        if dict is None:
            return
        for key in dict:
            if key in self.config and self.config_prior[key] > prior:
                pass
            else:
                self.config[key] = dict[key]
                self.config_prior[key] = prior

    def __getitem__(self, key):
        return self.config[key]

    def __setitem__(self, key, value):
        self.config[key] = self._para_format(value)

    def __call__(self, key, value=None):
        if value:
            self.config[key] = value
        else:
            return self.config[key]

    def __getattr__(self, item):
        if item in self.config:
            return self.config[item]
        else:
            return None

    def __setattr__(self, key, value):
        self.__dict__[key] = self._para_format(value)

    def contain(self,key):
        return key in self.config

    def _para_format(self, para):
        if not isinstance(para, str):
            return para
        try:
            value = eval(para)
            if value is not None and not isinstance(
                    value, (str, int, float, list, tuple, dict, bool)
            ):
                value = para
        except (NameError, SyntaxError, TypeError):
            if isinstance(para, str):
                if para.lower() == "true":
                    value = True
                elif para.lower() == "false":
                    value = False
                else:
                    value = para
            else:
                value = para
        return value

    def _load_config_files(self, file_list):
        file_config_dict = dict()
        if file_list:
            for file in file_list:
                with open(file, "r", encoding="utf-8") as f:
                    for ind, line in enumerate(f):
                        if line.strip() != '':
                            try:
                                key, value = line.strip().split('=')
                                file_config_dict[key] = value
                            except ValueError:
                                print('config file is not in the correct format! Error Line:%d' % ind)
        return file_config_dict

    def _load_cmd_line(self):
        cmd_config_dict = dict()
        unrecognized_args = []
        if "ipykernel_launcher" not in sys.argv[0]:
            for arg in sys.argv[1:]:
                if not arg.startswith("--") or len(arg[2:].split("=")) != 2:
                    unrecognized_args.append(arg)
                    continue
                cmd_arg_name, cmd_arg_value = arg[2:].split("=")
                if (
                    cmd_arg_name in cmd_config_dict
                    and cmd_arg_value != cmd_config_dict[cmd_arg_name]
                ):
                    raise SyntaxError(
                        "There are duplicate commend arg '%s' with different value."
                        % arg
                    )
                else:
                    cmd_config_dict[cmd_arg_name] = cmd_arg_value
        if len(unrecognized_args) > 0:
            logger = getLogger()
            logger.warning(
                "command line args [{}] will not be used in RecBole".format(
                    " ".join(unrecognized_args)
                )
            )
        # cmd_config_dict = self._convert_config_dict(cmd_config_dict)
        return cmd_config_dict

    def default_conf(self):
        tr_conf = {
            'epoch': 100,
            'clip_grad_norm': True,
            'max_grad_norm': 5.,
            'log_step': 1,
            'early_stop': 50,
            # 'metric_list': ['masked_MAE', 'masked_RMSE', 'masked_MAPE'],
            'metric_list': ['MAE', 'RMSE', 'MAPE'],
            'metric_len': 12,
            'metirc_threshold': 0, # when given a list, evaluate all values in list; evaluate truth greater (not greater or equal) than the value given; when given None, won't apply
            'short_valid': True,
            'task_name': 'ODFP',
            'trainer_type': 'base',
            'val_use_loss': False,
            'val_step': 1,
            'save_result':False,
            'save_dir':'cache/result/',
            'dual_evaluator':True
        }
        self.update(tr_conf, 1)
        lr_conf = {
            'optimizer': 'adam',
            'schdeuler': 'multisteplr',
            'learning_rate': 0.01,
            'lr_betas': (0.9, 0.999),
            'weight_decay': 0.0005,
            'lr_beta2': 0.999,
            'lr_alpha': 0.99,
            'lr_epsilon': 1e-8,
            'lr_momentum': 0,
            'lr_decay': True,
            'lr_decay_ratio': 0.3,
            'milestones': [50, 100, 150],
            'step_size': 50,
            'lr_lambda': lambda x: x,
            'lr_T_max': 30,
            'lr_eta_min': 0,
            'lr_patience': 10,
            'lr_threshold': 1e-4,
        }
        self.update(lr_conf, 1)
        ds_config = {
            'data_path': 'data/',
            'input_window': 12,
            'output_window': 12,
            'train_rate': 0.6,
            'test_rate': 0.2,
            'scaler_mode': 'standard',
            'need_data': False,
            'need_od': False
        }
        self.update(ds_config, 1)
        md_conf = {
            'batch_size': 1024,
            'use_weather': False,
            'use_weekday': False,
            'need_time': False,
            'num_heads': 8,
            'dropout': 0.1
        }
        self.update(md_conf, 1)

if __name__ == '__main__':
    config = Config(config_dict={'A':1,'B':'abc','C':'true'})
    print(config.A,config['B'],config('C'))
    config.A = 'true'
    config['C'] = 'FALSE'
    print(config.A, config['B'], config('C'), config.D)