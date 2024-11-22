from dataloader.dataloader import Data_OD

class DATA_nycbike(Data_OD):
    def __init__(self, config, dataset_name):
        super().__init__()
        conf = {
            'num_nodes': 75,
            'num_times': 48
        }
        config.update(conf, 7)
        conf = {
            'total_size': 17520,
            'train_size': 243 * 48,
            'test_size': 61 * 48,
            'start_date': 0
        }
        config.update(conf, 5)
        self.config = config
        self.file_name = f'{dataset_name}/'