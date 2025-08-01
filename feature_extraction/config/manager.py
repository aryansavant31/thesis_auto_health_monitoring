import os
import sys

ROOT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT_DIR) if ROOT_DIR not in sys.path else None

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CONFIG_DIR) if CONFIG_DIR not in sys.path else None

FEX_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(FEX_DIR, "logs")

# other imports
import shutil

# global imports
from data.settings import DataConfig

# local imports
from rank_settings import FeatureRankingSettings

class FeatureRankingManager(FeatureRankingSettings):
    def __init__(self):
        super().__init__()
        self.helper = HelperClass()

    def get_perf_log_path(self, data_config:DataConfig, check_version=True):
        """
        Returns the path for saving the feature performance logs.

        Note
        -----
        It is assumed that the `set_train_dataset` method is called before this method.

        Parameters
        ----------
        data_config : DataConfig
            The data configuration object. Since the data_config attribute values can change outside, 
            it is being passed as a parameter here to provide new log paths.
        """
        self.node_type = f"({'+'.join(data_config.node_type)})"
        
        base_path = os.path.join(LOGS_DIR, 
                                f'{data_config.application_map[data_config.application]}', 
                                f'{data_config.machine_type}',
                                f'{data_config.scenario}')
        
        # add node type
        perf_path = os.path.join(base_path, f'{self.node_type}')
        
        # addnfeat number
        self.perf_log_path = os.path.join(perf_path, f"{self.node_type}_perf_{self.version}")
        
        # add data type and subtype to the path
        perf_path = self.helper.set_ds_types_in_path(data_config, perf_path)

        # add timestep id and signal type to path
        self.perf_id = os.path.join(perf_path, f"T{data_config.window_length} [{', '.join(data_config.signal_types)}]")

        if check_version:
            self.check_if_perf_version_exists()

        return self.perf_log_path
    
    def get_ranking_log_path(self, data_config:DataConfig):
        perf_log_path = self.get_perf_log_path(data_config, check_version=False)

        if not os.path.exists(perf_log_path):
            raise FileNotFoundError(f"Performance log path {perf_log_path} does not exist. Please run the feature performance ranking first or type the correct version.")

        self.ranking_log_path = os.path.join(perf_log_path, f'rankings')
        return self.ranking_log_path

    def save_perf_id(self):
        """
        Saves the performance id.
        """
        if not os.path.exists(self.perf_log_path):
            os.makedirs(self.perf_log_path)

        # config_path = os.path.join(self.train_log_path, f'train_config.pkl')
        # with open(config_path, 'wb') as f:
        #     pickle.dump(self.__dict__, f)

        perf_path = os.path.join(self.perf_log_path, f'{self.node_type}_perf_{self.version}.txt')
        with open(perf_path, 'w') as f:
            f.write(self.perf_id)

        print(f"Feature performance id saved to {self.perf_log_path}.")

    def _remove_perf_version(self):
        """
        Removes the performance version from the log path.
        """
        if os.path.exists(self.perf_log_path):
            user_input = input(f"Are you sure you want to remove '{self.node_type}_perf_{self.version}' from the log path {self.perf_log_path}? (y/n): ")
            if user_input.lower() == 'y':
                shutil.rmtree(self.perf_log_path)
                print(f"Overwrote exsiting '{self.node_type}_perf_{self.version}' from the log path {self.perf_log_path}.")

            else:
                print(f"Operation cancelled. {self.node_type}_perf_{self.version} still remains.")

    def _get_next_perf_version(self):
        parent_dir = os.path.dirname(self.perf_log_path)

        # List all folders in parent_dir that match 'v<number>'
        perf_folders = [f for f in os.listdir(parent_dir) if f.startswith(f'{self.node_type}_perf_')]

        if perf_folders:
            # Extract numbers and find the max
            max_perf_version = max(int(f.split('_')[-1]) for f in perf_folders)
            self.version = max_perf_version + 1
            new_feature_perf = f"{self.node_type}_perf_{self.version}"
            print(f"Next feature performance folder will be: {new_feature_perf}")
        else:
            new_feature_perf = f"{self.node_type}_perf_1"

        return os.path.join(parent_dir, new_feature_perf)
    
    def check_if_perf_version_exists(self):
        """
        Checks if the performance version already exists in the log path.

        Parameters
        ----------
        log_path : str
            The path where the logs are stored. It can be for nri model or skeleton graph model.
        """
        if os.path.isdir(self.perf_log_path):
            print(f"'{self.node_type}_perf_{self.version}' already exists in the log path '{self.perf_log_path}'.")
            user_input = input("(a) Overwrite exsiting version, (b) create new version, (c) stop operation (Choose 'a', 'b' or 'c'):  ")

            if user_input.lower() == 'a':
                self._remove_perf_version()

            elif user_input.lower() == 'b':
                self.perf_log_path = self._get_next_perf_version()

            elif user_input.lower() == 'c':
                print("Stopped training.")
                sys.exit()  # Exit the program gracefully    


class HelperClass:
    def get_augmment_config_str_list(self, augment_configs):
        """
        Returns a list of strings representing the augment configurations.
        """
        augment_str_list = []
        
        for augment_config in augment_configs:
            idx = 0
            augment_str = f"{augment_config['type']}"
            for key, value in augment_config.items():
                if key != 'type':
                    if idx == 0:
                        augment_str += "_" 
                        idx += 1
                    augment_str += f"{key[0]}={value}"

            augment_str_list.append(augment_str)

        return augment_str_list
    
    def set_ds_types_in_path(self, data_config, log_path):
        """
        Takes into account both empty healthy and unhealthy config and sets the path accordingly.
        """
        if data_config.unhealthy_configs == {}:
            log_path = os.path.join(log_path, 'healthy')

        elif data_config.unhealthy_configs != {}:
            log_path = os.path.join(log_path, 'healthy_unhealthy')

        # add ds_subtype to path
        config_str = ''
        
        if data_config.healthy_configs != {}:    
            healthy_config_str_list = []

            for healthy_type, augment_configs  in data_config.healthy_configs.items():
                augment_str_list = self.get_augmment_config_str_list(augment_configs)                    
                augment_str_main = ', '.join(augment_str_list) 

                healthy_config_str_list.append(f'{healthy_type}[{augment_str_main}]')

            config_str = ' + '.join(healthy_config_str_list)

        # add unhealthy config to path if exists
        if data_config.unhealthy_configs != {}:
            unhealthy_config_str_list = []

            for unhealthy_type, augment_configs in data_config.unhealthy_configs.items():
                augment_str_list = self.get_augmment_config_str_list(augment_configs)
                augment_str_main = ', '.join(augment_str_list) 

                unhealthy_config_str_list.append(f'{unhealthy_type}[{augment_str_main}]')

            if config_str:
                config_str += f" + {' + '.join(unhealthy_config_str_list)}"
            else:
                config_str += ' + '.join(unhealthy_config_str_list)

        log_path = os.path.join(log_path, config_str)
        return log_path