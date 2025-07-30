import sys
import os

FEATURE_EXTRACTION_DIR = os.path.dirname(os.path.abspath(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(FEATURE_EXTRACTION_DIR, "logs")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import shutil
from data.config import DataConfig

def get_freq_fex_config(fex_type, **kwargs):
    """
    All the frequency feature extraction configurations are defined here.

    Parameters
    ----------
    fex_type : str
        The type of feature extraction to be used (e.g., 'first_n_modes', 'PCA').

    **kwargs : dict
        2 or more dimensional features
        - `first_n_modes` : **n_modes** (_int_) (will get 'mode values' and its 'frequency')
        - `full_spectrum` : **parameters** (_list_) (will get 'psd', 'mag', 'amp', 'freq')
    """
    config = {}
    config['type'] = fex_type

    # 2 or more dimensional features

    if fex_type == 'first_n_modes':
        config['n_modes'] = kwargs.get('n_modes', 5)  # default to 5 modes if not specified
    
    elif fex_type == 'full_spectrum':
        config['parameters'] = kwargs.get('parameters', ['psd', 'mag', 'amp', 'freq'])  # default to all parameters if not specified

    # 1 dimensional features

    return config

def get_time_fex_config(fex_type, **kwargs):
     """
     All the time feature extraction configurations are defined here.

    Parameters
    ----------
    fex_type : str
        The type of time feature extraction
    
    **kwargs : dict
     """
     pass

def get_reduc_config(reduc_type, **kwargs):
    """
    Parameters
    ----------
    reduc_type : str
        The type of feature reduction to be used (e.g., 'PCA').
    **kwargs : dict
        For all options of `reduc_type`:
        - `PCA`: **n_components** (_int_) (number of components to keep)
    """
    config = {}
    config['type'] = reduc_type

    if reduc_type == 'PCA':
        config['n_components'] = kwargs.get('n_components', 5)  # default to 5 components if not specified

    return config

class FeatureRankingConfig:
    def __init__(self):
        self.data_config = DataConfig()
        self.helper = HelperClass()
        self.data_config.set_train_dataset()

        self.version = 1
        self.is_log = True

    def get_ranking_log_path(self):
        
        base_path = os.path.join(LOGS_DIR, 
                                f'{self.data_config.application_map[self.data_config.application]}', 
                                f'{self.data_config.machine_type}',
                                f'{self.data_config.scenario}')
        
        # add fex number
        self.ranking_log_path = os.path.join(base_path, f"feature_ranking_{self.version}")
        
        # add data type and subtype to the path
        self.ranking_id = self.helper.set_ds_types_in_path(self.data_config, base_path)

        return self.ranking_log_path

    def save_feature_ranking_id(self):
        """
        Saves the feature ranking id.
        """
        if not os.path.exists(self.ranking_log_path):
            os.makedirs(self.ranking_log_path)

        # config_path = os.path.join(self.train_log_path, f'train_config.pkl')
        # with open(config_path, 'wb') as f:
        #     pickle.dump(self.__dict__, f)

        ranking_path = os.path.join(self.ranking_log_path, f'ranking_{self.version}.txt')
        with open(ranking_path, 'w') as f:
            f.write(self.ranking_id)

        print(f"Feature ranking id saved to {self.ranking_log_path}.")

    def _remove_version(self):
        """
        Removes the ranking version from the log path.
        """
        if os.path.exists(self.ranking_log_path):
            user_input = input(f"Are you sure you want to remove the feature_ranking_{self.version} from the log path {self.ranking_log_path}? (y/n): ")
            if user_input.lower() == 'y':
                shutil.rmtree(self.train_log_path)
                print(f"Overwrote exsiting 'feature_ranking_{self.version}' from the log path {self.ranking_log_path}.")

            else:
                print(f"Operation cancelled. feature_ranking_{self.version} still remains.")

    
    def _get_next_version(self):
        parent_dir = os.path.dirname(self.ranking_log_path)

        # List all folders in parent_dir that match 'v<number>'
        ranking_folders = [f for f in os.listdir(parent_dir) if f.startswith('feature_ranking_')]

        if ranking_folders:
            # Extract numbers and find the max
            max_ranking_version = max(int(f.split('_')[-1]) for f in ranking_folders)
            self.version = max_ranking_version + 1
            new_feature_ranking = f"feature_ranking_{self.version}"
            print(f"Next feature ranking folder will be: {new_feature_ranking}")
        else:
            new_feature_ranking = f"feature_ranking_1"

        return os.path.join(parent_dir, new_feature_ranking)
    

    def check_if_version_exists(self):
        """
        Checks if the model_num already exists in the log path.

        Parameters
        ----------
        log_path : str
            The path where the logs are stored. It can be for nri model or skeleton graph model.
        """
        if os.path.isdir(self.ranking_log_path):
            print(f"'feature_ranking_{self.version}' already exists in the log path '{self.ranking_log_path}'.")
            user_input = input("(a) Overwrite exsiting version, (b) create new version, (c) stop operation (Choose 'a', 'b' or 'c'):  ")

            if user_input.lower() == 'a':
                self._remove_version()

            elif user_input.lower() == 'b':
                self.train_log_path = self._get_next_version()

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
