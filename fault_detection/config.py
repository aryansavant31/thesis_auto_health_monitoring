import os
import sys
FAULT_DETECTION_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(FAULT_DETECTION_DIR, "logs")

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from feature_extraction import get_fex_config
from data.config import DataConfig
import shutil
import re
from pathlib import Path
from rich.tree import Tree
from rich.console import Console
import pickle


class PredictAnomalyDetectorConfig:
    def __init__(self):
        pass

class TrainAnomalyDetectornConfig:
    def __init__(self):
        self.data_config = DataConfig()
        self.helper = HelperClass()
        self.data_config.set_train_dataset()

        self.set_training_params()
        self.set_run_params()
        self.anom_config = self.get_anom_config('IF')

    def set_training_params(self):
        self.version = 1
        self.is_log = True

        # dataset parameters
        self.batch_size = 50

        self.train_rt   = 0.8
        self.test_rt    = 0.2

    def set_run_params(self):
        self.domain = 'time'   # freq-psd, freq-amp
        self.norm_type = None
        self.fex_configs = [
            get_fex_config('first_n_modes', n_modes=5)
        ]  

    def get_anom_config(self, anom_type, **kwargs):
        """
        Parameters
        ----------
        anom_type : str
            The type of fault detection algorithm to be used. 
            - 'SVM': Support Vector Machine
            - 'IF': Isolation Forest)
        **kwargs : dict
            For all options of `anom_type`:
            - 'SVM': `kernel`, `nu`, `gamma`
            - 'IF': `n_estimators`, `seed`, `contam`, `n_jobs`

        """
        anom_config = {}
        anom_config['type'] = anom_type

        if anom_type == 'SVM':
            anom_config['kernel'] = kwargs.get('kernel', 'rbf')
            anom_config['gamma'] = kwargs.get('gamma', 'scale')
            anom_config['nu'] = kwargs.get('nu', 0.5)

        elif anom_type == 'IF':
            anom_config['n_estimators'] = kwargs.get('n_estimators', 100)
            anom_config['seed'] = kwargs.get('seed', 42)
            anom_config['contam'] = kwargs.get('contam', 'auto')
            anom_config['n_jobs'] = kwargs.get('n_jobs', -1)

        return anom_config  
    
    def get_train_log_path(self, n_components, n_dim):
        """
        Returns the path to store the logs based on data and topology config

        Parameters
        ----------
        n_components : int
            The number of components in each datapoint/sample in the dataset
        n_dim : int
            The number of dimensions in each component in the dataset
        """
        
        base_path = os.path.join(LOGS_DIR, 
                                f'{self.data_config.application_map[self.data_config.application]}', 
                                f'{self.data_config.machine_type}',
                                f'{self.data_config.scenario}')

        # add node name to path
        train_log_path = os.path.join(base_path, f'node={self.data_config.node_type}')

        # add healthy or healthy_unhealthy config to path
        train_log_path = self.helper.set_ds_types_in_path(self.data_config, train_log_path)

        # add model type to path
        train_log_path = os.path.join(train_log_path, f'anom={self.anom_config['type']}')

        # add datastats to path
        train_log_path = os.path.join(train_log_path, f'T{self.data_config.window_length}_measures=[{'+'.join(self.data_config.signal_types)}]')

        # add domain type to path
        train_log_path = os.path.join(train_log_path, f'domain={self.domain}')

        # add feature type to path
        fex_types = [fex['type'] for fex in self.fex_configs]

        if fex_types:
            train_log_path = os.path.join(train_log_path, f'fex=[{'+'.join(fex_types)}]')
        else:
            train_log_path = os.path.join(train_log_path, 'fex=_no_fex')

        # add model shape compatibility stats to path
        train_log_path = os.path.join(train_log_path, f'anom(comps)={n_components*n_dim}')

        # add model version to path
        self.train_log_path = os.path.join(train_log_path, f'v{self.version}')

        # check if version already exists
        self.check_if_version_exists()

        return self.train_log_path
        
    def get_test_log_path(self):
        
        test_log_path = os.path.join(self.train_log_path, 'test')

        # add healthy or healthy_unhealthy config to path
        test_log_path = self._set_ds_types_in_path(test_log_path)

        # add timestep id to path
        test_log_path = os.path.join(test_log_path, f'T{self.data_config.window_length}')

        # add test version to path
        test_log_path = os.path.join(test_log_path, f'test_v0')

        return test_log_path
    
    def save_params(self):
        """
        Saves the training parameters to a pickle file in the log path.
        """
        if not os.path.exists(self.train_log_path):
            os.makedirs(self.train_log_path)

        param_path = os.path.join(self.train_log_path, f'train_config.pkl')
        with open(param_path, 'wb') as f:
            pickle.dump(self.__dict__, f)

    def _remove_version(self):
        """
        Removes the version from the log path.
        """
        if os.path.exists(self.train_log_path):
            user_input = input(f"Are you sure you want to remove the version {self.version} from the log path {self.train_log_path}? (y/n): ")
            if user_input.lower() == 'y':
                shutil.rmtree(self.train_log_path)
                print(f"Overwrote exsiting version 'v{self.version}' from the log path {self.train_log_path}.")

            else:
                print(f"Operation cancelled. Version 'v{self.version}' still remains.")

    
    def _get_next_version(self):
        parent_dir = os.path.dirname(self.train_log_path)

        # List all folders in parent_dir that match 'v<number>'
        folders = [f for f in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, f))]
        v_folders = [f for f in folders if re.match(r'^v\d+$', f)]

        if v_folders:
            # Extract numbers and find the max
            max_v = max(int(f[1:]) for f in v_folders)
            new_v = f'v{max_v + 1}'
            print(f"Next version folder will be: {new_v}")
        else:
            new_v = 'v1'  # If no v folders exist

        return os.path.join(parent_dir, new_v)
    
    def save_params(self):
        """
        Saves the training parameters to a pickle file in the log path.
        """
        if not os.path.exists(self.train_log_path):
            os.makedirs(self.train_log_path)

        param_path = os.path.join(self.train_log_path, f'train_config.pkl')
        with open(param_path, 'wb') as f:
            pickle.dump(self.__dict__, f)

    def check_if_version_exists(self):
        """
        Checks if the version already exists in the log path.

        Parameters
        ----------
        log_path : str
            The path where the logs are stored. It can be for nri model or skeleton graph model.
        """

        if os.path.isdir(self.train_log_path):
            print(f"'Version {self.version}' already exists in the log path '{self.train_log_path}'.")
            user_input = input("(a) Overwrite exsiting version, (b) create new version, (c) stop training (Choose 'a', 'b' or 'c'):  ")

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
                augment_str_main = '--'.join(augment_str_list) 

                healthy_config_str_list.append(f'{healthy_type}[{augment_str_main}]')

            config_str = '_+_'.join(healthy_config_str_list)

        # add unhealthy config to path if exists
        if data_config.unhealthy_configs != {}:
            unhealthy_config_str_list = []

            for unhealthy_type, augment_configs in data_config.unhealthy_configs.items():
                augment_str_list = self.get_augmment_config_str_list(augment_configs)
                augment_str_main = '--'.join(augment_str_list) 

                unhealthy_config_str_list.append(f'{unhealthy_type}[{augment_str_main}]')

            if config_str:
                config_str += f'_+_{'_+_'.join(unhealthy_config_str_list)}'
            else:
                config_str += '_+_'.join(unhealthy_config_str_list)

        log_path = os.path.join(log_path, config_str)
        return log_path
 
def get_model_pickle_path(log_path):
    """
    Returns the path to the training parameters pickle file.
    """
    model_path = os.path.join(log_path, 'anomaly_detector.pkl')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Training parameters file not found: {model_path}. Please check the log path.")
    
    return model_path

def get_param_pickle_path(log_path):
    """
    Returns the path to the training parameters pickle file.
    """
    param_path = os.path.join(log_path, 'train_config.pkl')
    if not os.path.exists(param_path):
        raise FileNotFoundError(f"Training parameters file not found: {param_path}. Please check the log path.")
    
    return param_path

class SelectFaultDetectionModel():
    def __init__(self, application=None, machine=None, scenario=None, logs_dir="logs"):
        data_config = DataConfig()

        self.logs_dir = Path(logs_dir)
        if application is None or machine is None or scenario is None:
            self.application = data_config.application_map[data_config.application]
            self.machine = data_config.machine_type
            self.scenario = data_config.scenario
        else:
            self.application = application
            self.machine = machine
            self.scenario = scenario
        self.structure = {}
        self.version_paths = []
        self._build_structure()

    def _build_structure(self):
        base = self.logs_dir / self.application / self.machine / self.scenario 
        if not base.exists():
            raise FileNotFoundError(f"Path does not exist: {base}")
        self.structure = self._explore(base, 0)

    def _explore(self, path, level):
        structure = {}
        if not path.is_dir():
            return structure
        # Sort by name, ascending (case-insensitive)
        for item in sorted(path.iterdir(), key=lambda x: x.name.lower()):
            if item.is_dir():
                key = item.name
                if key.startswith("v") and key[1:].isdigit():
                    rel_path = item.relative_to(self.logs_dir)
                    self.version_paths.append(str(rel_path))
                structure[key] = self._explore(item, level + 1)
        return structure

    def print_tree(self):
        console = Console()
        version_index_map = {os.path.normpath(v): idx for idx, v in enumerate(self.version_paths)}

        # Green up to and including scenario
        tree = Tree(f"[green]{self.application}[/green]")
        machine_node = tree.add(f"[green]{self.machine}[/green]")
        scenario_node = machine_node.add(f"[green]{self.scenario}[/green]")
        
        self._build_rich_tree(scenario_node, self.structure, 0, [], version_index_map)
        console.print(tree)
        print("\nAvailable version paths:")
        for idx, vpath in enumerate(self.version_paths):
            print(f"{idx}: logs/{vpath}")

    def _build_rich_tree(self, parent_node, structure, level, parent_keys, version_index_map):
        current_path = [self.application, self.machine, self.scenario] + parent_keys
        is_no_fex = any("_no_fex" in k for k in parent_keys)
        # Label maps
        
        label_map = {
            0: "<node_name>",
            1: "<ds_type>",
            2: "<ds_subtype>",
            3: "<model>",
            4: "<ds_stats>",
            5: "<domain>",
            6: "<fex_type>",
            7: "<shape_compatibility>",
            8: "<version>"
            }
        
        added_labels = set()
        for key, value in structure.items():
            # Escape brackets for Rich markup
            safe_key = key.replace('[', '\\[')
            # Add blue label if at the correct level and not already added
            if level in label_map and label_map[level] not in added_labels:
                parent_node.add(f"[blue]{label_map[level]}[/blue]")
                added_labels.add(label_map[level])
            # For directed graph, make the model folder yellow
            if level == 3:
                branch = parent_node.add(f"[bright_yellow]{safe_key}[/bright_yellow]")
                self._build_rich_tree(branch, value, level + 1, parent_keys + [key], version_index_map)
                continue

            # Version folders: bold italic yellow/green name, cyan index, do not recurse inside
            if key.startswith("v") and key[1:].isdigit():
                rel_path = os.path.normpath(os.path.join(
                    self.application, self.machine, self.scenario, *parent_keys, key
                ))
                idx = version_index_map[rel_path]
                if is_no_fex:
                    parent_node.add(f"[bold][italic][bright_yellow]{safe_key}[/bright_yellow][/italic][/bold] [bright_cyan][{idx}][/bright_cyan]")
                else:
                    parent_node.add(f"[bold][italic][bright_green]{safe_key}[/bright_green][/italic][/bold] [bright_cyan][{idx}][/bright_cyan]")
                continue
            # All other folders: white
            branch = parent_node.add(f"[white]{safe_key}[/white]")
            self._build_rich_tree(branch, value, level + 1, parent_keys + [key], version_index_map)
    
    def select_model_and_params(self):
        self.print_tree()
        if not self.version_paths:
            print("No version paths found.")
            return None
        idx = int(input("\nEnter the index number of the version path to select: "))
        if idx < 0 or idx >= len(self.version_paths):
            print("Invalid index.")
            return None
        selected_log_path = os.path.join("logs", self.version_paths[idx])
        model_file_path = get_model_pickle_path(selected_log_path)
        config_file_path = get_param_pickle_path(selected_log_path)

        with open(".\\docs\\loaded_model_path.txt", "w") as f:
            f.write(model_file_path)

        with open(".\\docs\\loaded_config_path.txt", "w") as f:
            f.write(config_file_path)

        print(f"\nSelected model file path: {model_file_path}")
        print(f"\nSelected logged config file path: {config_file_path}\n")