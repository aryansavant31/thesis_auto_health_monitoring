import os
import sys
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
        self.set_training_params()
        self.anom_config = self.get_anom_config('SVM', kernel='rbf', C=1.0, gamma='scale')

    def set_training_params(self):
        self.version = 1
        self.is_log = True

        # dataset parameters
        self.batch_size = 50

        self.train_rt   = 0.8
        self.test_rt    = 0.2

    def set_run_params(self):
        self.domain = 'time', # freq-psd, freq-amp
        self.norm_type = 'std'
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
            - 'SVM': `kernel`, `C`, `gamma`
            - 'IF': `n_estimators`, `max_depth`, `min_samples_split`

        """
        anom_config = {}
        anom_config['type'] = anom_type

        if anom_type == 'SVM':
            anom_config['kernel'] = kwargs.get('kernel', 'rbf')
            anom_config['gamma'] = kwargs.get('gamma', 'scale')

        return anom_config  
    
    def _set_ds_types_in_path(self, log_path):
        """
        Takes into account both empty healthy and unhealthy config and sets the path accordingly.
        """
        if self.data_config.unhealthy_config == []:
            log_path = os.path.join(log_path, 'healthy')

        elif self.data_config.unhealthy_config != []:
            log_path = os.path.join(log_path, 'healthy_unhealthy')

        # add ds_subtype to path
        config_str = ''
        
        if self.data_config.healthy_config != []:    
            healthy_config_str_list = []
            for config in self.data_config.healthy_config:
                healthy_type = config[0]
                augments = config[1]

                augment_str = '+'.join(augments) 

                healthy_config_str_list.append(f'{healthy_type}_[{augment_str}]')

            config_str = '_+_'.join(healthy_config_str_list)

        if self.data_config.unhealthy_config != []:
            unhealthy_config_str_list = []
            for config in self.data_config.unhealthy_config:
                unhealthy_type = config[0]
                augments = config[1]

                augment_str = '+'.join(augments) 

                unhealthy_config_str_list.append(f'{unhealthy_type}_[{augment_str}]')

            if config_str:
                config_str += f'_+_{'_+_'.join(unhealthy_config_str_list)}'
            else:
                config_str += '_+_'.join(unhealthy_config_str_list)

        log_path = os.path.join(log_path, config_str)
        return log_path
    
    def get_train_log_path(self, n_nodes, n_components, n_dim):
        """
        Returns the path to store the logs based on data and topology config

        Parameters
        ----------
        data_config : Object
            The data configuration object of class DataConfig
        n_components : int
            The number of components in each datapoint/sample in the dataset
        """
        
        base_path = os.path.join('logs', 
                                f'{self.data_config.application_map[self.data_config.application]}', 
                                f'{self.data_config.machine_type}',
                                f'{self.data_config.scenario}')

        # add node name to path
        train_log_path = os.path.join(base_path, f'node={self.data_config.node_type}')

        # add healthy or healthy_unhealthy config to path
        train_log_path = self._set_ds_types_in_path(train_log_path)

        # add model type to path
        train_log_path = os.path.join(train_log_path, f'anom={self.anom_config['type']}')

        # add datastats to path
        train_log_path = os.path.join(train_log_path, f'{self.data_config.timestep_id}_measures=[{'+'.join(self.data_config.signal_types)}]')

        # add domain type to path
        train_log_path = os.path.join(train_log_path, f'domain={self.domain}')

        # add feature type to path
        fex_types = [fex['type'] for fex in self.fex_configs]

        if fex_types:
            train_log_path = os.path.join(train_log_path, f'fex=[{'+'.join(fex_types)}]')
        else:
            train_log_path = os.path.join(train_log_path, 'fex=_no_fex')

        # add model shape compatibility stats to path
        train_log_path = os.path.join(train_log_path, f'anom(comps)={n_nodes*n_components*n_dim}')

        # add model version to path
        self.train_log_path = os.path.join(train_log_path, f'v{self.version}')

        return self.train_log_path
        
    def get_test_log_path(self):
        
        test_log_path = os.path.join(self.train_log_path, 'test')

        # add healthy or healthy_unhealthy config to path
        test_log_path = self._set_ds_types_in_path(test_log_path)

        # add timestep id to path
        test_log_path = os.path.join(test_log_path, f'{self.data_config.timestep_id}')

        # add test version to path
        test_log_path = os.path.join(test_log_path, f'test_v0')

        return test_log_path
    
    def _remove_version(self):
        """
        Removes the version from the log path.
        """
        if os.path.exists(self.train_log_path):
            user_input = input(f"Are you sure you want to remove the version {self.version} from the log path {self.train_log_path}? (y/n): ")
            if user_input.lower() == 'y':
                shutil.rmtree(self.train_log_path)
                print(f"Removed version {self.version} from the log path {self.train_log_path}.")

            else:
                print(f"Operation cancelled. Version {self.version} still remains.")

    
    def _get_next_version(self):
        parent_dir = os.path.dirname(self.train_log_path)

        # List all folders in parent_dir that match 'v<number>'
        folders = [f for f in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, f))]
        v_folders = [f for f in folders if re.match(r'^v\d+$', f)]

        if v_folders:
            # Extract numbers and find the max
            max_v = max(int(f[1:]) for f in v_folders)
            new_v = f'v{max_v + 1}'
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
            print(f"Version {self.version} already exists in the log path '{self.train_log_path}'.")
            user_input = input("(a) Overwrite exsiting version, (b) create new version, (c) stop training (Choose 'a', 'b' or 'c'):  ")

            if user_input.lower() == 'a':
                self._remove_version()

            elif user_input.lower() == 'b':
                self.log_path = self._get_next_version()

            elif user_input.lower() == 'c':
                print("Stopped training.")
                sys.exit()  # Exit the program gracefully
 
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

        # Green up to and including framework
        tree = Tree(f"[green]{self.application}[/green]")
        machine_node = tree.add(f"[green]{self.machine}[/green]")
        scenario_node = machine_node.add(f"[green]{self.scenario}[/green]")
        
        self._build_rich_tree(scenario_node, self.structure, 0, [], version_index_map)
        console.print(tree)
        print("\nAvailable version paths:")
        for idx, vpath in enumerate(self.version_paths):
            print(f"{idx}: logs/{vpath}")

    def _build_rich_tree(self, parent_node, structure, level, parent_keys, version_index_map):
        current_path = [self.application, self.machine, self.scenario, self.framework] + parent_keys
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
            # For directed graph, make the model folder under framework yellow
            if level == 2:
                branch = parent_node.add(f"[bright_yellow]{safe_key}[/bright_yellow]")
                self._build_rich_tree(branch, value, level + 1, parent_keys + [key], version_index_map)
                continue

            # Version folders: bold italic yellow/green name, cyan index, do not recurse inside
            if key.startswith("v") and key[1:].isdigit():
                rel_path = os.path.normpath(os.path.join(
                    self.application, self.machine, self.scenario, self.framework, *parent_keys, key
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
    
    def select_ckpt_and_params(self):
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

        with open(".\\docs\\loaded_ckpt_path.txt", "w") as f:
            f.write(model_file_path)

        with open(".\\docs\\loaded_config_path.txt", "w") as f:
            f.write(config_file_path)

        print(f"\nSelected .ckpt file path: {model_file_path}")
        print(f"\nSelected logged config file path: {config_file_path}\n")