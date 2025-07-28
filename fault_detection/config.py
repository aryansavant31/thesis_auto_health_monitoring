import os
import sys
FAULT_DETECTION_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(FAULT_DETECTION_DIR, "logs")

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from feature_extraction.extractor import get_fex_config
from data.config import DataConfig
import shutil
import re
from pathlib import Path
from rich.tree import Tree
from rich.console import Console
import pickle


class PredictAnomalyDetectorConfig:
    def __init__(self):
        self.helper = HelperClass()
        self.data_config = DataConfig()
        self.load_log_config()
        self.set_predict_params()
        self.set_ckpt_path()

        self.train_log_path = self.log_config.train_log_path
        self.node_type = self.log_config.node_type

        self.selected_model_num = f"{self.node_type}_fault_detector_{self.log_config.model_num}"

    def set_predict_params(self):
        self.version = 1


    def set_ckpt_path(self):
         with open(".\\docs\\loaded_model_path.txt", "r") as f:
            self.ckpt_path = f.read() 

    def load_log_config(self):
        self.log_config = TrainAnomalyDetectorConfig()

        with open(".\\docs\\loaded_config_path.txt", "r") as f:
            log_config_path = f.read()

        if not os.path.exists(log_config_path):
            raise ValueError(f"\nThe parameter file does not exists")
        
        with open(log_config_path, 'rb') as f:
            self.log_config.__dict__.update(pickle.load(f))
    
    def get_custom_test_log_path(self):
        """
        Sets the log path for the predict run.
        """
        self.data_config.set_custom_test_dataset()

        test_num_path = self.train_log_path.replace(f"{os.sep}train{os.sep}", f"{os.sep}custom_test{os.sep}")
        self.test_log_path = os.path.join(test_num_path, f'{self.node_type}_custom_test_{self.version}')

        # add healthy or healthy_unhealthy config to path
        test_num_path = self.helper.set_ds_types_in_path(self.data_config, test_num_path)

        # add timestep_id to path
        self.test_id = os.path.join(test_num_path, f'T{self.data_config.window_length}')

        # # add version
        # self.test_id = os.path.join(test_num_path, f'test_num_{self.version}')

        # check if version already exists
        self.check_if_version_exists(self.test_log_path, 'custom_test')

        return self.test_log_path
    
    def get_predict_log_path(self):
        """
        Sets the log path for the predict run.
        """
        self.data_config.set_predict_dataset()

        predict_num_path = self.train_log_path.replace(f"{os.sep}train{os.sep}", f"{os.sep}predict{os.sep}")
        self.predict_log_path = os.path.join(predict_num_path, f'{self.node_type}_predict_{self.version}')

        # add healthy or healthy_unhealthy config to path
        predict_num_path = self.helper.set_ds_types_in_path(self.data_config, predict_num_path)

        # add timestep_id to path
        self.predict_id = os.path.join(predict_num_path, f'T{self.data_config.window_length}')

        # # add version
        # self.predict_id = os.path.join(predict_num_path, f'predict_num_{self.version}')

        # check if version already exists
        self.check_if_version_exists(self.predict_log_path, 'predict')

        return self.predict_log_path
    
    def save_custom_test_params(self):
        """
        Saves the test parameters in the test log path.
        """
        if not os.path.exists(self.test_log_path):
            os.makedirs(self.test_log_path)

        config_path = os.path.join(self.test_log_path, f'custom_test_config.pkl')
        with open(config_path, 'wb') as f:
            pickle.dump(self.__dict__, f)

        test_num_path = os.path.join(self.test_log_path, f'{self.node_type}_custom_test_{self.version}.txt')
        with open(test_num_path, 'w') as f:
            f.write(self.test_id)

        print(f"Custom test parameters saved to {self.test_log_path}.")

    def save_predict_params(self):
        """
        Saves the predict parameters in the predict log path.
        """
        if not os.path.exists(self.predict_log_path):
            os.makedirs(self.predict_log_path)

        config_path = os.path.join(self.predict_log_path, f'predict_config.pkl')
        with open(config_path, 'wb') as f:
            pickle.dump(self.__dict__, f)

        predict_num_path = os.path.join(self.predict_log_path, f'{self.node_type}_predict_{self.version}.txt')
        with open(predict_num_path, 'w') as f:
            f.write(self.predict_id)
            
        print(f"Predict parameters saved to {self.predict_log_path}.")

    def _remove_version(self, log_path):
        """
        Removes the version from the log path.
        """
        if os.path.exists(log_path):
            user_input = input(f"Are you sure you want to remove the version {self.version} from the log path {log_path}? (y/n): ")
            if user_input.lower() == 'y':
                shutil.rmtree(log_path)
                print(f"Overwrote version {self.version} from the log path {log_path}.")

            else:
                print(f"Operation cancelled. Version {self.version} still remains.")

    def _get_next_version(self, log_path, run_type):
        parent_dir = os.path.dirname(log_path)

        # List all folders in parent_dir that match 'fault_detector_<number>'
        folders = [f for f in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, f))]
        model_folders = [f for f in folders if re.match(fr'.*{run_type}_\d+$', f)]

        if model_folders:
            # Extract numbers and find the max
            max_model = max(int(f.split('_')[-1]) for f in model_folders)
            self.version = max_model + 1
            new_model = f'{self.node_type}_{run_type}_{self.version}'
        else:
            new_model = f'{self.node_type}_{run_type}_1'  # If no v folders exist

        return os.path.join(parent_dir, new_model)
    
    def check_if_version_exists(self, log_path, run_type):
        """
        Checks if the version already exists in the log path.

        Parameters
        ----------
        log_path : str
            The path where the test and predict logs are stored.
        """
 
        if os.path.isdir(log_path):
            print(f"\n{run_type} number {self.version} for already exists for {self.selected_model_num} in the log path '{log_path}'.")
            user_input = input(f"(a) Overwrite exsiting version, (b) create new version, (c) stop {run_type} (Choose 'a', 'b' or 'c'):  ")

            if user_input.lower() == 'a':
                self._remove_version(log_path)

            elif user_input.lower() == 'b':
                if run_type == 'custom_test':
                    self.test_log_path = self._get_next_version(log_path, run_type)
                elif run_type == 'predict':
                    self.predict_log_path = self._get_next_version(log_path, run_type)

            elif user_input.lower() == 'c':
                print("Stopped operation.")
                sys.exit()  # Exit the program gracefully


class TrainAnomalyDetectorConfig:
    def __init__(self):
        self.data_config = DataConfig()
        self.helper = HelperClass()
        self.data_config.set_train_dataset()

        self.set_training_params()
        self.set_run_params()

        self.node_type = f"({'+'.join(self.data_config.node_type)})"

        self.anom_config = self.get_anom_config('IF', n_estimators=1000, contam=0.5)

    def set_training_params(self):
        self.model_num = 2
        self.is_log = False

        # dataset parameters
        self.batch_size = 50

        self.train_rt   = 0.8
        self.test_rt    = 0.2

    def set_run_params(self):
        self.domain = 'time'   # freq-psd, freq-amp
        self.norm_type = None
        self.fex_configs = [
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
        model_path = os.path.join(base_path, 'train', f'{self.node_type}')

        # get train_log_path
        self.train_log_path = os.path.join(model_path, f"{self.node_type}_fault_detector_{self.model_num}")

        # add healthy or healthy_unhealthy config to path
        model_path = self.helper.set_ds_types_in_path(self.data_config, model_path)

        # add model type to path
        model_path = os.path.join(model_path, f'[anom] {self.anom_config['type']}')

        # add datastats to path
        model_path = os.path.join(model_path, f"T{self.data_config.window_length} [{', '.join(self.data_config.signal_types)}]")

        # add domain type to path
        model_path = os.path.join(model_path, f'{self.domain}')

        # add feature type to path
        fex_types = [fex['type'] for fex in self.fex_configs]

        if fex_types:
            model_path = os.path.join(model_path, f"(anom) [{'+'.join(fex_types)}]")
        else:
            model_path = os.path.join(model_path, '(anom) [no_fex]')

        # add model shape compatibility stats to path
        self.model_id = os.path.join(model_path, f'anom (comps = {n_components*n_dim})')

        # # add model version to path
        # self.model_path = os.path.join(model_path, f'v{self.version}')

        # check if version already exists
        self.check_if_version_exists()

        return self.train_log_path
        
    def get_test_log_path(self):
        
        test_log_path = os.path.join(self.train_log_path, 'test')

        # # add healthy or healthy_unhealthy config to path
        # test_log_path = self._set_ds_types_in_path(test_log_path)

        # # add timestep id to path
        # test_log_path = os.path.join(test_log_path, f'T{self.data_config.window_length}')

        # # add test version to path
        # test_log_path = os.path.join(test_log_path, f'test_v0')

        return test_log_path
    
    def save_params(self):
        """
        Saves the training parameters.
        """
        if not os.path.exists(self.train_log_path):
            os.makedirs(self.train_log_path)

        config_path = os.path.join(self.train_log_path, f'train_config.pkl')
        with open(config_path, 'wb') as f:
            pickle.dump(self.__dict__, f)

        model_path = os.path.join(self.train_log_path, f'{self.node_type}_fault_detector_{self.model_num}.txt')
        with open(model_path, 'w') as f:
            f.write(self.model_id)

        print(f"Model parameters saved to {self.train_log_path}.")

    def _remove_version(self):
        """
        Removes the model number from the log path.
        """
        if os.path.exists(self.train_log_path):
            user_input = input(f"Are you sure you want to remove the model number {self.model_num} from the log path {self.train_log_path}? (y/n): ")
            if user_input.lower() == 'y':
                shutil.rmtree(self.train_log_path)
                print(f"Overwrote exsiting model number '{self.model_num}' from the log path {self.train_log_path}.")

            else:
                print(f"Operation cancelled. Version 'v{self.model_num}' still remains.")

    
    def _get_next_version(self):
        parent_dir = os.path.dirname(self.train_log_path)

        # List all folders in parent_dir that match 'v<number>'
        model_folders = os.listdir(parent_dir)

        print("model_folders:", model_folders)
        if model_folders:
            # Extract numbers and find the max
            max_model = max(int(f.split('_')[-1]) for f in model_folders)
            print("max_model:", max_model)
            self.model_num = max_model + 1
            new_model = f"{self.node_type}_fault_detector_{self.model_num}"
            print(f"Next model number folder will be: {new_model}")
        else:
            new_model = f"{self.node_type}_fault_detector_1"

        return os.path.join(parent_dir, new_model)
    

    def check_if_version_exists(self):
        """
        Checks if the model_num already exists in the log path.

        Parameters
        ----------
        log_path : str
            The path where the logs are stored. It can be for nri model or skeleton graph model.
        """
        if os.path.isdir(self.train_log_path):
            print(f"'Version {self.model_num}' already exists in the log path '{self.train_log_path}'.")
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
    def __init__(self, application=None, machine=None, scenario=None, run_type='train', logs_dir=LOGS_DIR):
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

        self.run_type = run_type  # 'train' or 'predict'

        if self.run_type == 'train':
            self.file_name = 'fault_detector'
        elif self.run_type == 'custom_test':
            self.file_name = 'custom_test'
        elif self.run_type == 'predict':
            self.file_name = 'predict'

        self.structure = {}
        self.version_paths = []
        self.version_txt_files = []
        self._build_structure_from_txt()

    def _build_structure_from_txt(self):
        """
        Build the tree structure from all model_x.txt files under the framework directory.
        """
        base = self.logs_dir / self.application / self.machine / self.scenario / self.run_type 
        os.makedirs(base, exist_ok=True)  # Ensure the base directory exists

        txt_files = list(base.rglob(f"*{self.file_name}_*.txt"))
        self.version_txt_files = txt_files

        path_map = {}
        for txt_file in txt_files:
            try:
                with open(txt_file, "r") as f:
                    model_path = f.read().strip()
            except Exception as e:
                print(f"Could not read {txt_file}: {e}")
                continue
            # Remove base logs dir and split by os.sep
            rel_path = os.path.relpath(model_path, str(self.logs_dir))
            path_parts = rel_path.split(os.sep)
            # Only keep the parts after framework (skip first 4: app, machine, scenario, train)
            path_parts = path_parts[4:]
            model_name = txt_file.stem  # e.g., model_3
            key = tuple(path_parts)
            if key not in path_map:
                path_map[key] = []
            path_map[key].append((str(txt_file), model_name))

        structure = {}
        for path_parts, versions in path_map.items():
            node = structure
            # Add all parts as nodes, including the last one
            for part in path_parts:
                node = node.setdefault(part, {})
            if "_versions" not in node:
                node["_versions"] = []
            # Sort versions by edge_estimator number before assigning vnum
            sorted_versions = sorted(
                versions,
                key=lambda x: int(re.search(fr'.*{self.file_name}_\d+\.(\d+)', x[1]).group(1)) if re.search(fr'.*{self.file_name}_\d+\.(\d+)', x[1]) else 0
            )
            for idx, (txt_file, model_name) in enumerate(sorted_versions, 1):
                node["_versions"].append({
                    "model_name": model_name,
                    "txt_file": txt_file,
                    "vnum": idx,
                    "full_path": path_parts,
                })
                self.version_paths.append(txt_file)
        self.structure = structure

    def print_tree(self):
        console = Console()
        # Green up to and including framework
        tree = Tree(f"[green]{self.application}[/green]")
        machine_node = tree.add(f"[green]{self.machine}[/green]")
        
        if self.run_type == 'train':
            bracket = '(trained models)'
        elif self.run_type == 'custom_test':
            bracket = '(custom tested models)'
        elif self.run_type == 'predict':
            bracket = '(predicted models)'

        scenario_node = machine_node.add(f"[green]{self.scenario}[/green] [magenta]{bracket}[/magenta]")
  
        self._build_rich_tree(scenario_node, self.structure, 0, [])
        console.print(tree)
        print("\nAvailable version paths:")
        for idx, txt_file in enumerate(self.version_paths):
            print(f"{idx}: {os.path.dirname(txt_file)}")

    def _build_rich_tree(self, parent_node, structure, level, parent_keys):
        is_no_fex = any("(anom) [no_fex]" in k for k in parent_keys)
       
        if self.run_type == 'train':
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
        elif self.run_type in ['custom_test', 'predict']:
            label_map = {
                0: "<node_name>",
                1: "<trained_model>",
                2: "<ds_type>",
                3: "<ds_subtype>",
                4: "ds_stats",
                5: "<versions>"
            }
                    
        added_labels = set()
        # Add all keys except _versions first
        for key, value in structure.items():
            if key == "_versions":
                continue
            safe_key = key.replace('[', '\\[')
            # Add blue label if at the correct level and not already added
            if level in label_map and label_map[level] not in added_labels:
                parent_node.add(f"[blue]{label_map[level]}[/blue]")
                added_labels.add(label_map[level])
            # For directed graph, make the model folder under framework yellow
            if self.run_type == "train" and level == 3:
                branch = parent_node.add(f"[bright_yellow]{safe_key}[/bright_yellow]")
                self._build_rich_tree(branch, value, level + 1, parent_keys + [key])
                continue
            if self.run_type in ['custom_test', 'predict'] and level == 1:
                branch = parent_node.add(f"[bright_yellow]{safe_key}[/bright_yellow]")
                self._build_rich_tree(branch, value, level + 1, parent_keys + [key])
                continue
            
            # All other folders: white
            branch = parent_node.add(f"[white]{safe_key}[/white]")
            self._build_rich_tree(branch, value, level + 1, parent_keys + [key])

        # Now add <versions> label and version nodes if present
        if "_versions" in structure:
            if "<versions>" not in added_labels:
                parent_node.add(f"[blue]<versions>[/blue]")
                added_labels.add("<versions>")
            # --- SORT VERSIONS BY MODEL NUMBER ---
            sorted_versions = sorted(
                structure["_versions"],
                key=lambda v: int(re.search(fr'.*{self.file_name}_\d+\.(\d+)', v['model_name']).group(1)) if re.search(fr'.*{self.file_name}_\d+\.(\d+)', v['model_name']) else 0
            )
            for v in sorted_versions:
                model_disp = f"{v['model_name']} (v{v['vnum']})"
                safe_model_disp = model_disp.replace('[', '\\[')
                idx = self.version_paths.index(v["txt_file"])
                if is_no_fex:
                    parent_node.add(f"[bright_yellow]{safe_model_disp}[/bright_yellow] [bright_cyan][{idx}][/bright_cyan]")
                else:
                    parent_node.add(f"[bright_green]{safe_model_disp}[/bright_green] [bright_cyan][{idx}][/bright_cyan]")

    def select_model_and_params(self):
        self.print_tree()
        if not self.version_paths:
            print("No version paths found.")
            return None
        
        if self.run_type == 'train':
            idx = int(input("\nEnter the index number of the version path to select: "))
            if idx < 0 or idx >= len(self.version_paths):
                print("Invalid index.")
                return None
            selected_log_path = os.path.dirname(self.version_paths[idx])
            # Use the directory containing model_x.txt as the log path
        
            model_file_path = get_model_pickle_path(selected_log_path)
            config_file_path = get_param_pickle_path(selected_log_path)

            with open(".\\docs\\loaded_model_path.txt", "w") as f:
                f.write(model_file_path)

            with open(".\\docs\\loaded_config_path.txt", "w") as f:
                f.write(config_file_path)

            print(f"\nSelected model file path: {model_file_path}")
            print(f"\nSelected logged config file path: {config_file_path}\n")

    
if __name__ == "__main__":
    user_text = "To view/select trained fault detection models, type (a)\nTo view custom tested models, type (b)\nTo view predicted models, type (c)\nEnter input: "
    user_input = input(user_text).strip("'\"")
    if user_input.lower() == 'a':
        run_type = 'train'
    elif user_input.lower() == 'b':
        run_type = 'custom_test'
    elif user_input.lower() == 'c':
        run_type = 'predict'
    else:
        raise ValueError("Invalid input. Please enter 'a', 'b', or 'c'.")
    model_selector = SelectFaultDetectionModel(run_type=run_type)
    model_selector.select_model_and_params()