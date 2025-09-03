import os, sys

# ROOT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# sys.path.insert(0, ROOT_DIR) if ROOT_DIR not in sys.path else None

SETTINGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
# sys.path.insert(0, SETTINGS_DIR) if SETTINGS_DIR not in sys.path else None

FDET_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(FDET_DIR, "logs")

# other imports
import pickle
import shutil
from pathlib import Path
import re
import itertools
from copy import deepcopy
from collections import defaultdict

# global imports
from data.config import DataConfig, DataSweep

# local imports
from .train_config import AnomalyDetectorTrainConfig, AnomalyDetectorTrainSweep
from .infer_config import AnomalyDetectorInferConfig, AnomalyDetectorInferSweep


class AnomalyDetectorTrainManager(AnomalyDetectorTrainConfig):
    def __init__(self, data_config:DataConfig, train_sweep_num=0):
        super().__init__(data_config)
        self.helper = HelperClass()
        self.train_sweep_num = train_sweep_num
        
    def get_train_log_path(self, n_components, n_dim, always_next_version=False):
        """
        Returns the path to store the logs based on data and topology config

        Parameters
        ----------
        n_components : int
            The number of components in each datapoint/sample in the dataset
        n_dim : int
            The number of dimensions in each component in the dataset
        always_next_version : bool, optional
            Whether to always create the next version if a version already exists, by default False
        """
        self.always_next_version = always_next_version

        self.node_type = f"{self.data_config.signal_types['node_group_name']}"
        self.signal_group = f"{self.data_config.signal_types['signal_group_name']}"
        self.set_id = self.data_config.set_id
        
        base_path = os.path.join(LOGS_DIR, 
                                f'{self.data_config.application_map[self.data_config.application]}', 
                                f'{self.data_config.machine_type}',
                                f'{self.data_config.scenario}')

        # add node name to path
        model_path = os.path.join(base_path, 'train', f'{self.node_type}', f'{self.signal_group}', f'set_{self.set_id}')

        # model name
        self.model_name = f"[{self.node_type}_({self.signal_group}+{self.set_id})]-{self.anom_config['anom_type']}_fdet"
        # get train_log_path
        self.train_log_path = os.path.join(model_path, self.anom_config['anom_type'], f"tswp_{self.train_sweep_num}", f"{self.model_name}_{self.model_num}")

        # add healthy or healthy_unhealthy config to path
        model_path = self.helper.set_ds_types_in_path(self.data_config, model_path)

        # add model type to path
        model_path = os.path.join(model_path, f'[anom] {self.anom_config['anom_type']}')

        # add datastats to path
        signal_types_str = ', '.join(
            f"{node_type}: ({', '.join(signal_types_list)})" for node_type, signal_types_list in self.data_config.signal_types['group'].items()
        )
        model_path = os.path.join(model_path, f"{signal_types_str}")

        # add timestepp id to path
        model_path = os.path.join(model_path, f'T{self.data_config.window_length}')

        # add model hparams to path
        # model_path = os.path.join(model_path, f"[{self.anom_config['anom_type']}] ({', '.join([f'{key}={value}' for key, value in self.anom_config.items() if key != 'type'])})")

        # add domain type to path
        model_path = os.path.join(model_path, f'{self.domain_config['type']}')

        # add feature type to path
        if self.feat_configs:
            feat_types = []
            for feat_config in self.feat_configs:
                if feat_config['type'] == 'from_ranks':
                    feat_types.append(f"from_ranks, ({' + '.join(feat_config['feat_list'])})")
                else:
                    feat_types.append(feat_config['type'])
        else:
            feat_types = ['no_feat']
            
        reduc_type = self.reduc_config['type'] if self.reduc_config else 'no_reduc'

        model_path = os.path.join(model_path, f"(anom) [{' + '.join(feat_types)}] [{reduc_type}]")
        
        # add train sweep number to path
        model_path = os.path.join(model_path, f'tswp_{self.train_sweep_num}')

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

        model_path = os.path.join(self.train_log_path, f'{self.model_name}_{self.model_num}.txt')
        with open(model_path, 'w') as f:
            f.write(self.model_id)

        print(f"Model parameters saved to {self.train_log_path}.")

    def _remove_version(self):
        if os.path.exists(self.train_log_path):
            user_input = input(f"Are you sure you want to remove the model number {self.model_num} from the log path {self.train_log_path}? (y/n): ")
            if user_input.lower() == 'y':
                shutil.rmtree(self.train_log_path)
                print(f"Overwrote exsiting model number '{self.model_num}' from the log path {self.train_log_path}.")

            else:
                print(f"Operation cancelled. Version 'v{self.model_num}' still remains.")
                sys.exit()  # Exit the program gracefully   

    def _get_next_version(self):
        sweep_dir = os.path.dirname(self.train_log_path)
        parent_dir = os.path.dirname(sweep_dir)

        model_folders = []
        for root, dirs, files in os.walk(parent_dir):
            # Only look at immediate subfolders of parent_dir
            if os.path.dirname(root) == parent_dir:
                for d in dirs:
                    model_folders.append(d)

        if model_folders:
            # Extract numbers and find the max
            max_model = max(int(f.split('_')[-1]) for f in model_folders)
            self.model_num = max_model + 1
            new_model = f"{self.model_name}_{self.model_num}"
            print(f"Next model number folder will be: {new_model}")
        else:
            self.model_num = 1
            new_model = f"{self.model_name}_{self.model_num}"

        return os.path.join(sweep_dir, new_model)
    

    def check_if_version_exists(self):
        """
        Checks if the model_num already exists in the log path.
        
        """
        if self.always_next_version:
            self.train_log_path = self._get_next_version()
            return
        
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

class AnomalyDetectorInferManager(AnomalyDetectorInferConfig):
    def __init__(self, data_config:DataConfig, run_type, infer_sweep_num=0, selected_model_path=None):
        super().__init__(data_config, run_type, selected_model_path)

        # check if data and model are compatible
        if not self._is_data_model_match():
            raise ValueError("Data and model configurations are not compatible. Please check the configurations.")

        self.helper = HelperClass()
        self.run_type = run_type
        self.infer_sweep_num = infer_sweep_num
        
        self.train_log_path = self.log_config.train_log_path
        self.node_type = self.log_config.node_type
        self.signal_group = self.log_config.signal_group

        self.selected_model_num = os.path.basename(self.train_log_path)

    def _is_data_model_match(self):
        #model_id = os.path.basename(os.path.dirname(model_path)).split('-')[0].strip('[]')

        node_group_model = self.log_config.data_config.signal_types['node_group_name']
        signal_group_model = self.log_config.data_config.signal_types['signal_group_name']
        set_id_model = self.log_config.data_config.set_id
        window_length_model = self.log_config.data_config.window_length
        stride_model = self.log_config.data_config.stride

        is_node_match = node_group_model == self.data_config.signal_types['node_group_name']
        is_signal_match = signal_group_model == self.data_config.signal_types['signal_group_name']
        is_setid_match = set_id_model == self.data_config.set_id
        is_window_match = window_length_model == self.data_config.window_length
        is_stride_match = stride_model == self.data_config.stride

        if is_node_match and is_signal_match and is_setid_match and is_window_match and is_stride_match:
            return True
        else:
            print(f"\n> Incompatible model found: {os.path.basename(os.path.dirname(self.selected_model_path))}")
            print(f"Incompatible parameters:")
            if not is_node_match:
                print(f"  - Node group mismatch: Model({node_group_model}) != Data({self.data_config.signal_types['node_group_name']})")
            if not is_signal_match:
                print(f"  - Signal group mismatch: Model({signal_group_model}) != Data({self.data_config.signal_types['signal_group_name']})")
            if not is_setid_match:
                print(f"  - Set ID mismatch: Model({set_id_model}) != Data({self.data_config.set_id})")
            if not is_window_match:
                print(f"  - Window length mismatch: Model({window_length_model}) != Data({self.data_config.window_length})")
            if not is_stride_match:
                print(f"  - Stride mismatch: Model({stride_model}) != Data({self.data_config.stride})")
            return False
    
    def get_infer_log_path(self, always_next_version=False):
        """
        Sets the log path for the run.
        """
        self.always_next_version = always_next_version

        parts = self.train_log_path.split(os.sep)
        train_idx = parts.index('train')
        node = parts[train_idx + 1]
        signal_group = parts[train_idx + 2]
        set_id = parts[train_idx + 3]
        anom_type = parts[train_idx + 4]
        model_name = parts[-1]

        infer_num_path = os.path.join(
            LOGS_DIR,
            *parts[train_idx-3:train_idx],
            self.run_type,
            node, signal_group, set_id,
            f'iswp_{self.infer_sweep_num}', 
        )

        self.infer_log_path = os.path.join(infer_num_path, anom_type, model_name, f'{self.run_type}_{self.version}')
        
        # add healthy or healthy_unhealthy config to path
        infer_num_path = self.helper.set_ds_types_in_path(self.data_config, infer_num_path)

        # add model type to path
        infer_num_path = os.path.join(infer_num_path, f'[anom] {anom_type}', model_name)

        # add timestep_id to path
        self.infer_id = os.path.join(infer_num_path, f'T{self.data_config.window_length}')

        # # add version
        # self.infer_id = os.path.join(infer_num_path, f'infer_num_{self.version}')

        # check if version already exists
        self.check_if_version_exists()

        return self.infer_log_path
    
    # def get_predict_log_path(self):
    #     """
    #     Sets the log path for the predict run.
    #     """
    #     self.data_config.set_predict_dataset()

    #     predict_num_path = self.train_log_path.replace(f"{os.sep}train{os.sep}", f"{os.sep}predict{os.sep}")
    #     self.predict_log_path = os.path.join(predict_num_path, f'{self.node_type}_predict_{self.version}')

    #     # add healthy or healthy_unhealthy config to path
    #     predict_num_path = self.helper.set_ds_types_in_path(self.data_config, predict_num_path)

    #     # add timestep_id to path
    #     self.predict_id = os.path.join(predict_num_path, f'T{self.data_config.window_length}')

    #     # # add version
    #     # self.predict_id = os.path.join(predict_num_path, f'predict_num_{self.version}')

    #     # check if version already exists
    #     self.check_if_version_exists(self.predict_log_path, 'predict')

    #     return self.predict_log_path
    
    def save_infer_params(self):
        """
        Saves the infer parameters in the infer log path.
        """
        if not os.path.exists(self.infer_log_path):
            os.makedirs(self.infer_log_path)

        config_path = os.path.join(self.infer_log_path, f'{self.run_type}_config.pkl')
        with open(config_path, 'wb') as f:
            pickle.dump(self.__dict__, f)

        infer_num_path = os.path.join(self.infer_log_path, f'{self.run_type}_{self.version}.txt')
        with open(infer_num_path, 'w') as f:
            f.write(self.infer_id)

        print(f"{self.run_type.capitalize()} parameters saved to {self.infer_log_path}.")


    def _remove_version(self):
        """
        Removes the version from the log path.
        """
        if os.path.exists(self.infer_log_path):
            user_input = input(f"Are you sure you want to remove the version {self.version} from the log path {self.infer_log_path}? (y/n): ")
            if user_input.lower() == 'y':
                shutil.rmtree(self.infer_log_path)
                print(f"Overwrote version {self.version} from the log path {self.infer_log_path}.")

            else:
                print(f"Operation cancelled. Version {self.version} still remains.")
                sys.exit()  # Exit the program gracefully

    def _get_next_version(self):
        parent_dir = os.path.dirname(self.infer_log_path)

        os.makedirs(parent_dir, exist_ok=True)
        # List all folders in parent_dir that match 'fault_detector_<number>'
        #folders = [f for f in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, f))]
        model_folders = [f for f in os.listdir(parent_dir) if f.startswith(f'{self.run_type}_')]

        # model_folders = []
        # for root, dirs, files in os.walk(parent_dir):
        #     # Only look at immediate subfolders of parent_dir
        #     if os.path.dirname(root) == parent_dir:
        #         for d in dirs:
        #             model_folders.append(d)

        if model_folders:
            # Extract numbers and find the max
            max_model = max(int(f.split('_')[-1]) for f in model_folders)
            self.version = max_model + 1
            new_model = f'{self.run_type}_{self.version}'
            print(f"Next fault detection infer folder will be: {new_model}")
        else:
            self.version = 1
            new_model = f'{self.run_type}_{self.version}'  # If no v folders exist

        return os.path.join(parent_dir, new_model)
    
    def check_if_version_exists(self):
        """
        Checks if the version already exists in the log path.
        """
        if self.always_next_version:
            self.infer_log_path = self._get_next_version()
            return
 
        if os.path.isdir(self.infer_log_path):
            print(f"\n{self.run_type} number {self.version} for already exists for {self.selected_model_num} in the log path '{self.infer_log_path}'.")
            user_input = input(f"(a) Overwrite exsiting version, (b) create new version, (c) stop {self.run_type} (Choose 'a', 'b' or 'c'):  ")

            if user_input.lower() == 'a':
                self._remove_version()

            elif user_input.lower() == 'b':
                self.infer_log_path = self._get_next_version()

            elif user_input.lower() == 'c':
                print("Stopped operation.")
                sys.exit()  # Exit the program gracefully  

class AnomalyDetectorTrainSweepManager(AnomalyDetectorTrainSweep):
    def __init__(self, data_configs:list):
        self.data_configs = data_configs

    def get_sweep_configs(self):
        """
        Generate all possible combinations of parameters for sweeping.
            
        Returns
        -------
        list
            List of AnomalyDetectorTrainConfig objects with different parameter combinations
        """
        train_configs = []
        idx_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        idx = -1
        
        for data_config in self.data_configs: 

            AnomalyDetectorTrainSweep.__init__(self, data_config)
            
            node_group_change = data_config.signal_types['node_group_name'] != train_configs[-1].data_config.signal_types['node_group_name'] if train_configs else False
            signal_group_change = data_config.signal_types['signal_group_name'] != train_configs[-1].data_config.signal_types['signal_group_name'] if train_configs else False
            set_id_change = data_config.set_id != train_configs[-1].data_config.set_id if train_configs else False

            # # First level idx reset
            # if signal_types_change or set_id_change:
            #     idx = -1 
            #     idx_dict = {}
            
            # Convert sweep config to dictionary
            sweep_dict = self._to_dict()
            
            # Get all parameter names and their values
            param_names = list(sweep_dict.keys())
            param_values = list(sweep_dict.values())
            
            # Generate all combinations using cartesian product
            combinations = list(itertools.product(*param_values))
            
            # Create configs for each combination
            for combo in combinations:
                anom_type_change = combo[param_names.index('anom_config')]['anom_type'] != train_configs[-1].anom_config['anom_type'] if train_configs else False

                # # Second level reset idx
                # if anom_type_change:
                #     idx = idx_dict.get(combo[param_names.index('anom_config')]['anom_type'], -1)
                if node_group_change or signal_group_change or set_id_change or anom_type_change:
                    idx = (
                        idx_dict.get(data_config.signal_types['node_group_name'], {})
                                .get(data_config.signal_types['signal_group_name'], {})
                                .get(data_config.set_id, {})
                                .get(combo[param_names.index('anom_config')]['anom_type'], -1)
                    )
                idx += 1

                # Create base config
                train_config = AnomalyDetectorTrainManager(self.data_config, train_sweep_num=self.train_sweep_num)

                # Update parameters based on current combination
                for param_name, param_value in zip(param_names, combo):
                    setattr(train_config, param_name, param_value)
                
                # Update model number 
                _ = train_config.get_train_log_path(n_components=0, n_dim=0, always_next_version=True)  # Dummy values for n_components and n_dim
                train_config.model_num = train_config.model_num + idx

                idx_dict[
                    data_config.signal_types['node_group_name']
                ][
                    data_config.signal_types['signal_group_name']
                ][
                    data_config.set_id
                ][
                    combo[param_names.index('anom_config')]['anom_type']
                ] = idx

                # Regenerate hyperparameters after updating parameters
                train_config.hparams = train_config.get_hparams()

                train_configs.append(train_config)
                        
        return train_configs
    
    def _to_dict(self):
        """
        Convert sweep config attributes to dictionary, excluding private methods and non-list attributes.
        """
        sweep_dict = {}
        for attr_name in dir(self):
            if not attr_name.startswith('_') and not callable(getattr(self, attr_name)):
                attr_value = getattr(self, attr_name)
                if isinstance(attr_value, list):
                    if attr_name != 'data_configs': 
                        sweep_dict[attr_name] = attr_value
        return sweep_dict
    
    def get_total_combinations(self):
        """
        Get the total number of parameter combinations.
        """
        sweep_dict = self._to_dict()
        total = 1
        for values in sweep_dict.values():
            total *= len(values)
        return total
    
    def print_sweep_summary(self):
        """
        Print a summary of the parameter sweep.
        """
        sweep_dict = self._to_dict()
        print(f"\nParameter Sweep Summary:")
        print(f"Total combinations: {self.get_total_combinations()}")
        print("\nParameters and their values:")
        for param_name, param_values in sweep_dict.items():
            print(f"  {param_name}: {len(param_values)} values -> {param_values}") 

class AnomalyDetectorInferSweepManager(AnomalyDetectorInferSweep):
    def __init__(self, data_configs:list, run_type):
        self.data_configs = data_configs
        self.run_type = run_type

    def get_sweep_configs(self):
        """
        Generate all possible combinations of parameters for sweeping.
            
        Returns
        -------
        list
            List of AnomalyDetectorTrainConfig objects with different parameter combinations
        """
        infer_configs = []
        idx_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        idx = -1
        
        for data_config in self.data_configs: 

            AnomalyDetectorInferSweep.__init__(self, data_config)
            
            node_group_change = data_config.signal_types['node_group_name'] != infer_configs[-1].data_config.signal_types['node_group_name'] if infer_configs else False
            signal_group_change = data_config.signal_types['signal_group_name'] != infer_configs[-1].data_config.signal_types['signal_group_name'] if infer_configs else False
            set_id_change = data_config.set_id != infer_configs[-1].data_config.set_id if infer_configs else False

            # # First level idx reset
            # if signal_types_change or set_id_change:
            #     idx = -1 
            #     idx_dict = {}
            
            # Convert sweep config to dictionary
            sweep_dict = self._to_dict()
            
            # Get all parameter names and their values
            param_names = list(sweep_dict.keys())
            param_values = list(sweep_dict.values())
            
            # Generate all combinations using cartesian product
            combinations = list(itertools.product(*param_values))
            
            # Create configs for each combination
            for combo in combinations:

                # Create base config
                try:
                    infer_config = AnomalyDetectorInferManager(
                        self.data_config, 
                        run_type=self.run_type, 
                        infer_sweep_num=self.infer_sweep_num,
                        selected_model_path=combo[param_names.index('selected_model_path')]
                    )
                except ValueError as e:
                    print(f"Since model {os.path.basename(os.path.dirname(combo[param_names.index('selected_model_path')]))} is incompatible, it is skipped...")
                    continue
    
                # when model is approved, check if model path has changed
                model_change = combo[param_names.index('selected_model_path')] != infer_configs[-1].selected_model_path if infer_configs else False

                # Second level reset idx
                if node_group_change or signal_group_change or set_id_change or model_change:
                    idx = (
                        idx_dict.get(data_config.signal_types['node_group_name'], {})
                                .get(data_config.signal_types['signal_group_name'], {})
                                .get(data_config.set_id, {})
                                .get(combo[param_names.index('selected_model_path')], -1)
                    )
                
                idx += 1

                # Update parameters based on current combination
                for param_name, param_value in zip(param_names, combo):
                    setattr(infer_config, param_name, param_value)
                
                # update domain config
                infer_config.set_domain_config()

                # Update version number 
                _ = infer_config.get_infer_log_path(always_next_version=True)
                infer_config.version = infer_config.version + idx

                idx_dict[
                    data_config.signal_types['node_group_name']
                ][
                    data_config.signal_types['signal_group_name']
                ][
                    data_config.set_id
                ][
                    infer_config.selected_model_path
                ] = idx

                # Regenerate hyperparameters after updating parameters
                infer_config.infer_hparams = infer_config.get_infer_hparams()

                infer_configs.append(infer_config)
                        
        return infer_configs
    
    # def _is_data_model_match(self, infer_config:AnomalyDetectorInferManager):
    #     #model_id = os.path.basename(os.path.dirname(model_path)).split('-')[0].strip('[]')

    #     node_group_model = infer_config.log_config.data_config.signal_types['node_group_name']
    #     signal_group_model = infer_config.log_config.data_config.signal_types['signal_group_name']
    #     set_id_model = infer_config.log_config.data_config.set_id
    #     window_length_model = infer_config.log_config.data_config.window_length
    #     stride_model = infer_config.log_config.data_config.stride

    #     is_node_match = node_group_model == self.data_config.signal_types['node_group_name']
    #     is_signal_match = signal_group_model == self.data_config.signal_types['signal_group_name']
    #     is_setid_match = set_id_model == self.data_config.set_id
    #     is_window_match = window_length_model == self.data_config.window_length
    #     is_stride_match = stride_model == self.data_config.stride

    #     if is_node_match and is_signal_match and is_setid_match and is_window_match and is_stride_match:
    #         return True
    #     else:
    #         print(f"\nIncompatible model found: {os.path.basename(os.path.dirname(infer_config.selected_model_path))}. Hence skipping this model...")
    #         print(f"Incompatible parameters:")
    #         if not is_node_match:
    #             print(f"  Node group mismatch: Model({node_group_model}) != Data({self.data_config.signal_types['node_group_name']})")
    #         if not is_signal_match:
    #             print(f"  Signal group mismatch: Model({signal_group_model}) != Data({self.data_config.signal_types['signal_group_name']})")
    #         if not is_setid_match:
    #             print(f"  Set ID mismatch: Model({set_id_model}) != Data({self.data_config.set_id})")
    #         if not is_window_match:
    #             print(f"  Window length mismatch: Model({window_length_model}) != Data({self.data_config.window_length})")
    #         if not is_stride_match:
    #             print(f"  Stride mismatch: Model({stride_model}) != Data({self.data_config.stride})")

    
    def _to_dict(self):
        """
        Convert sweep config attributes to dictionary, excluding private methods and non-list attributes.
        """
        sweep_dict = {}
        for attr_name in dir(self):
            if not attr_name.startswith('_') and not callable(getattr(self, attr_name)):
                attr_value = getattr(self, attr_name)
                if isinstance(attr_value, list):
                    if attr_name != 'data_configs': 
                        sweep_dict[attr_name] = attr_value

        return sweep_dict
    
    def get_total_combinations(self):
        """
        Get the total number of parameter combinations.
        """
        sweep_dict = self._to_dict()
        total = 1
        for values in sweep_dict.values():
            total *= len(values)
        return total
    
    def print_sweep_summary(self):
        """
        Print a summary of the parameter sweep.
        """
        sweep_dict = self._to_dict()
        print(f"\nParameter Sweep Summary:")
        print(f"Total combinations: {self.get_total_combinations()}")
        print("\nParameters and their values:")

        for param_name, param_values in sweep_dict.items():
            if param_name == 'selected_model_path':
                print(f"\nNumber of models selected: {len(param_values)}")
                print("Selected models are as follows:\n" + 30*'-')

                for i, path in enumerate(param_values):
                    print(f"  Model {i+1}: {os.path.basename(os.path.dirname(path))}")
            else:
                print(f"  {param_name}: {len(param_values)} values -> {param_values}")
    

class HelperClass:
    def get_augmment_config_str_list(self, augment_configs):
        """
        Returns a list of strings representing the augment configurations.
        """
        augment_strings = []
        for augment in augment_configs:
            additional_keys = ', '.join([f"{key}={value}" for key, value in augment.items() if key != 'type'])
            if additional_keys and augment['type'] != 'OG':
                augment_strings.append(f"{augment['type']}({additional_keys})")
            else:
                augment_strings.append(f"{augment['type']}")

        return augment_strings
    
    def set_ds_types_in_path(self, data_config:DataConfig, log_path):
        """
        Takes into account both empty healthy, unhealthy and unknown data config and sets the path accordingly.
        """
        if data_config.healthy_configs != {} and data_config.unhealthy_configs == {} and data_config.unknown_configs == {}:
            log_path = os.path.join(log_path, 'healthy')

        elif data_config.healthy_configs != {} and data_config.unhealthy_configs != {} and data_config.unknown_configs == {}:
            log_path = os.path.join(log_path, 'healthy_unhealthy')

        elif data_config.unknown_configs != {} and data_config.unhealthy_configs != {} and data_config.healthy_configs != {}:
            log_path = os.path.join(log_path, 'healthy_unhealthy_unknown')

        elif data_config.healthy_configs == {} and data_config.unhealthy_configs == {} and data_config.unknown_configs != {}:
            log_path = os.path.join(log_path, 'unknown')

        # add ds_subtype to path
        config_str = ''
        
        if data_config.healthy_configs != {}:    
            healthy_config_str_list = []

            for healthy_type, augment_configs  in data_config.healthy_configs.items():
                augment_str_list = self.get_augmment_config_str_list(augment_configs)                    
                augment_str_main = ', '.join(augment_str_list) 

                if data_config.application == 'ASM':
                    healthy_type = healthy_type.replace("set01_", "").replace("set02_", "")

                healthy_config_str_list.append(f'(OK):{healthy_type}[{augment_str_main}]')

            config_str = ' + '.join(healthy_config_str_list)

        # add unhealthy config to path if exists
        if data_config.unhealthy_configs != {}:
            unhealthy_config_str_list = []

            for unhealthy_type, augment_configs in data_config.unhealthy_configs.items():
                augment_str_list = self.get_augmment_config_str_list(augment_configs)
                augment_str_main = ', '.join(augment_str_list)

                if data_config.application == 'ASM':
                    unhealthy_type = unhealthy_type.replace("set01_", "").replace("set02_", "") 

                unhealthy_config_str_list.append(f'(NOK):{unhealthy_type}[{augment_str_main}]')

            if config_str:
                config_str += f" + {' + '.join(unhealthy_config_str_list)}"
            else:
                config_str += ' + '.join(unhealthy_config_str_list)

        # add unknown config to path if exists
        if data_config.unknown_configs != {}:
            unknown_config_str_list = []

            for unknown_type, augment_configs in data_config.unknown_configs.items():
                augment_str_list = self.get_augmment_config_str_list(augment_configs)
                augment_str_main = ', '.join(augment_str_list)

                if data_config.application == 'ASM':
                    unknown_type = unknown_type.replace("set01_", "").replace("set02_", "") 

                unknown_config_str_list.append(f'(UK):{unknown_type}[{augment_str_main}]')

            if config_str:
                config_str += f" + {' + '.join(unknown_config_str_list)}"
            else:
                config_str += ' + '.join(unknown_config_str_list)

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

def get_selected_model_path(is_multi=False):
    """
    Returns the selected model path(s) from the settings directory.
    
    Parameters
    ----------
    is_multi : bool, optional
        If True, returns a list of multiple model paths. If False, returns a single model path. Default is False.
    
    Returns
    -------
    str or list
        The selected model path(s).
    """
    if is_multi:
        with open(os.path.join(SETTINGS_DIR, "selections", "multi_model_paths.txt"), "r") as f:
            model_paths = [(line.strip()) for line in f if line.strip()]
        return model_paths

    else:
        with open(os.path.join(SETTINGS_DIR, "selections", "single_model_path.txt"), "r") as f:
            model_path = (f.read()) 
        return model_path
    

def load_log_config(model_path):
    log_config = AnomalyDetectorTrainManager(DataConfig())

    log_config_path = os.path.join(os.path.dirname(model_path), 'train_config.pkl')

    if not os.path.exists(log_config_path):
        raise ValueError(f"\nThe parameter file does not exists in {log_config_path}")
    
    with open(log_config_path, 'rb') as f:
        log_config.__dict__.update(pickle.load(f))
    
    return log_config

class SelectFaultDetectionModel:
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
            self.file_name = 'fdet'
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

            new_parts = model_path.split(os.sep)
            if "logs" in  new_parts:
                framework_index = new_parts.index("logs")
            else:
                raise ValueError(f"Framework directory 'AFD' or 'AFD_thesis' not found in path: {model_path}")
            
            new_root = model_path.split(os.sep)[:framework_index + 1]
            new_root_path = os.sep.join(new_root)

            # Remove base logs dir and split by os.sep
            rel_path = os.path.relpath(model_path, str(new_root_path))
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
                key=lambda x: int(re.search(fr'.*{self.file_name}_(\d+)', x[1]).group(1)) if re.search(fr'.*{self.file_name}_(\d+)', x[1]) else 0
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
        from rich.tree import Tree
        from rich.console import Console
        from rich.console import Console
        from rich import print

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
        is_no_feat = any("(anom) [no_feat]" in k for k in parent_keys)
       
        if self.run_type == 'train':
            label_map = {
                0: "<node_name>",
                1: "<signal_group>",
                2: "<set>",
                3: "<ds_type>",
                4: "<ds_subtype>",
                5: "<model>",
                6: "<signal_types>",
                7: "<timestep_id>",
                8: "<domain>",
                9: "<feat_type>",
                10: "<tswp_id>",
                11: "<shape_compatibility>",
                12: "<version>"
            }
        elif self.run_type in ['custom_test', 'predict']:
            label_map = {
                0: "<node_name>",
                1: "<signal_group>",
                2: "<set>",
                3: "<iswp_id>",
                4: "<ds_type>",
                5: "<ds_subtype>",
                6: "<trained_model>",
                7: "<model_id>",
                8: "<timestep_id>",
                9: "<versions>"
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
            if self.run_type == "train" and level == 5:
                branch = parent_node.add(f"[bright_yellow]{safe_key}[/bright_yellow]")
                self._build_rich_tree(branch, value, level + 1, parent_keys + [key])
                continue
            if self.run_type in ['custom_test', 'predict'] and level == 6:
                branch = parent_node.add(f"[bright_yellow]{safe_key}[/bright_yellow]")
                self._build_rich_tree(branch, value, level + 1, parent_keys + [key])
                continue
            if self.run_type in ['custom_test', 'predict'] and level == 7:
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
                key=lambda v: int(re.search(fr'.*{self.file_name}_(\d+)', v['model_name']).group(1)) if re.search(fr'.*{self.file_name}_(\d+)', v['model_name']) else 0
            )
            for v in sorted_versions:
                model_disp = f"{v['model_name']} (v{v['vnum']})"
                safe_model_disp = model_disp.replace('[', '\\[')
                idx = self.version_paths.index(v["txt_file"])
                if is_no_feat:
                    parent_node.add(f"[bright_yellow]{safe_model_disp}[/bright_yellow] [bright_cyan][{idx}][/bright_cyan]")
                else:
                    parent_node.add(f"[bright_green]{safe_model_disp}[/bright_green] [bright_cyan][{idx}][/bright_cyan]")

    def select_model_and_params(self):
        self.print_tree()
        if not self.version_paths:
            print("No version paths found.")
            return None
        
        if self.run_type == 'train':
            
            select_mode = input("\nSelection mode: single (s) or multi (m)? [s/m]: ").strip().lower()

            if select_mode == 'm':
                idxs_str = input("\nEnter the index numbers of the models to select (comma separated): ")
                try:
                    idxs_str = idxs_str.replace(" ", "")
                    idxs = [int(i.strip()) for i in idxs_str.split(',')]
                except ValueError:
                    print("Invalid input. Please enter comma separated integer indices.")
                    return None
                
                # check for invalid indices
                invalid_idxs = [i for i in idxs if i < 0 or i >= len(self.version_paths)]
                if invalid_idxs:
                    print(f"Invalid indices: {invalid_idxs}")
                    return None
                
                selected_log_paths = [os.path.dirname(self.version_paths[i]) for i in idxs]
                model_file_paths = [get_model_pickle_path(p) for p in selected_log_paths]

                with open(os.path.join(SETTINGS_DIR, "selections", "multi_model_paths.txt"), "w") as f:
                    for path in model_file_paths:
                        f.write(f"{path}\n")
                
                print(f"\nSelected model file paths saved to {os.path.join(SETTINGS_DIR, 'selections', 'multi_model_paths.txt')}\n")

            elif select_mode == 's':
                idx = int(input("\nEnter the index number of the version path to select: "))
                if idx < 0 or idx >= len(self.version_paths):
                    print("Invalid index.")
                    return None
                selected_log_path = os.path.dirname(self.version_paths[idx])
                # Use the directory containing model_x.txt as the log path
            
                model_file_path = get_model_pickle_path(selected_log_path)

                with open(os.path.join(SETTINGS_DIR, "selections", "single_model_path.txt"), "w") as f:
                    f.write(model_file_path)

                print(f"\nSelected model file path: {model_file_path}")