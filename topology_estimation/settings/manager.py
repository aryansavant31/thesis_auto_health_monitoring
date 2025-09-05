import os, sys

ROOT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT_DIR) if ROOT_DIR not in sys.path else None

SETTINGS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SETTINGS_DIR) if SETTINGS_DIR not in sys.path else None

TP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(TP_DIR, "logs")

# other imports
import shutil
import re
from pathlib import Path
from rich.tree import Tree
from rich.console import Console
from data.config import DataConfig
import pickle
import glob
from collections import defaultdict
import itertools

# global imports
from data.config import DataConfig

# local imports
from .train_config import NRITrainConfig, NRITrainSweep, DecoderTrainConfig, DecoderTrainSweep
from .infer_config import NRIInferConfig, NRIInferSweep, DecoderInferConfig, DecoderInferSweep

class NRITrainManager(NRITrainConfig):
    def __init__(self, data_config:DataConfig, train_sweep_num=0):
        super().__init__(data_config)
        self.helper = HelperClass()
        self.train_sweep_num = train_sweep_num

    def get_train_log_path(self, n_dim, n_comps, always_next_version=False):
        """
        Returns the path to store the logs based on data and topology config

        Parameters
        ----------
        n_components : int
            The number of components in each sample in the dataset
        n_dim : int
            The number of dimensions in each sample in the dataset
        always_next_version : bool, optional
            Whether to always create the next version if a version already exists, by default False
        """
        self.always_next_version = always_next_version
        
        self.node_group = f"{self.data_config.signal_types['node_group_name']}"
        self.signal_group = f"{self.data_config.signal_types['signal_group_name']}"
        self.set_id = self.data_config.set_id

        base_path = os.path.join(LOGS_DIR, 
                                f'{self.data_config.application_map[self.data_config.application]}', 
                                f'{self.data_config.machine_type}',
                                f'{self.data_config.scenario}')

        # For directed graph path 
        model_path = os.path.join(base_path, 'nri',)  # add framework type

        # add num of edge types and node types to path
        model_path = os.path.join(model_path, 'train', f'etypes={self.n_edge_types}', f"{self.node_group}", self.signal_group, f"set_{self.set_id}")

        # get train log path
        self.model_name = f"[{self.node_group}_({self.signal_group}+{self.set_id})]-(E={self.pipeline_type}_D={self.recur_emb_type})_edge_est_{self.n_edge_types}"
        self.train_log_path = os.path.join(model_path, f"E={self.pipeline_type}_D={self.recur_emb_type}", f"tswp_{self.train_sweep_num}", f"{self.model_name}.{self.model_num}")
                       
        # add healthy or healthy_unhealthy config to path
        model_path = self.helper.set_ds_types_in_path(self.data_config, model_path)

        # add model type to path
        model_path = os.path.join(model_path, f'[E] {self.pipeline_type}, [D] {self.recur_emb_type}',)

        # add datastats to path
        signal_types_str = ', '.join(
            f"{node_type}: ({', '.join(signal_types_list)})" for node_type, signal_types_list in self.data_config.signal_types['group'].items()
        )
        model_path = os.path.join(model_path, f"{signal_types_str}")

        # add timestep id to path
        model_path = os.path.join(model_path, f"T{self.data_config.window_length}, Tmax = {self.data_config.max_timesteps:,}")

        # add sparsifier type to path
        model_path = self.helper.set_sparsifier_in_path(self.spf_config, self.spf_domain_config['type'], self.spf_feat_configs, self.spf_reduc_config, model_path)

        # add domain type of encoder and decoder to path
        model_path = os.path.join(model_path, f'(E) {self.enc_domain_config['type']}, (D) {self.dec_domain_config['type']}')

        # add feature types for encoder and decoder:
        enc_feat_types = self.helper.get_feat_type_str_list(self.enc_feat_configs)
        dec_feat_types = self.helper.get_feat_type_str_list(self.dec_feat_configs)

        enc_reduc_type = self.enc_reduc_config['type'] if self.enc_reduc_config else 'no_reduc'
        dec_reduc_type = self.dec_reduc_config['type'] if self.dec_reduc_config else 'no_reduc'

        model_path = os.path.join(model_path, f"(E) [{' + '.join(enc_feat_types)}] [{enc_reduc_type}], (D) [{' + '.join(dec_feat_types)}] [{dec_reduc_type}]")

        # add train sweep number to path
        model_path = os.path.join(model_path, f'tswp_{self.train_sweep_num}')

        # add model shape compatibility stats to path
        self.model_id = os.path.join(model_path, f'(E) (comps = {n_comps}), (D) (dims = {n_dim})')

        # # add model version to path
        # self.model_id = os.path.join(model_path, f'edge_estimator_{self.model_num}')

        # check if version already exists
        self.check_if_version_exists()

        return self.train_log_path

    def get_test_log_path(self):
        
        test_log_path = os.path.join(self.train_log_path, 'test')

        # # add healthy or healthy_unhealthy config to path
        # test_log_path = self.helper.set_ds_types_in_path(self.data_config, test_log_path)

        # # add timestep id to path
        # test_log_path = os.path.join(test_log_path, f'T{self.data_config.window_length}')

        # # add sparsifier type to path
        # test_log_path = self.helper.set_sparsifier_in_path(self.sparsif_type, self.domain_sparsif, self.fex_configs_sparsif, test_log_path)

        # # add test version to path
        # test_log_path = os.path.join(test_log_path, f'test_v0')

        return test_log_path

    
    def _remove_version(self):
        """
        Removes the model from the log path.
        """
        if os.path.exists(self.train_log_path):
            user_input = input(f"Are you sure you want to remove the '{self.model_name}.{self.model_num}' from the log path {self.train_log_path}? (y/n): ")
            if user_input.lower() == 'y':
                shutil.rmtree(self.train_log_path)
                print(f"Overwrote '{self.model_name}.{self.model_num}' from the log path {self.train_log_path}.")

            else:
                print(f"Operation cancelled. {self.model_name}.{self.model_num} still remains.")
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
            max_model = max(int(f.split('_')[-1].split('.')[-1]) for f in model_folders)
            self.model_num = max_model + 1
            new_model = f"{self.model_name}.{self.model_num}"
            print(f"Next edge estimator folder will be: {new_model}")
        else:
            self.model_num = 1
            new_model = f"{self.model_name}.{self.model_num}"  # If no v folders exist

        return os.path.join(sweep_dir, new_model)
    
    def save_params(self):
        """
        Saves the training parameters to a pickle file in the log path.
        """
        if not os.path.exists(self.train_log_path):
            os.makedirs(self.train_log_path)

        config_path = os.path.join(self.train_log_path, f'train_config.pkl')
        with open(config_path, 'wb') as f:
            pickle.dump(self.__dict__, f)
        
        model_path = os.path.join(self.train_log_path, f'{self.model_name}.{self.model_num}.txt')
        with open(model_path, 'w') as f:
            f.write(self.model_id)

        print(f"Model parameters saved to {self.train_log_path}.")


    def check_if_version_exists(self):
        """
        Checks if the version already exists in the log path.

        Parameters
        ----------
        log_path : str
            The path where the logs are stored. It can be for nri model or skeleton graph model.
        """
        if self.always_next_version:
            self.train_log_path = self._get_next_version()
            return
        
        sweep_dir = os.path.dirname(self.train_log_path)
        parent_dir = os.path.dirname(sweep_dir)

        model_folders = []
        model_paths = []

        for root, dirs, files in os.walk(parent_dir):
            # Only look at immediate subfolders of parent_dir
            if os.path.dirname(root) == parent_dir:
                for d in dirs:
                    model_folders.append(d)
                    model_paths.append(root)

        if self.continue_training:
            if os.path.basename(self.train_log_path) in model_folders:
                path_idx = model_folders.index(os.path.basename(self.train_log_path))
                print(f"\nContinuing training from '{self.model_name}.{self.model_num}' in the log path '{os.path.join(model_paths[path_idx], model_folders[path_idx])}'.")
                
            else:
                print(f"\nWith continue training enabled, there is no existing version to continue train in the log path '{self.train_log_path}'.")       
        else:
            if os.path.basename(self.train_log_path) in model_folders:
                path_idx = model_folders.index(os.path.basename(self.train_log_path))

                print(f"\n'{self.model_name}.{self.model_num}' already exists in the log path '{os.path.join(model_paths[path_idx], model_folders[path_idx])}'.")
                user_input = input("(a) Overwrite exsiting version, (b) create new version, (c) stop training (Choose 'a', 'b' or 'c'):  ")

                if user_input.lower() == 'a':
                    self._remove_version()

                elif user_input.lower() == 'b':
                    self.train_log_path = self._get_next_version()

                elif user_input.lower() == 'c':
                    print("Stopped training.")
                    sys.exit()  # Exit the program gracefully


class DecoderTrainManager(DecoderTrainConfig):
    def __init__(self, data_config:DataConfig, train_sweep_num=0):
        super().__init__(data_config)
        self.helper = HelperClass()
        self.train_sweep_num = train_sweep_num

    def get_train_log_path(self, n_dim, always_next_version=False, **kwargs):
        """
        Returns the path to store the logs based on data and topology config

        Parameters
        ----------
        n_dim : int
            The number of dimensions in each sample in the dataset
        """
        self.always_next_version = always_next_version

        self.node_group = f"{self.data_config.signal_types['node_group_name']}"
        self.signal_group = f"{self.data_config.signal_types['signal_group_name']}"
        self.set_id = self.data_config.set_id

        base_path = os.path.join(LOGS_DIR, 
                                f'{self.data_config.application_map[self.data_config.application]}', 
                                f'{self.data_config.machine_type}',
                                f'{self.data_config.scenario}')

        # For directed graph path 
        model_path = os.path.join(base_path, 'decoder',)  # add framework type

        # add num of edge types and node types to path
        model_path = os.path.join(model_path, 'train', f'etypes={self.n_edge_types}', f"{self.node_group}", self.signal_group, f"set_{self.set_id}")

        # get train log path
        self.model_name = f"[{self.node_group}_({self.signal_group}+{self.set_id})]-{self.recur_emb_type}_dec_{self.n_edge_types}"
        self.train_log_path = os.path.join(model_path, f"D={self.recur_emb_type}", f"tswp_{self.train_sweep_num}", f"{self.model_name}.{self.model_num}")
                       
        # add healthy or healthy_unhealthy config to path
        model_path = self.helper.set_ds_types_in_path(self.data_config, model_path)

        # add model type to path
        model_path = os.path.join(model_path, f'[D] {self.recur_emb_type}',)

        # add datastats to path
        signal_types_str = ', '.join(
            f"{node_type}: ({', '.join(signal_types_list)})" for node_type, signal_types_list in self.data_config.signal_types['group'].items()
        )
        model_path = os.path.join(model_path, f"{signal_types_str}")

        # add timestep id to path
        model_path = os.path.join(model_path, f"T{self.data_config.window_length}, Tmax = {self.data_config.max_timesteps:,}")

        # add sparsifier type to path
        model_path = self.helper.set_sparsifier_in_path(self.spf_config, self.spf_domain_config['type'], self.spf_feat_configs, self.spf_reduc_config, model_path)

        # add domain type of encoder and decoder to path
        model_path = os.path.join(model_path, f'(D) {self.dec_domain_config['type']}')

        # add feature types for decoder:
        dec_feat_types = self.helper.get_feat_type_str_list(self.dec_feat_configs)
        dec_reduc_type = self.dec_reduc_config['type'] if self.dec_reduc_config else 'no_reduc'

        model_path = os.path.join(model_path, f"(D) [{' + '.join(dec_feat_types)}] [{dec_reduc_type}]")

        # add train sweep number to path
        model_path = os.path.join(model_path, f'tswp_{self.train_sweep_num}')
  
        # add model shape compatibility stats to path
        self.model_id = os.path.join(model_path, f'(D) (dims = {n_dim})')

        # # add model version to path
        # self.model_id = os.path.join(model_path, f'edge_estimator_{self.model_num}')

        # check if version already exists
        self.check_if_version_exists()

        return self.train_log_path

    def get_test_log_path(self):
        
        test_log_path = os.path.join(self.train_log_path, 'test')

        # # add healthy or healthy_unhealthy config to path
        # test_log_path = self.helper.set_ds_types_in_path(self.data_config, test_log_path)

        # # add timestep id to path
        # test_log_path = os.path.join(test_log_path, f'T{self.data_config.window_length}')

        # # add sparsifier type to path
        # test_log_path = self.helper.set_sparsifier_in_path(self.sparsif_type, self.domain_sparsif, self.fex_configs_sparsif, test_log_path)

        # # add test version to path
        # test_log_path = os.path.join(test_log_path, f'test_v0')

        return test_log_path

    
    def _remove_version(self):
        """
        Removes the model from the log path.
        """
        if os.path.exists(self.train_log_path):
            user_input = input(f"Are you sure you want to remove the '{self.model_name}.{self.model_num}' from the log path {self.train_log_path}? (y/n): ")
            if user_input.lower() == 'y':
                shutil.rmtree(self.train_log_path)
                print(f"Overwrote '{self.model_name}.{self.model_num}' from the log path {self.train_log_path}.")

            else:
                print(f"Operation cancelled. {self.model_name}.{self.model_num} still remains.")
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
            max_model = max(int(f.split('_')[-1].split('.')[1]) for f in model_folders)
            self.model_num = max_model + 1
            new_model = f'{self.model_name}.{self.model_num}'
            print(f"Next decoder folder will be: {new_model}")
        else:
            self.model_num = 1
            new_model = f'{self.model_name}.{self.model_num}'  # If no v folders exist

        return os.path.join(sweep_dir, new_model)
    
    def save_params(self):
        """
        Saves the training parameters to a pickle file in the log path.
        """
        if not os.path.exists(self.train_log_path):
            os.makedirs(self.train_log_path)

        config_path = os.path.join(self.train_log_path, f'train_config.pkl')
        with open(config_path, 'wb') as f:
            pickle.dump(self.__dict__, f)
        
        model_path = os.path.join(self.train_log_path, f'{self.model_name}.{self.model_num}.txt')
        with open(model_path, 'w') as f:
            f.write(self.model_id)

        print(f"Model parameters saved to {self.train_log_path}.")


    def check_if_version_exists(self):
        """
        Checks if the version already exists in the log path.

        Parameters
        ----------
        log_path : str
            The path where the logs are stored. It can be for nri model or skeleton graph model.
        """
        if self.always_next_version:
            self.train_log_path = self._get_next_version()
            return
        
        sweep_dir = os.path.dirname(self.train_log_path)
        parent_dir = os.path.dirname(sweep_dir)

        model_folders = []
        model_paths = []

        for root, dirs, files in os.walk(parent_dir):
            # Only look at immediate subfolders of parent_dir
            if os.path.dirname(root) == parent_dir:
                for d in dirs:
                    model_folders.append(d)
                    model_paths.append(root)

        if self.continue_training:
            if os.path.basename(self.train_log_path) in model_folders:
                path_idx = model_folders.index(os.path.basename(self.train_log_path))
                user_input = input(f"\nContinue training from '{self.model_name}.{self.model_num}' in the log path '{os.path.join(model_paths[path_idx], model_folders[path_idx])}'? (y/n):")
                if user_input.lower() == 'y':
                    print(f"\nContinuing training from '{self.model_name}.{self.model_num}'")
                elif user_input.lower() == 'n':
                    print("Stopped training.")
                    sys.exit()  # Exit the program gracefully
                          
            else:
                print(f"\nWith continue training enabled, there is no existing version to continue train in the log path '{self.train_log_path}'.")       
        else:
            if os.path.basename(self.train_log_path) in model_folders:
                path_idx = model_folders.index(os.path.basename(self.train_log_path))

                print(f"\n'{self.model_name}.{self.model_num}' already exists in the log path '{os.path.join(model_paths[path_idx], model_folders[path_idx])}'.")
                user_input = input("(a) Overwrite exsiting version, (b) create new version, (c) stop training (Choose 'a', 'b' or 'c'):  ")

                if user_input.lower() == 'a':
                    self._remove_version()

                elif user_input.lower() == 'b':
                    self.train_log_path = self._get_next_version()

                elif user_input.lower() == 'c':
                    print("Stopped training.")
                    sys.exit()  # Exit the program gracefully


class TopologyEstimationTrainSweepManager(NRITrainSweep, DecoderTrainSweep):
    def __init__(self, data_config:DataConfig, framework):
        self.framework = framework

        if self.framework == 'nri':
            NRITrainSweep.__init__(self, data_config)
        elif self.framework == 'decoder':
            DecoderTrainSweep.__init__(self, data_config)

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
        print(f"\n{self.framework.capitalize()} Parameter Sweep Summary:")
        print(f"Total combinations: {self.get_total_combinations()}")
        print("\nParameters and their values:")
        for param_name, param_values in sweep_dict.items():
            print(f"  {param_name}: {len(param_values)} values -> {param_values}") 
    

class NRITrainSweepManager(TopologyEstimationTrainSweepManager):
    framework = 'nri'

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
        idx_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
        idx = -1
        
        for data_config in self.data_configs: 

            TopologyEstimationTrainSweepManager.__init__(self, data_config, framework=NRITrainSweepManager.framework)

            node_group_change = data_config.signal_types['node_group_name'] != train_configs[-1].data_config.signal_types['node_group_name'] if train_configs else False
            signal_group_change = data_config.signal_types['signal_group_name'] != train_configs[-1].data_config.signal_types['signal_group_name'] if train_configs else False
            set_id_change = data_config.set_id != train_configs[-1].data_config.set_id if train_configs else False

            # Convert sweep config to dictionary
            sweep_dict = self._to_dict()
            
            # Get all parameter names and their values
            param_names = list(sweep_dict.keys())
            param_values = list(sweep_dict.values())
            
            # Generate all combinations using cartesian product
            combinations = list(itertools.product(*param_values))
            
            # Create configs for each combination
            for combo in combinations:
                pipeline_change = combo[param_names.index('pipeline_type')] != train_configs[-1].pipeline_type if train_configs else False
                recur_emb_change = combo[param_names.index('recur_emb_type')] != train_configs[-1].recur_emb_type if train_configs else False

                if node_group_change or signal_group_change or set_id_change or pipeline_change or recur_emb_change:
                    idx = (
                        idx_dict.get(data_config.signal_types['node_group_name'], {})
                                .get(data_config.signal_types['signal_group_name'], {})
                                .get(data_config.set_id, {})
                                .get(combo[param_names.index('pipeline_type')], {})
                                .get(combo[param_names.index('recur_emb_type')], -1)
                    )
                idx += 1

                # Create base config
                train_config = NRITrainManager(self.data_config, train_sweep_num=self.train_sweep_num)

                # Update parameters based on current combination
                for param_name, param_value in zip(param_names, combo):
                    setattr(train_config, param_name, param_value)

                # Update model number
                _ = train_config.get_train_log_path(n_dim=0, n_comps=0, always_next_version=True) # Dummy values for n_dim and n_comps
                train_config.model_num = train_config.model_num + idx

                # Update idx dictionary
                idx_dict[
                    data_config.signal_types['node_group_name']
                ][
                    data_config.signal_types['signal_group_name']
                ][
                    data_config.set_id
                ][
                    combo[param_names.index('pipeline_type')]
                ][
                    combo[param_names.index('recur_emb_type')]
                ] = idx

                # Regenerate emb configs and hyperparams
                train_config.set_nri_emb_configs()
                train_config.hyperparams = train_config.get_hyperparams()

                train_configs.append(train_config)
        
        return train_configs
    
class DecoderTrainSweepManager(TopologyEstimationTrainSweepManager):
    framework = 'decoder'

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

            TopologyEstimationTrainSweepManager.__init__(self, data_config, framework=DecoderTrainSweepManager.framework)

            node_group_change = data_config.signal_types['node_group_name'] != train_configs[-1].data_config.signal_types['node_group_name'] if train_configs else False
            signal_group_change = data_config.signal_types['signal_group_name'] != train_configs[-1].data_config.signal_types['signal_group_name'] if train_configs else False
            set_id_change = data_config.set_id != train_configs[-1].data_config.set_id if train_configs else False

            # Convert sweep config to dictionary
            sweep_dict = self._to_dict()
            
            # Get all parameter names and their values
            param_names = list(sweep_dict.keys())
            param_values = list(sweep_dict.values())
            
            # Generate all combinations using cartesian product
            combinations = list(itertools.product(*param_values))
            
            # Create configs for each combination
            for combo in combinations:
                recur_emb_change = combo[param_names.index('recur_emb_type')] != train_configs[-1].recur_emb_type if train_configs else False

                if node_group_change or signal_group_change or set_id_change or recur_emb_change:
                    idx = (
                        idx_dict.get(data_config.signal_types['node_group_name'], {})
                                .get(data_config.signal_types['signal_group_name'], {})
                                .get(data_config.set_id, {})
                                .get(combo[param_names.index('recur_emb_type')], -1)
                    )
                idx += 1

                # Create base config
                train_config = DecoderTrainManager(self.data_config, train_sweep_num=self.train_sweep_num)

                # Update parameters based on current combination
                for param_name, param_value in zip(param_names, combo):
                    setattr(train_config, param_name, param_value)

                # Update model number
                _ = train_config.get_train_log_path(n_dim=0, always_next_version=True) # Dummy values for n_dim
                train_config.model_num = train_config.model_num + idx

                # Update idx dictionary
                idx_dict[
                    data_config.signal_types['node_group_name']
                ][
                    data_config.signal_types['signal_group_name']
                ][
                    data_config.set_id
                ][
                    combo[param_names.index('recur_emb_type')]
                ] = idx

                # Regenerate emb configs and hyperparams
                train_config.set_dec_emb_configs()
                train_config.hyperparams = train_config.get_hyperparams()

                train_configs.append(train_config)
        
        return train_configs
    


class TopologyEstimationInferManager(NRIInferConfig, DecoderInferConfig):
    def __init__(self, data_config, framework, run_type, infer_sweep_num=0, selected_model_path=None):
        """
        Initializes the infer manager for topology estimation.

        Parameters
        ----------
        data_config : DataConfig
            The data configuration object.
        framework : str
            The framework to use ('nri' or 'decoder').
        run_type : str
            The type of run ('train', 'custom_test', or 'predict').
        """
        if framework == 'nri':
            NRIInferConfig.__init__(self, data_config, run_type, selected_model_path)
        elif framework == 'decoder':
            DecoderInferConfig.__init__(self, data_config, run_type, selected_model_path)

        # check if data and model are compatible
        if not self._is_data_model_match():
            raise ValueError("Data and model configurations are not compatible. Please check the configurations.")

        self.helper = HelperClass()
        self.framework = framework
        self.infer_sweep_num = infer_sweep_num
        
        self.train_log_path = self.log_config.train_log_path
        self.n_edge_types = self.log_config.n_edge_types

        self.selected_model_num = os.path.basename(self.train_log_path)

    def _is_data_model_match(self):
        #model_id = os.path.basename(os.path.dirname(model_path)).split('-')[0].strip('[]')

        node_group_model = self.log_config.data_config.signal_types['node_group_name']
        signal_group_model = self.log_config.data_config.signal_types['signal_group_name']
        set_id_model = self.log_config.data_config.set_id
        window_length_model = self.log_config.data_config.window_length
        stride_model = self.log_config.data_config.stride
        custom_max_timesteps_model = self.log_config.data_config.max_timesteps


        is_node_match = node_group_model == self.data_config.signal_types['node_group_name']
        is_signal_match = signal_group_model == self.data_config.signal_types['signal_group_name']
        is_setid_match = set_id_model == self.data_config.set_id
        is_window_match = window_length_model == self.data_config.window_length
        is_stride_match = stride_model == self.data_config.stride
        is_custom_max_timesteps_match = custom_max_timesteps_model == self.data_config.max_timesteps if self.data_config.use_custom_max_timesteps else True


        if is_node_match and is_signal_match and is_setid_match and is_window_match and is_stride_match and is_custom_max_timesteps_match:
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
            if not is_custom_max_timesteps_match:
                print(f"  - Custom max timesteps mismatch: Model({custom_max_timesteps_model}) != Data({self.data_config.custom_max_timesteps})")
            return False

    def get_infer_log_path(self, always_next_version=False):
        """
        Sets the log path for the run.
        """
        self.always_next_version = always_next_version

        # infer_num_path = self.train_log_path.replace(f"{os.sep}train{os.sep}", f"{os.sep}{self.run_type}{os.sep}")

        parts = self.train_log_path.split(os.sep)
        train_idx = parts.index('train')
        etypes = parts[train_idx + 1]
        node = parts[train_idx + 2]
        signal_group = parts[train_idx + 3]
        set_id = parts[train_idx + 4]
        model_type = parts[train_idx + 5]
        model_name = parts[-1]

        infer_num_path = os.path.join(
            LOGS_DIR,
            *parts[train_idx-4:train_idx],
            self.run_type,
            etypes, node, signal_group, set_id,
            f'iswp_{self.infer_sweep_num}', 
        )

        self.infer_log_path = os.path.join(infer_num_path, model_type, model_name, f'{self.run_type}_{self.version}')

        # add healthy or healthy_unhealthy config to path
        infer_num_path = self.helper.set_ds_types_in_path(self.data_config, infer_num_path)

        # add model type to path
        infer_num_path = os.path.join(infer_num_path, model_type, model_name)

        # add timestep_id to path
        infer_num_path = os.path.join(infer_num_path, f'T{self.data_config.window_length}, Tmax = {self.data_config.max_timesteps:,}')

        # add sparsifier type to path
        self.infer_id = self.helper.set_sparsifier_in_path(self.spf_config, self.spf_domain_config['type'], self.spf_feat_configs, self.spf_reduc_config, infer_num_path)

        # # add version
        # self.test_id = os.path.join(test_num_path, f'test_num_{self.version}')

        # check if version already exists
        self.check_if_version_exists()

        return self.infer_log_path
    
    
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
        # folders = [f for f in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, f))]
        model_folders = [f for f in os.listdir(parent_dir) if f.startswith(f'{self.run_type}_')]

        if model_folders:
            # Extract numbers and find the max
            max_version = max(int(f.split('_')[-1]) for f in model_folders)
            self.version = max_version + 1
            new_infer_folder = f'{self.run_type}_{self.version}'
            print(f"Next {self.framework} infer folder will be: {new_infer_folder}")
        else:
            self.version = 1
            new_infer_folder = f'{self.run_type}_{self.version}'  # If no v folders exist

        return os.path.join(parent_dir, new_infer_folder)
    
    
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

class TopologyEstimationInferSweepManager(NRIInferSweep, DecoderInferSweep):
    def __init__(self, data_configs:list, framework, run_type):
        self.data_configs = data_configs
        self.framework = framework
        self.run_type = run_type

    def get_sweep_configs(self):
        """
        Generate all possible combinations of parameters for sweeping.
            
        Returns
        -------
        list
            List of TopologyEstimationInferConfig (NRIInfer or DecoderInfer) objects with different parameter combinations
        """
        infer_configs = []
        idx_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        idx = -1

        for data_config in self.data_configs:

            if self.framework == 'nri':
                NRIInferSweep.__init__(self, data_config)
            elif self.framework == 'decoder':
                DecoderInferSweep.__init__(self, data_config)

            node_group_change = data_config.signal_types['node_group_name'] != infer_configs[-1].data_config.signal_types['node_group_name'] if infer_configs else False
            signal_group_change = data_config.signal_types['signal_group_name'] != infer_configs[-1].data_config.signal_types['signal_group_name'] if infer_configs else False
            set_id_change = data_config.set_id != infer_configs[-1].data_config.set_id if infer_configs else False

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
                    infer_config = TopologyEstimationInferManager(
                        self.data_config,
                        framework=self.framework,
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
                infer_config.update_infer_configs()

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
                infer_config.infer_hyperparams = infer_config.get_infer_hyperparams()

                infer_configs.append(infer_config)
                        
        return infer_configs
    
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
        
    
class SelectTopologyEstimatorModel:
    def __init__(self, framework, application=None, machine=None, scenario=None, run_type='train', logs_dir=LOGS_DIR):
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
        self.framework = framework
        self.run_type = run_type  # 'train' or 'predict'

        if self.run_type == 'train':
            if self.framework == 'nri':
                self.file_name = 'edge_est'
            elif self.framework == 'decoder':
                self.file_name = 'dec'
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
        base = self.logs_dir / self.application / self.machine / self.scenario / self.framework / self.run_type 
        os.makedirs(base, exist_ok=True)

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
            # Only keep the parts after framework (skip first 4: app, machine, scenario, framework, train)
            path_parts = path_parts[5:]
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
            if self.run_type == 'train':
                key_pattern = lambda x: int(re.search(fr'{self.file_name}_\d+\.(\d+)', x[1]).group(1)) if re.search(fr'{self.file_name}_\d+\.(\d+)', x[1]) else 0
            else:
                key_pattern = lambda x: int(re.search(fr'.*{self.file_name}_(\d+)', x[1]).group(1)) if re.search(fr'.*{self.file_name}_(\d+)', x[1]) else 0

            sorted_versions = sorted(
                versions,
                key=key_pattern
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
        scenario_node = machine_node.add(f"[green]{self.scenario}[/green]")
        if self.run_type == 'train':
            bracket = '(trained models)'
        elif self.run_type == 'custom_test':
            bracket = '(custom tested models)'
        elif self.run_type == 'predict':
            bracket = '(predicted models)'
        framework_node = scenario_node.add(f"[green]{self.framework}[/green] [magenta]{bracket}[/magenta]")
        self._build_rich_tree(framework_node, self.structure, 0, [])
        console.print(tree)
        print("\nAvailable version paths:")
        for idx, txt_file in enumerate(self.version_paths):
            print(f"{idx}: {os.path.dirname(txt_file)}")

    def _build_rich_tree(self, parent_node, structure, level, parent_keys):
        is_no_sparsif = any("(spf) no_spf" in k for k in parent_keys)
        if self.framework == "skeleton_graph":
            label_map = {
                0: "<ds_type>",
                1: "<ds_subtype>",
                2: "<sparsif_type>",
                3: "<n_components>",
                4: "<domain>",
                5: "<sparsif_fex_type>"
            }
        elif self.run_type == 'train':
            if is_no_sparsif:
                label_map = {
                    0: "<n_edge_types>",
                    1: "<node_group>",
                    2: "<signal_group>",
                    3: "<set>",
                    4: "<ds_type>",
                    5: "<ds_subtype>",
                    6: "<model>",
                    7: "<signal_types>",
                    8: "<timestep_id>",
                    9: "<sparsif_type>",
                    10: "<domain>",
                    11: "<nri_feat_type>",
                    12: "<tswp_id>",
                    13: "<shape_compatibility>",
                    14: "<versions>"
                }
            else:
                label_map = {
                    0: "<n_edge_types>",
                    1: "<node_group>",
                    2: "<signal_group>",
                    3: "<set>",
                    4: "<ds_type>",
                    5: "<ds_subtype>",
                    6: "<model>",
                    7: "<signal_types>",
                    8: "<timestep_id>",
                    9: "<sparsif_type>",
                    10: "<sparsif_fex_type>",
                    11: "<domain>",
                    12: "<nri_fex_type>",
                    13: "<tswp_id>",
                    14: "<shape_compatibility>",
                    15: "<versions>"
                }
        elif self.run_type in ['custom_test', 'predict']:
            if is_no_sparsif:
                label_map = {
                    0: "<n_edge_types>",
                    1: "<node_group>",
                    2: "<signal_group>",
                    3: "<set>",
                    4: "<iswp_id>",
                    5: "<ds_type>",
                    6: "<ds_subtype>",
                    7: "<trained_model>",
                    8: "<model_id>",
                    9: "<timestep_id>",
                    10: "<sparsif_type>",
                    11: "<versions>"
                }
            else:
                label_map = {
                    0: "<n_edge_types>",
                    1: "<node_group>",
                    2: "<signal_group>",
                    3: "<set>",
                    4: "<iswp_id>",
                    5: "<ds_type>",
                    6: "<ds_subtype>",
                    7: "<trained_model>",
                    8: "<model_id>",
                    9: "<timestep_id>",
                    10: "<sparsif_type>",
                    11: "<sparsif_fex_type>",
                    12: "<versions>"
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
            if self.run_type == "train" and level == 6:
                branch = parent_node.add(f"[bright_yellow]{safe_key}[/bright_yellow]")
                self._build_rich_tree(branch, value, level + 1, parent_keys + [key])
                continue
            if self.run_type in ['custom_test', 'predict'] and level == 8:
                branch = parent_node.add(f"[bright_yellow]{safe_key}[/bright_yellow]")
                self._build_rich_tree(branch, value, level + 1, parent_keys + [key])
                continue
            # For skeleton graph, make the model folder under framework yellow
            if self.framework == "skeleton_graph" and level == 2:
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
            if self.run_type == 'train':
                key_pattern = lambda v: int(re.search(fr'{self.file_name}_\d+\.(\d+)', v['model_name']).group(1)) if re.search(fr'{self.file_name}_\d+\.(\d+)', v['model_name']) else 0
            else:
                key_pattern = lambda v: int(re.search(fr'.*{self.file_name}_(\d+)', v['model_name']).group(1)) if re.search(fr'.*{self.file_name}_(\d+)', v['model_name']) else 0

            sorted_versions = sorted(
                structure["_versions"],
                key= key_pattern
            )
            for v in sorted_versions:
                model_disp = f"{v['model_name']} (v{v['vnum']})"
                idx = self.version_paths.index(v["txt_file"])
                if is_no_sparsif:
                    parent_node.add(f"[bright_yellow]{model_disp}[/bright_yellow] [bright_cyan][{idx}][/bright_cyan]")
                else:
                    parent_node.add(f"[bright_green]{model_disp}[/bright_green] [bright_cyan][{idx}][/bright_cyan]")

    def select_ckpt_and_params(self):
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
                model_file_paths = [get_model_ckpt_path(p) for p in selected_log_paths]

                with open(os.path.join(SETTINGS_DIR, f"{self.framework}_selections", "multi_model_paths.txt"), "w") as f:
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

                model_file_path = get_model_ckpt_path(selected_log_path)
      

                with open(os.path.join(SETTINGS_DIR, f"{self.framework}_selections", "single_model_path.txt"), "w") as f:
                    f.write(model_file_path)

                print(f"\nSelected .ckpt file path: {model_file_path}")
           

class SparsifierConfig:
    def __init__(self):
        self.data_config = DataConfig()
        self.set_sparsif_params()

    def set_sparsif_params(self):
        self.sparsif_type = 'knn'  # default sparsifier type
        self.domain = 'time'  # default domain type
        self.fex_configs = []

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
                config_str += f"_+_{'_+_'.join(unhealthy_config_str_list)}"
            else:
                config_str += '_+_'.join(unhealthy_config_str_list)

        log_path = os.path.join(log_path, config_str)
        return log_path
    

    def get_log_path(self, n_components):
        # For skeleton path
        base_path = os.path.join('logs', 
                            f'{self.data_config.application_map[self.data_config.application]}', 
                            f'{self.data_config.machine_type}',
                            f'{self.data_config.scenario}')
        
        infer_log_path_sk = os.path.join(base_path,
                                    'skeleton_graph', # add framework type
                                    f'sparsif={self.sparsif_type}',) # add sparsifier type
        
        # add healthy or healthy_unhealthy config to path
        infer_log_path_sk = self._set_ds_types_in_path(infer_log_path_sk)

        # add datapoints 
        infer_log_path_sk = os.path.join(infer_log_path_sk, f'dp={n_components}')

        # add domain type
        infer_log_path_sk = os.path.join(infer_log_path_sk, f'domain={self.domain}')

        # sparsifer features
        fex_types_sparsif = [fex['type'] for fex in self.fex_configs]
        if fex_types_sparsif:
            infer_log_path_sk = os.path.join(infer_log_path_sk, f"(sparsif)=[{'+'.join(fex_types_sparsif)}]")
        else:
            infer_log_path_sk = os.path.join(infer_log_path_sk, '(sparsif)=_no_fex')

        return infer_log_path_sk

class HelperClass:
    def get_feat_type_str_list(self, feat_configs):
        if feat_configs:
            feat_types = []
            for feat_config in feat_configs:
                if feat_config['type'] == 'from_ranks':
                    feat_types.append(f"from_ranks, ({' + '.join(feat_config['feat_list'])})")
                else:
                    feat_types.append(feat_config['type'])
        else:
            feat_types = ['no_feat']

        return feat_types

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
    
    def set_sparsifier_in_path(self, spf_config, spf_domain_config, spf_feat_configs, spf_reduc_config, log_path):
        expert_str = "True" if spf_config['is_expert'] else "False"
        
        if spf_config['type'] != 'no_spf':
            if spf_config['type'] != 'vanilla':
                log_path = os.path.join(log_path, f"(spf) {spf_config['type']} (is_exp_top = {expert_str}) ({spf_domain_config['type']})") 

                # sparsifer features
                feat_types = self.get_feat_type_str_list(spf_feat_configs)
                reduc_type = spf_reduc_config['type'] if spf_reduc_config else 'no_reduc'

                log_path = os.path.join(log_path, f"(spf) [{' + '.join(feat_types)}] [{reduc_type}]")           
            else:
                log_path = os.path.join(log_path, f"(spf) {spf_config['type']} (is_exp_top = {expert_str})")
        else:
            log_path = os.path.join(log_path, f"(spf) {spf_config['type']}")

        return log_path

# Functions to load checkpoint files

def get_model_ckpt_path(log_path, always_highest_version=False):
    ckpt_dir = os.path.join(log_path, 'checkpoints')

    contents = os.listdir(ckpt_dir)

    if len(contents) == 0:
        raise ValueError(f"No .ckpt files found in {ckpt_dir}. Please check the directory.")

    if always_highest_version:
        contents = sorted(contents, key=lambda x: int(x.split('-step=')[1].split('.ckpt')[0]))
        model_path = os.path.join(ckpt_dir, f"{contents[-1]}")
    
    else:
        print(f"\n.ckpt_files available in {ckpt_dir}:\n")
        print(contents)

        if len(contents) > 1:
            user_input = input("\nEnter the ckpt file to load (include the quotes) (e.g., 'epoch=1-step=1000.ckpt'): ").strip("'\"")

            if user_input not in contents:
                raise ValueError(f"Invalid ckpt file name: {user_input}. Available files: {contents}")
            
            model_path = os.path.join(ckpt_dir, f"{user_input}")

        elif len(contents) == 1:
            model_path = os.path.join(ckpt_dir, f"{contents[0]}")
    
    return model_path

def get_param_pickle_path(log_path):
    """
    Returns the path to the training parameters pickle file.
    """
    param_path = os.path.join(log_path, 'train_config.pkl')
    if not os.path.exists(param_path):
        raise FileNotFoundError(f"Training parameters file not found: {param_path}. Please check the log path.")
    
    return param_path


# Functions to load from selection folders

def get_selected_model_path(framework, is_multi=False):
    """
    Returns the path to the selected checkpoint (model) file for the given framework.
    The path is read from a text file in the settings directory.

    Parameters
    ----------
    framework : str
        The framework for which to load the checkpoint path (e.g., 'nri', 'decoder').
    is_multi : bool, optional
        If True, returns a list of multiple model paths. If False, returns a single model path. Default is False.
    """
    if is_multi:
        with open(os.path.join(SETTINGS_DIR, f"{framework}_selections", "multi_model_paths.txt"), "r") as f:
            model_paths = [(line.strip()) for line in f if line.strip()]
        return model_paths
    else:
        with open(os.path.join(SETTINGS_DIR, f"{framework}_selections", "single_model_path.txt"), "r") as f:
            model_path = f.read() 
        return model_path

def load_log_config(framework, model_path):
    """
    Loads the training config class from a pickle file for the given framework.
    The path to the pickle file is read from a text file in the settings directory.

    Parameters
    ----------
    framework : str
        The framework for which to load the training config class (e.g., 'nri', 'decoder').
    """
    if framework == 'nri':
        log_config = NRITrainManager(DataConfig())
    elif framework == 'decoder':
        log_config = DecoderTrainManager(DataConfig())

    log_config_path = os.path.join(os.path.dirname(model_path), 'train_config.pkl')

    if not os.path.exists(log_config_path):
        raise ValueError(f"\nThe parameter file does not exists")
    
    with open(log_config_path, 'rb') as f:
        log_config.__dict__.update(pickle.load(f))
    
    return log_config