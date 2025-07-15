import os
import shutil
import re
from pathlib import Path
from rich.tree import Tree
from rich.console import Console
import sys


class TopologyEstimatorConfig:
    def __init__(self):        
        # Sim params
        self.version = 1
        self.continue_training = False

        self.is_log = False
        self.is_sparsifier = False
        self.is_nri = True
        self.fex_type = None

    def set_tp_dataset_params(self):
        self.batch_size = 50
        self.train_rt   = 0.8
        self.test_rt    = 0.2
        self.val_rt     = 1 - (self.train_rt + self.test_rt)

    def set_training_params(self):
        self.max_epochs = 5
        self.lr = 0.001
        self.optimizer = 'adam'

        self.loss_type_encd = 'kld'
        self.prior = None
        self.add_const_kld = False  # if True, adds a constant term to the KL divergence

        self.loss_type_decd = 'nnl'

    def set_encoder_params(self):
        """
        Sets encoder parameters for the model.

        """
        # ------ Pipeline Parameters ------
        self.pipeline_type          = 'mlp_1'   # default pipeline type
        self.encoder_pipeline       = self._get_encoder_pipeline(self.pipeline_type)

        self.n_edge_types           = 2         # output size 
        self.is_residual_connection = True      # if True, then use residual connection in the last layer


        # ------ Embedding Function Parameters ------
        # embedding config
        edge_emd_config             = {'mlp': 'default',
                                       'cnn': 'default'}
        
        node_emb_config             = {'mlp': 'default',
                                       'cnn': 'default'}

        self.edge_emb_configs_enc   = self._get_encoder_emb_config(config_type=edge_emd_config)  
        self.node_emb_configs_enc   = self._get_encoder_emb_config(config_type=node_emb_config)
        
        # other embedding parameters
        self.dropout_prob_enc       = {'mlp': 0.0,
                                       'cnn': 0.0}
        
        self.batch_norm_enc         = {'mlp': False,
                                       'cnn': False}


        # ------ Attention Parameters ------
        self.attention_output_size  = 5        # output size for attention layer

        # ------ Gumble Softmax Parameters ------
        self.temp                   = 1.0       # temperature for Gumble Softmax
        self.is_hard                = True      # if True, use hard Gumble Softmax

    def set_decoder_params(self):

        self.msg_out_size           = 64
    
        # ---------- Embedding function parameters ----------
        # embedding config
        edge_mlp_config             = {'mlp': 'default'}
        self.edge_mlp_config_dec    = self._get_decoder_emb_config(config_type=edge_mlp_config)['mlp']

        output_mlp_config           = {'mlp': 'default'}
        self.out_mlp_config_dec     = self._get_decoder_emb_config(config_type=output_mlp_config)['mlp']

        # other embedding parameters
        self.skip_first_edge_type   = True 
        self.dropout_prob_dec       = 0
        self.is_batch_norm_dec      = True

        # ------ Recurrent Embedding Parameters ------
        self.recurrent_emd_type     = 'gru' # options: gru

    def set_sparsifier_params(self):
        pass

    
        
        
    # =======================================
    # Extra methods
    # =======================================

    def _get_encoder_pipeline(self, pipeline_type, custom_pipeline=None):
        """
        pipeline_type : str
        custom_pipeline : list or None
        """
        pipelines = {
            'mlp_1': [
                        ['1/node_emd.1', 'mlp'],
                        ['1/node_emd.2', 'mlp'],
                        ['1/pairwise_op', 'sum'],
                        ['1/edge_emd.1.@', 'mlp'],
                        ['2/aggregate', 'sum'],
                        ['2/node_emd.1', 'mlp'],
                        ['2/node_emd.2', 'mlp'],
                        ['2/pairwise_op', 'concat'],
                        ['2/edge_emd.1', 'mlp'],
                        ['2/edge_emd.2', 'mlp']
                     ],
            'cnn1': [] 
        }

        # ------- Validate pipeline_type -------
        if pipeline_type in pipelines:
            return pipelines[pipeline_type]
        
        elif pipeline_type == 'custom':
            if custom_pipeline is None:
                raise ValueError("Custom pipeline must be provided when pipeline_type is 'custom'.")
            else:
                # placeholder for validation logic of custom pipeline
                pass
            return custom_pipeline
        
        
    def _get_encoder_emb_config(self, config_type, custom_config=None): 
        """
        config_type : dict
        custom_config : dict

        Attributes
        ----------
        Write description of the mlp_config names, cnn_config names etc.
        """     

        # Dictionaries
        mlp_configs = {
            'default': [[64, 'relu'],
                        [32, 'relu'],
                        [16, 'relu'],
                        [8, None]],
        }

        cnn_configs = {
            'default': [[5, 2, 64], 
                        [8]] # the last list for CNN will have one element i.e. the output CHANNEL size
        }

        # ------ Validate config_type -------
        configs = {}
        for key, value in config_type.items():
            if key == 'mlp':
                if value in mlp_configs: 
                    configs[key] = mlp_configs[value]
                elif value == 'custom':
                    if custom_config is None:
                        raise ValueError("Custom MLP config must be provided when value is 'custom'.")
                    else:
                        configs[key] = custom_config[key] 
            elif key == 'cnn':    
                if value in cnn_configs:
                    configs[key] = cnn_configs[value]
                elif value == 'custom':
                    if custom_config is None:
                        raise ValueError("Custom CNN config must be provided when value is 'custom'.")
                    else:
                        configs[key] = custom_config[key]

        return configs
    
    def _get_decoder_emb_config(self, config_type, custom_config=None):
        """
        config_type : dict
        custom_config : dict

        Attributes
        ----------
        Write description of the mlp_config names, cnn_config names etc.
        """     

        # Dictionaries
        mlp_configs = {
            'default': [[64, 'tanh'],
                        [32, 'tanh'],
                        [16, 'tanh'],
                        [self.msg_out_size, None]], # the last layer should look like this for any configs for decoder
        }

        # ------ Validate config_type -------
        configs = {}
        for key, value in config_type.items():
            if key == 'mlp':
                if value in mlp_configs: 
                    configs[key] = mlp_configs[value]
                elif value == 'custom':
                    if custom_config is None:
                        raise ValueError("Custom MLP config must be provided when value is 'custom'.")
                    else:
                        configs[key] = custom_config[key] 
            else:
                raise ValueError(f"Unsupported config type: {key}. Supported types are 'mlp'.")

        return configs
    
    def set_log_path(self, data_config, n_datapoints):
        """
        Returns the path to store the logs based on data and topology config

        Parameters
        ----------
        data_config : Object
            The data configuration object of class DataConfig
        n_datapoints : int
            The number of data points in the dataset (time or frequency domain)
        """
        self.log_path = os.path.join('logs', 
                                f'{data_config.application_map[data_config.application]}', 
                                f'{data_config.machine_type}',
                                f'{data_config.scenario}')
        
        if self.is_nri and self.is_sparsifier:
            self.log_path = os.path.join(self.log_path, 'novel_nri')
        elif self.is_nri and not self.is_sparsifier:
            self.log_path = os.path.join(self.log_path, 'std_nri')
        elif self.is_sparsifier and not self.is_nri:
            self.log_path = os.path.join(self.log_path, 'skeleton_graph')
        elif not self.is_nri and not self.is_sparsifier:
            raise ValueError("Both is_nri and is_sparsifier cannot be False. At least one should be True.")

        self.log_path = os.path.join(self.log_path, f'enc={self.pipeline_type}_dec={self.recurrent_emd_type}',
                                f'dp={n_datapoints}')
        
        # add healthy or healthy_unhealthy config to path
        if data_config.unhealthy_config == []:
            self.log_path = os.path.join(self.log_path, 'healthy')

        elif data_config.unhealthy_config != []:
            self.log_path = os.path.join(self.log_path, 'healthy_unhealthy')

        healthy_config_str_list = []

        for config in data_config.healthy_config:
            healthy_type = config[0]
            augments = config[1]

            augment_str = '+'.join(augments) 

            healthy_config_str_list.append(f'{healthy_type}_[{augment_str}]')

        config_str = '_+_'.join(healthy_config_str_list)

        if data_config.unhealthy_config != []:
            unhealthy_config_str_list = []
            for config in data_config.unhealthy_config:
                unhealthy_type = config[0]
                augments = config[1]

                augment_str = '+'.join(augments) 

                unhealthy_config_str_list.append(f'{unhealthy_type}_[{augment_str}]')

            config_str += f'_+_{'_+_'.join(unhealthy_config_str_list)}'

        self.log_path = os.path.join(self.log_path, config_str)
                
        # add feature type to path
        if self.fex_type is not None:
            self.log_path = os.path.join(self.log_path, self.fex_type)
        else:
            self.log_path = os.path.join(self.log_path, '_no_fex')

        # add model version to path
        self.log_path = os.path.join(self.log_path, f'v{self.version}')
    
    def remove_version(self):
        """
        Removes the version from the log path.
        """
        if os.path.exists(self.log_path):
            user_input = input(f"Are you sure you want to remove the version {self.version} from the log path {self.log_path}? (y/n): ")
            if user_input.lower() == 'y':
                shutil.rmtree(self.log_path)
                print(f"Removed version {self.version} from the log path {self.log_path}.")

            else:
                print(f"Operation cancelled. Version {self.version} still remains.")

    
    def get_next_version(self):
        parent_dir = os.path.dirname(self.log_path)

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

    def check_if_version_exists(self):
        """
        Checks if the version already exists in the log path.
        """

        if os.path.isdir(self.log_path):
            print(f"Version {self.version} already exists in the log path '{self.log_path}'.")
            user_input = input("(a) Overwrite exsiting version, (b) create new version, (c) stop training (Choose 'a', 'b' or 'c'):  ")

            if user_input.lower() == 'a':
                self.remove_version()

            elif user_input.lower() == 'b':
                self.log_path = self.get_next_version()

            elif user_input.lower() == 'c':
                print("Stopped training.")
                sys.exit()  # Exit the program gracefully
                

    
# Functions to load checkpoint files

def get_checkpoint_path(log_path):
    ckpt_path = os.path.join(log_path, 'checkpoints')

    contents = os.listdir(ckpt_path)
    print(f".ckpt_files available in {ckpt_path}:")
    print(contents)

    if len(contents) > 1:
        user_input = input("Enter the ckpt file to load (e.g., 'epoch=1-step=1000.ckpt'): ")

        if user_input not in contents:
            raise ValueError(f"Invalid ckpt file name: {user_input}. Available files: {contents}")
        
        ckpt_path = os.path.join(ckpt_path, f"{user_input}")

    elif len(contents) == 1:
        ckpt_path = os.path.join(ckpt_path, f"{contents[0]}")

    else:
        raise ValueError(f"No .ckpt files found in {ckpt_path}. Please check the directory.")
    
    return ckpt_path

class SelectTopologyEstimatorModel():
    def __init__(self, application, machine, scenario, logs_dir="logs"):
        self.logs_dir = Path(logs_dir)
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
        for item in sorted(path.iterdir()):
            if item.is_dir():
                key = item.name
                if key.startswith("v") and key[1:].isdigit():
                    rel_path = item.relative_to(self.logs_dir)
                    self.version_paths.append(str(rel_path))
                structure[key] = self._explore(item, level + 1)
        return structure

    def print_tree(self):
        console = Console()
        # Build mapping from normalized version path to index
        version_index_map = {os.path.normpath(v): idx for idx, v in enumerate(self.version_paths)}

        tree = Tree(f"[green]{self.application}[/green]")
        machine_node = tree.add(f"[green]{self.machine}[/green]")
        scenario_node = machine_node.add(f"[green]{self.scenario}[/green]")
        self._build_rich_tree(scenario_node, self.structure, 0, [], version_index_map)
        console.print(tree)
        print("\nAvailable version paths:")
        for idx, vpath in enumerate(self.version_paths):
            print(f"{idx}: logs/{vpath}")

    def _build_rich_tree(self, parent_node, structure, level, parent_keys, version_index_map):
        label_map = {
            0: "<framework>",
            1: "<model>",
            2: "<n_datapoints>",
            3: "<ds_type>",
            4: "<ds_subtype>",
            5: "<fex_type>",
            6: "<versions>"
        }
        added_labels = set()
        for key, value in structure.items():
            # Add normal blue label if at the correct level and not already added
            if level in label_map and label_map[level] not in added_labels:
                parent_node.add(f"[blue]{label_map[level]}[/blue]")
                added_labels.add(label_map[level])
            # Model folder in yellow
            if level == 1:
                branch = parent_node.add(f"[bright_yellow]{key}[/bright_yellow]")
                self._build_rich_tree(branch, value, level + 1, parent_keys + [key], version_index_map)
                continue
            # Version folders: white name, yellow index, do not recurse inside
            if key.startswith("v") and key[1:].isdigit():
                # Prepend application, machine, scenario to match version_index_map keys
                rel_path = os.path.normpath(os.path.join(
                    self.application, self.machine, self.scenario, *parent_keys, key
                ))

                idx = version_index_map[rel_path]
                parent_node.add(f"[bright_yellow]{key}[/bright_yellow] [bright_cyan][{idx}][/bright_cyan]")
                continue
            # All other folders: white
            branch = parent_node.add(f"[white]{key}[/white]")
            self._build_rich_tree(branch, value, level + 1, parent_keys + [key], version_index_map)

    def select_and_get_ckpt(self):
        self.print_tree()
        if not self.version_paths:
            print("No version paths found.")
            return None
        idx = int(input("\nEnter the index number of the version path to select: "))
        if idx < 0 or idx >= len(self.version_paths):
            print("Invalid index.")
            return None
        selected_log_path = os.path.join("logs", self.version_paths[idx])
        ckpt_file_path = get_checkpoint_path(selected_log_path)
        print(f"\nSelected .ckpt file path: {ckpt_file_path}")
        return ckpt_file_path











        
        
