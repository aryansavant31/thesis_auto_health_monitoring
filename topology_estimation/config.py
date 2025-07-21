import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

import shutil
import re
from pathlib import Path
from rich.tree import Tree
from rich.console import Console
from feature_extraction import get_fex_config
from data.config import DataConfig
import pickle


class PredictNRIConfig:
    def __init__(self):
        self.data_config = DataConfig()
        self.set_predict_params()
        self.set_run_params()
        self.set_ckpt_path()
        
    def set_predict_params(self):
        self.version = 1
        self.is_custom_test = True
        self.batch_size = 50
        self.amt_rt = 0.8

    def set_custom_test_params(self):
        pass

    def set_run_params(self):
        log_config = self.load_log_config()

       # input graph paramters
        self.sparsif_type     = 'path'
        self.domain_sparsif   = log_config.domain_sparsif  # options: time, frequency
        self.fex_configs_sparsif = log_config.fex_configs_sparsif  # first feature extraction config type

        self.domain_encoder   = log_config.domain_encoder
        self.norm_type_encoder = log_config.norm_type_encoder
        self.fex_configs_encoder = log_config.fex_configs_encoder

        # Gumble Softmax Parameters
        self.temp = 1.0       # temperature for Gumble Softmax
        self.is_hard = True      # if True, use hard Gumble Softmax

        # decoder run parameters
        self.domain_decoder = log_config.domain_decoder   # options: time, frequency
        self.norm_type_decoder = log_config.norm_type_decoder
        self.fex_configs_decoder = log_config.fex_configs_decoder

        self.skip_first_edge_type = True 
        # TASK: add rest of the decoder run params

    def set_ckpt_path(self):
         with open(".\\docs\\loaded_ckpt_path.txt", "r") as f:
            self.ckpt_path = f.read() 

    def load_log_config(self):
        log_config = TrainNRIConfig()

        with open(".\\docs\\loaded_config_path.txt", "r") as f:
            log_config_path = f.read()

        if not os.path.exists(log_config_path):
            raise ValueError(f"\nThe parameter file does not exists")
        
        with open(log_config_path, 'rb') as f:
            log_config.__dict__.update(pickle.load(f))

        return log_config
    
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
    
    def get_infer_log_path(self):
        """
        Sets the log path for the predict run.
        """
        log_config = self.load_log_config()

        train_log_path = log_config.train_log_path
        if self.is_custom_test:
            self.data_config.set_custom_test_dataset()
            infer_log_path = os.path.join(train_log_path, 'test')
        else:
            self.data_config.set_predict_dataset()
            infer_log_path= os.path.join(train_log_path, 'predict')

        # add healthy or healthy_unhealthy config to path
        infer_log_path = self._set_ds_types_in_path(infer_log_path)

        # add timestep_id to path
        infer_log_path = os.path.join(infer_log_path, f'{self.data_config.timestep_id}')

        # add sparsifier type to path
        if self.sparsif_type is not None:
            infer_log_path = os.path.join(infer_log_path, f'sparsif=[{self.sparsif_type}+{self.domain_sparsif}]') 

            # sparsifer features
            fex_types_sparsif = [fex['type'] for fex in log_config.fex_configs_sparsif]
            if fex_types_sparsif:
                infer_log_path = os.path.join(infer_log_path, f'(sparsif)=[{'+'.join(fex_types_sparsif)}]')
            else:
                infer_log_path = os.path.join(infer_log_path, '(sparsif)=_no_fex')
        else:
            infer_log_path = os.path.join(infer_log_path, 'sparsif=_no_sparsif')

        # add version
        if self.is_custom_test:
            infer_log_path = os.path.join(infer_log_path, f'test_v{self.version}')
        else:
            infer_log_path = os.path.join(infer_log_path, f'predict_v{self.version}')

        return infer_log_path


class TrainNRIConfig:
    def __init__(self):
        self.set_training_params()
        self.set_run_params()

        self.data_config = DataConfig()
        self.data_config.set_train_dataset()

    def set_training_params(self):        
        self.version = 1
        self.continue_training = False

        self.is_log = False
        
        self.n_edge_types = 2

        # dataset parameters
        self.batch_size = 50
        self.train_rt   = 0.8
        self.test_rt    = 0.2
        self.val_rt     = 1 - (self.train_rt + self.test_rt)

        # optimization parameters
        self.max_epochs = 5
        self.lr = 0.001
        self.optimizer = 'adam'

        self.loss_type_enc = 'kld'
        self.prior = None
        self.add_const_kld = True  # this needs to be True, adds a constant term to the KL divergence

        self.loss_type_dec = 'nnl'

    def set_run_params(self):
        """
        Attributes
        ----------
        sparsif_type : str
            The type of sparsifier to use. 
            ('knn', 'none')

        norm_type_sparsif : str
        norm_type_encoder : str
        norm_type_decoder : str
            The normalization type to use for the sparsifier, encoder, and decoder.
            ('std', 'minmax', 'none')

        [TODO] Add all the attributes of this method in docstring

        """
        # input graph paramters
        self.sparsif_type = 'knn'  # [TODO]: Get sparsif type from get_sparsif_config() method
        self.domain_sparsif   = 'time'  # options: time, freq-psd, freq-amp
        self.fex_configs_sparsif = [
            get_fex_config('first_n_modes'),
            get_fex_config('lucas', height=9, age=20)
        ]    
        self.norm_type_sparsif = 'std'  # options: std, minmax, none

        # encoder run parameters
        self.domain_encoder   = 'freq'  # options: time, freq-psd, freq-amp
        self.norm_type_encoder = 'std'  # options: std, minmax, none

        self.fex_configs_encoder = [
            get_fex_config('first_n_modes', n_modes=69),
        ]
        # gumble softmax parameters
        self.temp = 1.0       # temperature for Gumble Softmax
        self.is_hard = True      # if True, use hard Gumble Softmax

        # decoder run parameters
        self.domain_decoder = 'time'   # options: time, freq-psd, freq-amp
        self.norm_type_decoder = 'std' # options: std, minmax, none

        self.fex_configs_decoder = [
            get_fex_config('first_n_modes', n_modes=10),
        ]

        self.skip_first_edge_type = True 
        # TASK: add rest of the decoder run params

    def set_encoder_params(self):
        """
        Sets encoder parameters for the model.

        """
        # ------ Pipeline Parameters ------
        self.pipeline_type          = 'mlp_1'   # default pipeline type
        self.encoder_pipeline       = self._get_encoder_pipeline(self.pipeline_type)
         
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

    def set_decoder_params(self):

        self.msg_out_size           = 64
        self.n_edge_types_dec       = self.n_edge_types
    
        # ---------- Embedding function parameters ----------
        # embedding config
        edge_mlp_config             = {'mlp': 'default'}
        self.edge_mlp_config_dec    = self._get_decoder_emb_config(config_type=edge_mlp_config)['mlp']

        output_mlp_config           = {'mlp': 'default'}
        self.out_mlp_config_dec     = self._get_decoder_emb_config(config_type=output_mlp_config)['mlp']

        # other embedding parameters
        self.dropout_prob_dec       = 0
        self.is_batch_norm_dec      = True

        # ------ Recurrent Embedding Parameters ------
        self.recurrent_emd_type     = 'gru' # options: gru, 'mlp', if mlp, then only output mlp

    def get_sparsif_config(self, sparsif_type, **kwargs):
        config = {}
        config['type'] = sparsif_type

        # [TODO]: define all the parameters depending on sparsif_type and attach it to config dict (like get_fex_config() method)
        
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
                        ['1/pairwise_op', 'mean'],
                        ['1/edge_emd.1.@', 'mlp'],
                        ['2/aggregate', 'mean'],
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
    
    def _set_sparsifier_in_path(self, log_path):
        if self.sparsif_type is not None:
            log_path = os.path.join(log_path, f'sparsif=[{self.sparsif_type}+{self.domain_sparsif}]') 

            # sparsifer features
            fex_types_sparsif = [fex['type'] for fex in self.fex_configs_sparsif]
            if fex_types_sparsif:
                log_path = os.path.join(log_path, f'(sparsif)=[{'+'.join(fex_types_sparsif)}]')
            else:
                log_path = os.path.join(log_path, '(sparsif)=_no_fex')           
        else:
            log_path = os.path.join(log_path, 'sparsif=_no_sparsif')

        return log_path
    
    def get_train_log_path(self, n_components, n_dim):
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

        # For directed graph path 
        train_log_path = os.path.join(base_path, 'directed_graph',)  # add framework type

        # add num of edge types to path
        train_log_path = os.path.join(train_log_path, f'etypes={self.n_edge_types}')
                       
        # add healthy or healthy_unhealthy config to path
        train_log_path = self._set_ds_types_in_path(train_log_path)

        # add model type to path
        train_log_path = os.path.join(train_log_path, f'enc={self.pipeline_type}-dec={self.recurrent_emd_type}',)

        # add datastats to path
        train_log_path = os.path.join(train_log_path, f'{self.data_config.timestep_id}_measures=[{'+'.join(self.data_config.signal_types)}]')

        # add sparsifier type to path
        train_log_path = self._set_sparsifier_in_path(train_log_path)

        # add domain type of encoder and decoder to path
        train_log_path = os.path.join(train_log_path, f'[enc]={self.domain_encoder}-[dec]={self.domain_decoder}')

        # add feature type to path
        fex_types_encoder = [fex['type'] for fex in self.fex_configs_encoder]
        fex_types_decoder = [fex['type'] for fex in self.fex_configs_decoder]

        if fex_types_encoder and fex_types_decoder:
            train_log_path = os.path.join(train_log_path, f'(enc)=[{'+'.join(fex_types_encoder)}]-(dec)=[{'+'.join(fex_types_decoder)}]')
        elif fex_types_encoder and not fex_types_decoder:
            train_log_path = os.path.join(train_log_path, f'(enc)=[{'+'.join(fex_types_encoder)}]-(dec)=_no_fex')
        elif not fex_types_encoder and fex_types_decoder:
            train_log_path = os.path.join(train_log_path, f'(enc)=_no_fex-(dec)=[{'+'.join(fex_types_decoder)}]')
        elif not fex_types_encoder and not fex_types_decoder:
            train_log_path = os.path.join(train_log_path, '(enc)=_no_fex-(dec)=_no_fex')

        # add model shape compatibility stats to path
        train_log_path = os.path.join(train_log_path, f'enc(comps)={n_components}-dec(dims)={n_dim}')

        # add model version to path
        self.train_log_path = os.path.join(train_log_path, f'v{self.version}')

        return self.train_log_path

    def get_test_log_path(self):
        
        test_log_path = os.path.join(self.train_log_path, 'test')

        # add healthy or healthy_unhealthy config to path
        test_log_path = self._set_ds_types_in_path(test_log_path)

        # add timestep id to path
        test_log_path = os.path.join(test_log_path, f'{self.data_config.timestep_id}')

        # add sparsifier type to path
        test_log_path = self._set_sparsifier_in_path(test_log_path)

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
                

    
# Functions to load checkpoint files

def get_checkpoint_path(log_path):
    ckpt_path = os.path.join(log_path, 'checkpoints')

    contents = os.listdir(ckpt_path)
    print(f"\n.ckpt_files available in {ckpt_path}:\n")
    print(contents)

    if len(contents) > 1:
        user_input = input("\nEnter the ckpt file to load (e.g., 'epoch=1-step=1000.ckpt'): ").strip("'\"")

        if user_input not in contents:
            raise ValueError(f"Invalid ckpt file name: {user_input}. Available files: {contents}")
        
        ckpt_path = os.path.join(ckpt_path, f"{user_input}")

    elif len(contents) == 1:
        ckpt_path = os.path.join(ckpt_path, f"{contents[0]}")

    else:
        raise ValueError(f"No .ckpt files found in {ckpt_path}. Please check the directory.")
    
    return ckpt_path

def get_param_pickle_path(log_path):
    """
    Returns the path to the training parameters pickle file.
    """
    param_path = os.path.join(log_path, 'train_config.pkl')
    if not os.path.exists(param_path):
        raise FileNotFoundError(f"Training parameters file not found: {param_path}. Please check the log path.")
    
    return param_path
    
class SelectTopologyEstimatorModel():
    def __init__(self, framework, application=None, machine=None, scenario=None, logs_dir="logs"):
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
        self.structure = {}
        self.version_paths = []
        self._build_structure()

    def _build_structure(self):
        base = self.logs_dir / self.application / self.machine / self.scenario / self.framework
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
        framework_node = scenario_node.add(f"[green]{self.framework}[/green]")
        self._build_rich_tree(framework_node, self.structure, 0, [], version_index_map)
        console.print(tree)
        print("\nAvailable version paths:")
        for idx, vpath in enumerate(self.version_paths):
            print(f"{idx}: logs/{vpath}")

    def _build_rich_tree(self, parent_node, structure, level, parent_keys, version_index_map):
        current_path = [self.application, self.machine, self.scenario, self.framework] + parent_keys
        is_no_sparsif = any("_no_sparsif" in k for k in parent_keys)

        # Label maps
        if self.framework == "skeleton_graph":
            label_map = {
                0: "<ds_type>",
                1: "<ds_subtype>",
                2: "<sparsif_type>",
                3: "<n_components>",
                4: "<domain>",
                5: "<sparsif_fex_type>"
            }
        elif is_no_sparsif:
            label_map = {
                0: "<n_edge_types>",
                1: "<ds_type>",
                2: "<ds_subtype>",
                3: "<model>",     
                4: "<ds_stats>",
                5: "<sparsif_type>",
                6: "<domain>",
                7: "<nri_fex_type>",
                8: "<shape_compatibility>",
                9: "<versions>"
            }
        else:
            label_map = {
                0: "<n_edge_types>",
                1: "<ds_type>",
                2: "<ds_subtype>",
                3: "<model>", 
                4: "<ds_stats>",
                5: "<sparsif_type>",
                6: "<sparsif_fex_type>",
                7: "<domain>",
                8: "<nri_fex_type>",
                9: "<shape_compatibility>",
                10: "<versions>"
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
            if self.framework == "directed_graph" and level == 3:
                branch = parent_node.add(f"[bright_yellow]{safe_key}[/bright_yellow]")
                self._build_rich_tree(branch, value, level + 1, parent_keys + [key], version_index_map)
                continue
            # For skeleton graph, make the model folder under framework yellow
            if self.framework == "skeleton_graph" and level == 2:
                branch = parent_node.add(f"[bright_yellow]{safe_key}[/bright_yellow]")
                self._build_rich_tree(branch, value, level + 1, parent_keys + [key], version_index_map)
                continue

            # Version folders: bold italic yellow/green name, cyan index, do not recurse inside
            if key.startswith("v") and key[1:].isdigit():
                rel_path = os.path.normpath(os.path.join(
                    self.application, self.machine, self.scenario, self.framework, *parent_keys, key
                ))
                idx = version_index_map[rel_path]
                if is_no_sparsif:
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
        ckpt_file_path = get_checkpoint_path(selected_log_path)
        config_file_path = get_param_pickle_path(selected_log_path)

        with open(".\\docs\\loaded_ckpt_path.txt", "w") as f:
            f.write(ckpt_file_path)

        with open(".\\docs\\loaded_config_path.txt", "w") as f:
            f.write(config_file_path)

        print(f"\nSelected .ckpt file path: {ckpt_file_path}")
        print(f"\nSelected logged config file path: {config_file_path}\n")


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
                config_str += f'_+_{'_+_'.join(unhealthy_config_str_list)}'
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
            infer_log_path_sk = os.path.join(infer_log_path_sk, f'(sparsif)=[{'+'.join(fex_types_sparsif)}]')
        else:
            infer_log_path_sk = os.path.join(infer_log_path_sk, '(sparsif)=_no_fex')

        return infer_log_path_sk


if __name__ == "__main__":

    model_selector = SelectTopologyEstimatorModel(framework='directed_graph',)
    model_selector.select_ckpt_and_params()




        
        
