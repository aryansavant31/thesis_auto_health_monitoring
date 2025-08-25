import os, sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT_DIR) if ROOT_DIR not in sys.path else None

SETTINGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SETTINGS_DIR) if SETTINGS_DIR not in sys.path else None

# global imports
from data.config import DataConfig, get_domain_config
from feature_extraction.settings.feature_config import get_freq_feat_config, get_time_feat_config, get_reduc_config


class DecoderTrainConfig:
    def __init__(self, data_config:DataConfig):
        ext = ExtraSettings()
        self.data_config = data_config

    # 1: Training parameters   

        self.model_num = 1
        self.continue_training = False
        self.is_log = True
        
        self.n_edge_types = 1

        # dataset parameters
        self.batch_size = 50
        self.train_rt = 0.8
        self.test_rt = 0.1
        self.val_rt = 0.1
        self.num_workers = 1

        # optimization parameters
        self.max_epochs = 5
        self.lr = 0.001
        self.optimizer = 'adam'
        self.loss_type = 'nll'

    # 2: Decoder parameters

        self.msg_out_size = 64
    
        # embedding function parameters 
        self.edge_mlp_config = {'mlp': 'default'}
        self.out_mlp_config = {'mlp': 'default'}

        self.do_prob = 0
        self.is_batch_norm = True

        # recurrent embedding parameters
        self.recur_emb_type = 'gru'
        
        # Run parameters
        self.dec_domain_config = get_domain_config('time')
        self.dec_raw_data_norm = None
        self.dec_feat_configs = [
            # get_time_feat_config('first_n_modes', data_config=self.data_config, n_modes=10),
        ]
        self.dec_reduc_config = None # get_reduc_config('PCA', n_components=10) # or None
        self.dec_feat_norm = None

        self.skip_first_edge_type = False
        self.pred_steps = 1
        self.is_burn_in = False
        self.burn_in_steps = 1
        self.is_dynamic_graph = False
        self.temp = 1.0       # temperature for Gumble Softmax
        self.is_hard = True      

        self.dec_edge_mlp_config = ext.get_dec_emb_config(config_type=self.edge_mlp_config, msg_out_size=self.msg_out_size)['mlp']
        self.dec_out_mlp_config = ext.get_dec_emb_config(config_type=self.out_mlp_config, msg_out_size=self.msg_out_size)['mlp']

    # 3: Sparsifier parameters

        self.spf_config = get_spf_config('no_spf', is_expert=False)

        self.spf_domain_config = get_domain_config('time')
        self.spf_raw_data_norm = None 
        self.spf_feat_configs = [
           # get_time_feat_config('first_n_modes', data_config=self.data_config),
        ]    
        self.spf_reduc_config = None # get_reduc_config('PCA', n_components=10) # or None
        self.spf_feat_norm = None

    # 4: Hyperparameters and plots
        self.hyperparams = self.get_hyperparams()


    def get_hyperparams(self):
        """
        Sets the hyperparameters for the decoder model.
        """
        domain_dec_str = get_config_str([self.dec_domain_config])
        feat_dec_str = get_config_str(self.dec_feat_configs)
        reduc_dec_str = get_config_str([self.dec_reduc_config]) if self.dec_reduc_config else 'None'

        spf_domain_str = get_config_str([self.spf_domain_config])
        spf_feat_str = get_config_str(self.spf_feat_configs)
        spf_reduc_str = get_config_str([self.spf_reduc_config]) if self.spf_reduc_config else 'None'

        hyperparams = {
            'batch_size': self.batch_size,
            'train_rt': self.train_rt,
            'test_rt': self.test_rt,
            'val_rt': self.val_rt,
            'max_epochs': self.max_epochs,
            'lr': self.lr,
            'optimizer': self.optimizer,
            'loss_type': self.loss_type,
            'n_edge_types': self.n_edge_types,
            'window_length': self.data_config.window_length,
            'stride': self.data_config.stride,

            # decoder parameters
            'dec/msg_out_size': self.msg_out_size,
            'dec/recur_emb_type': self.recur_emb_type,
            'dec/do_prob': self.do_prob,
            'dec/batch_norm': self.is_batch_norm,
            'dec/domain': domain_dec_str,
            'dec/raw_data_norm': self.dec_raw_data_norm,
            'dec/feats': f"[{feat_dec_str}]",
            'dec/reduc': reduc_dec_str,
            'dec/feat_norm': self.dec_feat_norm,
            'dec/skip_first_edge': self.skip_first_edge_type,
            'dec/pred_steps': self.pred_steps,
            'dec/is_burn_in': self.is_burn_in,
            'dec/burn_in_steps': self.burn_in_steps,
            'dec/is_dynamic_graph': self.is_dynamic_graph,
            'enc/temp': self.temp,
            'enc/is_hard': self.is_hard,
            'dec/edge_mlp_config': f"{self.edge_mlp_config}",
            'dec/out_mlp_config': f"{self.out_mlp_config}",

            # sparsifier parameters
            'spf/config': f"{self.spf_config['type']} (expert={self.spf_config['is_expert']})" if self.spf_config['type'] != 'no_spf' else 'no_spf',
            'spf/domain': spf_domain_str,
            'spf/raw_data_norm': self.spf_raw_data_norm,
            'spf/feats': f"[{spf_feat_str}]",
            'spf/reduc': spf_reduc_str,
            'spf/feat_norm': self.spf_feat_norm
        }

        for key, value in hyperparams.items():
            if isinstance(value, list):
                hyperparams[key] = ', '.join(map(str, value))
            elif isinstance(value, (int, float)):
                hyperparams[key] = str(value)
            elif value is None:
                hyperparams[key] = 'None'

        return hyperparams
    

class NRITrainConfig:
    def __init__(self, data_config:DataConfig):
        """
        1: Training Attributes
        -----------------------
        model_num : int
            Model number for training.
        continue_training : bool
            Whether to continue training from a previous checkpoint.
        is_log : bool
            Whether to log training progress.
        n_edge_types : int
            Number of edge types to consider in the nri model.

        - **_Dataset parameters_**

        batch_size : int
            Batch size for training.
        train_rt : float
            Ratio of training data.
        test_rt : float
            Ratio of testing data.
        val_rt : float
            Ratio of validation data.
        
        - **_Optimization parameters_**
        max_epochs : int
            Maximum number of epochs for training.
        lr : float
            Learning rate for the optimizer.
        optimizer : str
            Type of optimizer to use (`adam`)
        loss_type_enc : str
            Type of loss function for the encoder (`kld`)

        2: Encoder Attributes
        -----------------------
        - **_Pipeline parameters_**
        pipeline_type : str
            Type of pipeline to use for the encoder (`mlp_1`)
        is_residual_connection : bool
            if True, then use residual connection in the last layer

        - **_Embedding function parameters_**
        do_prob_enc : dict
            Dropout (do) probabilities for encoder layers.
        bn_enc : dict
            Whether to use batch normalization (bn) in encoder layers.

        - **_Run parameters_**
        enc_domain_config : str
            Domain configuration for the encoder (`time`, `freq`)
        enc_norm : str
            Normalization type for the encoder (`std`, `minmax`, `None`)

        3: Decoder Attributes
        -----------------------
        recur_emd_type : str
            Type of recurrent embedding to use in the decoder (`gru`, `mlp`) 
            ( if `mlp`, then only output mlp)
        """
        ext = ExtraSettings()
        self.data_config = data_config

    # 1: Training parameters   

        self.model_num = 1
        self.continue_training = False
        self.is_log = True
        
        self.n_edge_types = 1

        # dataset parameters
        self.batch_size = 50
        self.train_rt = 0.8
        self.test_rt = 0.1
        self.val_rt = 0.1
        self.num_workers = 1

        # optimization parameters
        self.max_epochs = 5
        self.lr = 0.001
        self.optimizer = 'adam'

        self.loss_type_enc = 'kld'
        self.prior = None
        self.add_const_kld = True  # this needs to be True, adds a constant term to the KL divergence

        self.loss_type_dec = 'nll'

    # 2: Encoder parameters

        # pipeline parameters
        self.pipeline_type = 'mlp_1' 
        self.is_residual_connection = True 

        # embedding function parameters
        self.edge_emb_config = {
            'mlp': 'default',
            'cnn': 'default'
            }
        self.node_emb_config = {
            'mlp': 'default',
            'cnn': 'default'
            }

        self.enc_do_prob = {
            'mlp': 0.0,
            'cnn': 0.0
            }
        self.enc_is_batch_norm = {
            'mlp': True,
            'cnn': False
            }
        # attention parameters
        self.attention_output_size = 5   

        # Run parameters
        self.enc_domain_config = get_domain_config('time')
        self.enc_raw_data_norm = None  
        self.enc_feat_configs = []
        self.enc_reduc_config = None # get_reduc_config('PCA', n_components=10) # or None
        self.enc_feat_norm = None

        # gumble softmax parameters
        self.temp = 1.0       
        self.is_hard = True   

        self.pipeline = ext.get_enc_pipeline(self.pipeline_type)  
        self.enc_edge_emb_configs = ext.get_enc_emb_config(config_type=self.edge_emb_config)  
        self.enc_node_emb_configs = ext.get_enc_emb_config(config_type=self.node_emb_config)

    # 3: Decoder parameters

        self.msg_out_size = 64
    
        # embedding function parameters 
        self.edge_mlp_config = {'mlp': 'default'}
        self.out_mlp_config = {'mlp': 'default'}

        self.dec_do_prob = 0
        self.dec_is_batch_norm = True

        # recurrent embedding parameters
        self.recur_emb_type = 'gru'
        
        # Run parameters
        self.dec_domain_config = get_domain_config('time')
        self.dec_raw_data_norm = None 
        self.dec_feat_configs = [
            # get_time_feat_config('first_n_modes', data_config=self.data_config, n_modes=10),
        ]
        self.dec_feat_norm = None
        self.dec_reduc_config = None # get_reduc_config('PCA', n_components=10) # or None
        
        self.skip_first_edge_type = True 
        self.pred_steps = 1
        self.is_burn_in = False
        self.burn_in_steps = 1
        self.is_dynamic_graph = False

        self.dec_edge_mlp_config = ext.get_dec_emb_config(self.edge_mlp_config, self.msg_out_size)['mlp']
        self.dec_out_mlp_config = ext.get_dec_emb_config(self.out_mlp_config, self.msg_out_size)['mlp']

    # 4: Sparsifier parameters

        self.spf_config = get_spf_config('vanilla', is_expert=True)
        
        self.spf_domain_config   = get_domain_config('time')
        self.spf_raw_data_norm = None 
        self.spf_feat_configs = [
            # get_time_feat_config('first_n_modes'),
        ]    
        self.spf_feat_norm = None
        self.spf_reduc_config = None # get_reduc_config('PCA', n_components=10) # or None
        
    # 5: Hyperparameters and plots
        self.hyperparams = self.get_hyperparams()

    def get_hyperparams(self):
        """
        Sets the hyperparameters for the NRI model.
        """
        domain_enc_str = get_config_str([self.enc_domain_config])
        feat_enc_str = get_config_str(self.enc_feat_configs)
        reduc_enc_str = get_config_str([self.enc_reduc_config]) if self.enc_reduc_config else 'None'

        domain_dec_str = get_config_str([self.dec_domain_config])
        feat_dec_str = get_config_str(self.dec_feat_configs)
        reduc_dec_str = get_config_str([self.dec_reduc_config]) if self.dec_reduc_config else 'None'

        spf_domain_str = get_config_str([self.spf_domain_config])
        spf_feat_str = get_config_str(self.spf_feat_configs)
        spf_reduc_str = get_config_str([self.spf_reduc_config]) if self.spf_reduc_config else 'None'

        hyperparams = {
            'batch_size': self.batch_size,
            'train_rt': self.train_rt,
            'test_rt': self.test_rt,
            'val_rt': self.val_rt,
            'window_length': self.data_config.window_length,
            'stride': self.data_config.stride,
            'max_epochs': self.max_epochs,
            'lr': self.lr,
            'optimizer': self.optimizer,
            'enc/loss_type': self.loss_type_enc,
            'dec/loss_type': self.loss_type_dec,
            'n_edge_types': self.n_edge_types,
            'enc/prior': self.prior,
            'enc/add_const_kld': self.add_const_kld,

            # encoder parameters
            'enc/pipeline_type': self.pipeline_type,
            'enc/is_residual_connection': self.is_residual_connection,
            'enc/do_prob': f"{self.enc_do_prob}",
            'enc/is_batch_norm': f"{self.enc_is_batch_norm}",
            'enc/domain': domain_enc_str,
            'enc/raw_data_norm': self.enc_raw_data_norm,
            'enc/feats': f"[{feat_enc_str}]",
            'enc/reduc': reduc_enc_str,
            'enc/feat_norm': self.enc_feat_norm,
            'enc/edge_emb_configs_enc': f"{self.edge_emb_config}",
            'enc/node_emb_configs_enc': f"{self.node_emb_config}",
            'enc/temp': self.temp,
            'enc/is_hard': self.is_hard,
            'enc/attention_output_size': self.attention_output_size,

            # decoder parameters
            'dec/msg_out_size': self.msg_out_size,
            'dec/do_prob': self.dec_do_prob,
            'dec/is_batch_norm': self.dec_is_batch_norm,
            'dec/recur_emb_type': self.recur_emb_type,
            'dec/domain': domain_dec_str,
            'dec/raw_data_norm': self.dec_raw_data_norm,
            'dec/feats': f"[{feat_dec_str}]",
            'dec/reduc': reduc_dec_str,
            'dec/feat_norm': self.dec_feat_norm,
            'dec/edge_mlp_config': f"{self.edge_mlp_config}",
            'dec/out_mlp_config': f"{self.out_mlp_config}",
            'dec/dec_skip_first_edge': self.skip_first_edge_type,
            'dec/pred_steps': self.pred_steps,
            'dec/is_burn_in': self.is_burn_in,
            'dec/burn_in_steps': self.burn_in_steps,
            'dec/is_dynamic_graph': self.is_dynamic_graph,

            # sparsifier parameters
            'spf/config': f"{self.spf_config['type']} (expert={self.spf_config['is_expert']})" if self.spf_config['type'] != 'no_spf' else 'no_spf',
            'spf/domain': spf_domain_str,
            'spf/raw_data_norm': self.spf_raw_data_norm,
            'spf/feats': f"[{spf_feat_str}]",
            'spf/reduc': spf_reduc_str,
            'spf/feat_norm': self.spf_feat_norm
        }

        for key, value in hyperparams.items():
            if isinstance(value, list):
                hyperparams[key] = ', '.join(map(str, value))
            elif isinstance(value, (int, float)):
                hyperparams[key] = str(value)
            elif value is None:
                hyperparams[key] = 'None'

        return hyperparams
        

def get_config_str(configs:list):
    """
    Get a neat string that has the type of config and its parameters.
    Eg: "PCA(comps=3)"
    """
    config_strings = []

    for config in configs:
        additional_keys = ', '.join([f"{key}={value}" for key, value in config.items() if key not in ['fs', 'type', 'feat_list']])
        if additional_keys:
            config_strings.append(f"{config['type']}({additional_keys})")
        else:
            config_strings.append(f"{config['type']}")

    return ', '.join(config_strings)


class ExtraSettings:
    def get_enc_pipeline(self, pipeline_type, custom_pipeline=None):
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
        
        
    def get_enc_emb_config(self, config_type, custom_config=None): 
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
    
    def get_dec_emb_config(self, config_type, msg_out_size, custom_config=None):
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
                        [msg_out_size, None]], # the last layer should look like this for any configs for decoder
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
    
def get_spf_config(spf_type, **kwargs):
    """
    spf_type : str
        Type of sparsifier to use.
    **kwargs : dict
        For all options of `spf_type`:
        - `no_spf`: None, *_is_expert = False_*
        - `vanilla`: None, *_is_expert = True_*
        
    
    Returns
    -------
    config : dict
        Configuration dictionary for the specified sparsifier type.
    """
    config = {}
    config['type'] = spf_type
    config['is_expert'] = kwargs.get('is_expert', False) 

    if spf_type == 'no_spf':
        config['is_expert'] = False
    elif spf_type == 'vanilla':
        config['is_expert'] = True
    
    return config
    
if __name__ == "__main__":
    from topology_estimation.settings.manager import SelectTopologyEstimatorModel
    user_text = "To view/select trained nri models, type (a)\nTo view/select trained decoder models, type (b)\nEnter input: "
    user_input = input(user_text).strip("'\"")
    if user_input.lower() == 'a':
        framework = 'nri'
    elif user_input.lower() == 'b':
        framework = 'decoder'
    else:
        raise ValueError("Invalid input. Please enter 'a', 'b', or 'c'.")
    model_selector = SelectTopologyEstimatorModel(framework=framework, run_type='train')
    model_selector.select_ckpt_and_params()




        
        
