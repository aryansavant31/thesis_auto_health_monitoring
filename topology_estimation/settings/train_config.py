import os, sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT_DIR) if ROOT_DIR not in sys.path else None

SETTINGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SETTINGS_DIR) if SETTINGS_DIR not in sys.path else None

import torch

# global imports
from data.config import DataConfig, get_domain_config
from feature_extraction.settings.feature_config import get_freq_feat_config, get_time_feat_config, get_reduc_config


class DecoderTrainConfig:
    framework = 'decoder'

    def __init__(self, data_config:DataConfig):
        """
        1: Training Attributes
        -----------------------

        2: Decoder Attributes
        -----------------------
        - **_Pipeline parameters_**
        msg_out_size : int
            Size of the output layer of the message embedding function. 

        - **_Embedding function parameters_**

        - **_Run parameters_**

        pred_steps : int   
            Controls the frequency of ground truth injection during sequence prediction.
    
            - **pred_step = 1**: Use ground truth at every step (full teacher forcing)
            - **pred_step = N, (N > 1)**: Use ground truth every **N** steps, predictions for intermediate steps
            
            Higher values increase autoregressive behavior, lower values provide more stability but can cause exposure bias.

        is_burn_in : bool
            - if True, then use first `n_comps - final_pred_steps` steps as ground truth inputs. After that, use model predictions as inputs for last `final_pred_steps`.
            - if False, then control ground truth injection using `pred_steps` only.

        final_pred_steps : int
            Number of final steps to use as prediction inputs if `is_burn_in` is True.

        3: Sparsifier Attributes
        -----------------------
        - **_Sparsifier parameters_**

        always_fully_connected_rel : bool
            - if True, then relation matrices are set to fully connected graph in every batch. 
            - However, edge matrix adapts to the sparsifier config. 
            - If False, then relation matrices are set as per the sparsifier output.
        """
        self.ext = ExtraSettings()
        self.data_config = data_config

    # 1: Training parameters   

        self.model_num = 1  # 15 [test failed due to error](correct tp: mae loss, long prediction horizon), 16 - incorrect
        self.continue_training = True
        self.is_log = True
        
        self.n_edge_types = 1

        # dataset parameters
        self.batch_size = 50
        self.train_rt = 0.8
        self.test_rt = 0.1
        self.val_rt = 0.1
        self.num_workers = 1

        # optimization parameters
        self.max_epochs = 50
        self.lr = 0.001
        self.optimizer = 'adam'
        self.momentum = 0.9
        self.loss_type = 'mae'

    # 2: Decoder parameters

        self.msg_out_size = 256
    
        # embedding function parameters 
        self.edge_mlp_config = {'mlp': 'edge_nri_4'}
        self.out_mlp_config = {'mlp': 'out_nri_4'}

        self.do_prob = 0
        self.is_batch_norm = False
        self.is_xavier_weights = False

        # recurrent embedding parameters
        self.recur_emb_type = 'gru'
        
        # input processor parameters
        self.dec_domain_config = get_domain_config('time')
        self.dec_raw_data_norm = 'std'
        self.dec_feat_configs = [
            # get_time_feat_config('first_n_modes', data_config=self.data_config, n_modes=10),
        ]
        self.dec_reduc_config = None 
        self.dec_feat_norm = None

        # run parameters
        self.skip_first_edge_type = False
        self.pred_steps = 10
        self.is_burn_in = True
        self.final_pred_steps = 50
        self.is_dynamic_graph = False

        # if dynamic graph is true
        self.temp = 1.0    # temperature for Gumble Softmax
        self.is_hard = True   

        # plotting parameters
        self.show_conf_band = False

        self.set_dec_emb_configs()   

    # 3: Sparsifier parameters

        self.spf_config = get_spf_config('vanilla', is_expert=True)
        self.always_fully_connected_rel = True # if True, then relation matrices are set to fully connected graph in every batch. However, edge matrix adapts to the sparsifier config. If False, then relation matrices are set as per the sparsifier output.

        self.spf_domain_config = get_domain_config('time')
        self.spf_raw_data_norm = None 
        self.spf_feat_configs = [
           # get_time_feat_config('first_n_modes', data_config=self.data_config),
        ]    
        self.spf_reduc_config = None # get_reduc_config('PCA', n_components=10) # or None
        self.spf_feat_norm = None

    # 4: Hyperparameters and plots
        self.hyperparams = self.get_hyperparams()

    def set_dec_emb_configs(self):
        """
        Sets the decoder embedding function configurations based on the provided config types.
        """
        self.dec_edge_mlp_config = self.ext.get_dec_emb_config(config_type=self.edge_mlp_config, msg_out_size=self.msg_out_size)['mlp']
        self.dec_out_mlp_config = self.ext.get_dec_emb_config(config_type=self.out_mlp_config, msg_out_size=self.msg_out_size)['mlp']

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
            'momentum': self.momentum,
            'loss_type': self.loss_type,
            'n_edge_types': self.n_edge_types,
            'window_length': self.data_config.window_length,
            'stride': self.data_config.stride,

            # decoder parameters
            'dec/msg_out_size': self.msg_out_size,
            'dec/recur_emb_type': self.recur_emb_type,
            'dec/do_prob': self.do_prob,
            'dec/batch_norm': self.is_batch_norm,
            'dec/is_xavier_weights': self.is_xavier_weights,
            'dec/domain': domain_dec_str,
            'dec/raw_data_norm': self.dec_raw_data_norm,
            'dec/feats': f"[{feat_dec_str}]",
            'dec/reduc': reduc_dec_str,
            'dec/feat_norm': self.dec_feat_norm,
            'dec/skip_first_edge': self.skip_first_edge_type,
            'dec/pred_steps': self.pred_steps,
            'dec/is_burn_in': self.is_burn_in,
            'dec/final_pred_steps': self.final_pred_steps,
            'dec/is_dynamic_graph': self.is_dynamic_graph,
            'enc/temp': self.temp,
            'enc/is_hard': self.is_hard,
            'dec/edge_mlp_config': f"{self.edge_mlp_config}",
            'dec/out_mlp_config': f"{self.out_mlp_config}",

            # sparsifier parameters
            'spf/config': f"{self.spf_config['type']} (expert={self.spf_config['is_expert']})" if self.spf_config['type'] != 'no_spf' else 'no_spf',
            'spf/always_fully_connected_rel': self.always_fully_connected_rel,
            'spf/domain': spf_domain_str,
            'spf/raw_data_norm': self.spf_raw_data_norm,
            'spf/feats': f"[{spf_feat_str}]",
            'spf/reduc': spf_reduc_str,
            'spf/feat_norm': self.spf_feat_norm
        }

        for key, value in hyperparams.items():
            if isinstance(value, list):
                hyperparams[key] = ', '.join(map(str, value))
            elif isinstance(value, (int, float, dict)):
                hyperparams[key] = str(value)
            elif value is None:
                hyperparams[key] = 'None'

        return hyperparams
    
class DecoderTrainSweep:
    def __init__(self, data_config:DataConfig):
        self.data_config = data_config
        self.train_sweep_num = 1

    # 1: Training parameters   
        # dataset parameters
        self.batch_size = [50, 100]
        self.train_rt = [0.8]
        self.test_rt = [0.1]
        self.val_rt = [0.1]

        # optimization parameters
        self.max_epochs = [5]
        self.lr = [0.001]
        self.optimizer = ['adam']
        self.loss_type = ['nll']

    # 2: Decoder parameters

        self.msg_out_size = [64]
    
        # embedding function parameters 
        self.edge_mlp_config = [
            {'mlp': 'default'}
            ]
        self.out_mlp_config = [
            {'mlp': 'default'}
            ]

        self.do_prob = [0]
        self.is_batch_norm = [True]

        # recurrent embedding parameters
        self.recur_emb_type = ['gru']
        
        # input processor parameters
        self.dec_domain_config = [get_domain_config('time')]
        self.dec_raw_data_norm = ['min_max']
        self.dec_feat_configs = [
            []
            # [get_time_feat_config('first_n_modes', data_config=self.data_config, n_modes=10)]
        ]
        self.dec_reduc_config = [None] # get_reduc_config('PCA', n_components=10) # or None
        self.dec_feat_norm = [None]

        # run parameters
        self.skip_first_edge_type = [False]
        self.pred_steps = [1]
        self.is_burn_in = [False]
        self.final_pred_steps = [1]
        self.is_dynamic_graph = [False]

        # if dynamic graph is true
        self.temp = [1.0]    # temperature for Gumble Softmax
        self.is_hard = [True]   

        # self.dec_edge_mlp_config = ext.get_dec_emb_config(config_type=self.edge_mlp_config, msg_out_size=self.msg_out_size)['mlp']
        # self.dec_out_mlp_config = ext.get_dec_emb_config(config_type=self.out_mlp_config, msg_out_size=self.msg_out_size)['mlp']

    # 3: Sparsifier parameters

        self.spf_config = [get_spf_config('vanilla', is_expert=True)]

        # self.spf_domain_config = get_domain_config('time')
        # self.spf_raw_data_norm = None 
        # self.spf_feat_configs = [
        #    # get_time_feat_config('first_n_modes', data_config=self.data_config),
        # ]    
        # self.spf_reduc_config = None # get_reduc_config('PCA', n_components=10) # or None
        # self.spf_feat_norm = None



class NRITrainConfig:
    framework = 'nri'

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
        self.ext = ExtraSettings()
        self.data_config = data_config

    # 1: Training parameters   

        self.model_num = 31 # 2 raw time data, 3 psd feats, 4 time feats
        self.continue_training = False
        self.is_log = True
        self.train_sweep = 4.3 # 3.1 - mse loss # 2 - using some possible configuration, 3 - most of tp config
        
        self.n_edge_types = 2

        # dataset parameters
        self.batch_size = 50
        self.train_rt = 0.8
        self.test_rt = 0.1
        self.val_rt = 0.1
        self.num_workers = 1

        # optimization parameters
        self.max_epochs = 30
        self.optimizer = 'adam'

        ## encoder
        self.lr_enc = 0.002
        self.loss_type_enc = 'kld'

        self.is_beta_annealing = True
        self.final_beta = 0.0005        # final value of beta after annealing
        self.warmup_frac_beta = 0.8     # fraction of total steps for warmup

        ### for kld loss
        self.prior = torch.tensor([0.5, 0.5])  # prior distribution for edge types
        self.add_const_kld = True               # this needs to be True, adds a constant term to the KL divergence

        ## decoder
        self.lr_dec = 0.002
        self.loss_type_dec = 'mae'


        # warmup parameters
        self.is_enc_warmup = True       # if True, then only train encoder with cross entrop loss until accuracy reaches warmup_acc_cutoff

        # if encoder warmup is True
        self.warmup_acc_cutoff = 1.1   # accuracy cutoff for encoder warmup
        self.sustain_enc_warmup = True  # if True, then re-enable encoder warmup if edge accuracy drops below cutoff during training
        self.final_gamma = 1
        self.warmup_frac_gamma = 0.1    # fraction of total steps for warmup

        self.dec_loss_stabilize_steps = 80  # Number of steps with constant loss to consider decoder stabilized
        self.dec_loss_bound_update_interval = 5  # Interval (in steps) to update loss bounds. Less value means more frequent updates.
        self.dec_loss_window_size = 200  # Window size for storing decoder loss. More value means longer memory.

    # 2: Encoder parameters

        # pipeline parameters
        self.pipeline_type = 'f_mlp1_og' 
        self.is_residual_connection = True 

        # embedding function parameters
        self.n_hidden_mlp = 256
        self.edge_emb_config = {
            'mlp': 'nri_og',
            'cnn': 'default'
            }
        self.node_emb_config = {
            'mlp': 'nri_og',
            'cnn': 'default'
            }

        self.enc_do_prob = {
            'mlp': 0.1,
            'cnn': 0.0
            }
        self.enc_is_batch_norm = {
            'mlp': True,
            'cnn': False
            }
        
        self.enc_is_xavier_weights = True

        # attention parameters
        self.attention_output_size = 5   

        # input processor parameters
        self.enc_domain_config = get_domain_config('time+freq')
        self.enc_raw_data_norm = None
        self.enc_feat_configs = [
            get_freq_feat_config('first_n_modes', n_modes=5),
            get_time_feat_config('mean_abs'),
            get_time_feat_config('energy'),
            get_time_feat_config('variance'),
            get_time_feat_config('std'),
            get_time_feat_config('rms'),
        ]

        self.enc_reduc_config = None # get_reduc_config('PCA', n_components=10) # or None
        self.enc_feat_norm = 'std'

        # gumble softmax parameters
        self.is_hard = False   

        self.init_temp = 1.0    # initial temperature for Gumble Softmax
        self.min_temp = 0.05     # minimum temperature for Gumble Softmax
        self.decay_temp = 0.001  # exponential decay rate for temperature  

    # 3: Decoder parameters

        self.msg_out_size = 128
    
        # embedding function parameters 
        self.edge_mlp_config = {'mlp': 'edge_nri_og'}
        self.out_mlp_config = {'mlp': 'out_nri_og'}

        self.dec_do_prob = 0
        self.dec_is_batch_norm = False
        self.dec_is_xavier_weights = False

        # recurrent embedding parameters
        self.recur_emb_type = 'gru'
        
        # input processor parameters
        self.dec_domain_config = get_domain_config('time')
        self.dec_raw_data_norm = 'std' 
        self.dec_feat_configs = [
            # get_time_feat_config('first_n_modes', data_config=self.data_config, n_modes=10),
        ]
        self.dec_feat_norm = None
        self.dec_reduc_config = None # get_reduc_config('PCA', n_components=10) # or None
        
        # run parameters
        self.skip_first_edge_type = False
        self.pred_steps = 10
        self.is_burn_in = True
        self.final_pred_steps = 75
        self.dynamic_rel = False
        self.is_dynamic_graph = False

        # plotting parameters
        self.show_conf_band = False

        self.set_nri_emb_configs()

    # 4: Sparsifier parameters

        self.spf_config = get_spf_config('no_spf', is_expert=False)
        
        self.spf_domain_config   = get_domain_config('time')
        self.spf_raw_data_norm = None 
        self.spf_feat_configs = [
            # get_time_feat_config('first_n_modes'),
        ]    
        self.spf_feat_norm = None
        self.spf_reduc_config = None # get_reduc_config('PCA', n_components=10) # or None
        
    # 5: Hyperparameters and plots
        self.hyperparams = self.get_hyperparams()

    def set_nri_emb_configs(self):
        """
        Sets the encoder and decoder embedding function configurations based on the provided config types.
        """
        # set encoder pipeline and embedding configs
        self.pipeline = self.ext.get_enc_pipeline(self.pipeline_type)  
        self.enc_edge_emb_configs = self.ext.get_enc_emb_config(config_type=self.edge_emb_config, n_hidden=self.n_hidden_mlp)  
        self.enc_node_emb_configs = self.ext.get_enc_emb_config(config_type=self.node_emb_config, n_hidden=self.n_hidden_mlp)

        # set decoder embedding configs
        self.dec_edge_mlp_config = self.ext.get_dec_emb_config(self.edge_mlp_config, self.msg_out_size)['mlp']
        self.dec_out_mlp_config = self.ext.get_dec_emb_config(self.out_mlp_config, self.msg_out_size)['mlp']


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
            'lr_enc': self.lr_enc,
            'lr_dec': self.lr_dec,
            'optimizer': self.optimizer,
            'is_beta_annealing': self.is_beta_annealing,
            'final_beta': self.final_beta,
            'warmup_frac_beta': self.warmup_frac_beta,
            'enc/is_enc_warmup': self.is_enc_warmup,
            'enc/warmup_acc_cutoff': self.warmup_acc_cutoff,
            'enc/sustain_enc_warmup': self.sustain_enc_warmup,
            'enc/final_gamma': self.final_gamma,
            'enc/warmup_frac_gamma': self.warmup_frac_gamma,
            'enc/loss_type': self.loss_type_enc,
            'dec/loss_type': self.loss_type_dec,
            'n_edge_types': self.n_edge_types,
            'enc/prior': self.prior,
            'enc/add_const_kld': self.add_const_kld,
            'dec/dec_loss_stabilize_steps': self.dec_loss_stabilize_steps,
            'dec/dec_loss_bound_update_interval': self.dec_loss_bound_update_interval,
            'dec/dec_loss_window_size': self.dec_loss_window_size,

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
            'enc/n_hidden_mlp': self.n_hidden_mlp,
            'enc/is_xavier_weights': self.enc_is_xavier_weights,
            'enc/is_hard': self.is_hard,
            'enc/init_temp': self.init_temp,
            'enc/min_temp': self.min_temp,
            'enc/decay_temp': self.decay_temp,
            'enc/attention_output_size': self.attention_output_size,

            # decoder parameters
            'dec/msg_out_size': self.msg_out_size,
            'dec/do_prob': self.dec_do_prob,
            'dec/is_batch_norm': self.dec_is_batch_norm,
            'dec/is_xavier_weights': self.dec_is_xavier_weights,
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
            'dec/final_pred_steps': self.final_pred_steps,
            'dec/dynamic_rel': self.dynamic_rel,
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
            elif isinstance(value, (int, float, dict)):
                hyperparams[key] = str(value)
            elif value is None:
                hyperparams[key] = 'None'

        return hyperparams
    
class NRITrainSweep:
    def __init__(self, data_config:DataConfig):
        self.data_config = data_config
        self.train_sweep_num = 1

    # 1: Training parameters   
        # dataset parameters
        self.batch_size = [50]
        self.train_rt = [0.8]
        self.test_rt = [0.1]
        self.val_rt = [0.1]

        # optimization parameters
        self.max_epochs = [5]
        self.lr = [0.001]
        self.optimizer = ['adam']

        self.loss_type_enc = ['kld']
        self.prior = [None]
        self.add_const_kld = [True]  # this needs to be True, adds a constant term to the KL divergence

        self.loss_type_dec = ['nll']

    # 2: Encoder parameters

        # pipeline parameters
        self.pipeline_type = ['mlp_1'] 
        self.is_residual_connection = [True] 

        # embedding function parameters
        self.edge_emb_config = [
            {'mlp': 'default', 'cnn': 'default'}
        ]
        self.node_emb_config = [
            {'mlp': 'default', 'cnn': 'default'}
        ]

        self.enc_do_prob = [
            {'mlp': 0.0, 'cnn': 0.0}
        ]
        self.enc_is_batch_norm = [
            {'mlp': True, 'cnn': False}
        ]
        # attention parameters
        self.attention_output_size = [5]  

        # input processor parameters
        self.enc_domain_config = [get_domain_config('time')]
        self.enc_raw_data_norm = [None] 
        self.enc_feat_configs = [
            []
            ]
        self.enc_reduc_config = [None] # get_reduc_config('PCA', n_components=10) # or None
        self.enc_feat_norm = [None]

        # gumble softmax parameters
        self.temp = [1.0 ]     
        self.is_hard = [True]   

    # 3: Decoder parameters

        self.msg_out_size = [64]
    
        # embedding function parameters 
        self.edge_mlp_config = [{'mlp': 'default'}]
        self.out_mlp_config = [{'mlp': 'default'}]

        self.dec_do_prob = [0]
        self.dec_is_batch_norm = [True]

        # recurrent embedding parameters
        self.recur_emb_type = ['gru']
        
        # input processor parameters
        self.dec_domain_config = [get_domain_config('time')]
        self.dec_raw_data_norm = [None]
        self.dec_feat_configs = [
            []
            # [get_time_feat_config('first_n_modes', data_config=self.data_config, n_modes=10)]
        ]
        self.dec_feat_norm = [None]
        self.dec_reduc_config = [None] # get_reduc_config('PCA', n_components=10) # or None
        
        # run parameters
        self.skip_first_edge_type = [True] 
        self.pred_steps = [1]
        self.is_burn_in = [False]
        self.final_pred_steps = [1]
        self.is_dynamic_graph = [False]

    # 4: Sparsifier parameters

        self.spf_config = [get_spf_config('vanilla', is_expert=True)]
        
        # self.spf_domain_config   = get_domain_config('time')
        # self.spf_raw_data_norm = None 
        # self.spf_feat_configs = [
        #     # get_time_feat_config('first_n_modes'),
        # ]    
        # self.spf_feat_norm = None
        # self.spf_reduc_config = None # get_reduc_config('PCA', n_components=10) # or None
    
        

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
            'f_mlp1_og': [
                        ['1/node_emd.1', 'mlp'],
                        ['1/pairwise_op', 'concat'],
                        ['1/edge_emd.1.@', 'mlp'],
                        ['2/aggregate', 'mean'],
                        ['2/node_emd.1', 'mlp'],
                        ['2/pairwise_op', 'concat'],
                        ['2/edge_emd.1', 'mlp'],
                     ],
            'mlp1_og': [
                        ['1/node_emd.1', 'mlp'],
                        ['1/pairwise_op', 'concat'],
                        ['1/edge_emd.1.@', 'mlp'],
                        ['1/edge_emd.2', 'mlp'],
                        ['1/edge_emd.3', 'mlp'],
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
        
        
    def get_enc_emb_config(self, config_type, n_hidden, custom_config=None): 
        """
        config_type : dict
        custom_config : dict

        Attributes
        ----------
        Write description of the mlp_config names, cnn_config names etc.
        """     

        # Dictionaries
        mlp_configs = {
            'nri_og': [[n_hidden, 'elu'],
                        [n_hidden, 'elu']],
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
            'edge_nri_og':  [[msg_out_size, 'tanh'],
                            [msg_out_size, 'tanh']], # the last layer should look like this for any configs for decoder

            'out_nri_og': [[msg_out_size, 'relu'],
                            [msg_out_size, 'relu']],

            'edge_nri_4':  [[msg_out_size, 'tanh'],
                            [msg_out_size, 'tanh'],
                            [msg_out_size, 'tanh'],
                            [msg_out_size, 'tanh'],], 

            'out_nri_4': [[msg_out_size, 'relu'],
                            [msg_out_size, 'relu'],
                            [msg_out_size, 'relu'],
                            [msg_out_size, 'relu']],

            'edge_nri_6':  [[msg_out_size, 'tanh'],
                            [msg_out_size, 'tanh'],
                            [msg_out_size, 'tanh'],
                            [msg_out_size, 'tanh'],
                            [msg_out_size, 'tanh'],
                            [msg_out_size, 'tanh'],], 

            'out_nri_6': [[msg_out_size, 'relu'],
                            [msg_out_size, 'relu'],
                            [msg_out_size, 'relu'],
                            [msg_out_size, 'relu'],
                            [msg_out_size, 'relu'],
                            [msg_out_size, 'relu']],
                            
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




        
        
