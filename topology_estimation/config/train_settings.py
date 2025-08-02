import os
import sys
from manager import SelectTopologyEstimatorModel, load_selected_config

TOPOLOGY_ESTIMATION_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(TOPOLOGY_ESTIMATION_DIR))

from feature_extraction.config import get_freq_feat_config

class NRITrainSettings:
    def __init__(self):
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

    # 1: Training parameters   

        self.model_num = 1
        self.continue_training = False
        self.is_log = True
        
        self.n_edge_types = 2

        # dataset parameters
        self.batch_size = 50
        self.train_rt = 0.8
        self.test_rt = 0.1
        self.val_rt = 0.1

        # optimization parameters
        self.max_epochs = 5
        self.lr = 0.001
        self.optimizer = 'adam'

        self.loss_type_enc = 'kld'
        self.prior = None
        self.add_const_kld = True  # this needs to be True, adds a constant term to the KL divergence

        self.loss_type_dec = 'nnl'

    # 2: Encoder parameters

        # pipeline parameters
        self.pipeline_type = 'mlp_1' 
        self.is_residual_connection = True 

        # embedding function parameters
        edge_emd_config = {
            'mlp': 'default',
            'cnn': 'default'
            }
        node_emb_config = {
            'mlp': 'default',
            'cnn': 'default'
            }

        self.do_prob_enc = {
            'mlp': 0.0,
            'cnn': 0.0
            }
        self.bn_enc = {
            'mlp': False,
            'cnn': False
            }
        # attention parameters
        self.attention_output_size = 5   

        # Run parameters
        self.enc_domain_config = 'freq' 
        self.enc_norm = None  
        self.enc_feat_configs = []

        # gumble softmax parameters
        self.temp = 1.0       
        self.is_hard = True   

        self.encoder_pipeline = ext.get_enc_pipeline(self.pipeline_type)  
        self.edge_emb_configs_enc = ext.get_enc_emb_config(config_type=edge_emd_config)  
        self.node_emb_configs_enc = ext.get_enc_emb_config(config_type=node_emb_config)

    # 3: Decoder parameters

        self.msg_out_size = 64
    
        # embedding function parameters 
        edge_mlp_config = {'mlp': 'default'}
        out_mlp_config = {'mlp': 'default'}

        self.do_prob_dec = 0
        self.is_bn_dec = True

        # recurrent embedding parameters
        self.recur_emd_type = 'gru'
        
        # Run parameters
        self.dec_domain_config = 'freq'  
        self.dec_norm = None 
        self.dec_fex_configs = [
            get_freq_feat_config('first_n_modes', n_modes=10),
        ]

        self.skip_first_edge_type = True 

        # [TODO]: add rest of the decoder run params

        self.edge_mlp_config_dec = ext.get_dec_emb_config(config_type=edge_mlp_config)['mlp']
        self.out_mlp_config_dec = ext.get_dec_emb_config(config_type=out_mlp_config)['mlp']

    # 4: Sparsifier parameters

        self.spf_type = None 
        
        self.spf_domain_config   = 'time' 
        self.spf_fex_configs = [
            get_freq_feat_config('first_n_modes'),
        ]    
        self.spf_norm = None 

        # [TODO]: define all the parameters depending on sparsif_type and attach it to config dict (like get_fex_config() method)
        # [TODO]: Add domain config, raw_norm and fex_norm, reduc_config for sparsifier, encoder and decoder (see fault detection config)
        
class DecoderTrainSettings:
    pass 

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
    
    def get_dec_emb_config(self, config_type, custom_config=None):
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

    
if __name__ == "__main__":
    user_text = "To view/select trained topology (edge) estimation models, type (a)\nTo view custom tested models, type (b)\nTo view predicted models, type (c)\nEnter input: "
    user_input = input(user_text).strip("'\"")
    if user_input.lower() == 'a':
        run_type = 'train'
    elif user_input.lower() == 'b':
        run_type = 'custom_test'
    elif user_input.lower() == 'c':
        run_type = 'predict'
    else:
        raise ValueError("Invalid input. Please enter 'a', 'b', or 'c'.")
    model_selector = SelectTopologyEstimatorModel(framework='directed_graph', run_type=run_type)
    model_selector.select_ckpt_and_params()




        
        
