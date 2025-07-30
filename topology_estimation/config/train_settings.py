import os
import sys
from manager import SelectTopologyEstimatorModel, load_selected_config

TOPOLOGY_ESTIMATION_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(TOPOLOGY_ESTIMATION_DIR))

from feature_extraction.config import get_freq_fex_config

class TrainNRIConfig:
    def __init__(self):
        self.set_training_params()
        self.set_run_params()

    def set_training_params(self):        
        self.model_num = 1
        self.continue_training = False

        self.is_log = True
        
        self.n_edge_types = 2

        # dataset parameters
        self.batch_size = 50
        self.train_rt   = 0.8
        self.test_rt    = 0.1
        self.val_rt     = 0.1

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
        self.sparsif_type = None  # [TODO]: Get sparsif type from get_sparsif_config() method
        self.domain_sparsif_config   = 'time'  # options: time, freq-psd, freq-amp
        self.fex_configs_sparsif = [
            get_freq_fex_config('first_n_modes'),
        ]    
        self.norm_type_sparsif = None  # options: std, minmax, none

        # [TODO]: Add domain config, raw_norm and fex_norm, reduc_config for sparsifier, encoder and decoder (see fault detection config)

        # encoder run parameters
        self.domain_encoder_config   = 'freq'  # options: time, freq-psd, freq-amp
        self.norm_type_encoder = None  # options: std, minmax, none

        self.fex_configs_encoder = [
        ]
        # gumble softmax parameters
        self.temp = 1.0       # temperature for Gumble Softmax
        self.is_hard = True      # if True, use hard Gumble Softmax

        # decoder run parameters
        self.domain_decoder_config = 'freq'   # options: time, freq-psd, freq-amp
        self.norm_type_decoder = None # options: std, minmax, none

        self.fex_configs_decoder = [
            get_freq_fex_config('first_n_modes', n_modes=10),
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




        
        
