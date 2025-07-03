class TopologyEstimatorConfig:
    def __init__(self):
        self.tp = TopologyConfigUtils()
        
        # Sim params
        self.sim_num = 1.1
        self.is_save = False

    def set_tp_dataset_params(self):
        self.batch_size = 50
        self.train_rt   = 0.8
        self.test_rt    = 0.2
        self.val_rt     = 1 - (self.train_rt + self.test_rt)

    def encoder_params(self):
        """
        Sets encoder parameters for the model.

        """
        # ------ Pipeline Parameters ------
        pipeline_type               = 'mlp_1'   # default pipeline type
        self.encoder_pipeline       = self._get_encoder_pipeline(pipeline_type)

        self.n_edge_types           = 2         # output size 
        self.is_residual_connection    = True      # if True, then use residual connection in the last layer


        # ------ Embedding Function Parameters ------
        # embedding config
        edge_emd_config             = {'mlp': 'default',
                                       'cnn': 'default'}
        
        node_emb_config             = {'mlp': 'default',
                                       'cnn': 'default'}

        self.edge_emb_configs_enc   = self._get_emb_config(config_type=edge_emd_config)  
        self.node_emb_configs_enc   = self._get_emb_config(config_type=node_emb_config)
        
        # other embedding parameters
        self.dropout_prob_mlp_enc   = {'mlp': 0.0,
                                       'cnn': 0.0}
        
        self.batch_norm_mlp_enc     = {'mlp': False,
                                       'cnn': False}


        # ------ Attention Parameters ------
        self.attention_output_size  = 6         # output size for attention layer

        # ------ Gumble Softmax Parameters ------
        self.temp                   = 1.0       # temperature for Gumble Softmax
        self.is_hard                = True      # if True, use hard Gumble Softmax

    def set_decoder_params(self):

        self.msg_out_size           = 64

        # edge embedding config
        edge_mlp_config             = {'mlp': 'default'}
        self.edge_mlp_configs_dec   = self._get_emb_config(config_type=edge_mlp_config)['mlp']

        output_mlp_config           = {'mlp': 'default'}
        self.out_mlp_config_dec     = self._get_emb_config(config_type=output_mlp_config)['mlp']

        self.reccurent_emd_type     = 'lstm'
        self.n_layers_recurrent     = 1

    def set_sparsifier_params(self):
        pass

    def set_training_params(self):
        self.epochs = 100
        self.lr = 0.001
        self.optimizer = 'adam'

        self.loss_type_encd = 'kld'
        self.prior = None
        self.add_const_kld = False  # if True, adds a constant term to the KL divergence

        self.loss_type_decd = 'nnl'
        
        
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
                        ['1/pairwise_op', 'concat'],
                        ['1/edge_emd.1.@', 'mlp'],
                        ['2/aggregate', 'weighted_sum'],
                        ['2/combine', 'concat'],
                        ['2/node_emd.1', 'mlp'],
                        ['2/pairwise_op', 'concat'],
                        ['2/edge_emd.1', 'mlp']
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
        
        
    def _get_emb_config(self, config_type, custom_config=None): 
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
                        [32, None]] # the last layer is the output layer here
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

        

class TopologyConfigUtils():
    def __init__(self):
        super().__init__()


    def set_result_path(self):
        pass
        
