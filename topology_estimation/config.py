class TopologyConfig:
    def __init__(self):
        self.tp = TopologyConfigUtils()
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

        Attributes
        ----------
        emb_type : str
            Type of embedding model to use:
            - 'mlp': Multi-layer perceptron
            - 'cnn': Convolutional neural network

        use_shared_weights : bool
            Whether to use shared embedding function weights for each message passing step

        message_pass_type : str
            Type of message passing to use:
            - 'concat': Concatenation of messages
            - 'sum': Summation of messages

        num_message_pass : int
            Number of message passing steps. 
            (1 message pass = --> node --> edge)
        """

        self.use_shared_weights = False
        self.n_edge_types = 2  # output size 

        pipeline_type = 'mlp_1'  # default pipeline type
        self.encoder_pipeline = self.tp.get_encoder_pipeline(pipeline_type)

        self.residual_connection = True  # if True, then use residual connection in the last layer

        # eg: edge_emd_config = {'mlp': custom, 'cnn': 'default}
        # custom_config = {'mlp': [[2, 3]]}
        # same for node_emb_config
        edge_emd_config = {'mlp': 'default'}
        node_emb_config = {'mlp': 'default'}

        self.edge_emb_configs = self.tp.get_emb_config(config_type=edge_emd_config)  
        self.node_emb_configs = self.tp.get_emb_config(config_type=node_emb_config)
        
        # others
        # MLP
        self.dropout_prob_mlp = 0.0
        self.batch_norm_mlp = False


        # if using attention based aggregation
        self.attention_output_size = 6  # output size for attention layer


        # # for mlp type emd_fn
        # self.edge_config = self.edge_mlp_config_default
        # self.node_mlp_config = self.node_mlp_config_default

        # # for cnn type emd_fn
        # self.edge_conv_config = self.edge_conv_config_default
        # self.node_conv_config = self.node_conv_config_default
            

        # node to edge transformation
        # self.pairwise_op = 'concat'
        # self.edge_emb_fn_type = ['mlp', 'mlp']        # check encoder_params_extras() to configure
        
        # edge to node transformation 
        # self.agg_type = 'sum'
        # self.node_emb_fn_type = 'mlp'
        # self.use_prev_node_emb = True
        # self.combine_type = 'sum'  # if self.use_prev_node_emb == True

        


    # def encoder_params_extras(self):
    #     """
    #     Attributes
    #     ----------
    #     edge_mlp_config : list or None
    #         Configuration for MLP edge embedding.

    #     edge_conv_config : list or None
    #         Configuration for CNN edge embedding.
    #     """
    #     # ------- emb function config params -------
    #     # MLP
    #     # edge_mlp
    #     self.edge_mlp_config_default = []

    #     # node_mlp
    #     self.node_mlp_config_default = []

    #     # CNN
    #     # edge_conv
    #     self.edge_conv_config_default = []

    #     # node_conv
    #     self.node_conv_config_default = []


    #     # ------- pipelines --------

    #     self.mlp1_encoder_pipeline = [
    #         ['1.node_emd_1','mlp'],
    #         ['1.pairwise_op', 'concat'], 
    #         ['1.edge_emd_1', 'mlp',]
    #         ['2.agg', 'weighted_sum'], # if weighted, then calculate attention weights,
    #         ['2.combine', 'concat'], # if combine is ther, it means i want ot use exsiting node emb with M
    #         ['2.node_emd', 'mlp'],
    #         ['2.pairwise_op', 'concat'], 
    #         ['2.edge_emd', 'mlp']
    #     ]
            


    def set_decoder_params(self):
        pass

    def set_sparsifier_params(self):
        pass

    def set_training_params(self):
        self.epochs = 100
        self.lr = 0.001
        self.optimizer = 'adam'
        self.loss_type_encd = 'kld'
        self.loss_type_decd = 'nnl'
        

class TopologyConfigUtils():
    def __init__(self):
        super().__init__()

    def get_encoder_pipeline(self, pipeline_type, custom_pipeline=None):
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
        
        if pipeline_type in pipelines:
            return pipelines[pipeline_type]
        
        elif pipeline_type == 'custom':
            if custom_pipeline is None:
                raise ValueError("Custom pipeline must be provided when pipeline_type is 'custom'.")
            else:
                # placeholder for validation logic of custom pipeline
                pass
            return custom_pipeline
        
    def get_emb_config(self, config_type, custom_config=None): 
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

    def set_result_path(self):
        pass
        
