"""
main class for topology estimation with both training and inference code

will contain train loop, test loop and run/prediction loop
"""
from config import TopologyEstimatorConfig
from encoder_blocks import Encoder
from decoder_blocks import Decoder
from nri import TopologyEstimator
from data.config import DataConfig
from data.load import load_spring_particle_data

class TopologyEstimatiorMain:
    def __init__(self, run_type, model_num=None):
        self.par = TopologyEstimatorConfig()
        data_config = DataConfig()
        data_config.set_train_valid_dataset()

        # get dataset paths
        node_ds_paths, edge_ds_paths = data_config.get_dataset_paths()

        # load data
        self.train_loader, self.valid_loader, self.test_loader = load_spring_particle_data(node_ds_paths, edge_ds_paths, self.par.batch_size)

        # get number of timesteps and dimensions from the data
        dataiter = iter(self.train_loader)
        data = next(dataiter)
        
        self.n_timesteps = data[0].shape[2]
        self.n_dims = data[0].shape[3]

        if run_type == 'train':
            self.train_model()
            self.test_model()

        elif run_type == 'run':
            self.run_model()

    def train_model(self):
        # Initialize encoder and decoder
        encoder = Encoder(self.par.encoder_pipeline, 
                        self.par.n_edge_types, 
                        self.par.is_residual_connection, 
                        self.n_timesteps, 
                        self.n_dims, 
                        self.par.edge_emb_configs_enc, 
                        self.par.node_emb_configs_enc, 
                        self.par.dropout_prob_mlp_enc, 
                        self.par.batch_norm_mlp_enc, 
                        self.par.attention_output_size)

        decoder = Decoder()

        # Initialize topology estimator
        topology_estimator = TopologyEstimator(encoder, decoder)

    def test_model():
        """
        Loads and tests the trained model"""  
        pass

    def run_model(self):
        """
        Loads and runs the trained model
        """
        pass
        
        


