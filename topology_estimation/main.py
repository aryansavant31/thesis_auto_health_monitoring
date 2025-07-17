"""
main class for topology estimation with both training and inference code

will contain train loop, test loop and run/prediction loop
"""
from config import TopologyEstimatorConfig, get_checkpoint_path
from encoder_blocks import Encoder
from decoder_blocks import Decoder
from nri import NRI
from data.config import DataConfig
from graph_structures import FullyConnectedGraph, SparisifiedGraph
from data.load import load_spring_particle_data
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import os
from data.transform import DataTransformer

class TopologyEstimatiorMain:
    def __init__(self, run_type, model_num=None):
        self.tp_config = TopologyEstimatorConfig(run_type)
        self.data_config = DataConfig()
        
        self.load_data(run_type)
        self.set_relation_matrices() # over her, i can pass teh data for sparsification

    def load_data(self, run_type):
        # set train data parameters
        if run_type == 'train':  
            # set parameters  
            self.data_config.set_train_dataset()
            # get dataset paths
            node_ds_paths, edge_ds_paths = self.data_config.get_dataset_paths()
            # load data
            self.train_loader, self.valid_loader, self.test_loader, self.data_stats = load_spring_particle_data(node_ds_paths, edge_ds_paths, self.tp_config.batch_size)

            if self.data_config.custom_test_ds:
                self.data_config.set_custom_test_dataset()
                # get custom dataset paths
                node_ds_paths, edge_ds_paths = self.data_config.get_dataset_paths()
                # load custom test data
                self.test_loader = "make_dataloader(node_ds_path, edge_ds_path)" # Some dataloader function that loads just 1 dataloader

            # for getting data stats
            dataiter = iter(self.train_loader)
            data = next(dataiter)

            # transform data

        # set predict data parameters
        elif run_type == 'predict':
            self.data_config.set_predict_dataset()
            # get predict dataset paths
            node_ds_paths, edge_ds_paths = self.data_config.get_dataset_paths()
            # load predict data
            self.predict_loader = "make_dataloader(node_ds_path, edge_ds_path)" # Some dataloader function that loads just 1 dataloader

            # for getting data stats
            dataiter = iter(self.predict_loader)
            data = next(dataiter)
        
        # process the input data to get correct data shape for the model initialization
        self.transform = DataTransformer(domain=self.tp_config.domain_encoder, 
                                         norm_type=self.tp_config.norm_type_encoder, 
                                        data_stats=self.data_stats)
        data = self.process_input_data(data)    

        # set data stats
        self.batch_size = data[0].shape[0]
        self.n_nodes = data[0].shape[1]
        self.n_datapoints = data[0].shape[2]
        self.n_dims = data[0].shape[3]

        self._verbose_load_data()

        self.log_path_nri, self.log_path_sk = self.tp_config.get_log_path(self.data_config, self.n_datapoints, self.n_dims)

    def process_input_data(self, data):
        """
        Transform the data
            - domain change
            - normalization
        Feature extraction
        """
        # transform data
        data = self.transform(data)

        # [TODO]: Implement feature extraction logic here

        return data
    
    def set_relation_matrices(self, run_type):
            if self.tp_config.is_sparsifier:
                # some sparsifier model may have to be loaded
                self.rec_rel, self.send_rel = SparisifiedGraph(self.n_nodes).get_relation_matrix()

            # if no sparsifier, then fully connected graph
            else:
                self.rec_rel, self.send_rel = FullyConnectedGraph(self.n_nodes, self.batch_size).get_relation_matrices()

    def init_model(self):
        # initialize encoder
        self.tp_config.set_encoder_params() 
        encoder = Encoder(n_datapoints=self.n_datapoints, 
                        n_dims=self.n_dims,
                        pipeline=self.tp_config.encoder_pipeline, 
                        n_edge_types=self.tp_config.n_edge_types_enc, 
                        is_residual_connection=self.tp_config.is_residual_connection,
                        edge_emd_configs=self.tp_config.edge_emb_configs_enc, 
                        node_emd_configs=self.tp_config.node_emb_configs_enc, 
                        drop_out_prob=self.tp_config.dropout_prob_enc,
                        batch_norm=self.tp_config.batch_norm_enc, 
                        attention_output_size=self.tp_config.attention_output_size)
        
        # initialize decoder
        self.tp_config.set_decoder_params()
        decoder = Decoder(n_dim=self.n_dims,
                        msg_out_size=self.tp_config.msg_out_size,
                        n_edge_types=self.tp_config.n_edge_types_dec,
                        edge_mlp_config=self.tp_config.edge_mlp_config_dec,
                        recurrent_emd_type=self.tp_config.recurrent_emd_type,
                        out_mlp_config=self.tp_config.out_mlp_config_dec,
                        do_prob=self.tp_config.dropout_prob_dec,
                        is_batch_norm=self.tp_config.is_batch_norm_dec)
        
        # initialize NRI model
        nri_model = NRI(encoder, decoder)
        nri_model.set_run_params()  # [TODO]: pass params from tp.config for set_run_params() 
        nri_model.set_input_graph(self.rec_rel, self.send_rel)  

        self._verbose_init_model(encoder, decoder, nri_model)

        return nri_model

    def train_nri(self):
        # initialize model
        untrained_nri_model = self.init_model()

        # if continue training, load ckpt path of untrained model 
        if self.tp_config.continue_training:
            ckpt_path = get_checkpoint_path(self.log_path_nri)
        else:
            ckpt_path = None

        # set training parameters
        untrained_nri_model.set_training_params() # [TODO]: pass params from tp.config for set_training_params()

        if self.tp_config.is_log:
            logger = TensorBoardLogger(os.path.dirname(self.log_path_nri), name=None, version=os.path.basename(self.log_path_nri))
        else:
            logger = None

        trainer = Trainer(
            max_epochs=self.tp_config.max_epochs,
            logger=logger,
            enable_progress_bar=True,
            log_every_n_steps=1,)
        
        # train the model
        trainer.fit(untrained_nri_model, self.train_loader, self.valid_loader, ckpt_path=ckpt_path) 

    def train_sparsifier(self):
        """
        Train the sparsifier model if needed.
        """
        # [TODO]: Implement the logic to train the sparsifier model
        pass

    def log_hyperparameters(self):
        """
        Logs the topology model hypperparametrs
        """
        # [TODO]: Implement the logic to log all hyperparameters of the model.
        # Some hyperparameters can be logged seperately to track its correlation with model accuracy
        pass

    def load_model(self, log_path):
        """
        Loads the trained model from the log path.
        """
        ckpt_path = get_checkpoint_path(log_path)  
        # [TODO]: Implement the logic to load the model from the ckpt path

    def test_model():
        """
        Loads and tests the trained model"""  
        # [TODO]: Implement the logic to test the model
        pass

    def predict(self):
        model = self.load_model(self.tp_config.load_path_nri)

        # [TODO]: Implement the logic to predict results using the loaded model

    # ======================================================
    # Verbose methods for printing data and model stats
    # ======================================================
    
    def _verbose_load_data(self):
        """
        Prints the data stats for the loaded data.
        """
        print(5*'-', 'Data Stats', 5*'-')
        print(f"\nBatch size: {self.batch_size}")
        print(f"Number of nodes: {self.n_nodes}")
        print(f"Number of datapoints: {self.n_datapoints}")  
        print(f"Number of dimensions: {self.n_dims}")

    def _verbose_init_model(self, encoder, decoder, nri_model):
        """
        Prints the model summary for the encoder, decoder and NRI model.
        """
        print(5*'-', 'Topology Estimator Model Summary', 5*'-')
        # print encoder summary
        print(2*'-', 'Encoder Summary')
        print("\n Enocder pipeline:\n")
        for layer in encoder.pipeline:
            print(layer)

        print("\n Encoder embedding function summary:\n")         
        print(encoder)

        # print decoder summary
        print(2*'-', 'Decoder Summary')
        print("\n Decoder embedding function summary:\n")
        print(decoder)

        # print nri model summary
        print(2*'-', 'NRI Model Summary')
        print(nri_model)
        


