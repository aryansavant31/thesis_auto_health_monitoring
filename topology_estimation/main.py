"""
main class for topology estimation with both training and inference code

will contain train loop, test loop and run/prediction loop
"""
from config import TopologyEstimatorConfig
from encoder_blocks import Encoder
from decoder_blocks import Decoder
from nri import NRI
from data.config import DataConfig
from graph_structures import FullyConnectedGraph, SparisifiedGraph
from data.load import load_spring_particle_data
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import os

class TopologyEstimatiorMain:
    def __init__(self, run_type, model_num=None):
        self.tp_config = TopologyEstimatorConfig()
        self.data_config = DataConfig()

        self.load_data(run_type)
        self.set_relation_matrices()

        if run_type == 'train':
            untrained_nri_model = self.init_model()
            self.train_model(untrained_nri_model)
            self.test_model() # TASK: Set up the test loop

        elif run_type == 'run':
            model = self.load_model()
            self.run_model(model)


    def load_data(self, run_type):
        # set data parameters
        if run_type == 'train':    
            self.data_config.set_train_dataset()
            # get dataset paths
            node_ds_paths, edge_ds_paths = self.data_config.get_dataset_paths()
            # load data
            self.train_loader, self.valid_loader, self.test_loader = load_spring_particle_data(node_ds_paths, edge_ds_paths, self.tp_config.batch_size)

            if self.data_config.custom_test_ds:
                self.data_config.set_custom_test_dataset()
                # get custom dataset paths
                node_ds_paths, edge_ds_paths = self.data_config.get_dataset_paths()
                # load custom test data
                self.test_loader = "make_dataloader(node_ds_path, edge_ds_path)" # Some dataloader function that loads just 1 dataloader

            # for getting data stats
            dataiter = iter(self.train_loader)
            data = next(dataiter)

        elif run_type == 'run':
            self.data_config.set_predict_dataset()
            # get predict dataset paths
            node_ds_paths, edge_ds_paths = self.data_config.get_dataset_paths()
            # load predict data
            self.predict_loader = "make_dataloader(node_ds_path, edge_ds_path)" # Some dataloader function that loads just 1 dataloader

            # for getting data stats
            dataiter = iter(self.predict_loader)
            data = next(dataiter)
        
        # set data stats
        self.batch_size = data[0].shape[0]
        self.n_nodes = data[0].shape[1]
        self.n_datapoints = data[0].shape[2]
        self.n_dims = data[0].shape[3]

        self._verbose_load_data()

        self.log_path = self.tp_config.get_log_path(self.data_config, self.n_datapoints)

    def set_relation_matrices(self):
        if self.tp_config.is_sparsifier:
            self.rec_rel, self.send_rel = SparisifiedGraph(self.n_nodes).get_relation_matrix()

        # if no sparsifier, then fully connected graph
        else:
            self.rec_rel, self.send_rel = FullyConnectedGraph(self.n_nodes, self.batch_size).get_relation_matrices()

    def init_model(self):
        # initialize encoder
        encoder = Encoder(n_timesteps=self.n_datapoints, 
                        n_dims=self.n_dims,
                        pipeline=self.tp_config.encoder_pipeline, 
                        n_edge_types=self.tp_config.n_edge_types, 
                        is_residual_connection=self.tp_config.is_residual_connection,
                        edge_emd_configs=self.tp_config.edge_emb_configs_enc, 
                        node_emd_configs=self.tp_config.node_emb_configs_enc, 
                        drop_out_prob=self.tp_config.dropout_prob_enc,
                        batch_norm=self.tp_config.batch_norm_enc, 
                        attention_output_size=self.tp_config.attention_output_size)
        
        # initialize decoder
        decoder = Decoder(n_dim=self.n_dims,
                        msg_out_size=self.tp_config.msg_out_size,
                        n_edge_types=self.tp_config.n_edge_types,
                        skip_first=self.tp_config.skip_first_edge_type,
                        edge_mlp_config=self.tp_config.edge_mlp_config_dec,
                        recurrent_emd_type=self.tp_config.recurrent_emd_type,
                        out_mlp_config=self.tp_config.out_mlp_config_dec,
                        do_prob=self.tp_config.dropout_prob_dec,
                        is_batch_norm=self.tp_config.is_batch_norm_dec)
        
        # initialize NRI model
        nri_model = NRI(encoder, decoder)
        nri_model.set_run_params()  # TASK: pass params from tp.config for set_run_params() 
        nri_model.set_input_graph(self.rec_rel, self.send_rel)  

        self._verbose_init_model(encoder, decoder, nri_model)

        return nri_model
    
    def load_model(self):
        """
        Loads the trained model from the log path.
        """
        # TASK: Implement the logic to load the model from the log path
        pass    

    def train_model(self, untrained_nri_model):
        self.tp_config.set_training_params()
        untrained_nri_model.set_training_params() # TASK: pass params from tp.config for set_training_params()

        if self.tp_config.is_log:
            logger = TensorBoardLogger(os.path.dirname(self.log_path), name=None, version=os.path.basename(self.log_path))
        else:
            logger = None

        trainer = Trainer(
            max_epochs=self.tp_config.max_epochs,
            logger=logger,
            enable_progress_bar=True,
            log_every_n_steps=1,)
        
        trainer.fit(untrained_nri_model, self.train_loader, self.valid_loader) # TASK: Set up the validation and test loop

    def test_model():
        """
        Loads and tests the trained model"""  
        pass

    def run_model(self, model):
        """
        Loads and runs the trained model
        """
        # TASK: Implement the logic to run the model on the predict dataset
        pass

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
        


