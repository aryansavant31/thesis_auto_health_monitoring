"""
main class for topology estimation with both training and inference code

will contain train loop, test loop and run/prediction loop
"""
from config import TrainNRIConfig, PredictNRIConfig, get_checkpoint_path
from nri import NRI
from data.config import DataConfig
from graph_structures import FullyConnectedGraph, SparisifiedGraph
from data.load import load_spring_particle_data
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import os
from data.transform import DataTransformer
from feature_extraction import FeatureExtractor

class PredictNRIMain:
    def __init__(self):
        self.data_config = DataConfig()
        self.tp_config = PredictNRIConfig()

        self.load_data()
        self.set_relation_matrices()  # over here, I can pass the data for sparsIF

    def load_data(self):
        # set parameters
        if self.tp_config.is_custom_test:
            self.data_config.set_custom_test_dataset()
        else:
            self.data_config.set_predict_dataset()

        # get predict dataset paths
        node_ds_paths, edge_ds_paths = self.data_config.get_dataset_paths()
        # load predict data
        self.custom_loader, self.data_stats = "make_dataloader(node_ds_path, edge_ds_path)" # Some dataloader function that loads just 1 dataloader
        
        # for getting data stats
        dataiter = iter(self.custom_loader)
        data = next(dataiter)
    
        # process the input data to get correct data shape for the model initialization  
        data = self.process_input_data(data)    

        # set data stats
        self.batch_size = data[0].shape[0]
        self.n_nodes = data[0].shape[1]
        self.n_components = data[0].shape[2]
        self.n_dims = data[0].shape[3]

        self._verbose_load_data()

    def process_input_data(self, data):
        """
        Transform the data
            - domain change
            - normalization
        Feature extraction
        """
        # transform data
        data = DataTransformer(domain=self.tp_config.domain_encoder, 
                                        norm_type=self.tp_config.norm_type_encoder, 
                                        data_stats=self.data_stats)(data)

        # extract features from data if fex_configs are provided
        if self.tp_config.fex_configs:
            data = FeatureExtractor(fex_configs=self.tp_config.fex_configs)(data)

        return data
    
    def set_relation_matrices(self):
        if self.tp_config.sparsif_type:
            self.rec_rel, self.send_rel = SparisifiedGraph(self.n_nodes).get_relation_matrix()
        else:
            self.rec_rel, self.send_rel = FullyConnectedGraph(self.n_nodes, self.batch_size).get_relation_matrices()

    def load_model(self):
        trained_model = NRI.load_from_checkpoint(self.tp_config.ckpt_path)
        run_params = {} # [TODO]: pass params from tp_pred_config for set_run_params()
        
        trained_model.set_run_params()  # [TODO]: pass params from tp.config for set_run_params() 
        trained_model.set_input_graph(self.rec_rel, self.send_rel)

        self._verbose_init_model(trained_model)

        return trained_model

    def run_model(self):
        # [TODO]: Implement the logic to test the model
        trained_model = self.load_model()

        log_path = self.tp_config.get_infer_log_path()

        logger = TensorBoardLogger(os.path.dirname(log_path), name=None, version=os.path.basename(log_path))
        trainer = Trainer(logger=logger)

        if self.tp_config.is_custom_test:
            trainer.test(trained_model, self.custom_loader)
        else:
            trainer.predict(trained_model, self.custom_loader)

    def log_hyperparameters(self):
        """
        Logs the topology model hypperparametrs
        """
        # [TODO]: Implement the logic to log all hyperparameters of the model.
        # Some hyperparameters can be logged seperately to track its correlation with model accuracy
        pass

    
    def _verbose_load_data(self):
        """
        Prints the data stats for the loaded data.
        """
        print(5*'-', 'Data Stats', 5*'-')
        print(f"\nBatch size: {self.batch_size}")
        print(f"Number of nodes: {self.n_nodes}")
        print(f"Number of datapoints: {self.n_components}")  
        print(f"Number of dimensions: {self.n_dims}")

    def _verbose_init_model(self, nri_model):
        """
        Prints the model summary for the encoder, decoder and NRI model.
        """
        print(5*'-', 'Topology Estimator Model Summary', 5*'-')
        # # print encoder summary
        # print(2*'-', 'Encoder Summary')
        # print("\n Enocder pipeline:\n")
        # for layer in encoder.pipeline:
        #     print(layer)

        # print("\n Encoder embedding function summary:\n")         
        # print(encoder)

        # # print decoder summary
        # print(2*'-', 'Decoder Summary')
        # print("\n Decoder embedding function summary:\n")
        # print(decoder)

        # print nri model summary
        print(2*'-', 'NRI Model Summary')
        print(nri_model)


class TrainNRIMain:
    def __init__(self):
        self.tp_config = TrainNRIConfig()
        self.data_config = DataConfig()
        
        self.load_data()
        self.set_relation_matrices() # over her, i can pass teh data for sparsification

    def load_data(self):
        # set train data parameters
        self.data_config.set_train_dataset()
        # get dataset paths
        node_ds_paths, edge_ds_paths = self.data_config.get_dataset_paths()
        # load data
        self.train_loader, self.valid_loader, self.test_loader, self.data_stats = load_spring_particle_data(node_ds_paths, edge_ds_paths, self.tp_config.batch_size)

        # for getting data stats
        dataiter = iter(self.train_loader)
        data = next(dataiter)

        # process the input data to get correct data shape for the model initialization
        data = self.process_input_data(data)    

        # set data stats
        self.batch_size = data[0].shape[0]
        self.n_nodes = data[0].shape[1]
        self.n_components = data[0].shape[2]
        self.n_dims = data[0].shape[3]

        self._verbose_load_data()

    def process_input_data(self, data):
        """
        Transform the data
            - domain change
            - normalization
        Feature extraction
        """
        # transform data
        data = DataTransformer(domain=self.tp_config.domain_encoder, 
                                        norm_type=self.tp_config.norm_type_encoder, 
                                        data_stats=self.data_stats)(data)

        # extract features from data if fex_configs are provided
        if self.tp_config.fex_configs:
            data = FeatureExtractor(fex_configs=self.tp_config.fex_configs)(data)

        return data
    
    def set_relation_matrices(self):
        if self.tp_config.sparsif_type:
            # some sparsifier model may have to be loaded
            self.rec_rel, self.send_rel = SparisifiedGraph(self.n_nodes).get_relation_matrix()
        # if no sparsifier, then fully connected graph
        else:
            self.rec_rel, self.send_rel = FullyConnectedGraph(self.n_nodes, self.batch_size).get_relation_matrices()

    def init_model(self):
        # if self.run_type == 'train':
        #     self.tp_config = self.tp_train_config
        # elif self.run_type == 'predict':
        #     self.tp_config = self.tp_pred_config.load_log_config()

        # initialize encoder
        self.tp_config.set_encoder_params() 
        encoder_params = {
            "n_components": self.n_components,
            "n_dims": self.n_dims,
            "pipeline": self.tp_config.encoder_pipeline,
            "n_edge_types": self.tp_config.n_edge_types,
            "is_residual_connection": self.tp_config.is_residual_connection,
            "edge_emd_configs": self.tp_config.edge_emb_configs_enc,
            "node_emd_configs": self.tp_config.node_emb_configs_enc,
            "drop_out_prob": self.tp_config.dropout_prob_enc,
            "batch_norm": self.tp_config.batch_norm_enc,
            "attention_output_size": self.tp_config.attention_output_size
        }

        # initialize decoder
        self.tp_config.set_decoder_params()
        decoder_params = {
            "n_dim": self.n_dims,
            "msg_out_size": self.tp_config.msg_out_size,
            "n_edge_types": self.tp_config.n_edge_types_dec,
            "edge_mlp_config": self.tp_config.edge_mlp_config_dec,
            "recurrent_emd_type": self.tp_config.recurrent_emd_type,
            "out_mlp_config": self.tp_config.out_mlp_config_dec,
            "do_prob": self.tp_config.dropout_prob_dec,
            "is_batch_norm": self.tp_config.is_batch_norm_dec
        }

        # initialize NRI model
        nri_model = NRI(encoder_params, decoder_params)
        nri_model.set_run_params()  # [TODO]: pass params from tp.config for set_run_params() 
        nri_model.set_input_graph(self.rec_rel, self.send_rel)  

        self._verbose_init_model(nri_model)

        return nri_model
    
    def load_model(self):
        trained_model = NRI.load_from_checkpoint(get_checkpoint_path(self.train_log_path))
        run_params = {} # [TODO]: pass params from tp_train_config for set_run_params()

        trained_model.set_run_params()  # [TODO]: pass params from tp.config for set_run_params() 
        trained_model.set_input_graph(self.rec_rel, self.send_rel)

        return trained_model

    def train(self):
        # initialize model
        untrained_nri_model = self.init_model()

        self.train_log_path = self.tp_config.get_train_log_path(self.n_components, self.n_dims)

        # if continue training, load ckpt path of untrained model 
        if self.tp_config.continue_training:
            ckpt_path = get_checkpoint_path(self.train_log_path)
        else:
            ckpt_path = None
            self.tp_config.check_if_version_exists()

        # set training parameters
        untrained_nri_model.set_training_params() # [TODO]: pass params from tp.config for set_training_params()

        if self.tp_config.is_log:
            logger = TensorBoardLogger(os.path.dirname(self.train_log_path), name=None, version=os.path.basename(self.train_log_path))
        else:
            logger = None

        trainer = Trainer(
            max_epochs=self.tp_config.max_epochs,
            logger=logger,
            enable_progress_bar=True,
            log_every_n_steps=1,)
        
        # train the model
        trainer.fit(untrained_nri_model, self.train_loader, self.valid_loader, ckpt_path=ckpt_path) #[TODO]: check if continue training works
        
        # test the model
        self.test_model()

    def test_model(self):
        # Load the trained model from checkpoint
        trained_model = self.load_model()
        test_log_path = self.tp_config.get_test_log_path()

        if self.tp_config.is_log:
            logger = TensorBoardLogger(os.path.dirname(test_log_path), name=None, version=os.path.basename(test_log_path))

        else:
            logger = None

        trainer = Trainer(logger=logger)

        trainer.test(trained_model, self.test_loader)

    def log_hyperparameters(self):
        """
        Logs the topology model hypperparametrs
        """
        # [TODO]: Implement the logic to log all hyperparameters of the model.
        # Some hyperparameters can be logged seperately to track its correlation with model accuracy
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
        print(f"Number of datapoints: {self.n_components}")  
        print(f"Number of dimensions: {self.n_dims}")

    def _verbose_init_model(self, nri_model):
        """
        Prints the model summary for the encoder, decoder and NRI model.
        """
        print(5*'-', 'Topology Estimator Model Summary', 5*'-')
        # # print encoder summary
        # print(2*'-', 'Encoder Summary')
        # print("\n Enocder pipeline:\n")
        # for layer in encoder.pipeline:
        #     print(layer)

        # print("\n Encoder embedding function summary:\n")         
        # print(encoder)

        # # print decoder summary
        # print(2*'-', 'Decoder Summary')
        # print("\n Decoder embedding function summary:\n")
        # print(decoder)

        # print nri model summary
        print(2*'-', 'NRI Model Summary')
        print(nri_model)
        


