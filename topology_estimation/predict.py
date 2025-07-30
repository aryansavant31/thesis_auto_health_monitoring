from .config.manager import PredictNRIConfigMain
from .nri import NRI
from data.config import DataConfig
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
import os
from .graph_structures import FullyConnectedGraph, SparisifiedGraph
from data.transform import DataTransformer
from feature_extraction.config import FeatureExtractor

class PredictNRIMain:
    def __init__(self):
        self.data_config = DataConfig()
        self.tp_config = PredictNRIConfigMain()

    def test_model(self):
        # load data
        test_loader = self.load_data('test')

        # load model
        trained_model = self.load_model()

        test_log_path = self.tp_config.get_custom_test_log_path()

        logger = TensorBoardLogger(os.path.dirname(test_log_path), name=None, version=os.path.basename(test_log_path))
        trainer = Trainer(logger=logger)

        trainer.test(trained_model, test_loader)
 
    def predict(self):
        # load data
        predict_loader = self.load_data('predict')
        
        # load model
        trained_model = self.load_model()

        predict_log_path = self.tp_config.get_predict_log_path()

        logger = TensorBoardLogger(os.path.dirname(predict_log_path), name=None, version=os.path.basename(predict_log_path))
        trainer = Trainer(logger=logger)

        trainer.predict(trained_model, predict_loader)

    def load_data(self, run_type):
        # set parameters
        if run_type == 'test':
            self.data_config.set_custom_test_dataset()
        else:
            self.data_config.set_predict_dataset()

        # get predict dataset paths
        node_ds_paths, edge_ds_paths = self.data_config.get_dataset_paths()
        # load predict data
        custom_loader, self.data_stats = "make_dataloader(node_ds_path, edge_ds_path)" # Some dataloader function that loads just 1 dataloader
        
        # for getting data stats
        dataiter = iter(custom_loader)
        data = next(dataiter)
    
        # process the input data to get correct data shape for the model initialization  
        data = self.process_input_data(data)    

        # set data stats
        self.batch_size = data[0].shape[0]
        self.n_nodes = data[0].shape[1]
        self.n_components = data[0].shape[2]
        self.n_dims = data[0].shape[3]

        self._verbose_load_data()

        return custom_loader

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
        if self.tp_config.fex_configs_encoder:
            data = FeatureExtractor(fex_configs=self.tp_config.fex_configs_encoder)(data)

        return data
    
    def load_model(self):
        trained_model = NRI.load_from_checkpoint(self.tp_config.ckpt_path)
        run_params = {} # [TODO]: pass params from tp_pred_config for set_run_params()
        
        trained_model.set_run_params()  # [TODO]: pass params from tp.config for set_run_params() 

        # set relation matrices
        rec_rel, send_rel = self.set_relation_matrices()
        trained_model.set_input_graph(rec_rel, send_rel)

        self._verbose_init_model(trained_model)

        return trained_model
    
    def set_relation_matrices(self):
        if self.tp_config.sparsif_type:
            rec_rel, send_rel = SparisifiedGraph(self.n_nodes).get_relation_matrix()
        else:
            rec_rel, send_rel = FullyConnectedGraph(self.n_nodes, self.batch_size).get_relation_matrices()

        return rec_rel, send_rel

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