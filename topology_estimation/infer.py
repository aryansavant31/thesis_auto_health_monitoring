import sys, os

# other imports
import inspect
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import argparse

# global imports
from data.config import DataConfig
from data.prep import DataPreprocessor
from console_logger import ConsoleLogger

# local imports
from .settings.manager import TopologyEstimationInferManager
from .relations import RelationMatrixMaker
from .utils.custom_loader import CombinedDataLoader
from .nri import NRI
from .decoder import Decoder
from .encoder import Encoder

class TopologyEstimationInferHelper:
    """
    Base class for NRI and Decoder infer pipelines.
    """
    def __init__(self, data_config:DataConfig, tp_config:TopologyEstimationInferManager):
        self.data_config = data_config
        self.tp_config = tp_config

        self.data_preprocessor = DataPreprocessor("topology_estimation")
        self.rm = RelationMatrixMaker(self.tp_config.spf_config)

    def load_training_data(self):
        """
        Load the training, validation, and test data.

        Attributes
        ----------
        custom_loader : DataLoader
            DataLoader for the custom data.
        custom_data_stats : dict
            Statistics of the custom data.
        """
        custom_data = self.data_preprocessor.get_custom_data_package(
            self.data_config, 
            batch_size=self.tp_config.batch_size,
            )
        # unpack data_loaders and data_stats
        self.custom_loader, self.custom_data_stats = custom_data

    # def load_relation_matrix_loaders(self):
    #     """
    #     Load the relation matrix loaders for training, validation, and test data.

    #     Attributes
    #     ----------
    #     rel_loader : DataLoader
    #         DataLoader for the relation matrices of the training data.
    #     """
    #     self.rel_loader = self.rm.get_relation_matrix_loader(self.custom_loader)
        
    def get_decoder_params(self):
        """
        Get the decoder parameters

        Returns
        -------
        dec_run_params : dict
            Dictionary containing the decoder run parameters.
        """
        # decoder params
        req_dec_run_params = inspect.signature(Decoder.set_run_params).parameters.keys()
        dec_run_params = {
            key.removeprefix('dec_'): value for key, value in self.tp_config.__dict__.items() if key.removeprefix('dec_') in req_dec_run_params and key not in ['data_config']
        }

        print("\nDecoder run parameters:")
        print(15 * "-")
        for key, value in dec_run_params.items():
            print(f"{key}: {value}")

        # print('\n' + 35*'-')

        return dec_run_params
    
    def _prep_for_inference(self):
        """
        Prepare the inference environment before starting the inference process.
        """
        # if logging enabled, save parameters and initialize TensorBoard logger
        if self.tp_config.is_log:
            self.infer_log_path = self.tp_config.get_infer_log_path()

            print("INFER LOG PATH", self.infer_log_path) #DEBUG 
            self.tp_config.save_infer_params()
            logger = TensorBoardLogger(os.path.dirname(self.infer_log_path), name="", version=os.path.basename(self.infer_log_path))

            # log all the attributes of train_config
            # formatted_params = "\n".join([f"{key}: {value}" for key, value in self.tp_config.__dict__.items()])
            # logger.experiment.add_text(os.path.basename(self.infer_log_path), formatted_params)

            # log dataset selected
            data_text = self.data_preprocessor.get_data_selection_text()
            rel_text = self.rm.get_relation_matrices_summary()
            data_text += rel_text
            
            base_name = f"{self.tp_config.selected_model_num} + {os.path.basename(self.infer_log_path)}"
            logger.experiment.add_text(base_name, data_text)

            print(f"\n{self.tp_config.run_type.capitalize()} environment set. {self.tp_config.run_type.capitalize()} will be logged at: {self.infer_log_path}")

        else:
            self.infer_log_path = None
            logger = None
            print(f"\n{self.tp_config.run_type.capitalize()} environment set. Logging is disabled.")

        print('\n' + 75*'-')
        return logger
    

class NRIInferPipeline(TopologyEstimationInferHelper):
    """
    Main class for NRI inference.
    """
    def __init__(self, data_config:DataConfig, nri_config:TopologyEstimationInferManager):
        """
        Initialize the NRI inference main class.

        Parameters
        ----------
        data_config : DataConfig
            Configuration object for the data.
        nri_config : TopologyEstimationInferManager
            Configuration object for the NRI inference.
        """
        super().__init__(data_config, nri_config)

    def infer(self, device='auto'):
        """
        Perform inference using the NRI model.
        """
        # load data
        self.load_training_data()

        # load relation matrices
        rec_rel, send_rel = self.rm.get_relation_matrix(self.custom_loader)

        # initialize model
        dec_run_params = self.get_decoder_params()
        
        nri_model = self._load_model(
            dec_run_params, 
            rec_rel, send_rel, self.custom_data_stats
            )

        # infer using the nri model
        logger = self._prep_for_inference()

        tester = Trainer(
            accelerator=device,
            logger=logger
            )
        
        if self.tp_config.run_type == 'custom_test':
            preds = tester.test(nri_model, self.custom_loader)

        elif self.tp_config.run_type == 'predict':
            preds = tester.predict(nri_model, self.custom_loader)

        return preds

    
    def _load_model(self, dec_run_params, rec_rel, send_rel, data_stats):
        """
        Load the NRI model from the checkpoint path.

        Parameters
        ----------
        dec_run_params : dict
            Dictionary containing the decoder run parameters.
        rec_rel : torch.Tensor, shape (n_edges, n_nodes)
            Receiver relation matrix that indicate which edges are on reciever end of nodes.
        send_rel : torch.Tensor, shape (n_edges, n_nodes)
            Sender relation matrix that indicate which edges are senders of nodes.
        data_stats : dict
            Statistics of the custom data.

        Returns
        -------
        trained_nri_model : NRI
            The loaded NRI model.
        """
        trained_nri_model = NRI.load_from_checkpoint(self.tp_config.selected_model_path)

        # update model with domain config
        trained_nri_model.encoder.domain_config = self.tp_config.enc_domain_config
        trained_nri_model.decoder.domain_config = self.tp_config.dec_domain_config

        # update model with infer hyperparams
        trained_nri_model.hyperparams.update(self.tp_config.infer_hyperparams)
        trained_nri_model.hyperparams.update({
            f"max_timesteps/{self.tp_config.run_type}" : f"{int(self.data_config.max_timesteps):,}"
        })
        

        # update the model with new run params and custom data statistics
        trained_nri_model.set_input_graph(rec_rel, send_rel)

        trained_nri_model.set_run_params(
            dec_run_params=dec_run_params, 
            data_config=self.data_config, 
            data_stats=data_stats, 
            init_temp=self.tp_config.init_temp,
            min_temp=self.tp_config.min_temp,
            decay_temp=self.tp_config.decay_temp,
            is_hard=self.tp_config.is_hard,
            dynamic_rel=self.tp_config.dynamic_rel
            )

        print(f"\nNRI model loaded for '{self.tp_config.run_type}' from {self.tp_config.ckpt_path}")
        print(f"\nModel type: {type(trained_nri_model).__name__}, Model ID: {trained_nri_model.hyperparams['model_id']}, No. of input features req.: {trained_nri_model.encoder.n_comps}, No. of dimensions req.: {trained_nri_model.decoder.n_dims}") 
        print('\n' + 75*'-')

        return trained_nri_model
    
class DecoderInferPipeline(TopologyEstimationInferHelper):
    """
    Main class for Decoder inference.
    """
    def __init__(self, data_config:DataConfig, decoder_config:TopologyEstimationInferManager):
        """
        Initialize the Decoder inference main class.

        Parameters
        ----------
        data_config : DataConfig
            Configuration object for the data.
        decoder_config : TopologyEstimationInferManager
            Configuration object for the Decoder inference.
        """
        super().__init__(data_config, decoder_config)

    def infer(self, device='auto'):
        """
        Perform inference using the Decoder model.
        """
        # load data
        self.load_training_data()
        
        # load relation matrices
        rec_rel, send_rel = self.rm.get_relation_matrix(self.custom_loader)

        # initialize model
        dec_run_params = self.get_decoder_params()
        decoder_model = self._load_model(
            dec_run_params, 
            rec_rel, send_rel, self.custom_data_stats
            )

        # infer using the decoder model
        logger = self._prep_for_inference()

        tester = Trainer(
            accelerator=device,
            logger=logger)

        if self.tp_config.run_type == 'custom_test':
            preds = tester.test(decoder_model, self.custom_loader)

        elif self.tp_config.run_type == 'predict':
            preds = tester.predict(decoder_model, self.custom_loader)
            preds['adj_matrix_label'] = send_rel.T @ rec_rel

        return preds
        
    def _load_model(self, dec_run_params, rec_rel, send_rel, data_stats):
        """
        Load the Decoder model from the checkpoint path.

        Parameters
        ----------
        dec_run_params : dict
            Dictionary containing the decoder run parameters.
        rec_rel : torch.Tensor, shape (n_edges, n_nodes)
            Receiver relation matrix that indicate which edges are on reciever end of nodes.
        send_rel : torch.Tensor, shape (n_edges, n_nodes)
            Sender relation matrix that indicate which edges are senders of nodes.
        data_stats : dict
            Statistics of the custom data.

        Returns
        -------
        Decoder
            The loaded Decoder model.
        """
        trained_decoder_model = Decoder.load_from_checkpoint(self.tp_config.selected_model_path)

        # update model with domain config
        trained_decoder_model.domain_config = self.tp_config.dec_domain_config
        
        # update model with infer hyperparams
        trained_decoder_model.hyperparams.update(self.tp_config.infer_hyperparams)
        trained_decoder_model.hyperparams.update({
            f"max_timesteps/{self.tp_config.run_type}" : f"{int(self.data_config.max_timesteps):,}"
        })

        # update the model with new run params and custom data statistics
        trained_decoder_model.set_input_graph(
            rec_rel, send_rel, 
            make_edge_matrix=True, 
            always_fully_connected_rel=self.tp_config.always_fully_connected_rel,
            batch_size=self.custom_loader.batch_size
            )

        trained_decoder_model.set_run_params(
            **dec_run_params, 
            data_config=self.data_config, 
            data_stats=data_stats
            )

        print(f"\nDecoder model loaded for '{self.tp_config.run_type}' from {self.tp_config.selected_model_path}")
        print(f"\nModel type: {type(trained_decoder_model).__name__}, Model ID: {trained_decoder_model.hyperparams['model_id']}, No. of dimensions req.: {trained_decoder_model.n_dims}")
        print('\n' + 75*'-')

        return trained_decoder_model
    

if __name__ == "__main__":
    # create console logger to log all the outputs in terminal
    console_logger = ConsoleLogger()
    parser = argparse.ArgumentParser(description="Infer NRI or Decoder models.")

    parser.add_argument('--framework', type=str, 
                        choices=['nri', 'decoder', 'full_tp'],
                        default='nri',
                        required=True, help="Framework to infer: nri, decoder or full_tp")
    
    parser.add_argument('--run-type', type=str, 
                        choices=['custom_test', 'predict'],
                        default='predict',
                        required=True, help="Run type: custom_test or predict")

    args = parser.parse_args()

    data_config = DataConfig(run_type=args.run_type)

    if args.framework in ['decoder', 'full_tp']:
        decoder_config = TopologyEstimationInferManager(data_config, 'decoder', args.run_type)
    if args.framework in ['nri', 'full_tp']:
        nri_config = TopologyEstimationInferManager(data_config, 'nri', args.run_type)
    
    use_nri = False

    with console_logger.capture_output():
        print(f"\nStarting {args.framework} model to {args.run_type}...")

        if args.framework == 'decoder' or args.framework == 'full_tp':
            decoder_infer_pipeline = DecoderInferPipeline(data_config, decoder_config)
            preds_dec = decoder_infer_pipeline.infer()
            base_name = f"{decoder_config.selected_model_num}/{os.path.basename(decoder_infer_pipeline.infer_log_path)}" if decoder_infer_pipeline.infer_log_path else f"{decoder_config.selected_model_num}/{args.run_type}"
            
            print('\n' + 75*'=')
            print(f"\n{args.run_type.capitalize()} of decoder model '{base_name}' completed.")

            if args.run_type == 'predict':
                if preds_dec['dec/residual'] > decoder_config.residual_thresh:
                    print(f"\nDecoder residual {preds_dec['dec/residual']:,.4f} > {decoder_config.residual_thresh}. Hence using NRI model for topology prediction.")
                    use_nri = True
                else:
                    print(f"\nDecoder residual {preds_dec['dec/residual']:,.4f} <= {decoder_config.residual_thresh}. Hence given topology to decoder is correct.")
                    use_nri = False

            if decoder_config.is_log:
                # save the captured output to a file
                file_path = os.path.join(decoder_infer_pipeline.infer_log_path, "console_output.txt")
                console_logger.save_to_file(file_path, script_name="topology_estimation.infer.py", base_name=base_name)


        if args.framework == 'nri' or (args.framework == 'full_tp' and use_nri):
            nri_infer_pipeline = NRIInferPipeline(data_config, nri_config)
            preds_enc = nri_infer_pipeline.infer()
            base_name = f"{nri_config.selected_model_num}/{os.path.basename(nri_infer_pipeline.infer_log_path)}" if nri_infer_pipeline.infer_log_path else f"{nri_config.selected_model_num}/{args.run_type}"
            
            print('\n' + 75*'=')
            print(f"\n{args.run_type.capitalize()} of nri model '{base_name}' completed.")

            if nri_config.is_log:
                # save the captured output to a file
                file_path = os.path.join(nri_infer_pipeline.infer_log_path, "console_output.txt")
                console_logger.save_to_file(file_path, script_name="topology_estimation.infer.py", base_name=base_name)
        