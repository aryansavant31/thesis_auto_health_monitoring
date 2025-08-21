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
from .settings.manager import TopologyEstimationInferManager, get_checkpoint_path
from .graph_structures import RelationMatrixMaker
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

    def load_relation_matrix_loaders(self):
        """
        Load the relation matrix loaders for training, validation, and test data.

        Attributes
        ----------
        rel_loader : DataLoader
            DataLoader for the relation matrices of the training data.
        """
        self.rel_loader = self.rm.get_relation_matrix_loader(self.custom_loader)

    def get_encoder_params(self):
        """
        Get the encoder parameters required for initializing the NRI model.

        Returns
        -------
        enc_run_params : dict
            Dictionary containing the encoder run parameters.
        """
        # encoder params
        req_enc_run_params = inspect.signature(Encoder.set_run_params).parameters.keys()
        enc_run_params = {
            key.removeprefix('enc_'): value for key, value in self.tp_config.__dict__.items() if key.removeprefix('enc_') in req_enc_run_params
        }
        return enc_run_params
        
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
            key.removeprefix('dec_'): value for key, value in self.tp_config.__dict__.items() if key.removeprefix('dec_') in req_dec_run_params
        }

        return dec_run_params
    
    def _prep_for_inference(self):
        """
        Prepare the inference environment before starting the inference process.
        """
        # if logging enabled, save parameters and initialize TensorBoard logger
        if self.tp_config.is_log:
            self.infer_log_path = self.tp_config.get_infer_log_path()

            self.tp_config.save_infer_params()
            logger = TensorBoardLogger(os.path.dirname(self.infer_log_path), name="", version=os.path.basename(self.infer_log_path))

            # log all the attributes of train_config
            formatted_params = "\n".join([f"{key}: {value}" for key, value in self.tp_config.__dict__.items()])
            logger.experiment.add_text(os.path.basename(self.train_log_path), formatted_params)
            print(f"\n{self.tp_config.run_type.capitalize()} environment set. {self.tp_config.run_type.capitalize()} will be logged at: {self.infer_log_path}")

        else:
            self.infer_log_path = None
            logger = None
            print(f"\n{self.tp_config.run_type.capitalize()} environment set. Logging is disabled.")

        print('\n' + 75*'-')
        return logger
    

class NRIInferMain(TopologyEstimationInferHelper):
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
        self.nri_config = nri_config

    def infer(self):
        """
        Perform inference using the NRI model.
        """
        # load data
        self.load_training_data()
        self.load_relation_matrix_loaders()

        # initialize model
        enc_run_params = self.get_encoder_params()
        dec_run_params = self.get_decoder_params()
        
        nri_model = self._load_model(enc_run_params, dec_run_params, self.custom_data_stats)

        # infer using the nri model
        logger = self._prep_for_inference()

        tester = Trainer(logger=logger)
        if self.nri_config.run_type == 'custom_test':
            tester.test(nri_model, CombinedDataLoader(self.custom_loader, self.rel_loader))

        elif self.nri_config.run_type == 'predict':
            preds = tester.predict(nri_model, CombinedDataLoader(self.custom_loader, self.rel_loader))
            return preds

    
    def _load_model(self, enc_run_params, dec_run_params, data_stats):
        """
        Load the NRI model from the checkpoint path.

        Parameters
        ----------
        enc_run_params : dict
            Dictionary containing the encoder run parameters.
        dec_run_params : dict
            Dictionary containing the decoder run parameters.
        data_stats : dict
            Statistics of the custom data.

        Returns
        -------
        trained_nri_model : NRI
            The loaded NRI model.
        """
        trained_nri_model = NRI.load_from_checkpoint(self.nri_config.ckpt_path)

        # update the model with new run params and custom data statistics
        trained_nri_model.set_run_params(enc_run_params, dec_run_params, data_stats, self.nri_config.temp, self.nri_config.is_hard)

        print(f"\nNRI model loaded for '{self.tp_config.run_type}' from {self.tp_config.ckpt_path}")
        print(f"\nModel type: {type(trained_nri_model).__name__}, Model ID: {trained_nri_model.hparams['model_id']}, No. of input features req.: {trained_nri_model.encoder.n_components}, No. of dimensions req.: {trained_nri_model.decoder.n_dims}") 
        print('\n' + 75*'-')

        return trained_nri_model
    
class DecoderInferMain(TopologyEstimationInferHelper):
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
        self.decoder_config = decoder_config

    def infer(self):
        """
        Perform inference using the Decoder model.
        """
        # load data
        self.load_training_data()
        self.load_relation_matrix_loaders()

        # initialize model
        dec_run_params = self.get_decoder_params()
        decoder_model = self._load_model(dec_run_params, self.custom_data_stats)

        # infer using the decoder model
        logger = self._prep_for_inference()

        tester = Trainer(logger=logger)
        if self.decoder_config.run_type == 'custom_test':
            tester.test(decoder_model, CombinedDataLoader(self.custom_loader, self.rel_loader))

        elif self.decoder_config.run_type == 'predict':
            preds = tester.predict(decoder_model, CombinedDataLoader(self.custom_loader, self.rel_loader))
            return preds
        
    def _load_model(self, dec_run_params, data_stats):
        """
        Load the Decoder model from the checkpoint path.

        Parameters
        ----------
        dec_run_params : dict
            Dictionary containing the decoder run parameters.
        data_stats : dict
            Statistics of the custom data.

        Returns
        -------
        Decoder
            The loaded Decoder model.
        """
        trained_decoder_model = Decoder.load_from_checkpoint(self.decoder_config.ckpt_path)

        # update the model with new run params and custom data statistics
        trained_decoder_model.set_run_params(dec_run_params, data_stats, self.decoder_config.temp, self.decoder_config.is_hard)

        print(f"\nDecoder model loaded for '{self.decoder_config.run_type}' from {self.decoder_config.ckpt_path}")
        print(f"\nModel type: {type(trained_decoder_model).__name__}, Model ID: {trained_decoder_model.hparams['model_id']}, No. of dimensions req.: {trained_decoder_model.n_dims}")
        print('\n' + 75*'-')

        return trained_decoder_model
    

if __name__ == "__main__":
    # create console logger to log all the outputs in terminal
    console_logger = ConsoleLogger()
    parser = argparse.ArgumentParser(description="Infer NRI or Decoder models.")

    parser.add_argument('--framework', type=str, 
                        choices=['nri', 'decoder'],
                        default='nri',
                        required=True, help="Framework to infer: nri or decoder")
    
    parser.add_argument('--run-type', type=str, 
                        choices=['custom_test', 'predict'],
                        default='predict',
                        required=True, help="Run type: custom_test or predict")

    args = parser.parse_args()

    data_config = DataConfig(run_type=args.run_type)
    tp_config = TopologyEstimationInferManager(data_config, args.framework, args.run_type)

    with console_logger.capture_output():
        print(f"\nStarting {args.framework} model to {args.run_type}...")

        infer_pipeline = NRIInferMain(data_config, tp_config)
        if args.run_type == 'custom_test':
            infer_pipeline.infer()

        elif args.run_type == 'predict':
            preds = infer_pipeline.infer()

        base_name = f"{tp_config.selected_model_num}/{os.path.basename(infer_pipeline.infer_log_path)}" if infer_pipeline.infer_log_path else f"{tp_config.selected_model_num}/{args.run_type}"
        print('\n' + 75*'=')
        print(f"\n{args.run_type.capitalize()} of {args.framework} model '{base_name}' completed.")

    if tp_config.is_log:
        # save the captured output to a file
        file_path = os.path.join(infer_pipeline.infer_log_path, "console_output.txt")
        console_logger.save_to_file(file_path, script_name="topology_estimation.infer.py", base_name=base_name)
        