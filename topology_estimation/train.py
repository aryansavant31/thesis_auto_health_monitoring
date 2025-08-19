import sys, os

# other imports


# global imports
from data.config import DataConfig
from data.prep import DataPreprocessor
from console_logger import ConsoleLogger

# local imports
from .settings.manager import NRITrainManager, DecoderTrainManager, get_checkpoint_path
from .nri import NRI
from .decoder import Decoder
from .encoder import Encoder

class NRITrainMain:
    def __init__(self, data_config:DataConfig, nri_config:NRITrainManager):
        """
        Initialize the NRI training main class.

        Parameters
        ----------
        data_config : DataConfig
            The data configuration object.
        nri_config : NRITrainManager
            The NRI training configuration object.
        """
        self.data_config = data_config
        self.nri_config = nri_config
        self.data_preprocessor = DataPreprocessor("topology_estimation")

    def train(self):
        """
        Main method to train the NRI model.
        """
    # 1. Load data
        train_data, test_data, val_data = self.data_preprocessor.get_training_data_package(
            self.data_config, 
            batch_size=self.nri_config.batch_size,
            train_rt=self.nri_config.train_rt,
            test_rt=self.nri_config.test_rt,
            val_rt=self.nri_config.val_rt
            )
        # unpack data_loaders and data_stats
        train_loader, train_data_stats = train_data
        test_loader, test_data_stats = test_data
        val_loader, val_data_stats = val_data

    # 2. Initialize the NRI model
        # prepare the model params
