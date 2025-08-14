import os, sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR) if ROOT_DIR not in sys.path else None

FDET_DIR = os.path.dirname((os.path.abspath(__file__)))
sys.path.insert(0, FDET_DIR) if FDET_DIR not in sys.path else None

# other imports
import inspect 

# global imports
from data.config import DataConfig
from data.prep import DataPreprocessor
from console_logger import ConsoleLogger
from torch.utils.tensorboard import SummaryWriter

# local imports
from settings.manager import AnomalyDetectorTrainManager, get_model_pickle_path
from detector import AnomalyDetector, TrainerAnomalyDetector


class AnomalyDetectorTrainMain:
    def __init__(self, data_config:DataConfig, fdet_config:AnomalyDetectorTrainManager):
        """
        Initialize the anomaly detector training main class.

        Parameters
        ----------
        data_config : DataConfig
            The data configuration object.
        fdet_config : AnomalyDetectorTrainManager
            The anomaly detector training configuration object.
        """
        self.data_config = data_config
        self.fdet_config = fdet_config
        self.data_preprocessor = DataPreprocessor("fault_detection")

    def train(self):
        """
        Main method to train the anomaly detector.
        """
    # 1. Load data
        train_data, test_data, _, = self.data_preprocessor.get_training_data_package(self.data_config)

        # unpack data_loaders and data_stats
        train_loader, train_data_stats = train_data
        test_loader, test_data_stats = test_data

    # 2. Initialize the anomaly detector model
        anomaly_detector = self._init_model(train_data_stats)

    # 3. Train the anomaly detector model
        train_logger = self._prep_for_training(anomaly_detector, train_loader)

        trainer = TrainerAnomalyDetector(logger=train_logger)
        trainer.fit(anomaly_detector, train_loader)

        # plot training results if required
        if self.fdet_config.is_log:
            pass # [TODO] add ploting code here


    # 4. Test the trained anomaly detector model
        trained_anomaly_detector = self._load_model(self.train_log_path, test_data_stats)
        
        test_logger = self._prep_for_testing()

        # test the trained model
        tester = TrainerAnomalyDetector(logger=test_logger)
        tester.test(trained_anomaly_detector, test_loader)

        # plot testing results if required
        if self.fdet_config.is_log:
            pass # [TODO] add ploting code here


    def _prep_for_training(self, anomaly_detector, train_loader):
        """
        Prepare the training environment before starting the training process.
        """
        # get log path to save trained model
        n_comp, n_dims = TrainerAnomalyDetector().process_input_data(anomaly_detector, train_loader, get_data_shape=True)
        self.train_log_path = self.fdet_config.get_train_log_path(n_comp, n_dims)

        # if logging enabled, save parameters and initialize TensorBoard logger
        if self.fdet_config.is_log:
            self.fdet_config.save_params()
            train_logger = SummaryWriter(log_dir=self.train_log_path)

            # log all the attributes of train_config
            formatted_params = "\n".join([f"{key}: {value}" for key, value in self.fdet_config.__dict__.items()])
            train_logger.add_text(os.path.basename(self.train_log_path), formatted_params)
        else:
            train_logger = None

        return train_logger
    
    def _prep_for_testing(self):
        """
        Prepare the testing environment before starting the testing process.
        """
        # get log path to save trained model
        test_log_path = self.fdet_config.get_test_log_path()

        # if logging enabled, initialize TensorBoard logger
        if self.fdet_config.is_log:
            test_logger = SummaryWriter(log_dir=test_log_path)
        else:
            test_logger = None

        return test_logger

    def _init_model(self, train_data_stats):
        """
        Load the anomaly detector model and set the requried run params.
        """
        anomaly_detector = AnomalyDetector(self.fdet_config.anom_config)
        
        req_run_params = inspect.signature(anomaly_detector.set_run_params).parameters.keys()
        run_config = {key: value for key, value in self.fdet_config.__dict__.items() if key in req_run_params}

        anomaly_detector.set_run_params(**run_config, data_stats=train_data_stats)

        # print model info
        print("Anomaly Detector Model Initialized with the following configurations:")
        anomaly_detector.print_model_info()

        return anomaly_detector
    
    def _load_model(self, train_log_path, test_data_stats):
        """
        Load the anomaly detector model from the saved pickle file.
        """
        # load trained model
        trained_anomaly_detector = AnomalyDetector.load_from_pickle(get_model_pickle_path(train_log_path))

        # update its data stats
        trained_anomaly_detector.data_stats = test_data_stats

        return trained_anomaly_detector
        

if __name__ == "__main__":
    # create console logger to log all the outputs in terminal
    console_logger = ConsoleLogger()

    data_config = DataConfig(run_type="train")
    fdet_config = AnomalyDetectorTrainManager(data_config)

    with console_logger.capture_output():
        print("\nStarting fault detection model training...")

        train_pipeline = AnomalyDetectorTrainMain(data_config, fdet_config)
        train_pipeline.train()

        print("\nFault detection model training completed.")

    if fdet_config.is_log:
        # save the captured output to a file
        file_path = os.path.join(train_pipeline.train_log_path, "console_output.txt")
        base_name = os.path.basename(train_pipeline.train_log_path)
        console_logger.save_to_file(file_path, script_name="fault_detection.train.py", base_name=base_name)
