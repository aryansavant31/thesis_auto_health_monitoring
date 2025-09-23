import os, sys

# ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.insert(0, ROOT_DIR) if ROOT_DIR not in sys.path else None

# FDET_DIR = os.path.dirname((os.path.abspath(__file__)))
# sys.path.insert(0, FDET_DIR) if FDET_DIR not in sys.path else None

# print(f"Root directory set to: {ROOT_DIR}")
# print(f"Fault detection directory set to: {FDET_DIR}")

# other imports
import inspect 
from torch.utils.tensorboard import SummaryWriter

# global imports
from data.config import DataConfig
from data.prep import DataPreprocessor
from console_logger import ConsoleLogger
from feature_extraction.selector import FeatureSelector

# local imports
from .settings.manager import AnomalyDetectorTrainManager, get_model_pickle_path
from .detector import AnomalyDetector, TrainerAnomalyDetector


class AnomalyDetectorTrainPipeline:
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
        train_data, test_data, _, = self.data_preprocessor.get_training_data_package(
            self.data_config, 
            batch_size=self.fdet_config.batch_size,
            train_rt=self.fdet_config.train_rt,
            test_rt=self.fdet_config.test_rt
            )

        # unpack data_loaders and data_stats
        train_loader, train_data_stats = train_data
        test_loader, test_data_stats = test_data

    # 2. Initialize the anomaly detector model
        anomaly_detector = self._init_model(train_data_stats)

    # 3. Feature selection (if enabled)
        if self.fdet_config.feat_select_config is not None:
            feat_selector = FeatureSelector(self.fdet_config.feat_select_config)
            anomaly_detector = feat_selector.select_features(anomaly_detector, train_loader, self.data_config)
        else:
            print("\nFeature selection is disabled.")
            feat_selector = None

        train_logger = self._prep_for_training(anomaly_detector, train_loader)

        # plot feature ranking
        if feat_selector is not None:
            if self.fdet_config.train_plots['feat_ranking_plot'][0]:
                feat_selector.feat_ranking_histogram(logger=train_logger)

    # 4. Train the anomaly detector model
        trainer = TrainerAnomalyDetector(logger=train_logger)
        trained_anomaly_detector = trainer.fit(anomaly_detector, train_loader)

        # plot the train results
        for plot_name, plot_config in self.fdet_config.train_plots.items():
            if plot_name != 'feat_ranking_plot':  # already plotted
                if plot_config[0]:
                    getattr(trainer, plot_name.split('-')[0])(**plot_config[1])

    # 5. Test the trained anomaly detector model
        # update its data stats
        test_logger = self._prep_for_testing()

        # test the trained model
        tester = TrainerAnomalyDetector(logger=test_logger)
        tester.test(trained_anomaly_detector, test_loader)

        # plot the test results
        for plot_name, plot_config in self.fdet_config.test_plots.items():
            if plot_config[0]:
                getattr(tester, plot_name.split('-')[0])(**plot_config[1])


    def _prep_for_training(self, anomaly_detector, train_loader):
        """
        Prepare the training environment before starting the training process.
        """
        # if logging enabled, save parameters and initialize TensorBoard logger
        if self.fdet_config.is_log:

            # get log path to save trained model
            n_comp, n_dims = TrainerAnomalyDetector().process_input_data(anomaly_detector, train_loader, get_data_shape=True)
            self.train_log_path = self.fdet_config.get_train_log_path(n_comp, n_dims)
            
            self.fdet_config.save_params()
            train_logger = SummaryWriter(log_dir=self.train_log_path)

            # log all the attributes of train_config
            # formatted_params = "\n".join([f"{key}: {value}" for key, value in self.fdet_config.__dict__.items()])
            # train_logger.add_text(os.path.basename(self.train_log_path), formatted_params)

            # log dataset selected
            data_text = self.data_preprocessor.get_data_selection_text()   
            train_logger.add_text(f"{os.path.basename(self.train_log_path)} + train", data_text)     

            print(f"\nTraining environment set. Training will be logged at: {self.train_log_path}")
        
        else:
            self.train_log_path = None
            train_logger = None
            print("\nTraining environment set. Logging is disabled.")

        print('\n' + 75*'-')

        return train_logger
    
    def _prep_for_testing(self):
        """
        Prepare the testing environment before starting the testing process.
        """
        # if logging enabled, initialize TensorBoard logger
        if self.fdet_config.is_log:
            # get log path to save trained model
            test_log_path = self.fdet_config.get_test_log_path()
            test_logger = SummaryWriter(log_dir=test_log_path)
            print(f"\nTesting environment set. Testing will be logged at: {test_log_path}")
        else:
            test_logger = None
            print("\nTesting environment set. Logging is disabled.")
        
        return test_logger

    def _init_model(self, train_data_stats):
        """
        Load the anomaly detector model and set the requried run params.
        """
        # prep hyperparams
        self.fdet_config.hparams.update({
            'max_timesteps': f"{int(self.data_config.max_timesteps):,}", 
            })
        
        anomaly_detector = AnomalyDetector(self.fdet_config.anom_config, self.fdet_config.hparams)
        anomaly_detector.ok_percentage = self.fdet_config.ok_percentage
        
        req_run_params = inspect.signature(anomaly_detector.set_run_params).parameters.keys()
        run_config = {key: value for key, value in self.fdet_config.__dict__.items() if key in req_run_params and key not in ['data_config']}

        anomaly_detector.set_run_params(**run_config, data_config=self.data_config)

        # print model info
        print("\nAnomaly Detector Model Initialized with the following configurations:")
        anomaly_detector.print_model_info()
        print('\n' + 75*'-')

        return anomaly_detector
    
    # def _load_model(self, train_log_path, test_data_stats):
    #     """
    #     Load the anomaly detector model from the saved pickle file.
    #     """
    #     # load trained model
    #     trained_anomaly_detector = AnomalyDetector.load_from_pickle(get_model_pickle_path(train_log_path))

    #     # update its data stats
    #     trained_anomaly_detector.data_stats = test_data_stats

    #     print("\nTrained anomaly Detector model loaded for testing:")
    #     return trained_anomaly_detector
        

if __name__ == "__main__":
    # create console logger to log all the outputs in terminal
    console_logger = ConsoleLogger()

    data_config = DataConfig(run_type="train")
    fdet_config = AnomalyDetectorTrainManager(data_config)

    with console_logger.capture_output():
        print("\nStarting fault detection model training...")

        train_pipeline = AnomalyDetectorTrainPipeline(data_config, fdet_config)
        train_pipeline.train()

        base_name = os.path.basename(train_pipeline.train_log_path) if train_pipeline.train_log_path else fdet_config.anom_config['anom_type']
        print('\n' + 75*'=')
        print(f"\nFault detection model '{base_name}' training completed.")

    if fdet_config.is_log:
        # save the captured output to a file
        file_path = os.path.join(train_pipeline.train_log_path, "console_output.txt")
        console_logger.save_to_file(file_path, script_name="fault_detection.train.py", base_name=base_name)
