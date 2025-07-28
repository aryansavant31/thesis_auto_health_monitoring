from fault_detection.anomaly_detector import AnomalyDetector, TrainerAnomalyDetector
from config import TrainAnomalyDetectorConfig, PredictAnomalyDetectorConfig, get_model_pickle_path
from data.config import DataConfig
from data.prep import DataPreprocessor, load_spring_particle_data
from data.transform import DataTransformer
from feature_extraction.extractor import FeatureExtractor
from torch.utils.tensorboard import SummaryWriter

class PredictAnomalyDetectorMain:
    pass

class TrainAnomalyDetectorMain:
    def __init__(self):
        self.data_preprocessor = DataPreprocessor(package='fault_detection')
        self.config = TrainAnomalyDetectorConfig()

    def load_data(self):
        # load data
        self.train_loader, self.test_loader, _, self.data_stats = self.data_preprocessor.get_training_dataloaders(
            self.config.train_rt,
            self.config.test_rt,
            self.config.batch_size,
        )

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
        data = DataTransformer(domain=self.config.domain, 
                                        norm_type=self.config.norm_type, 
                                        data_stats=self.data_stats)(data)

        # extract features from data if fex_configs are provided
        if self.config.fex_configs:
            data = FeatureExtractor(fex_configs=self.config.fex_configs)(data)

        return data
    
    def init_model(self):
        anomaly_detector = AnomalyDetector(anom_config=self.config.anom_config)
        anomaly_detector.set_run_params(data_stats=self.data_stats,
                                        domain=self.config.domain, 
                                        norm_type=self.config.norm_type, 
                                        fex_configs=self.config.fex_configs)
        # print model info
        print("Anomaly Detector Model Initialized with the following configurations:")
        anomaly_detector.print_model_info()

        return anomaly_detector
    
    def train(self):
        """
        Train the anomaly detection model.
        """
        # load data
        self.load_data()

        # initialize anomaly detector
        untrained_anomaly_detector = self.init_model()
        self.train_log_path = self.config.get_train_log_path()

        # initialize logger
        if self.config.is_log:
            self.config.save_params()
            logger = SummaryWriter(log_dir=self.train_log_path)
        else:
            logger = None

        # initialize trainer
        trainer = TrainerAnomalyDetector(
            log_path=self.train_log_path,
            logger=logger,
            )
        trainer.fit(untrained_anomaly_detector, self.train_loader)

    def test(self):
        trained_anomaly_detector = AnomalyDetector.load_from_pickle(get_model_pickle_path(self.train_log_path))

        test_log_path = self.config.get_test_log_path()
        trainer = TrainerAnomalyDetector(log_path=test_log_path)
        trainer.test(trained_anomaly_detector, self.test_loader)

    def log_hyperparameters(self):
        """
        Logs the topology model hypperparametrs
        """
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