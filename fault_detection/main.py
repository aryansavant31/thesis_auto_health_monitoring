from anomaly_detection import AnomalyDetector, TrainerAnomalyDetector
from config import TrainAnomalyDetectornConfig, PredictAnomalyDetectorConfig, get_model_pickle_path
from data.config import DataConfig
from data.load import load_spring_particle_data
from data.transform import DataTransformer
from feature_extraction import FeatureExtractor
from torch.utils.tensorboard import SummaryWriter

class PredictAnomalyDetectorMain:
    pass

class TrainAnomalyDetectorMain:
    def __init__(self):
        self.data_config = DataConfig()
        self.fdet_config = TrainAnomalyDetectornConfig()

        # load data
        self.load_data()

    def load_data(self):
        # set train data parameters
        self.data_config.set_train_dataset()
        # get dataset paths
        node_ds_paths, edge_ds_paths = self.data_config.get_dataset_paths()
        # load data
        self.train_loader, _, self.test_loader, self.data_stats = load_spring_particle_data(node_ds_paths, edge_ds_paths, self.fdet_config.batch_size)

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
        data = DataTransformer(domain=self.fdet_config.domain, 
                                        norm_type=self.fdet_config.norm_type, 
                                        data_stats=self.data_stats)(data)

        # extract features from data if fex_configs are provided
        if self.fdet_config.fex_configs:
            data = FeatureExtractor(fex_configs=self.fdet_config.fex_configs)(data)

        return data
    
    def init_model(self):
        anomaly_detector = AnomalyDetector(anom_config=self.fdet_config.anom_config)
        anomaly_detector.set_run_params(data_stats=self.data_stats,
                                        domain=self.fdet_config.domain, 
                                        norm_type=self.fdet_config.norm_type, 
                                        fex_configs=self.fdet_config.fex_configs)
        # print model info
        print("Anomaly Detector Model Initialized with the following configurations:")
        anomaly_detector.print_model_info()

        return anomaly_detector
    
    def train(self):
        """
        Train the anomaly detection model.
        """
        # initialize anomaly detector
        untrained_anomaly_detector = self.init_model()
        self.train_log_path = self.fdet_config.get_train_log_path()

        # initialize logger
        if self.fdet_config.is_log:
            self.fdet_config.save_params()
            logger = SummaryWriter(log_dir=self.train_log_path)
        else:
            logger = None

        # initialize trainer
        trainer = TrainerAnomalyDetector(log_path=self.train_log_path)
        trainer.fit(untrained_anomaly_detector, self.train_loader)

    def test(self):
        trained_anomaly_detector = AnomalyDetector.load_from_pickle(get_model_pickle_path(self.train_log_path))

        test_log_path = self.fdet_config.get_test_log_path()
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