import os, sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR) if ROOT_DIR not in sys.path else None

FDET_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, FDET_DIR) if FDET_DIR not in sys.path else None

# other imports
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import time
import numpy as np
import pickle
from torch.utils.tensorboard import SummaryWriter

# global imports
from data.transform import DomainTransformer, DataNormalizer
from feature_extraction.extractor import FrequencyFeatureExtractor, TimeFeatureExtractor, FeatureReducer

# local imports
from settings.manager import AnomalyDetectorTrainManager

class AnomalyDetector:
    def __init__(self, anom_config):
        self.anom_config = anom_config

        if anom_config['type'] == 'IF':
            self.model = IsolationForest(contamination=anom_config['contam'], 
                                         random_state=anom_config['seed'],
                                         n_jobs=anom_config['n_jobs'], 
                                         n_estimators=anom_config['n_estimators'],)
            
        elif anom_config['type'] == 'SVM':
            self.model = OneClassSVM(kernel=anom_config['kernel'],  
                                     gamma=anom_config['gamma'],
                                     nu=anom_config['nu'])
            
            
    def set_run_params(self, domain_config, data_stats=None, raw_data_norm=None, feat_norm=None, feat_configs=[], reduc_config=None):
        """
        Set the run parameters for the anomaly detection model

        Parameters
        ----------
        domain_config : str
            Domain configuration of the data (e.g., 'time', 'frequency')
        """
        self.domain = domain_config['type']

        self.domain_config = domain_config
        self.data_stats = data_stats
        self.raw_data_norm = raw_data_norm
        self.feat_norm = feat_norm
        self.feat_configs = feat_configs
        self.reduc_config = reduc_config

    def init_process_blocks(self):
        self.domain_transformer = DomainTransformer(domain_config=self.domain_config)
        self.raw_data_normalizer = DataNormalizer(norm_type=self.raw_data_norm, data_stats=self.data_stats) if self.raw_data_norm else None
        self.feat_normalizer = DataNormalizer(norm_type=self.feat_norm) if self.feat_norm else None

        # define feature objects
        if self.domain == 'time':
            self.time_fex = TimeFeatureExtractor(self.feat_configs) if self.feat_configs else None
        elif self.domain == 'freq':
            self.freq_fex = FrequencyFeatureExtractor(self.feat_configs) if self.feat_configs else None
        
        self.feat_reducer = FeatureReducer(reduc_config=self.reduc_config) if self.reduc_config else None

    def print_model_info(self):
        """
        Print the model information such as number of input features, support vectors, kernel type, etc.
        """
        print("Model type:", type(self.model).__name__)
        
        if self.anom_config['type'] == 'IF':
            print("Number of trees in the forest:", self.model.n_estimators)
        
        elif self.anom_config['type'] == 'SVM':
            print("Number of support vectors:", self.model.n_support_)
            print("Support vectors shape:", self.model.support_vectors_.shape)
            print("Kernel:", self.model.kernel)
            print("Gamma:", self.model.gamma)
            print("Nu:", self.model.nu)

    @staticmethod
    def load_from_pickle(model_path):
        """
        Load the anomaly detection model from a pickle file.

        Parameters
        ----------
        file_path : str
            Path to the pickle file containing the model.
        """
        if not os.path.exists(model_path):
            raise ValueError(f"\nThe model file does not exist at {model_path}")
        
        with open(model_path, 'rb') as f:
            return pickle.load(f)

    
class TrainerAnomalyDetector:
    def __init__(self, logger:SummaryWriter=None):
        self.logger = logger

    def process_input_data(self, anomaly_detector:AnomalyDetector, data_loader, get_data_shape=False):
        """
        Parameters
        ----------
        anomaly_detector : AnomalyDetector
            The anomaly detection model to be trained.
        loader : DataLoader
            DataLoader containing the training data.
            data : torch.Tensor, shape (batch_size, n_nodes, n_timesteps, n_dims)
                Input **time** data tensor containing the trajectory data

        Returns
        -------
        data_np : np.ndarray
            Numpy array of shape (batch_size * n_nodes, n_components * n_dims) ready for fitting.
            (shape meaning (total number of signals/samples, total number of features))
        label_np : np.ndarray
            Numpy array of shape (batch_size,) containing the labels for each sample.
        """
        data_list = []
        label_list = []

        anomaly_detector.init_process_blocks()

        for time_data, label in data_loader:
            # domain transform data (mandatory)
            if anomaly_detector.domain == 'time':
                data = anomaly_detector.domain_transformer.transform(time_data)
            elif anomaly_detector.domain == 'freq':
                data, freq_bins = anomaly_detector.domain_transformer.transform(time_data)

            # normalize raw data (optional)
            if anomaly_detector.raw_data_normalizer:
                if anomaly_detector.domain == 'time':
                    data = anomaly_detector.raw_data_normalizer.normalize(data)
                elif anomaly_detector.domain == 'freq':
                    print("\nFrequency data cannot be normalzied before feature extraction.")

            # extract features from data (optional)
            is_fex = False
            if anomaly_detector.domain == 'time':
                if anomaly_detector.time_fex:
                    data = anomaly_detector.time_fex.extract(data)
                    is_fex = True
            elif anomaly_detector.domain == 'freq':
                if anomaly_detector.freq_fex:
                    data = anomaly_detector.freq_fex.extract(data, freq_bins)
                    is_fex = True

            # normalize features (optional : if feat_norm is provided)
            if anomaly_detector.feat_normalizer:
                if is_fex:
                    data = anomaly_detector.feat_normalizer.normalize(data)
                else:
                    print("\nNo features extracted, so feature normalization is skipped.")

            # reduce features (optional : if reduc_config is provided)
            if anomaly_detector.feat_reducer:
                data = anomaly_detector.feat_reducer.reduce(data)
 
            # get data shape if required (used to get log path)
            if get_data_shape:
                n_comps = data.shape[2] 
                n_dims = data.shape[3]   
                return n_comps, n_dims

            # convert data to numpy array for fitting
            data_np = data.view(data.size(0)*data.size(1), -1).detach().numpy() # shape (batch size * n_nodes, n_components*n_dims)
            label_np = label.view(-1).numpy()  # shape (batch size,)

            # make labels optimized for sklearn models (convert label to 1 for normal data and -1 for anomalies)
            label_skl = np.where(label_np == 0, 1, -1)  # assuming 0 is normal and 1 is anomaly

            data_list.append(data_np)
            label_list.append(label_skl)

        return np.vstack(data_list), np.hstack(label_list)
    
    def fit(self, anomaly_detector:AnomalyDetector, train_loader):
        """
        Fit the anomaly detection model on the provided data.

        Parameters
        ----------
        
        """
        start_time = time.time()

        # process the input data
        train_data, labels = self.process_input_data(anomaly_detector, train_loader)

        # fit the model
        print("\nFitting anomaly detection model...")
        anomaly_detector.model.fit(train_data)

        # calculate and print the training time
        training_time = time.time() - start_time
        print(f"\nModel fitted successfully in {training_time:.2f} seconds")

        # training accuracy
        train_pred = anomaly_detector.model.predict(train_data)
        train_scores = anomaly_detector.model.decision_function(train_data)

        accuracy = np.mean(train_pred == labels)
        print(f"Training accuracy: {accuracy:.2f}")

        # save model
        if self.logger:
            self.logger.add_scalar("train/accuracy", accuracy)
            
            model_path = os.path.join(self.logger.log_dir, 'anomaly_detector.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(anomaly_detector, f)
        
            print(f"\nModel saved at {model_path}")


        # # model stats 
        # model_log = {}
        # model_log['metrics'] = {}
        # model_log['metrics']['training_time'] = training_time

        # if isinstance(anomaly_detector.model, IsolationForest):
        #     model_log['model_type'] = 'IsolationForest'
        #     model_log['metrics']['n_trees'] = anomaly_detector.model.n_estimators

        # elif isinstance(anomaly_detector.model, OneClassSVM):
        #     model_log['model_type'] = 'OneClassSVM'
        #     model_log['metrics']['n_in_features'] = anomaly_detector.model.n_features_in_
        #     model_log['metrics']['n_support_vectors'] = anomaly_detector.model.n_support_  

        # self.log_model(anomaly_detector, model_log)
   
    def predict(self, anomaly_detector:AnomalyDetector, predict_loader):
        """
        Predict anomalies in the provided data.

        Parameters
        ----------
        loader : DataLoader
            DataLoader containing the data to predict anomalies on.
            data : torch.Tensor, shape (batch_size, n_nodes, n_timesteps, n_dims)
                Input data tensor containing the trajectory data
        """
        # process the input data
        data = self.process_input_data(anomaly_detector, predict_loader)

        # predict anomalies
        y_pred = anomaly_detector.model.predict(data)
        scores = anomaly_detector.model.decision_function(data)

        return y_pred, scores
    
    def test(self, anomaly_detector:AnomalyDetector, test_loader):
        """
        Along with prediction, this method calcualtes model accuracy, precision, recall, and F1 score.
        """
        # process the input data
        data, labels = self.process_input_data(anomaly_detector, test_loader)

        # predict anomalies
        test_pred = anomaly_detector.model.predict(data)
        scores = anomaly_detector.model.decision_function(data)

        # calculate accuracy
        accuracy = np.mean(test_pred == labels)
        print(f"Test accuracy: {accuracy:.2f}")

        if self.logger:
            self.logger.add_scalar("test/accuracy", accuracy)
                
    
    def log_model(self, anomaly_detector, model_log):
        """
        Log the model information such as number of input features, support vectors, kernel type, etc.
        """
        if self.logger:
            # for key, value in model_log['metrics'].items():
            #     self.logger.add_scalar(f"{model_log['model_type']}/{key}", value)

            # # save model
            # if not os.path.exists(self.log_path):
            #     os.makedirs(self.log_path)

            model_path = os.path.join(self.log_path, 'anomaly_detector.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(anomaly_detector, f)
        
            print(f"\nModel saved at {model_path}")

        

        