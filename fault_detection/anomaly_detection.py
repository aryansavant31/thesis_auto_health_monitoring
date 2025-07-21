from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from data.transform import DataTransformer
from feature_extraction import FeatureExtractor
import time
import numpy as np
import pickle
import os

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
            
            
    def set_run_params(self, data_stats, domain='time', norm_type=None, fex_configs=[]):
        """
        Set the run parameters for the anomaly detection model

        Parameters
        ----------
        domain : str
            Domain of the data (e.g., 'time', 'frequency')
        norm_type : str
            Normalization type (e.g., 'std', 'minmax')
        fex_configs : list
            List of feature extraction configurations.
        """
        self.fex_configs = fex_configs

        self.transformer = DataTransformer(domain=domain, norm_type=norm_type, data_stats=data_stats)
        self.feature_extractor = FeatureExtractor(fex_configs=fex_configs)

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
    def __init__(self, log_path, logger=None):
        self.log_path = log_path
        self.logger = logger

    def process_input_data(self, anomaly_detector, loader):
        """
        Transform the data
            - domain change
            - normalization
        Feature extraction
        Conversion from tensor to numpy array for fitting
        """
        data_list = []
        for data, _ in loader:
            # transform data
            data = anomaly_detector.transformer(data)

            # extract features from data if fex_configs are provided
            if anomaly_detector.fex_configs:
                data = anomaly_detector.feature_extractor(data)

            # convert data to numpy array for fitting
            data_np = data.view(data.size(0), -1).detach().numpy() # shape (batch size, n_nodes*n_components*n_dims)
            data_list.append(data_np)

        return np.vstack(data_list)
    
    def fit(self, anomaly_detector, loader):
        """
        Fit the anomaly detection model on the provided data.

        Parameters
        ----------
        trainloader : DataLoader
            DataLoader containing the training data.
            data : torch.Tensor, shape (batch_size, n_nodes, n_timesteps, n_dims)
                Input data tensor containing the trajectory data
        """
        start_time = time.time()

        # process the input data
        data = self.process_input_data(anomaly_detector, loader)

        # fit the model
        print("\nFitting anomaly detection model...")
        anomaly_detector.model.fit(data)

        # calculate and print the training time
        training_time = time.time() - start_time
        print(f"\nModel fitted successfully in {training_time:.2f} seconds")

        # model stats 
        model_log = {}
        model_log['metrics'] = {}
        model_log['metrics']['training_time'] = training_time

        if isinstance(anomaly_detector.model, IsolationForest):
            model_log['model_type'] = 'IsolationForest'
            model_log['metrics']['n_trees'] = anomaly_detector.model.n_estimators

        elif isinstance(anomaly_detector.model, OneClassSVM):
            model_log['model_type'] = 'OneClassSVM'
            model_log['metrics']['n_in_features'] = anomaly_detector.model.n_features_in_
            model_log['metrics']['n_support_vectors'] = anomaly_detector.model.n_support_  

        self.log_model(anomaly_detector, model_log)
   
    def predict(self, anomaly_detector, loader):
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
        data = self.process_input_data(anomaly_detector, loader)

        # predict anomalies
        y_pred = anomaly_detector.model.predict(data)
        scores = anomaly_detector.model.decision_function(data)

        return y_pred, scores
    
    def test(self, anomaly_detector, loader):
        """
        Along with prediction, this method calcualtes model accuracy, precision, recall, and F1 score.
        """
        # process the input data
        data = self.process_input_data(anomaly_detector, loader)

        # predict anomalies
        y_pred = anomaly_detector.model.predict(data)
        scores = anomaly_detector.model.decision_function(data)
        
    
    def log_model(self, anomaly_detector, model_log):
        """
        Log the model information such as number of input features, support vectors, kernel type, etc.
        """
        if self.logger:
            for key, value in model_log['metrics'].items():
                self.logger.add_scalar(f"{model_log['model_type']}/{key}", value)

            # save model
            if not os.path.exists(self.log_path):
                os.makedirs(self.log_path)

            model_path = os.path.join(self.log_path, 'anomaly_detector.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(anomaly_detector, f)
        
            print(f"\nModel saved at {model_path}")

        

        