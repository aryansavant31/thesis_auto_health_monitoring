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
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve

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
        data_stats : dict, optional
            Statistics of the data (mean, std, etc.) for normalization
        raw_data_norm : str, optional
            Normalization type for raw data (e.g., 'min_max', 'standard')
        """
        self._domain_config = domain_config
        self._raw_data_norm = raw_data_norm
        self._feat_norm = feat_norm
        self._feat_configs = feat_configs
        self._reduc_config = reduc_config

        self._domain = domain_config['type']
        self._feat_names = self._get_feature_names() if self._feat_configs else None

        self.data_stats = data_stats

    def _get_feature_names(self):
        """
        Get the names of the features that will be used in the anomaly detection model.
        """
        non_rank_feats = [feat_config['type'] for feat_config in self._feat_configs if feat_config['type'] != 'from_ranks']
        rank_feats = next((feat_config['feat_list'] for feat_config in self._feat_configs if feat_config['type'] == 'from_ranks'), [])

        return non_rank_feats + rank_feats
            
    def init_input_processors(self):
        self.domain_transformer = DomainTransformer(domain_config=self._domain_config)
        self.raw_data_normalizer = DataNormalizer(norm_type=self._raw_data_norm, data_stats=self.data_stats) if self._raw_data_norm else None
        self.feat_normalizer = DataNormalizer(norm_type=self._feat_norm) if self._feat_norm else None

        # define feature objects
        if self._domain == 'time':
            self.time_fex = TimeFeatureExtractor(self._feat_configs) if self._feat_configs else None
        elif self._domain == 'freq':
            self.freq_fex = FrequencyFeatureExtractor(self._feat_configs) if self._feat_configs else None
        
        self.feat_reducer = FeatureReducer(reduc_config=self._reduc_config) if self._reduc_config else None

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
    def __init__(self, logger:SummaryWriter=None, hparams=None):
        self.logger = logger
        self.hparams = hparams

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

        anomaly_detector.init_input_processors()

        for time_data, label in data_loader:
            # domain transform data (mandatory)
            if anomaly_detector._domain == 'time':
                data = anomaly_detector.domain_transformer.transform(time_data)
            elif anomaly_detector._domain == 'freq':
                data, freq_bins = anomaly_detector.domain_transformer.transform(time_data)

            # normalize raw data (optional)
            if anomaly_detector.raw_data_normalizer:
                if anomaly_detector._domain == 'time':
                    data = anomaly_detector.raw_data_normalizer.normalize(data)
                elif anomaly_detector._domain == 'freq':
                    print("\nFrequency data cannot be normalzied before feature extraction.")

            # extract features from data (optional)
            is_fex = False
            if anomaly_detector._domain == 'time':
                if anomaly_detector.time_fex:
                    data = anomaly_detector.time_fex.extract(data)
                    is_fex = True
            elif anomaly_detector._domain == 'freq':
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
 
            n_comps = data.shape[2] 
            n_dims = data.shape[3]
            # get data shape if required (used to get log path)
            if get_data_shape:   
                return n_comps, n_dims

            # convert data to numpy array for fitting
            data_np = data.view(data.size(0)*data.size(1), data.size(2)*data.size(3)).detach().numpy() # shape (batch size * n_nodes, n_components*n_dims)
            label_np = label.view(-1).numpy()  # shape (batch size,) (0 for OK, 1 for NOK, -1 for UK)

            # # make labels optimized for sklearn models (convert label to 1 for normal data and -1 for anomalies)
            # label_skl = np.where(label_np == 0, 1, -1)  # assuming 0 is normal and 1 is anomaly

            data_list.append(data_np)
            label_list.append(label_np)

        data_np_all, label_np_all = np.vstack(data_list), np.hstack(label_list)

        # convert np data into pd dataframe
        if anomaly_detector._feat_names and anomaly_detector.feat_reducer is None:
            self.comp_cols = [f"{feat}_{dim}" for feat in anomaly_detector._feat_names for dim in range(n_dims)]
        else:
            self.comp_cols = [f"comp{comp}_dim{dim}" for comp in range(n_comps) for dim in range(n_dims)]

        df = pd.DataFrame(data_np_all, columns=self.comp_cols)
        df['given_label'] = label_np_all

        return df
    
    def fit(self, anomaly_detector:AnomalyDetector, train_loader):
        """
        Fit the anomaly detection model on the provided data.

        Parameters
        ----------
        anomaly_detector : AnomalyDetector
            The anomaly detection model to be trained.
        train_loader : DataLoader
            DataLoader containing the training data.
            data : torch.Tensor, shape (batch_size, n_nodes, n_timesteps, n_dims)
                Input data tensor containing the trajectory data
        """
        start_time = time.time()

        # process the input data
        self.df = self.process_input_data(anomaly_detector, train_loader)

        # fit the model
        print("\nFitting anomaly detection model...")
        anomaly_detector.model.fit(self.df[self.comp_cols])

        # calculate and print the training time
        training_time = time.time() - start_time
        print(f"\nModel fitted successfully in {training_time:.2f} seconds")

        # training accuracy and scores
        self.df['scores'] = anomaly_detector.model.decision_function(self.df[self.comp_cols])
        self.df['pred_label'] = anomaly_detector.model.predict(self.df[self.comp_cols])

        # preprocess pred label to match the given label notations
        self.df['pred_label'] = np.where(self.df['pred_label'] == -1, 1, 0)  # convert -1 to 1 (anomaly) and 1 to 0 (normal)
        valid_rows = self.df['given_label'] != -1

        # filter out rows where given_label is -1 (unknown) - not needed for accuracy calculation
        filtered_df = self.df[valid_rows]

        accuracy = np.mean(filtered_df['pred_label'] == filtered_df['given_label'])
        print(f"Training accuracy: {accuracy:.2f}")

        anomaly_detector.anom_config['train_accuracy'] = accuracy
        # save model
        if self.logger:
            self.logger.add_scalar("train/accuracy", accuracy)
            self.logger.add_hparams({**anomaly_detector.anom_config, **self.hparams}, {})
            model_path = os.path.join(self.logger.log_dir, 'anomaly_detector.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(anomaly_detector, f)
        
            print(f"\nModel saved at {model_path}")

        self.model_type = type(anomaly_detector.model).__name__
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

    def pair_plot(self, feat_cols=None):
        """
        Create a pair plot of the features.
        """
        if not hasattr(self, 'df'):
            raise ValueError("DataFrame is not available. Please train, test or predict the model first.")
        feat_cols = self.comp_cols if feat_cols is None else feat_cols

        palette = ['#ff7f0e', '#1f77b4']
        pair_plot = sns.pairplot(self.df, vars=feat_cols, hue='pred_label', palette=palette)
        pair_plot.figure.suptitle("Pair Plot of Features", y=1.02)
        plt.show()

        # save the pair plot if logger is available
        if self.logger:
            fig = pair_plot.figure
            fig.savefig(os.path.join(self.logger.log_dir, 'pair_plot.png'), dpi=500)
            self.logger.add_figure(f"{self.model_type}/pair_plot", fig, close=True)
            print(f"\nPair plot logged at {self.logger.log_dir}")

    def confusion_matrix(self):
        """
        Create a confusion matrix of the predictions.
        """
        if not hasattr(self, 'df'):
            raise ValueError("DataFrame is not available. Please train, test or predict the model first.")

        x_label = ['OK (prediction)', 'NOK (prediction)']
        y_label = ['OK (truth)', 'NOK (truth)']

        cm = pd.crosstab(
            self.df['given_label'], self.df['pred_label'], 
            rownames=['Actual'], colnames=['Predicted'], dropna=False
            ).reindex(index=[1, -1], columns=[1, -1], fill_value=0)
        
        plt.figure(figsize=(8, 6))
        cm_plot = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=x_label, yticklabels=y_label)

        # add tp, fp, tn, fn labels
        cell_labels = [['TN', 'FP'],
                        ['FN', 'TP']]

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                cm_plot.text(j + 0.5, i + 0.3, cell_labels[i][j],
                        ha='center', va='center', color='grey', fontsize=10)
                
        plt.title('Confusion Matrix')
        plt.show()

        # save the confusion matrix if logger is available
        if self.logger:
            fig = cm_plot.get_figure()
            fig.savefig(os.path.join(self.logger.log_dir, 'confusion_matrix.png'), dpi=500)
            self.logger.add_figure(f"{self.model_type}/confusion_matrix", fig, close=True)
            print(f"\nConfusion matrix logged at {self.logger.log_dir}")

    def roc_curve(self):
        """
        Create a ROC curve of the predictions.
        """
        if not hasattr(self, 'df'):
            raise ValueError("DataFrame is not available. Please train, test or predict the model first.")

        # calculate the ROC curve
        fpr, tpr, thresholds = roc_curve(self.df['given_label'], self.df['scores'], pos_label=-1)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', label='ROC curve')
        plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random guess')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.show()
        
        
        
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

        

        