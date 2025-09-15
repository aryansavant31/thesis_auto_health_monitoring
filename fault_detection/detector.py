import os, sys
# ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.insert(0, ROOT_DIR) if ROOT_DIR not in sys.path else None

# FDET_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.insert(0, FDET_DIR) if FDET_DIR not in sys.path else None

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
import torch

# global imports
from data.transform import DomainTransformer, DataNormalizer
from feature_extraction.extractor import FrequencyFeatureExtractor, TimeFeatureExtractor, FeatureReducer

# local imports
from .settings.manager import AnomalyDetectorTrainManager

class AnomalyDetector:
    def __init__(self, anom_config, hparams):
        self.anom_config = anom_config
        self.hparams = hparams
        self.ok_percentage = 1
        self.nok_percentage = 1

        self.ok_lower_bound = None
        self.nok_upper_bound = None

        if anom_config['anom_type'] == 'IF':
            self.model = IsolationForest(contamination=anom_config['IF/contam'], 
                                         random_state=anom_config['IF/seed'],
                                         n_jobs=anom_config['IF/n_jobs'], 
                                         n_estimators=anom_config['IF/n_estimators'],
                                         verbose=anom_config['IF/verbose'],
                                         max_samples=anom_config['IF/max_samples'],
                                         bootstrap=anom_config['IF/bootstrap'],
                                         warm_start=anom_config['IF/warm_start'],
                                         max_features=anom_config['IF/max_features'])
            
        elif anom_config['anom_type'] == '1SVM':
            self.model = OneClassSVM(kernel=anom_config['1SVM/kernel'],  
                                     gamma=anom_config['1SVM/gamma'],
                                     nu=anom_config['1SVM/nu'],
                                     shrinking=anom_config['1SVM/is_shrinking'],
                                     tol=anom_config['1SVM/tol'],
                                     cache_size=anom_config['1SVM/cache_size'],
                                     verbose=anom_config['1SVM/is_verbose'],
                                     max_iter=anom_config['1SVM/max_iter'],
                                     coef0=anom_config['1SVM/coef0'],
                                     degree=anom_config['1SVM/degree'])
            
    def set_run_params(self, data_config, domain_config, data_stats=None, raw_data_norm=None, feat_norm=None, feat_configs=[], reduc_config=None):
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
        self._data_config = data_config
        self._raw_data_norm = raw_data_norm
        self._feat_norm = feat_norm
        self._feat_configs = feat_configs
        self._reduc_config = reduc_config

        
        self._feat_names = self._get_feature_names() if self._feat_configs else None

        self.data_stats = data_stats
        self.domain_config = domain_config

    def _get_feature_names(self):
        """
        Get the names of the features that will be used in the anomaly detection model.
        """
        # non_rank_feats = [feat_config['type'] for feat_config in self._feat_configs if feat_config['type'] not in ['from_ranks', 'first_n_modes', 'full_spectrum']]
        # first_n_freq_feats = [f"freq{freq_bin+1}" for feat_config in self._feat_configs if feat_config['type'] == 'first_n_modes' for freq_bin in range(feat_config['n_modes'])] 
        # first_n_modes_feats = [f"mode{mode+1}" for feat_config in self._feat_configs if feat_config['type'] == 'first_n_modes' for mode in range(feat_config['n_modes'])]

        # rank_feats = next((feat_config['feat_list'] for feat_config in self._feat_configs if feat_config['type'] == 'from_ranks'), [])

        # Order the feature names according to the order in self._feat_configs
        feat_names = []

        for feat_config in self._feat_configs:
            if feat_config['type'] not in ['from_ranks', 'first_n_modes', 'full_spectrum']:
                feat_names.append(feat_config['type'])

            elif feat_config['type'] == 'first_n_modes':
                feat_names.extend([f"freq{freq_bin+1}" for freq_bin in range(feat_config['n_modes'])])
                feat_names.extend([f"mode{mode+1}" for mode in range(feat_config['n_modes'])])

            elif feat_config['type'] == 'from_ranks':
                feat_names.extend(feat_config['feat_list'])

        return feat_names
            
    def init_input_processors(self, is_verbose=True):
        print(f"\nInitializing input processors for anomaly detection model...") if is_verbose else None
        
        self._domain = self.domain_config['type']

        domain_str = self._get_config_str([self.domain_config])
        feat_str = self._get_config_str(self._feat_configs) if self._feat_configs else 'None'
        reduc_str = self._get_config_str([self._reduc_config]) if self._reduc_config else 'None'

        self.domain_transformer = DomainTransformer(domain_config=self.domain_config, data_config=self._data_config)
        # if self._domain == 'time':
        print(f"\n>> Domain transformer initialized: {domain_str}") if is_verbose else None
        # elif self._domain == 'freq':
        #     print(f"\n>> Domain transformer initialized for 'frequency' domain") if is_verbose else None

        # initialize data normalizers
        if self._raw_data_norm:
            self.raw_data_normalizer = DataNormalizer(norm_type=self._raw_data_norm, data_stats=self.data_stats)
            print(f"\n>> Raw data normalizer initialized with '{self._raw_data_norm}' normalization") if is_verbose else None
        else:
            self.raw_data_normalizer = None
            print("\n>> No raw data normalization is applied") if is_verbose else None

        if self._feat_norm:
            self.feat_normalizer = DataNormalizer(norm_type=self._feat_norm)
            print(f"\n>> Feature normalizer initialized with '{self._feat_norm}' normalization") if is_verbose else None
        else:
            self.feat_normalizer = None
            print("\n>> No feature normalization is applied") if is_verbose else None

        # define feature objects
        if self._domain == 'time':
            if self._feat_configs:
                self.time_fex = TimeFeatureExtractor(self._feat_configs)
                print(f"\n>> Time feature extractor initialized with features: {feat_str}") if is_verbose else None
            else:
                self.time_fex = None
                print("\n>> No time feature extraction is applied") if is_verbose else None

        elif self._domain == 'freq':
            if self._feat_configs:
                self.freq_fex = FrequencyFeatureExtractor(self._feat_configs, data_config=self._data_config)
                print(f"\n>> Frequency feature extractor initialized with features: {feat_str}") if is_verbose else None
            else:
                self.freq_fex = None
                print("\n>> No frequency feature extraction is applied") if is_verbose else None
        
        # define feature reducer
        if self._reduc_config:
            self.feat_reducer = FeatureReducer(reduc_config=self._reduc_config)
            print(f"\n>> Feature reducer initialized with '{reduc_str}' reduction") if is_verbose else None
        else:
            self.feat_reducer = None
            print("\n>> No feature reduction is applied") if is_verbose else None

        print('\n' + 75*'-') if is_verbose else None

    def _get_config_str(self, configs:list):
        """
        Get a neat string that has the type of config and its parameters.
        Eg: "PCA(comps=3)"
        """
        config_strings = []

        for config in configs:
            additional_keys = ', '.join([f"{key}={value}" for key, value in config.items() if key not in ['fs', 'type', 'feat_list']])
            if additional_keys:
                config_strings.append(f"{config['type']}({additional_keys})")
            else:
                config_strings.append(f"{config['type']}")

        return ', '.join(config_strings)

    def print_model_info(self):
        """
        Print the model information such as number of input features, support vectors, kernel type, etc.
        """
        print("Model type:", type(self.model).__name__)
        
        if self.anom_config['anom_type'] == 'IF':
            print("Number of trees in the forest:", self.model.n_estimators)
            print("Contamination:", self.model.contamination)
        
        elif self.anom_config['anom_type'] == '1SVM':
            # print("Number of support vectors:", self.model.n_support_)
            # print("Support vectors shape:", self.model.support_vectors_.shape)
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
        rep_num_list = []

        anomaly_detector.init_input_processors(is_verbose = not get_data_shape)

        for idx, (time_data, label, rep_num) in enumerate(data_loader):
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
                    print("\nFrequency data cannot be normalized before feature extraction, hence skipping raw data normalization.") if not get_data_shape and idx == 0 else None

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
                    print("\nNo features extracted, so feature normalization is skipped.") if not get_data_shape and idx == 0 else None

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

            # match n_samples of labels with data
            label_np = np.repeat(label.view(-1).numpy(), data.size(1))  # shape (batch size*n_nodes,) (0 for OK, 1 for NOK, -1 for UK)

            # match n_samples of rep_num with data
            rep_num = np.repeat(rep_num.view(-1).numpy(), data.size(1))
            
            # # make labels optimized for sklearn models (convert label to 1 for normal data and -1 for anomalies)
            # label_skl = np.where(label_np == 0, 1, -1)  # assuming 0 is normal and 1 is anomaly

            data_list.append(data_np)
            label_list.append(label_np)
            rep_num_list.append(rep_num)

        data_np_all, label_np_all, rep_num_np_all = np.vstack(data_list), np.hstack(label_list), np.hstack(rep_num_list)

        # add datashape to hparams
        anomaly_detector.hparams['n_comps'] = str(int(n_comps))
        anomaly_detector.hparams['n_dims'] = str(int(n_dims))
        anomaly_detector.hparams['n_comps_total'] = str(int(n_comps*n_dims))

        # convert np data into pd dataframe
        if anomaly_detector._feat_names and anomaly_detector.feat_reducer is None:
            self.comp_cols = [f"{feat}_{dim}" for feat in anomaly_detector._feat_names for dim in range(n_dims)]

            if len(self.comp_cols) != n_comps * n_dims:
                n_remaining_feats = n_comps * n_dims - len(self.comp_cols)
                self.comp_cols.extend([f"ext_feat{idx+1}_dim{dim}" for idx in range(n_remaining_feats) for dim in range(n_dims)])

            print(f"\nFeature names: {self.comp_cols}")

        else:
            self.comp_cols = [f"comp{comp}_dim{dim}" for comp in range(n_comps) for dim in range(n_dims)]
            if anomaly_detector.feat_reducer:
                print(f"\nReduced feature names: {self.comp_cols}")
            else:
                print(f"\nUsing components as features: [{', '.join(self.comp_cols[:4])}...{', '.join(self.comp_cols[-4:])}]")

        df = pd.DataFrame(data_np_all, columns=self.comp_cols)
        df['given_label'] = label_np_all
        df['rep_num'] = rep_num_np_all

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

    # 1. Process the input data
        self.df = self.process_input_data(anomaly_detector, train_loader)

    # 2. Train model

        # fit the model
        print("\nFitting anomaly detection model...")
        anomaly_detector.model.fit(self.df[self.comp_cols])

        # calculate and print the training time
        training_time = time.time() - start_time
        print(f"\nModel fitted successfully in {training_time:.2f} seconds")

        start_time = time.time()
        # training accuracy and scores
        self.df['scores'] = anomaly_detector.model.decision_function(self.df[self.comp_cols])
        self.df['pred_label'] = anomaly_detector.model.predict(self.df[self.comp_cols])

        infer_time = time.time() - start_time
        print(f"\nTraining inference completed in {infer_time:.2f} seconds")

        # # preprocess pred label to match the given label notations
        # self.df['pred_label'] = np.where(self.df['pred_label'] == -1, 1, 0)  # convert -1 to 1 (anomaly) and 1 to 0 (normal)
        valid_rows = self.df['given_label'] != 0

        # filter out rows where given_label is -1 (unknown) - not needed for accuracy calculation
        filtered_df = self.df[valid_rows]

        accuracy = np.mean(filtered_df['pred_label'] == filtered_df['given_label'])

        print(f"\nTraining accuracy: {accuracy:.2f}")

        # calculate ok_lower_bound
        scores_pred_ok = self.df[self.df['pred_label'] == 1]['scores']
        anomaly_detector.ok_lower_bound = np.percentile(scores_pred_ok, (1 - anomaly_detector.ok_percentage) * 100) if len(scores_pred_ok) > 0 else None
        
        # calculate nok_upper_bound
        scores_pred_nok = self.df[self.df['pred_label'] == -1]['scores']
        anomaly_detector.nok_upper_bound = np.percentile(scores_pred_nok, anomaly_detector.nok_percentage * 100) if len(scores_pred_nok) > 0 else None

        # assign is_final_pred
        self.df['final_pred_label'] = self.df.apply(self._assign_final_pred, axis=1, args=(anomaly_detector,))

        self.ok_percentage = anomaly_detector.ok_percentage
        self.nok_percentage = anomaly_detector.nok_percentage
        self.ok_lower_bound = anomaly_detector.ok_lower_bound
        self.nok_upper_bound = anomaly_detector.nok_upper_bound

        print(f"\nOK lower bound (at {self.ok_percentage*100:.1f} percentile): {self.ok_lower_bound:.4f}" if anomaly_detector.ok_lower_bound is not None else "\nOK lower bound could not be determined as there are no samples predicted as OK.")
        print(f"\nNOK upper bound (at {self.nok_percentage*100:.1f} percentile): {self.nok_upper_bound:.4f}" if anomaly_detector.nok_upper_bound is not None else "\nNOK upper bound could not be determined as there are no samples predicted as NOK.")

        print(f"\nDataframe is as follows:")
        print(self.df)

    # 3. Log model information
        self.model_type = type(anomaly_detector.model).__name__
        self.model_id = os.path.basename(self.logger.log_dir) if self.logger else self.model_type
        self.run_type = 'train'
        self.tb_tag = self.model_id.split('-')[0].strip('[]').replace('_(', "  (").replace('+', " + ") if self.logger else self.model_type

        # update hparams
        anomaly_detector.hparams['train_accuracy'] = accuracy
        anomaly_detector.hparams['model_id'] = self.model_id
        anomaly_detector.hparams['training_time'] = training_time
        anomaly_detector.hparams['model_num'] = int(self.logger.log_dir.split('_')[-1]) if self.logger else 0
        anomaly_detector.hparams['db_delta_ok/train'] = anomaly_detector.ok_lower_bound

        if self.logger:
            self.logger.add_scalar(f"{self.tb_tag}/train_accuracy", accuracy)

            print(f"\nTraining hyperparameters logged for tensorboard at {self.logger.log_dir}")

            # save model
            model_path = os.path.join(self.logger.log_dir, 'anomaly_detector.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(anomaly_detector, f)

            # save dataframe
            df_path = os.path.join(self.logger.log_dir, f'dataframe_{self.run_type}.pkl')
            self.df.to_pickle(df_path)  
        
            print(f"\nModel saved at {model_path}")
        else:
            print("\nTraining hyperparameters not logged as logging is disabled.")
            print("Model not saved as logging is disabled. Please enable logging to save the model.")

        print('\n' + 75*'-')

        return anomaly_detector
        
    def predict(self, anomaly_detector:AnomalyDetector, predict_loader):
        """
        Predict anomalies in the provided data.

        Parameters
        ----------
        anomaly_detector : AnomalyDetector
            The anomaly detection model to be used for prediction.
        loader : DataLoader
            DataLoader containing the data to predict anomalies on.
            data : torch.Tensor, shape (batch_size, n_nodes, n_timesteps, n_dims)
                Input data tensor containing the trajectory data

        Returns
        -------
        dict
            A dictionary containing:
            - `pred_labels`: torch.Tensor of shape (n_samples,) containing predicted classes (0 for normal, 1 for anomaly)
            - `scores`: torch.Tensor of shape (n_samples,) containing anomaly scores for each sample
            - `reps`: torch.Tensor of shape (n_samples,) containing rep numbers for each sample
        
        """
        start_time = time.time()
        print("\nPredicting anomalies using the trained model...")

    # 1. Process the input data
        self.df = self.process_input_data(anomaly_detector, predict_loader)

    # 2. Predict anomalies
        # scores
        self.df['scores'] = anomaly_detector.model.decision_function(self.df[self.comp_cols])
        self.df['pred_label'] = anomaly_detector.model.predict(self.df[self.comp_cols])

        infer_time = time.time() - start_time
        print(f"\nPrediction completed in {infer_time:.2f} seconds")
        
        # # preprocess pred label to match the given label notations
        # self.df['pred_label'] = np.where(self.df['pred_label'] == -1, 1, 0)  # convert -1 to 1 (anomaly) and 1 to 0 (normal)

        # calculate nok_upper_bound
        scores_pred_nok = self.df[self.df['pred_label'] == -1]['scores']
        anomaly_detector.nok_upper_bound = np.percentile(scores_pred_nok, anomaly_detector.nok_percentage * 100) if len(scores_pred_nok) > 0 else None
        print(f"\nNOK upper bound (at {anomaly_detector.nok_percentage*100:.1f} percentile): {anomaly_detector.nok_upper_bound:.4f}" if anomaly_detector.nok_upper_bound is not None else "\nNOK upper bound could not be determined as there are no samples predicted as NOK.")

        # assign is_final_pred
        self.df['final_pred_label'] = self.df.apply(self._assign_final_pred, axis=1, args=(anomaly_detector,))

        self.ok_percentage = anomaly_detector.ok_percentage
        self.nok_percentage = anomaly_detector.nok_percentage
        self.ok_lower_bound = anomaly_detector.ok_lower_bound
        self.nok_upper_bound = anomaly_detector.nok_upper_bound

        print(f"\nOK lower bound (from training): {self.ok_lower_bound:.4f}" if self.ok_lower_bound is not None else "\nOK lower bound could not be determined as there are no samples predicted as OK during training.")
        print(f"\nNOK upper bound (from testing): {self.nok_upper_bound:.4f}" if self.nok_upper_bound is not None else "\nNOK upper bound could not be determined as there are no samples predicted as NOK during testing.")

        print(f"\nDataframe is as follows:")
        print(self.df)

        # convert predictions to tensor
        pred_labels = torch.tensor(self.df['pred_label'].values, dtype=torch.int64)
        scores = torch.tensor(self.df['scores'].values, dtype=torch.float32)
        rep_nums = torch.tensor(self.df['rep_num'].values, dtype=torch.float32)

        print(f"\nPredictions: {pred_labels}")

    # 3. Log model information

        self.model_type = type(anomaly_detector.model).__name__
        self.model_id = anomaly_detector.hparams.get('model_id', 'unknown_model')
        self.run_type = os.path.basename(self.logger.log_dir) if self.logger else 'predict'
        self.tb_tag = self.model_id.split('-')[0].strip('[]').replace('_(', "  (").replace('+', " + ") if self.logger else self.model_type

        if self.logger:
            # save dataframe
            df_path = os.path.join(self.logger.log_dir, f'dataframe_{self.run_type}.pkl')
            self.df.to_pickle(df_path)  

        print('\n' + 75*'-')

        return {'pred_labels': pred_labels, 
                'scores': scores,
                'reps': rep_nums}
    
    def test(self, anomaly_detector:AnomalyDetector, test_loader):
        """
        Test the anomaly detection model on the provided data.
        """
        start_time = time.time()
        print("\nTesting anomaly detection model...")

    # 1. Process the input data
        self.df = self.process_input_data(anomaly_detector, test_loader)

    # 2. Predict anomalies
        # scores
        self.df['scores'] = anomaly_detector.model.decision_function(self.df[self.comp_cols])
        self.df['pred_label'] = anomaly_detector.model.predict(self.df[self.comp_cols])

        infer_time = time.time() - start_time
        print(f"\nTest inference completed in {infer_time:.2f} seconds")

        # # preprocess pred label to match the given label notations
        # self.df['pred_label'] = np.where(self.df['pred_label'] == -1, 1, 0)  # convert -1 to 1 (anomaly) and 1 to 0 (normal)
        valid_rows = self.df['given_label'] != 0

        # filter out rows where given_label is -1 (unknown) - not needed for accuracy calculation
        filtered_df = self.df[valid_rows]

        # calculate test accuracy
        accuracy = np.mean(filtered_df['pred_label'] == filtered_df['given_label'])

        # calculate nok_upper_bound
        scores_pred_nok = self.df[self.df['pred_label'] == -1]['scores']
        anomaly_detector.nok_upper_bound = np.percentile(scores_pred_nok, anomaly_detector.nok_percentage * 100) if len(scores_pred_nok) > 0 else None
        print(f"\nNOK upper bound (at {anomaly_detector.nok_percentage*100:.1f} percentile): {anomaly_detector.nok_upper_bound:.4f}" if anomaly_detector.nok_upper_bound is not None else "\nNOK upper bound could not be determined as there are no samples predicted as NOK.")

        # assign is_final_pred
        self.df['final_pred_label'] = self.df.apply(self._assign_final_pred, axis=1, args=(anomaly_detector,))

        self.ok_percentage = anomaly_detector.ok_percentage
        self.nok_percentage = anomaly_detector.nok_percentage
        self.ok_lower_bound = anomaly_detector.ok_lower_bound
        self.nok_upper_bound = anomaly_detector.nok_upper_bound

        print(f"\nOK lower bound (from training): {self.ok_lower_bound:.4f}" if self.ok_lower_bound is not None else "\nOK lower bound could not be determined as there are no samples predicted as OK during training.")
        print(f"\nNOK upper bound (from testing): {self.nok_upper_bound:.4f}" if self.nok_upper_bound is not None else "\nNOK upper bound could not be determined as there are no samples predicted as NOK during testing.")

        # calculate precison, recall, f1-score
        tp = np.sum((filtered_df['pred_label'] == -1) & (filtered_df['given_label'] == -1))  # True Positives
        fp = np.sum((filtered_df['pred_label'] == -1) & (filtered_df['given_label'] == 1))  # False Positives
        fn = np.sum((filtered_df['pred_label'] == 1) & (filtered_df['given_label'] == -1))  # False Negatives
        tn = np.sum((filtered_df['pred_label'] == -1) & (filtered_df['given_label'] == -1))  # True Negatives

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print(f"\nDataframe is as follows:")
        print(self.df)

        print(f"\nTest accuracy: {accuracy:.2f}")
        print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-score: {f1_score:.2f}")

    # 3. Log model information
        self.model_type = type(anomaly_detector.model).__name__
        self.model_id = anomaly_detector.hparams.get('model_id', 'unknown_model')
        self.run_type = os.path.basename(self.logger.log_dir) if self.logger else 'test'
        self.tb_tag = self.model_id.split('-')[0].strip('[]').replace('_(', "  (").replace('+', " + ") if self.logger else self.model_type

        # update hparams
        anomaly_detector.hparams['test_accuracy'] = accuracy
        anomaly_detector.hparams['precision'] = precision
        anomaly_detector.hparams['recall'] = recall
        anomaly_detector.hparams['f1_score'] = f1_score
        anomaly_detector.hparams['run_type'] = self.run_type
        anomaly_detector.hparams['infer_time'] = infer_time
        anomaly_detector.hparams['tp'] = int(tp)
        anomaly_detector.hparams['fp'] = int(fp)
        anomaly_detector.hparams['tn'] = int(tn)
        anomaly_detector.hparams['fn'] = int(fn)
        anomaly_detector.hparams[f'db_delta_nok/test'] = anomaly_detector.nok_upper_bound

        if self.logger:
            self.logger.add_scalar(f"{self.tb_tag}/test_accuracy", accuracy)
            self.logger.add_scalar(f"{self.tb_tag}/precision", precision)
            self.logger.add_scalar(f"{self.tb_tag}/recall", recall)
            self.logger.add_scalar(f"{self.tb_tag}/f1_score", f1_score)

            self.logger.add_hparams(anomaly_detector.hparams, {})

            # save dataframe
            df_path = os.path.join(self.logger.log_dir, f'dataframe_{self.run_type}.pkl')
            self.df.to_pickle(df_path)  

            print(f"\nTesting hyperparameters logged for tensorboard at {self.logger.log_dir}")
        else:
            print("\nTesting hyperparameters not logged as logging is disabled.")

        print('\n' + 75*'-')

    def _assign_final_pred(self, row, anomaly_detector):
            if row['pred_label'] == -1:
                if anomaly_detector.nok_upper_bound is not None and row['scores'] <= anomaly_detector.nok_upper_bound:
                    return -1
                elif anomaly_detector.nok_upper_bound is not None and row['scores'] > anomaly_detector.nok_upper_bound:
                    return -0.5
            
            if row['pred_label'] == 1:
                if anomaly_detector.ok_lower_bound is not None and row['scores'] >= anomaly_detector.ok_lower_bound:
                    return 1
                elif anomaly_detector.ok_lower_bound is not None and row['scores'] < anomaly_detector.ok_lower_bound:
                    return 0.5
            return np.nan
                
# ================== Visualization Methods =======================

    def pair_plot(self, feat_cols=None):
        """
        Create a pair plot of the features.
        """
        if not hasattr(self, 'df'):
            raise ValueError("DataFrame is not available. Please train, test or predict the model first.")
        
        print("\n" + 12*"<" + " PAIR PLOT " + 12*">")
        print(f"\n> Creating pair plot for {self.model_id} / {self.run_type}...")

        feat_cols = self.comp_cols[:5] if feat_cols is None else feat_cols
        n_feats = len(feat_cols)

        df_sns = self.df.copy()
        df_sns['pred_label'] = df_sns['pred_label'].map({0: 'OK', 1: 'NOK'})
        # Dynamically set height (default is 2.5, you can adjust as needed)
        #height = max(2.5, min(2.5 + 0.5 * (n_feats - 2), 5.0))

        # if self.df[self.df['pred_label'] == 0].empty:
        #     palette = ['#ff7f0e']
        # elif self.df[self.df['pred_label'] == 1].empty:
        #     palette = ['#1f77b4']
        # else:
        palette = ['#1f77b4', '#ff7f0e']

        pair_plot = sns.pairplot(df_sns, vars=feat_cols, hue='pred_label', hue_order=['OK', 'NOK'], palette=palette, height=2.5)
        pair_plot.figure.suptitle(f"Pair Plot of Features : [{self.model_id} / {self.run_type}]", y=0.99, fontsize=13)
        plt.tight_layout(rect=[0, 0, 0.93, 0.99])  # Reserve space for the title
        # plt.subplots_adjust(right=0.94)

        # save the pair plot if logger is available
        if self.logger:
            fig = pair_plot.figure
            fig.savefig(os.path.join(self.logger.log_dir, f'pair_plot_({self.model_id}_{self.run_type}).png'), dpi=500)
            self.logger.add_figure(f"{self.tb_tag}/{self.model_id}/{self.run_type}/pair_plot", fig, close=True)
            print(f"\nPair plot logged at {self.logger.log_dir}\n")
        else:
            print("\nPair plot not logged as logging is disabled.\n")
            

    def confusion_matrix_simple(self):
        """
        Create a confusion matrix of the predictions.
        """
        if not hasattr(self, 'df'):
            raise ValueError("DataFrame is not available. Please train, test or predict the model first.")

        print("\n" + 12*"<" + " CONFUSION MATRIX SIMPLE" + 12*">")
        print(f"\n> Creating confusion matrix for {self.model_id} / {self.run_type}...")

        x_label = ['OK (prediction)', 'NOK (prediction)']
        y_label = ['OK (truth)', 'NOK (truth)']

        # given_labels = np.where(self.df['given_label'] == 1, -1, 1) # convert 1 to -1 (anomaly) and 0 to 1 (normal)
        # pred_labels = np.where(self.df['pred_label'] == 1, -1, 1) 

        cm = pd.crosstab(
            self.df['given_label'], self.df['pred_label'],   
            rownames=['Actual'], colnames=['Predicted'], dropna=False
            ).reindex(index=[1, -1], columns=[1, -1], fill_value=0)
        
        # update font settings for plots
        plt.rcParams.update({
            "text.usetex": False,   # No external LaTeX
            "font.family": "serif",
            "mathtext.fontset": "cm",  # Computer Modern math
        })
        
        plt.figure(figsize=(8, 6), dpi=100)
        cm_plot = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=x_label, yticklabels=y_label)

        # add tp, fp, tn, fn labels
        cell_labels = [['TN', 'FP'],
                        ['FN', 'TP']]

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                cm_plot.text(j + 0.5, i + 0.3, cell_labels[i][j],
                        ha='center', va='center', color='grey', fontsize=10)
                
        plt.title(f'Confusion Matrix : [{self.model_id} / {self.run_type}]')

        # save the confusion matrix if logger is available
        if self.logger:
            fig = cm_plot.get_figure()
            fig.savefig(os.path.join(self.logger.log_dir, f'cm_simple({self.model_id}_{self.run_type}).png'), dpi=500)
            self.logger.add_figure(f"{self.tb_tag}/{self.model_id}/{self.run_type}/confusion_matrix", fig, close=True)
            print(f"\nConfusion matrix logged at {self.logger.log_dir}\n")
        else:
            print("\nConfusion matrix not logged as logging is disabled.\n")


    def confusion_matrix_advance(self):
        """
        Create a confusion matrix that distinguishes between hard and soft predictions.
        """
        if not hasattr(self, 'df'):
            raise ValueError("DataFrame is not available. Please train, test or predict the model first.")

        print("\n" + 12*"<" + " CONFUSION MATRIX ADVANCE " + 12*">")
        print(f"\n> Creating confusion matrix (hard/soft) for {self.model_id} / {self.run_type}...")

        # Define mapping for display
        x_label = ['OK (prediction)', 'NOK (prediction)']
        y_label = ['OK (truth)', 'NOK (truth)']

        # Prepare masks for each cell
        # Rows: given_label (1=OK, -1=NOK)
        # Cols: is_final_pred (1=Hard OK, 0.5=Soft OK, -1=Hard NOK, -0.5=Soft NOK)
        matrix = {
            (1, 1): {}, (1, 0.5): {}, (1, -1): {}, (1, -0.5): {},
            (-1, 1): {}, (-1, 0.5): {}, (-1, -1): {}, (-1, -0.5): {},
        }
        for (truth, pred) in matrix.keys():
            mask = (self.df['given_label'] == truth) & (self.df['final_pred_label'] == pred)
            matrix[(truth, pred)]['count'] = mask.sum()

        # For each cell, sum hard and soft
        cell_counts = {
            (1, 1): matrix[(1, 1)]['count'],
            (1, 0.5): matrix[(1, 0.5)]['count'],
            (1, -1): matrix[(1, -1)]['count'],
            (1, -0.5): matrix[(1, -0.5)]['count'],
            (-1, 1): matrix[(-1, 1)]['count'],
            (-1, 0.5): matrix[(-1, 0.5)]['count'],
            (-1, -1): matrix[(-1, -1)]['count'],
            (-1, -0.5): matrix[(-1, -0.5)]['count'],
        }

        # Build 2x2 grid: rows = truth (OK, NOK), cols = pred (OK, NOK)
        # Each cell: A = total, B = hard, C = soft
        grid = [
            [  # Truth = OK
                {
                    'A': cell_counts[(1, 1)] + cell_counts[(1, 0.5)],
                    'B': cell_counts[(1, 1)],
                    'C': cell_counts[(1, 0.5)],
                },
                {
                    'A': cell_counts[(1, -1)] + cell_counts[(1, -0.5)],
                    'B': cell_counts[(1, -1)],
                    'C': cell_counts[(1, -0.5)],
                },
            ],
            [  # Truth = NOK
                {
                    'A': cell_counts[(-1, 1)] + cell_counts[(-1, 0.5)],
                    'B': cell_counts[(-1, 1)],
                    'C': cell_counts[(-1, 0.5)],
                },
                {
                    'A': cell_counts[(-1, -1)] + cell_counts[(-1, -0.5)],
                    'B': cell_counts[(-1, -1)],
                    'C': cell_counts[(-1, -0.5)],
                },
            ],
        ]

        # TN, FP, FN, TP cell labels
        cell_labels = [['TN', 'FP'], ['FN', 'TP']]

        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        plt.rcParams.update({
            "text.usetex": False,
            "font.family": "serif",
            "mathtext.fontset": "cm",
        })

        fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
        ax.set_xticks([0.5, 1.5])
        ax.set_yticks([0.5, 1.5])
        ax.set_xticklabels(x_label)
        ax.set_yticklabels(y_label)
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 2)
        ax.invert_yaxis()

        # Draw grid
        for i in range(3):
            ax.axhline(i, color='black', lw=1)
            ax.axvline(i, color='black', lw=1)

        # Write cell contents
        for i in range(2):  # rows
            for j in range(2):  # cols
                cell = grid[i][j]
                y = i + 0.5
                x = j + 0.5
                # A: total (center, large)
                ax.text(x, y-0.1, f"{cell['A']}", ha='center', va='center', fontsize=20, fontweight='bold', color='black')
                # B: hard (below, bold black)
                ax.text(x, y+0.13, f"H: {cell['B']}", ha='center', va='center', fontsize=13, fontweight='normal', color='black')
                # C: soft (below, light gray)
                ax.text(x, y+0.28, f"S: {cell['C']}", ha='center', va='center', fontsize=13, fontweight='normal', color='grey')
                # Cell label (TN, FP, etc)
                ax.text(x-0.38, y-0.38, cell_labels[i][j], ha='left', va='top', fontsize=12, color='dimgray', fontweight='bold')

        ax.set_xlabel('Prediction')
        ax.set_ylabel('Ground Truth')
        plt.title(f'Confusion Matrix (Hard/Soft) : [{self.model_id} / {self.run_type}]')
        plt.tight_layout()

        # Save if logger is available
        if self.logger:
            fig.savefig(os.path.join(self.logger.log_dir, f'cm_advance_({self.model_id}_{self.run_type}).png'), dpi=500)
            self.logger.add_figure(f"{self.tb_tag}/{self.model_id}/{self.run_type}/confusion_matrix_advance", fig, close=True)
            print(f"\nConfusion matrix (hard/soft) logged at {self.logger.log_dir}\n")
        else:
            plt.show()
            print("\nConfusion matrix (hard/soft) not logged as logging is disabled.\n")


    def roc_curve(self):
        """
        Create a ROC curve of the predictions.
        """
        if not hasattr(self, 'df'):
            raise ValueError("DataFrame is not available. Please train, test or predict the model first.")

        print("\n" + 12*"<" + " ROC CURVE " + 12*">")
        print(f"\nCreating ROC curve for {self.model_id} / {self.run_type}...")

        # calculate the ROC curve
        fpr, tpr, thresholds = roc_curve(self.df['given_label'], self.df['scores'], pos_label=1)
        
        # update font settings for plots
        plt.rcParams.update({
            "text.usetex": False,   # No external LaTeX
            "font.family": "serif",
            "mathtext.fontset": "cm",  # Computer Modern math
        })

        plt.figure(figsize=(8, 6), dpi=100)
        plt.plot(fpr, tpr, color='blue', label='ROC curve')
        plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random guess')

        # annotate threshold values at selected points
        for i in range(len(thresholds)):
            if i % (len(thresholds) // 10) == 0:  # Annotate every 10% of the points
                plt.text(fpr[i], tpr[i], f"{thresholds[i]:.2f}", fontsize=8, color='black')

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve with Thresholds : [{self.model_id} / {self.run_type}]')
        plt.legend()

        # save the ROC curve if logger is available
        if self.logger:
            fig = plt.gcf()
            fig.savefig(os.path.join(self.logger.log_dir, f'roc_({self.model_id}_{self.run_type}).png'), dpi=500)
            self.logger.add_figure(f"{self.tb_tag}/{self.model_id}/{self.run_type}/roc_curve", fig, close=True)
            print(f"\nROC curve logged at {self.logger.log_dir}\n")
        else:
            print("\nROC curve not logged as logging is disabled.\n")
        

    def anomaly_score_dist_simple(self, is_pred=False, is_log_x=False, bins=50, num=1):
        """
        Create a histogram of the anomaly scores for Ok and NOK classes.
        
        Parameters
        ----------
        is_pred : bool, optional
            If True, the scores are from predictions, by default False
        bins : int, optional
            Number of bins for the histogram, by default 100
        """
        if not hasattr(self, 'df'):
            raise ValueError("DataFrame is not available. Please train, test or predict the model first.")
        
        label_col = 'pred_label' if is_pred else 'given_label'

        print("\n" + 12*"<" + f" ANOMALY SCORE DISTRIBUTION SIMPLE {num} ({label_col.upper()}) " + 12*">")
        print(f"\nCreating anomaly score distribution plot for {self.model_id} / {self.run_type}...")

        ok_lower = self.ok_lower_bound if self.ok_lower_bound is not None else None
        nok_upper = self.nok_upper_bound if self.nok_upper_bound is not None else None

        # separate scores for OK and NOK classes
        scores_ok = self.df[self.df[label_col] == 1]['scores']
        scores_nok = self.df[self.df[label_col] == -1]['scores']
        indices_ok = self.df[self.df[label_col] == 1].index
        indices_nok = self.df[self.df[label_col] == -1].index

        if is_log_x:
            # handle negative scores by shifting them to positive range
            min_score_ok = scores_ok.min() if not scores_ok.empty else 0
            min_score_nok = scores_nok.min() if not scores_nok.empty else 0
            min_score = min(min_score_ok, min_score_nok)

            if min_score <= 0:
                shift = abs(min_score) + 1 # boundary = 0 + shift = shift
                scores_ok += shift
                scores_nok += shift
                ok_lower = ok_lower + shift if ok_lower is not None else None
                nok_upper = nok_upper + shift if nok_upper is not None else None
            else:
                shift = 0
        else:
            shift = 0

        # calculate means
        # mean_ok = np.mean(scores_ok)
        # mean_nok = np.mean(scores_nok)

        # db_delta values
        db_delta_ok = ok_lower - shift if ok_lower is not None else None
        db_delta_nok = shift - nok_upper if nok_upper is not None else None

        ok_included = scores_ok[scores_ok >= ok_lower] if ok_lower is not None else scores_ok
        ok_excluded = scores_ok[scores_ok < ok_lower] if ok_lower is not None else np.array([])
        
        nok_included = scores_nok[scores_nok <= nok_upper] if nok_upper is not None else scores_nok
        nok_excluded = scores_nok[scores_nok > nok_upper] if nok_upper is not None else np.array([])
        
        
        # # create bins and map sample indices to bins
        # bins_edges = np.histogram_bin_edges(np.concatenate([scores_ok, scores_nok]), bins=bins)
        # bin_indices_ok = np.digitize(scores_ok, bins_edges) - 1
        # bin_indices_nok = np.digitize(scores_nok, bins_edges) - 1

        # # map sample indices to bins
        # bin_samples_ok = {i: [] for i in range(len(bins_edges) - 1)}
        # bin_samples_nok = {i: [] for i in range(len(bins_edges) - 1)}

        # for idx, bin_idx in zip(indices_ok, bin_indices_ok):
        #     if 0 <= bin_idx < len(bins_edges) - 1:  # Ensure bin_idx is within valid range
        #         bin_samples_ok[bin_idx].append(idx)

        # for idx, bin_idx in zip(indices_nok, bin_indices_nok):
        #     if 0 <= bin_idx < len(bins_edges) - 1:
        #         bin_samples_nok[bin_idx].append(idx)

        # Combine all scores for bin edges
        all_scores = np.concatenate([ok_included, ok_excluded, nok_included, nok_excluded])
        bins_edges = np.histogram_bin_edges(all_scores, bins=bins)

        # update font settings for plots
        plt.rcParams.update({
            "text.usetex": False,   # No external LaTeX
            "font.family": "serif",
            "mathtext.fontset": "cm",  # Computer Modern math
        })

        # create the histogram
        plt.figure(figsize=(12, 8), dpi=100)
        # Plot included/excluded for OK and NOK
        counts_ok_in, _, _ = plt.hist(ok_included, bins=bins_edges, color='blue', label=f'OK hard ({self.ok_percentage * 100}% OK of train)', alpha=0.5)
        counts_ok_ex, _, _ = plt.hist(ok_excluded, bins=bins_edges, color='blue', label='OK soft', alpha=0.2)
        counts_nok_in, _, _ = plt.hist(nok_included, bins=bins_edges, color='orange', label=f'NOK hard ({self.nok_percentage * 100}% NOK)', alpha=0.5)
        counts_nok_ex, _, _ = plt.hist(nok_excluded, bins=bins_edges, color='orange', label='NOK soft', alpha=0.2)

        # # add vertical lines for means and boundary
        # plt.axvline(mean_ok, color='blue', linestyle='--', linewidth=1, label=f'Mean OK: {mean_ok:.4f}')
        # plt.axvline(mean_nok, color='orange', linestyle='--', linewidth=1, label=f'Mean NOK: {mean_nok:.4f}')
        plt.axvline(shift, color='red', linestyle=':', linewidth=1.5, label=f'Boundary: {shift:.4f}')
        plt.axvline(ok_lower, color='teal', linestyle=':', linewidth=1.5, label=f'OK Lower Bound (from train): {ok_lower:.4}') if ok_lower is not None else None
        plt.axvline(nok_upper, color='brown', linestyle=':', linewidth=1.5, label=f'NOK Upper Bound: {nok_upper:.4}') if nok_upper is not None else None
        
        # db_delta_ok
        if ok_lower is not None:
            plt.hlines(y=5, xmin=min(ok_lower, shift), xmax=max(ok_lower, shift), colors='teal', alpha=0.8, linestyles='--', linewidth=1, label=f'DB delta OK ({self.ok_percentage * 100}% OK of train): {db_delta_ok:.4f}')
        # db_delta_nok
        if nok_upper is not None:
            plt.hlines(y=8, xmin=min(nok_upper, shift), xmax=max(nok_upper, shift), colors='brown', alpha=0.6, linestyles='--', linewidth=1, label=f'DB delta NOK ({self.nok_percentage * 100}% NOK): {db_delta_nok:.4f}')

         # Add bin indices on top of each bar for all histograms
        for i in range(len(bins_edges) - 1):
            if counts_ok_in[i] > 0:
                plt.text((bins_edges[i] + bins_edges[i + 1]) / 2, counts_ok_in[i], str(i), ha='center', va='bottom', fontsize=8, color='blue')
            if counts_ok_ex[i] > 0:
                plt.text((bins_edges[i] + bins_edges[i + 1]) / 2, counts_ok_ex[i], str(i), ha='center', va='bottom', fontsize=8, color='blue', alpha=0.4)
            if counts_nok_in[i] > 0:
                plt.text((bins_edges[i] + bins_edges[i + 1]) / 2, counts_nok_in[i], str(i), ha='center', va='bottom', fontsize=8, color='orange')
            if counts_nok_ex[i] > 0:
                plt.text((bins_edges[i] + bins_edges[i + 1]) / 2, counts_nok_ex[i], str(i), ha='center', va='bottom', fontsize=8, color='orange', alpha=0.4)

        
        xlabel_text = " (log scale) (shifted)" if is_log_x else ""

        plt.xlabel(f'Anomaly Score{xlabel_text}')
        plt.ylabel('Number of Samples (log scale)')
        plt.title(f'Anomaly Score Distribution ({label_col.replace('_', ' ').capitalize()}) : [{self.model_id} / {self.run_type}]')
        plt.legend()
        plt.grid(True)
        plt.yscale('log') 
        if is_log_x: 
            plt.xscale('log')

        # write sample indices in each bin
        text = f"## Bin details for Anomaly Score Distribution Simple {num} ({label_col})\n"
        text += 25 * "-"

        # text += f"\n### Total OK samples (from {label_col}) : {len(scores_ok)}\n"
        # text += 10*"-" + " Samples in OK bins " + 10*"-" + "\n"
        # for bin_idx, samples in bin_samples_ok.items():
        #     if samples:  # Only print non-empty bins
        #         low, high = bins_edges[bin_idx], bins_edges[bin_idx + 1]
        #         sample_info = ", ".join([f"{idx} ({float(self.df.loc[idx, 'rep_num']):,.3f})" for idx in samples])
        #         text += f"- **Bin {bin_idx} ({low-shift:.5f}, {high-shift:.5f}) [blue]**: {sample_info}\n"

        # text += f"\n### Total NOK samples (from {label_col}) : {len(scores_nok)}\n"
        # text += 10*"-" + " Samples in NOK bins " + 10*"-" + "\n"
        # for bin_idx, samples in bin_samples_nok.items():
        #     if samples:  # Only print non-empty bins
        #         low, high = bins_edges[bin_idx], bins_edges[bin_idx + 1]
        #         sample_info = ", ".join([f"{idx} ({float(self.df.loc[idx, 'rep_num']):,.3f})" for idx in samples])
        #         text += f"- **Bin {bin_idx} ({low-shift:.5f}, {high-shift:.5f}) [orange]**: {sample_info}\n"

        def write_bin_samples(indices, scores, bins_edges, label, color):
            nonlocal text
            bin_indices = np.digitize(scores, bins_edges) - 1
            bin_samples = {i: [] for i in range(len(bins_edges) - 1)}
            for idx, bin_idx in zip(indices, bin_indices):
                if 0 <= bin_idx < len(bins_edges) - 1:
                    bin_samples[bin_idx].append(idx)

            text += f"\n### Total {label} samples: {sum(len(samples) for samples in bin_samples.values())}\n"
            text += 10*"-" + f" Samples in {label} bins " + 10*"-" + "\n"
            for bin_idx, samples in bin_samples.items():
                if samples:
                    low, high = bins_edges[bin_idx], bins_edges[bin_idx + 1]
                    sample_info = ", ".join([f"{idx} ({float(self.df.loc[idx, 'rep_num']):,.3f})" for idx in samples])
                    text += f"- **Bin {bin_idx} ({low-shift:.5f}, {high-shift:.5f}) [{color}]**: {sample_info}\n"
        
        write_bin_samples(indices_ok, scores_ok, bins_edges, "OK", "blue")
        write_bin_samples(indices_nok, scores_nok, bins_edges, "NOK", "orange")

        # Print db_delta values
        text += f"\nDB delta OK ({self.ok_percentage * 100}% OK of train): {db_delta_ok:.4f} (DB = {shift:.4f})" if db_delta_ok is not None else "No DB delta OK"
        text += f"\nDB delta NOK ({self.nok_percentage * 100}% NOK): {db_delta_nok:.4f} (DB = {shift:.4f})\n" if db_delta_nok is not None else "No DB delta NOK"

        print(text)

        # save the distribution plot if logger is available
        if self.logger:
            fig = plt.gcf()
            fig.savefig(os.path.join(self.logger.log_dir, f'anom_score_simple_{num}_{label_col.split('_')[0]}({self.model_id}_{self.run_type}).png'), dpi=500)
            self.logger.add_figure(f"{self.tb_tag}/{self.model_id}/{self.run_type}/anomaly_score_dist_simple_{num}_{label_col}", fig, close=True)
            self.logger.add_text(f"{self.model_id} + {self.run_type}", text)
            print(f"\nAnomaly score distribution plot logged at {self.logger.log_dir}\n")
        else:
            print("\nAnomaly score distribution plot not logged as logging is disabled.\n")


    def anomaly_score_dist_advance(self, bins=50, is_log_x=False, num=1):
        """
        Create a histogram of the anomaly scores for all combinations of given and predicted labels.
        Plots:
            - True Negative (OK correctly classified): blue
            - True Positive (NOK correctly classified): orange
            - False Positive (OK misclassified as NOK): red
            - False Negative (NOK misclassified as OK): purple

        Also shows db_delta_ok and db_delta_nok for TN and TP using percentiles.

        Parameters
        ----------
        bins : int, optional
            Number of bins for the histogram, by default 50
        is_log_x : bool, optional
            If True, the x-axis is logarithmic, by default False
        """

        if not hasattr(self, 'df'):
            raise ValueError("DataFrame is not available. Please train, test or predict the model first.")

        print("\n" + 12*"<" + F" ANOMALY SCORE DISTRIBUTION ADVANCE {num} (GIVEN vs PRED) " + 12*">")
        print(f"\n(Using '{self.ok_percentage * 100}%' OK of train and '{self.nok_percentage * 100}%' NOK)")
        print(f"\nCreating anomaly score distribution plot for {self.model_id} / {self.run_type}...")

        df = self.df
        ok_lower = self.ok_lower_bound if self.ok_lower_bound is not None else None
        nok_upper = self.nok_upper_bound if self.nok_upper_bound is not None else None

        # Masks for each category
        tn_mask = (df['given_label'] == 1) & (df['pred_label'] == 1)  # True Negative (OK correctly classified)
        tp_mask = (df['given_label'] == -1) & (df['pred_label'] == -1)  # True Positive (NOK correctly classified)
        fp_mask = (df['given_label'] == 1) & (df['pred_label'] == -1)  # False Positive (OK misclassified as NOK)
        fn_mask = (df['given_label'] == -1) & (df['pred_label'] == 1)  # False Negative (NOK misclassified as OK)

        # Scores and indices for each category
        scores_tn = df[tn_mask]['scores']
        scores_tp = df[tp_mask]['scores']
        scores_fp = df[fp_mask]['scores']
        scores_fn = df[fn_mask]['scores']

        indices_tn = df[tn_mask].index
        indices_tp = df[tp_mask].index
        indices_fp = df[fp_mask].index
        indices_fn = df[fn_mask].index

        if is_log_x:
            # Shift scores if needed
            min_score = min([
                scores_tn.min() if not scores_tn.empty else 0,
                scores_tp.min() if not scores_tp.empty else 0,
                scores_fp.min() if not scores_fp.empty else 0,
                scores_fn.min() if not scores_fn.empty else 0
            ])
            shift = abs(min_score) + 1 if min_score <= 0 else 0
            
            scores_tn = scores_tn + shift
            scores_tp = scores_tp + shift
            scores_fp = scores_fp + shift
            scores_fn = scores_fn + shift
            ok_lower = ok_lower + shift if ok_lower is not None else None
            nok_upper = nok_upper + shift if nok_upper is not None else None

        else:
            shift = 0

        # Percentile cutoff for OK (from max to min)
        # ok_sorted = np.sort(scores_given_ok)[::-1]
        # ok_cutoff_idx = int(np.ceil(len(ok_sorted) * (percentile_ok / 100.0))) - 1
        # ok_included_min = ok_sorted[ok_cutoff_idx] if ok_sorted.size > 0 else None

        # # Percentile cutoff for NOK (from min to max)
        # nok_sorted = np.sort(scores_given_nok)
        # nok_cutoff_idx = int(np.ceil(len(nok_sorted) * (percentile_nok / 100.0))) - 1
        # nok_included_max = nok_sorted[nok_cutoff_idx] if nok_sorted.size > 0 else None

        # db_delta values
        db_delta_ok = ok_lower - shift if ok_lower is not None else None
        db_delta_nok = shift - nok_upper if nok_upper is not None else None

        # Filter TN/TP samples based on percentile cutoffs
        tn_included = scores_tn[scores_tn >= ok_lower] if ok_lower is not None else scores_tn
        tn_excluded = scores_tn[scores_tn < ok_lower] if ok_lower is not None else np.array([])

        tp_included = scores_tp[scores_tp <= nok_upper] if nok_upper is not None else scores_tp
        tp_excluded = scores_tp[scores_tp > nok_upper] if nok_upper is not None else np.array([])

        fn_included = scores_fn[scores_fn >= ok_lower] if ok_lower is not None else scores_fn
        fn_excluded = scores_fn[scores_fn < ok_lower] if ok_lower is not None else np.array([])

        fp_included = scores_fp[scores_fp <= nok_upper] if nok_upper is not None else scores_fp
        fp_excluded = scores_fp[scores_fp > nok_upper] if nok_upper is not None else np.array([])

        # Combine all scores for bin edges
        all_scores = np.concatenate([tn_included, tn_excluded, tp_included, tp_excluded, scores_fp, scores_fn])
        bins_edges = np.histogram_bin_edges(all_scores, bins=bins)

        # Update font settings for plots
        plt.rcParams.update({
            "text.usetex": False,
            "font.family": "serif",
            "mathtext.fontset": "cm",
        })

        plt.figure(figsize=(12, 8), dpi=100)

        # Plot included samples
        counts_tn, _, _ = plt.hist(tn_included, bins=bins_edges, color='blue', label=f'TN hard ({self.ok_percentage * 100}% OK of train) [OK, correct]', alpha=0.5)
        counts_tn_ex, _, _ = plt.hist(tn_excluded, bins=bins_edges, color='blue', alpha=0.2, label=f'TN soft')

        counts_tp, _, _ = plt.hist(tp_included, bins=bins_edges, color='orange', label=f'TP hard ({self.nok_percentage * 100}% NOK) [NOK, correct]', alpha=0.5)
        counts_tp_ex, _, _ = plt.hist(tp_excluded, bins=bins_edges, color='orange', alpha=0.2, label=f'TP soft')
        
        counts_fp, _, _ = plt.hist(fp_included, bins=bins_edges, color='red', label='FP hard [OK, misclassified]', alpha=0.5)
        counts_fp_ex, _, _ = plt.hist(fp_excluded, bins=bins_edges, color='red', alpha=0.2, label='FP soft')
        
        counts_fn, _, _ = plt.hist(fn_included, bins=bins_edges, color='purple', label='FN hard [NOK, misclassified]', alpha=0.5)
        counts_fn_ex, _, _ = plt.hist(fn_excluded, bins=bins_edges, color='purple', alpha=0.2, label='FN soft')

        # vertical lines for boundary
        plt.axvline(shift, color='red', linestyle=':', linewidth=1.5, label=f'Boundary: {shift:.4f}')
        plt.axvline(ok_lower, color='teal', linestyle=':', linewidth=1.5, label=f'OK Lower Bound (from train): {ok_lower:.4}') if ok_lower is not None else None
        plt.axvline(nok_upper, color='brown', linestyle=':', linewidth=1.5, label=f'NOK Upper Bound: {nok_upper:.4}') if nok_upper is not None else None

        # db_delta_ok and db_delta_nok
        if db_delta_ok is not None:
            plt.hlines(y=5, xmin=min(ok_lower, shift), xmax=max(ok_lower, shift), color='teal', linestyle='--', alpha=0.8, linewidth=1,
                        label=f'DB delta OK ({self.ok_percentage * 100}% OK of train): {db_delta_ok:.4f}')
        if db_delta_nok is not None:
            plt.hlines(y=8, xmin=min(nok_upper, shift), xmax=max(nok_upper, shift), color='brown', linestyle='--', alpha=0.6, linewidth=1,
                        label=f'DB delta NOK ({self.nok_percentage * 100}% NOK): {db_delta_nok:.4f}')

        # Add bin indices on top of each bar for all histograms
        for i in range(len(bins_edges) - 1):
            y_offset = 0
            if counts_tn[i] > 0:
                plt.text((bins_edges[i] + bins_edges[i + 1]) / 2, counts_tn[i], str(i), ha='center', va='bottom', fontsize=8, color='blue')
                y_offset = max(y_offset, counts_tn[i])
            if counts_tp[i] > 0:
                plt.text((bins_edges[i] + bins_edges[i + 1]) / 2, counts_tp[i], str(i), ha='center', va='bottom', fontsize=8, color='orange')
                y_offset = max(y_offset, counts_tp[i])
            if counts_fp[i] > 0:
                plt.text((bins_edges[i] + bins_edges[i + 1]) / 2, counts_fp[i], str(i), ha='center', va='bottom', fontsize=8, color='red')
                y_offset = max(y_offset, counts_fp[i])
            if counts_fn[i] > 0:
                plt.text((bins_edges[i] + bins_edges[i + 1]) / 2, counts_fn[i], str(i), ha='center', va='bottom', fontsize=8, color='purple')
                y_offset = max(y_offset, counts_fn[i])
            if tn_excluded.size > 0 and counts_tn_ex[i] > 0:
                plt.text((bins_edges[i] + bins_edges[i + 1]) / 2, counts_tn_ex[i], str(i), ha='center', va='bottom', fontsize=8, color='blue', alpha=0.4)
            if tp_excluded.size > 0 and counts_tp_ex[i] > 0:
                plt.text((bins_edges[i] + bins_edges[i + 1]) / 2, counts_tp_ex[i], str(i), ha='center', va='bottom', fontsize=8, color='orange', alpha=0.4)
            if fp_excluded.size > 0 and counts_fp_ex[i] > 0:
                plt.text((bins_edges[i] + bins_edges[i + 1]) / 2, counts_fp_ex[i], str(i), ha='center', va='bottom', fontsize=8, color='red', alpha=0.4)
            if fn_excluded.size > 0 and counts_fn_ex[i] > 0:
                plt.text((bins_edges[i] + bins_edges[i + 1]) / 2, counts_fn_ex[i], str(i), ha='center', va='bottom', fontsize=8, color='purple', alpha=0.4)

        xlabel_text = " (log scale) (shifted)" if is_log_x else ""

        plt.xlabel(f'Anomaly Score{xlabel_text}')
        plt.ylabel('Number of Samples (log scale)')
        plt.title(f'Anomaly Score Distribution (Given vs Pred Label) : [{self.model_id} / {self.run_type}]')
        plt.legend()
        plt.grid(True)
        plt.yscale('log')
        if is_log_x:
            plt.xscale('log')

        # Write sample indices for each bin and category
        text = f"## Bin details for Anomaly Score Distribution Advance {num} ({self.ok_percentage * 100}% OK of train, {self.nok_percentage * 100}% NOK) \n"
        text += 25 * "-"

        def write_bin_samples(indices, scores, bins_edges, label, color):
            nonlocal text
            bin_indices = np.digitize(scores, bins_edges) - 1
            bin_samples = {i: [] for i in range(len(bins_edges) - 1)}
            for idx, bin_idx in zip(indices, bin_indices):
                if 0 <= bin_idx < len(bins_edges) - 1:
                    bin_samples[bin_idx].append(idx)
            text += f"\n### Total {label} samples: {sum(len(samples) for samples in bin_samples.values())}\n"
            text += 10*"-" + f" Samples in {label} bins " + 10*"-" + "\n"
            for bin_idx, samples in bin_samples.items():
                if samples:
                    low, high = bins_edges[bin_idx], bins_edges[bin_idx + 1]
                    sample_info = ", ".join([f"{idx} ({float(df.loc[idx, 'rep_num']):,.3f})" for idx in samples])
                    text += f"- **Bin {bin_idx} ({low-shift:.5f}, {high-shift:.5f}) [{color}]**: {sample_info}\n"

        write_bin_samples(indices_tn, scores_tn, bins_edges, "True Negative (OK, correct)", "blue")
        write_bin_samples(indices_tp, scores_tp, bins_edges, "True Positive (NOK, correct)", "orange")
        write_bin_samples(indices_fp, scores_fp, bins_edges, "False Positive (OK, misclassified)", "red")
        write_bin_samples(indices_fn, scores_fn, bins_edges, "False Negative (NOK, misclassified)", "purple")

        # Print db_delta values
        text += f"\nDB delta OK ({self.ok_percentage * 100}% OK of train): {db_delta_ok:.4f} (DB = {shift:.4f})" if db_delta_ok is not None else "No DB delta OK"
        text += f"\nDB delta NOK ({self.nok_percentage * 100}% NOK): {db_delta_nok:.4f} (DB = {shift:.4f})\n" if db_delta_nok is not None else "No DB delta NOK"

        print(text)

        # Save the plot if logger is available
        if self.logger:
            fig = plt.gcf()
            fig.savefig(os.path.join(self.logger.log_dir, f'anom_score_advance_{num}({self.model_id}_{self.run_type}).png'), dpi=500)
            self.logger.add_figure(f"{self.tb_tag}/{self.model_id}/{self.run_type}/anomaly_score_dist_advance_{num}", fig, close=True)
            self.logger.add_text(f"{self.model_id} + {self.run_type}", text)
            print(f"\nAnomaly score distribution plot logged at {self.logger.log_dir}\n")
        else:
            print("\nAnomaly score distribution plot not logged as logging is disabled.\n")


    def pred_plot(self):
        """
        Plot samples divided into OK and NOK categories based on predicted labels.
        """
        if not hasattr(self, 'df'):
            raise ValueError("DataFrame is not available. Please train, test or predict the model first.")
        
        print("\n" + 12*"<" + " PREDICTION PLOT " + 12*">")
        print(f"\nCreating prediction plot for {self.model_id} / {self.run_type}...")

        # separate samples into OK and NOK
        ok_samples = self.df[self.df['pred_label'] == 0].index.tolist()
        nok_samples = self.df[self.df['pred_label'] == 1].index.tolist()

        # create the bar chart
        categories = ['OK', 'NOK']
        counts = [len(ok_samples), len(nok_samples)]

        # update font settings for plots
        plt.rcParams.update({
            "text.usetex": False,   # No external LaTeX
            "font.family": "serif",
            "mathtext.fontset": "cm",  # Computer Modern math
        })
        
        plt.bar(categories, counts, color=['blue', 'orange'], alpha=0.7)

        # annotate the bar chart with sample indices
        for i, samples in enumerate([ok_samples, nok_samples]):
            plt.text(i, counts[i] / 2, f"Samples: {', '.join(map(str, samples))}", 
                    ha='center', va='center', fontsize=10, color='grey', wrap=True)

        plt.title(f"Predictions OK v/s NOK : [{self.model_id} / {self.run_type}]")
        plt.ylabel("Number of Samples")

        # save the bar plot if logger is available
        if self.logger:
            fig = plt.gcf()
            fig.savefig(os.path.join(self.logger.log_dir, f'pred_plot_({self.model_id}_{self.run_type}).png'), dpi=500)
            self.logger.add_figure(f"{self.tb_tag}/{self.model_id}/{self.run_type}/pred_plot", fig, close=True)
            print(f"\nPrediction plot logged at {self.logger.log_dir}\n")
        else:
            print("\nPrediction plot not logged as logging is disabled.\n")