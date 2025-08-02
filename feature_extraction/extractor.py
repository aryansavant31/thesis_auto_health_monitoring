"""
This module contains:
- Pipeline class
- Feature extraction Functions
"""

import sys
import os

FEX_DIR = os.path.join((os.path.abspath(__file__)))
sys.path.insert(0, FEX_DIR) if FEX_DIR not in sys.path else None

# other imports
import numpy as np
import torch
from sklearn.decomposition import PCA

# global imports
from data.settings import DataConfig

# local imports
import tf # time features
import ff # frequency features
from config.manager import FeatureRankingManager


class TimeFeatureExtractor:
    def __init__(self, feat_configs, run_type='train'):
        """
        Initialize the time feature extractor model with the specified type.

        Parameters
        ----------
        feat_configs : list
            List of time feature extraction configurations.
        """
        self.feat_configs = feat_configs
        self.data_config = DataConfig(run_type)

    def extract(self, time_data):
        """
        Apply the time feature extraction to the data.

        Parameters
        ----------
        time_data : torch.Tensor, shape (batch_size, n_nodes, n_timesteps, n_dims)
            Input data in time domain.
        
        Returns
        -------
        final_tensor : torch.Tensor, shape (batch_size, n_nodes, n_components, n_dims)
            Tensor with extracted features based on the specified configurations.
        """
        features_list = []

        # loop through each feature extraction configuration
        for feat_config in self.feat_configs:
            feat_type = feat_config['type']

            # 1. features from ranks
            if feat_type == 'from_ranks':
                top_feature_names = get_feature_list(
                    n=feat_config['n'],
                    perf_v=feat_config['perf_v'],
                    rank_v=feat_config['rank_v'],
                    domain='time',
                    data_config=self.data_config)
                
                for feature_name in top_feature_names:
                    if hasattr(tf, feature_name):
                        feat_fn = getattr(tf, feature_name)
                        features = feat_fn(time_data)  
                        features_list.append(features)
                    else:
                        raise ValueError(f"Unknown time feature name in ranks: {feature_name}")

            # 2. features apart from ranks
            elif hasattr(tf, feat_type):
                feat_fn = getattr(tf, feat_type)
                features = feat_fn(time_data)  
                features_list.append(features)
            else:
                raise ValueError(f"Unknown time feature extraction type: {feat_type}")
            
        # concatenate all features into a single tensor
        final_tensor = torch.cat(features_list, axis=2)  # shape: (batch_size, n_nodes, n_components, n_dims)

        return final_tensor 
             

class FrequencyFeatureExtractor:
    def __init__(self, feat_configs, run_type='train'):
        """
        Initialize the feature extractor model with the specified type.
        
        Parameters
        ----------
        feat_config : list
            Type of features to be used (e.g., 'first_n_modes').
        """
        self.feat_configs = feat_configs
        self.data_config = DataConfig(run_type)
        self.fs = self.data_config.fs  # sampling frequency

    def extract(self, freq_mag, freq_bins):
        """
        Apply the frequency feature extraction to the data.

        Parameters
        ----------
        freq_mag : torch.Tensor, shape (batch_size, n_nodes, n__bins, n_dims)
            Frequency magnitude
        freq_bins : torch.Tensor, shape (batch_size, n_nodes, n_bins, n_dims)
        
        Returns
        -------
        final_tensor : torch.Tensor, shape (batch_size, n_nodes, n_components, n_dims)
            Tensor with extracted features based on the specified configurations (if `feat_configs` is not empty).   
        """
        # initialize an empty list to hold features
        features_list = []

        # convert frequency magnitude to frequency amplitude and PSD
        freq_amp = ff.get_freq_amp(freq_mag) 
        freq_psd = ff.get_freq_psd(freq_mag, self.fs)
    
        # loop through each feature extraction configuration
        for feat_config in self.feat_configs:
            feat_type = feat_config['type']

            # 1. features from ranks
            if feat_type == 'from_ranks':
                top_feature_names = get_feature_list(
                    n=feat_config['n'],
                    perf_v=feat_config['perf_v'],
                    rank_v=feat_config['rank_v'],
                    domain='time',
                    data_config=self.data_config)
                
                for feature_name in top_feature_names:
                    if hasattr(tf, feature_name):
                        feat_fn = getattr(ff, feature_name)

                        if feat_fn.__code__.co_argcount == 1:
                            features = feat_fn(freq_bins)
                        elif feat_fn.__code__.co_argcount == 2:
                            features = feat_fn(freq_bins, freq_amp)  

                        features_list.append(features)
                    else:
                        raise ValueError(f"Unknown time feature name in ranks: {feature_name}")

            # 2. first n modes
            elif feat_type == 'first_n_modes':
                top_psd, top_freq = ff.first_n_modes(freq_bins, freq_psd, feat_config['n_modes']) 
                features_list.append(top_freq)
                features_list.append(top_psd)

            # 3. full spectrum features
            elif feat_type == 'full_spectrum':
                for param in feat_config['parameters']:
                    if param == 'psd':
                        features_list.append(freq_psd)
                    elif param == 'mag':
                        features_list.append(freq_mag)
                    elif param == 'amp':
                        features_list.append(freq_amp)
                    elif param == 'freq':
                        features_list.append(freq_bins)
                    else:
                        raise ValueError(f"Unknown frequency feature parameter: {param}")

            # 4. features apart from ranks
            elif hasattr(ff, feat_type):
                feat_fn = getattr(ff, feat_type)

                if feat_fn.__code__.co_argcount == 1:
                    features = feat_fn(freq_bins)
                elif feat_fn.__code__.co_argcount == 2:
                    features = feat_fn(freq_bins, freq_amp)  

                features_list.append(features)   
            else:
                raise ValueError(f"Unknown frequency feature extraction type: {feat_config['type']}")

        # concatenate all features into a single tensor
        final_tensor = torch.cat(features_list, axis=2)  # shape: (batch_size, n_nodes, n_components, n_dims)

        return final_tensor       
        
class FeatureReducer:
    def __init__(self, reduc_config):

        if reduc_config['type'] == 'PCA':
            self.model = PCA(n_components=reduc_config['n_components'])

    def reduce(self, data):
        """
        Apply feature reduction to the data.

        Parameters
        ----------
        data : torch.Tensor, shape (batch_size, n_nodes, n_components, n_dims)
        """
        # convert data to 2D numpy array
        batch_size, n_nodes, n_components, n_dims = data.shape
        device = data.device
        data_np = data.view(-1, data.shape[-2] * data.shape[-1]).detach().cpu().numpy()  # shape (batch_size * n_nodes, n_components * n_dims)

        features = self.model.fit_transform(data_np)  
        
        # convert features back to torch tensor and reshape
        features = torch.from_numpy(features).to(device)
        final_tensor = features.view(batch_size, n_nodes, -1, n_dims)

        return final_tensor
        


def get_feature_list(n, perf_v, rank_v, domain, data_config:DataConfig):
    """
    Get the list of features based on the ranks from the specified version.

    Parameters
    ----------
    n : int
        First 'n' features to extract.
    perf_v : int
        Performance version to get the feature ranks from.
    rank_v : int
        Ranking version to get the feature ranks from.
    domain : str
        Domain of the features (`time` or `freq`)
    data_config : DataConfig
        Data configuration object containing the data settings.
    """
    rank_config = FeatureRankingManager()

    rank_config.perf_version = perf_v
    rank_config.rank_version = rank_v

    rank_log_path = rank_config.get_ranking_log_path(data_config, is_avail=True)
    rank_dict = np.load(os.path.join(rank_log_path, f"{domain}_feature_ranking.npy"), allow_pickle=True).item()

    # sort the features based on the ranks in decending order
    sorted_features = sorted(rank_dict.items(), key=lambda x: x[1], reverse=True)

    # get the top 'n' features
    top_features = [feature_name.lower() for feature_name, _ in sorted_features[:n]]
    return top_features