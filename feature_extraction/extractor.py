"""
This module contains:
- Pipeline class
- Feature extraction Functions
"""
import numpy as np
import torch
from data.config import DataConfig
from sklearn.decomposition import PCA
import time_features as tf
import frequency_features as ff


class TimeFeatureExtractor:
    def __init__(self, fex_configs):
        """
        Initialize the time feature extractor model with the specified type.

        Parameters
        ----------
        fex_configs : list
            List of time feature extraction configurations.
        """
        self.fex_configs = fex_configs

    def preprocess_input(self, data):
        """
        Flatten the input data to 2D and convert to numpy array.

        Parameters
        ----------
        data : torch.Tensor, shape (batch_size, n_nodes, n_timesteps, n_dims)
            Input data in time domain.

        Returns
        -------
        data_np : np.ndarray, shape (batch_size * n_nodes, n_timesteps, n_dims)
            Flattened input data.
        """
        self.original_shape = data.shape  # save original shape for later reshaping
        self.device = data.device  # save device for later use

        data_np = data.view(-1, data.shape[-2], data.shape[-1]).detach().cpu().numpy()  # (batch_size * n_nodes, n_timesteps, n_dims)
        return data_np
    
    def postprocess_output(self, features):
        """
        Reshape the features to match the original input shape.

        Parameters
        ----------
        features : np.ndarray, shape (batch_size * n_nodes, n_components, n_dims)
            Extracted features.

        Returns
        -------
        reshaped_features : torch.Tensor, shape (batch_size, n_nodes, n_components, n_dims)
            Reshaped features.
        """
        features = torch.from_numpy(features).to(self.device)
        return features.view(self.original_shape)

    def __call__(self, data):
        """
        Apply the time feature extraction to the data.

        Parameters
        ----------
        data : torch.Tensor, shape (batch_size, n_nodes, n_timesteps, n_dims)
            Input data in time domain.
        
        Returns
        -------
        features : torch.Tensor, shape (batch_size, n_nodes, n_components, n_dims)
            Extracted features based on the specified configurations.
        """
        features_list = []
        data_np = self.preprocess_input(data)

        # loop through each feature extraction configuration
        for fex_config in self.fex_configs:
            fex_type = fex_config['type']

            if hasattr(tf, fex_type):
                fex = getattr(tf, fex_type)
                features = fex(data_np)  
                features_list.append(features)
            else:
                raise ValueError(f"Unknown time feature extraction type: {fex_type}")
            
        # concatenate all features into a single tensor
        features_np = np.concatenate(features_list, axis=1)  # shape: (batch_size * n_nodes, n_components, n_dims)
        # convert to input format
        features = self.postprocess_output(features_np)

        return features 
             

class FrequencyFeatureExtractor:
    def __init__(self, fex_configs):
        """
        Initialize the feature extractor model with the specified type.
        
        Parameters
        ----------
        fex_config : list
            Type of feature extraction to be used (e.g., 'first_n_modes').
        """
        self.fex_configs = fex_configs
        self.fs = DataConfig().fs  # sampling frequency

    def preprocess_input(self, data):
        """
        Returns
        -------
        freq_mag : np.ndarray, shape (batch_size * n_nodes, n_freq_bins, n_dims)
            Frequency magnitude data.
        freq_bins : np.ndarray, shape (batch_size * n_nodes, n_freq_bins, n_dims)
        
        """
        self.original_shape = data.shape  # save original shape for later reshaping
        self.device = data.device  # save device for later use

        data_np = data.view(-1, data.shape[-2]/2, 2, data.shape[-1]).detach().cpu().numpy()  # (batch_size * n_nodes, n_freq_bins, 2, n_dims)
        
        # seperate frequnecy data and frequency bins
        freq_mag = data_np[:, :, 0, :]
        freq_bins = data_np[:, :, 1, :]

        return freq_mag, freq_bins 
    
    def postprocess_output(self, features):
        """
        Reshape the features to match the original input shape.

        Parameters
        ----------
        features : np.ndarray, shape (batch_size * n_nodes, n_components, n_dims)
            Extracted features.

        Returns
        -------
        reshaped_features : torch.Tensor, shape (batch_size, n_nodes, n_components, n_dims) 
        """
        features = torch.from_numpy(features).to(self.device)
        return features.view(self.original_shape)
    
    def infer_freq_mag(self, freq_mag):
        """
        Parameters
        ----------
        freq_mag : np.ndarray, shape (batch_size * n_nodes, n_freq_bins, n_dims)
            Frequency magnitude data.
        
        Returns
        -------
        freq_amp : np.ndarray, shape (batch_size * n_nodes, n_freq_bins, n_dims)
            Frequency amplitude data.
        freq_psd : np.ndarray, shape (batch_size * n_nodes, n_freq_bins, n_dims)
            Power spectral density data.
        """
        # get frequency amplitude
        freq_amp = ff.get_freq_amp(freq_mag)

        # get psd
        freq_psd_list = [
            ff.get_freq_psd(freq_mag[:, :, dim], self.fs[dim]) for dim in range(len(self.fs))
            ]  
        freq_psd = np.stack(freq_psd_list, axis=-1)

        return freq_amp, freq_psd

    def __call__(self, data):
        """
        Apply the frequency feature extraction to the data.

        Note
        ----
        Data should be in frequency domain

        Parameters
        ----------
        data : torch.Tensor, shape (batch_size, n_nodes, n_freq_bins*2, n_dims)
            Input data in frequency domain, where the seond last dimension contains frequency magnitude and frequency bins.
        
        Returns
        -------
        features : torch.Tensor, shape (batch_size, n_nodes, n_components, n_dims)
            Extracted features based on the specified configurations (if `fex_configs` is not empty).   
        """
        # initialize an empty list to hold features
        features_list = []

        # preprocess input to get frequency magnitude and frequency bins
        freq_mag, freq_bins = self.preprocess_input(data)

        # convert frequency magnitude to frequency amplitude and PSD
        freq_amp, freq_psd = self.infer_freq_mag(freq_mag)  
    
        # loop through each feature extraction configuration
        for fex_config in self.fex_configs:
            fex_type = fex_config['type']
            n_dims = freq_mag.shape[-1]  # number of dimensions

            # first n modes
            if fex_type == 'first_n_modes':
                # assumes each dimension has its own sampling frequency, hence differnt freq_bins
                top_results = [
                    ff.first_n_modes(freq_bins[0, :, dim], freq_psd[:, :, dim], fex_config['n_modes']) for dim in range(n_dims)
                    ]
                top_psd_list, top_freq_list = zip(*top_results) 

                top_psd = np.stack(top_psd_list, axis=-1)  # shape: (batch_size * n_nodes, n_modes, n_dims)
                top_freq = np.stack(top_freq_list, axis=-1)  

                features_list.append(top_psd)
                features_list.append(top_freq)

            # full spectrum features
            elif fex_type == 'full_spectrum':
                for param in fex_config['parameters']:
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

            # 1 dimensional features
            elif hasattr(ff, fex_type):
                fex = getattr(ff, fex_type)
                features_dims = [fex(freq_bins[0, :, dim], freq_amp[:, :, dim]) for dim in range(n_dims)]   

                # stack features for each dimension 
                features = np.stack(features_dims, axis=-1)  # shape: (batch_size * n_nodes, n_components, n_dims)
                features_list.append(features)   

            else:
                raise ValueError(f"Unknown frequency feature extraction type: {fex_config['type']}")

        # concatenate all features into a single tensor
        features_np = np.concatenate(features, axis=1)  # shape: (batch_size * n_nodes, n_components, n_dims)
        # convert to input format
        features = self.postprocess_output(features_np)

        return features       
        
class FeatureReducer:
    def __init__(self, reduc_config):

        if reduc_config['type'] == 'PCA':
            self.model = PCA(n_components=reduc_config['n_components'])

    def preprocess_input(self, data):
        """
        Flatten the input data to 2D and convert to numpy array.

        Returns
        -------
        data_np : np.ndarray, shape (batch_size * n_nodes, n_components * n_dims)
        """
        self.original_shape = data.shape
        self.device = data.device

        data_np = data.view(-1, data.shape[-2] * data.shape[-1]).detach().cpu().numpy()  
        return data_np
    
    def postprocess_output(self, features):
        """
        Reshape the features to match the original input shape.

        Returns
        -------
        reshaped_features : torch.Tensor, shape (batch_size, n_nodes, n_components, n_dims)
        """
        features = torch.from_numpy(features).to(self.device)
        return features.view(self.original_shape)

    def __call__(self, data):
        """
        Apply feature reduction to the data.
        Parameters
        ----------
        data : torch.Tensor, shape (batch_size, n_nodes, n_components, n_dims)
        """
        data_np = self.preprocess_input(data)

        features_np = self.model.fit_transform(data_np)  
        features = self.postprocess_output(features_np)

        return features
        



    

# [TODO] Add rest of the LUCAS's feature extraction functions below