import os
import sys

ROOT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT_DIR) if ROOT_DIR not in sys.path else None

FEX_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, FEX_DIR) if FEX_DIR not in sys.path else None

# other imports
import numpy as np
import torch

# global imports
from data.prep import DataPreprocessor
from data.transform import DomainTransformer
from data.settings import DataConfig, get_domain_config

# sub-local imports 
from extractor import FrequencyFeatureExtractor
from config.feature_settings import get_freq_feat_config


class DataAdapter:
    def __init__(self):
        self.data_preprocessor = DataPreprocessor(package='fault_detection')
        self.domain_transformer = DomainTransformer(
            get_domain_config('freq')
        )
        self.amp_extractor = FrequencyFeatureExtractor(
            [get_freq_feat_config('full_spectrum', parameters=['amp'])]
        )

    def load_all_data(self):
        """
        Load the time data, frequency amplitude, and frequency bins from the dataset.

        Returns:
        -------
        time_data : np.ndarray, shape (n_samples, n_nodes, n_timesteps, n_dims)
        freq_amp : np.ndarray, shape (n_samples, n_nodes, n_freq_bins, n_dims)
        freq_bins : np.ndarray, shape (n_samples, n_nodes, n_freq_bins, n_dims)
        labels : np.ndarray, shape (n_samples,)
        """
        data_loader = self.data_preprocessor.get_custom_dataloader(self.data_config)
        time_data, labels = self._process_dataloader(data_loader)

        # domain transform
        freq_mag, freq_bins = self.domain_transformer.transform(time_data)
        freq_amp = self.amp_extractor.extract(freq_mag, freq_bins)

        # convert to numpy arrays
        time_data_np = time_data.detach().cpu().numpy()
        freq_amp_np = freq_amp.detach().cpu().numpy()
        freq_bins_np = freq_bins.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()

        return time_data_np, freq_amp_np, freq_bins_np, labels_np
    
    def get_dictionaries(self, data_config:DataConfig):
        """
        Get dictionaries for time data, frequency amplitude, and frequency bins.

        Returns
        -------
            Calc_0 : dictionary
                Contains time data for healthy and unhealthy samples.
            Calc_FFT : dictionary
                Contains frequency amplitude data for healthy and unhealthy samples.
            fftFreq : dictionary
                Contains frequency bins for healthy and unhealthy samples.
        """
        self.data_config = data_config

        # load all data
        time_data, freq_amp, freq_bins, labels = self.load_all_data()

        # reshape time, freq_amp, freq_bins to (n_samples * n_nodes, n_components * n_dims)
        time_data = time_data.reshape(-1, time_data.shape[-2] * time_data.shape[-1])
        freq_amp = freq_amp.reshape(-1, freq_amp.shape[-2] * freq_amp.shape[-1])
        freq_bins = freq_bins.reshape(-1, freq_bins.shape[-2] * freq_bins.shape[-1])

        # create dictionaries
        Calc_0, Calc_FFT, fftFreq = self._optimize_data_for_ranking(
            time_data, freq_amp, freq_bins, labels)
        
        return Calc_0, Calc_FFT, fftFreq


    def _process_dataloader(self, data_loader):
        """
        Converts a dataloader to stacked tensors.
        Parameters:
        ----------
        data_loader : torch.utils.data.Dataset
            Dataset containing time data and labels.

        Returns:
        -------
        data_tensor : torch.tensor, shape (n_samples, n_nodes, n_timesteps, n_dims)
        label_tensor : torch.tensor, shape (n_samples,)
        """
        data_list = []
        label_list = []

        for data, label in data_loader:
            data_list.append(data)
            label_list.append(label)

        # Stack into arrays
        data_tensor = torch.cat(data_list, axis=0)   # shape: (n_samples, n_nodes, n_components, n_dims)
        label_tensor = torch.cat(label_list, dim=0).squeeze()

        print("Data shape:", data_tensor.shape)
        print("Labels shape:", label_tensor.shape)

        return data_tensor, label_tensor


    def _optimize_data_for_ranking(self, time_data, freq_amp, freq_bins, labels):
        """
        Convert the time data, frequency amplitude, and frequency bins into dictionaries for ranking.
        
        Parameters
        ----------
        time_data : tensor, shape (n_samples, n_timesteps)
        freq_amp : tensor, shape (n_samples, n_freq_bins)
        freq_bins : tensor, shape (n_samples, n_freq_bins)
        labels : tensor, shape (n_samples,) 
            with 1 for healthy, -1 for unhealthy

        Returns
        -------
            Calc_0 : dictionary
                Contains time data for healthy and unhealthy samples.
            Calc_FFT : dictionary
                Contains frequency amplitude data for healthy and unhealthy samples.
            fftFreq : dictionary
                Contains frequency bins for healthy and unhealthy samples.
        """
        
        Calc_0 = {}
        Calc_FFT = {}
        fftFreq = {}
        
        # Separate healthy and unhealthy samples
        healthy_indices = np.where(labels == 0)[0]
        unhealthy_indices = np.where(labels == 1)[0]
        
        # Process healthy samples
        for i, idx in enumerate(healthy_indices):
            key = f'N_{i}'

            time_sample = time_data[idx, :].flatten()
            Calc_0[key] = time_sample

            freq_amp_sample = freq_amp[idx, :].flatten()
            Calc_FFT[f'FFT ({key})'] = freq_amp_sample

            freq_bin_sample = freq_bins[idx, :].flatten()
            fftFreq[f'FFT ({key})'] = freq_bin_sample
        
        # Process unhealthy samples
        for i, idx in enumerate(unhealthy_indices):
            key = f'U_{i}'

            time_sample = time_data[idx, :].flatten()
            Calc_0[key] = time_sample

            freq_amp_sample = freq_amp[idx, :].flatten()
            Calc_FFT[f'FFT ({key})'] = freq_amp_sample

            freq_bin_sample = freq_bins[idx, :].flatten()
            fftFreq[f'FFT ({key})'] = freq_bin_sample
            
        
        return Calc_0, Calc_FFT, fftFreq