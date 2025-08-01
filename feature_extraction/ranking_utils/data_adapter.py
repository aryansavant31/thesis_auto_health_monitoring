import numpy as np
import torch
from data.prep import DataPreprocessor
from data.transform import DomainTransformer
from data.config import get_domain_config
from feature_extraction.extractor import FrequencyFeatureExtractor
from feature_extraction.config.feature_settings import get_freq_feat_config
from data.config import DataConfig

class DataAdapter:
    def __init__(self):
        self.data_preprocessor = DataPreprocessor(package='fault_detection')
        self.domain_transformer = DomainTransformer(
            get_domain_config('freq')
        )

        self.amp_extractor = FrequencyFeatureExtractor(
            [get_freq_feat_config('full_spectrum', parameters=['amp'])]
        )
        self.freq_extractor = FrequencyFeatureExtractor(
            [get_freq_feat_config('full_spectrum', parameters=['freq'])]
        )

    def load_all_data(self, data_config:DataConfig, run_type='train', amt_of_data=1):
        """
        Load the time data, frequency amplitude, and frequency bins from the dataset.

        Returns:
        -------
        time_data : np.ndarray, shape (n_samples, n_nodes, n_timesteps, n_dims)
        freq_amp : np.ndarray, shape (n_samples, n_nodes, n_freq_bins, n_dims)
        freq_bins : np.ndarray, shape (n_samples, n_nodes, n_freq_bins, n_dims)
        labels : np.ndarray, shape (n_samples,)
        """
        time_dataset = self.data_preprocessor.load_dataset(data_config, run_type)
        time_data, labels = self._process_time_dataset(time_dataset, amt_of_data)

        # domain transform
        freq_data = self.domain_transformer.transform(time_data)

        freq_amp = self.amp_extractor.extract(freq_data)
        freq_bins = self.freq_extractor.extract(freq_data)

        # convert to numpy arrays
        time_data_np = time_data.detach().cpu().numpy()
        freq_amp_np = freq_amp.detach().cpu().numpy()
        freq_bins_np = freq_bins.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()

        return time_data_np, freq_amp_np, freq_bins_np, labels_np
    
    def get_dictionaries(self, data_config:DataConfig, run_type='train'):
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
        time_data, freq_amp, freq_bins, labels = self.load_all_data(data_config, run_type)

        # reshape time, freq_amp, freq_bins to (n_samples * n_nodes, n_components * n_dims)
        time_data = time_data.reshape(-1, time_data.shape[-2] * time_data.shape[-1])
        freq_amp = freq_amp.reshape(-1, freq_amp.shape[-2] * freq_amp.shape[-1])
        freq_bins = freq_bins.reshape(-1, freq_bins.shape[-2] * freq_bins.shape[-1])

        # create dictionaries
        Calc_0, Calc_FFT, fftFreq = self._optimize_data_for_ranking(
            time_data, freq_amp, freq_bins, labels)
        
        return Calc_0, Calc_FFT, fftFreq


    def _process_time_dataset(self, time_dataset, amt_of_data=1):
        """
        Converts a TensorDataset to stacked tensors.
        Parameters:
        ----------
        time_dataset : torch.utils.data.Dataset
            Dataset containing time data and labels.
        amt_of_data : int, optional
            Amount of data to process. Default is 1, which processes all data.

        Returns:
        -------
        data_tensor : torch.tensor, shape (n_samples, n_nodes, n_timesteps, n_dims)
        label_tensor : torch.tensor, shape (n_samples,)
        """
        data_list = []
        label_list = []

        if amt_of_data > 1:
            raise ValueError("'amt_of_data' cannot be greater than 1")
        
        n_samples = int(len(time_dataset) * amt_of_data)
        subset = [time_dataset[i] for i in range(n_samples)]

        for sample in subset:
            data, label = sample
            data_list.append(data)
            label_list.append(label)  # If label is a scalar tensor

        # Stack into arrays
        data_tensor = torch.stack(data_list, axis=0)   # shape: (n_samples, n_nodes, n_components, n_dims)
        label_tensor = torch.stack(label_list, dim=0).squeeze()

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