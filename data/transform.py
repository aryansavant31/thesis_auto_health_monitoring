"""
This module contains 
- pipeline class `DataTransformer` to apply domain transformations and normalization to the data.
- domain transofrm functions and 
- normalization functions.
"""

from scipy.signal import butter, filtfilt
import numpy as np
from .config import DataConfig
import torch

class DomainTransformer:
    """
    Domain transform (time, freq)
    |
    v
    Normalization (std, min-max)
    """
    def __init__(self, domain_config):
        """
        Parameters
        ----------
        domain_config : str
        
        """
        self.fs = DataConfig().fs
        self.domain_config = domain_config
        self.domain = domain_config['type'] 

    def preprocess_input(self, data):
        """
        Converts data to numpy array and reshapes it to (batch_size * n_nodes, n_timesteps, n_dims).
        """
        self.original_shape = data.shape
        self.device = data.device

        return data.view(-1, data.shape[-2], data.shape[-1]).detach().cpu().numpy() 
    
    def postprocess_output(self, data):
        """
        Converts data back to original shape and moves it to the original device.
        """
        data = torch.from_numpy(data).to(self.device)
        return data.view(self.original_shape[0], self.original_shape[1], data.shape[1], self.original_shape[-1])  # (batch_size, n_nodes, n_components, n_dims)
    
    def postprocess_freq_output(self, freq_data, freq_bins):
        """
        - Flattens frequency data and frequency bins along freq. axis
        - Converts it back to original shape and moves it to the original device.
        """
        freq_bins_expanded = np.tile(freq_bins[None, :, :], (freq_data.shape[0], 1, 1))  # (batch_size * n_nodes, n_freq_bins, n_dims)
        data_np = np.concatenate([freq_data, freq_bins_expanded], axis=1)  # (batch_size * n_nodes, 2*n_freq_bins, n_dims)

        # convert to tensor and reshape back to original shape
        return self.postprocess_output(data_np)
    
    def transform(self, data):
        """
        Apply the domain transform to the data.
        
        Parameters
        ----------
        data : torch.Tensor
            Input data tensor of shape (batch_size, n_nodes, n_timesteps, n_dims).

        Returns
        -------
        data : torch.Tensor
            Transformed data tensor of shape (batch_size, n_nodes, n_components, n_dims).
        """
        # reshape data to (batch_size * n_nodes, n_timesteps, n_dims)
        data_np = self.preprocess_input(data)

        if self.domain == 'time':
            if self.domain_config['cutoff_freq'] > 0:
                data_np_list = [high_pass_filter(data_np[:, :, dim], self.domain_config['cutoff_freq'], self.fs[dim]) for dim in range(len(self.fs))]
                data_np = np.stack(data_np_list, axis=-1)  # shape: (batch_size * n_nodes, n_timesteps, n_dims)

            data = self.postprocess_output(data_np)

        elif self.domain == 'freq':
            if self.domain_config['cutoff_freq'] > 0:
                data_np_list = [high_pass_filter(data_np[:, :, dim], self.domain_config['cutoff_freq'], self.fs[dim]) for dim in range(len(self.fs))] 
                data_np = np.stack(data_np_list, axis=-1)

            # convert to frequency domain for each dimension (assumes each dimensnion has its own sampling frequency)
            freq_results = [
                to_freq_domain(data_np[:, :, dim], self.fs[dim]) for dim in range(len(self.fs))
                ]
            freq_data_list, freq_bins_list = zip(*freq_results)

            freq_data = np.stack(freq_data_list, axis=-1)  # (batch_size * n_nodes, n_freq_bins, n_dims)
            freq_bins = np.stack(freq_bins_list, axis=-1) 

            data = self.postprocess_freq_output(freq_data, freq_bins)
        
        return data
    
class DataNormalizer:
    """
    Normalize data based on the specified normalization type.
    """
    def __init__(self, norm_type, data_stats=None):
        """
        Parameters
        ----------
        norm_type : str
            The type of normalization to be applied (e.g., 'std', 'min_max').
        data_stats : dict
            Statistics for normalization (e.g., mean, std, min, max).
        """
        self.norm_type = norm_type
        self.data_stats = data_stats

    def __call__(self, data):
        """
        Apply normalization to the data.
        
        Parameters
        ----------
        data : torch.Tensor
            Input data tensor of shape (batch_size, n_nodes, n_components, n_dims).
        """
        if self.data_stats is None:
            data_stats = {
                'mean': torch.mean(data, dim=(0, 2), keepdim=True),
                'std': torch.std(data, dim=(0, 2), keepdim=True),
                'min': torch.min(data, dim=(0, 2), keepdim=True).values,
                'max': torch.max(data, dim=(0, 2), keepdim=True).values
            }
            for k in data_stats:
                data_stats[k] = data_stats[k].squeeze(0)

        if self.norm_type == 'min_max':
            data = min_max_normalize(data, self.data_stats['min'], self.data_stats['max'])

        elif self.norm_type == 'std':
            data = std_normalize(data, self.data_stats['mean'], self.data_stats['std'])

        else:
            raise ValueError(f"Unknown normalization type: {self.norm_type}")

        return data 
    
def high_pass_filter(data, cutoff_freq, fs):
        """
        Apply a high-pass Butterworth filter to the data.

        Parameters
        ----------
        data : np.ndarray, shape (batch_size * n_nodes, n_timesteps)
        cutoff_freq : float
            Cutoff frequency for the high-pass filter.
        fs : float
            Sampling frequency of the data.

        Returns
        -------
        filtered_data : np.ndarray, shape (batch_size * n_nodes, n_timesteps)
            Filtered data.
        """
        b, a = butter(4, cutoff_freq / (0.5 * fs), btype='high')  

        # initialize output array
        filtered_data = np.zeros_like(data)

        # apply filter to each sample and its dimensions
        for i in range(data.shape[0]):
            filtered_data[i, :] = filtfilt(b, a, data[i, :])

        return filtered_data

def to_freq_domain(data, fs):
    """
    Convert data to frequency domain using FFT.

    Parameters
    ----------
    data : np.ndarray, shape (batch_size * n_nodes, n_timesteps, n_dims)
        Input data array.
    fs : float
        Sampling frequency of the data.

    Returns
    -------
    freq_data : np.ndarray, shape (batch_size * n_nodes, n_freq_bins, n_dims)
        Frequency domain representation of the data.
    freq_bins : np.ndarray, shape (n_freq_bins,)
    """
    n_comps = data.shape[1]  # numper of components/samples
    
    freq_bins = np.fft.rfftfreq(n_comps, 1/fs)  
    freq_data = np.fft.rfft(data, axis=1)  # FFT along the time dimension

    # remove Nyquist frequency if n_comps is even
    if n_comps % 2 == 0:
        freq_bins = freq_bins[:-1]
        freq_data = freq_data[:, :-1]

    return freq_data, freq_bins

        
def min_max_normalize(data, min, max):
    """
    Normalize the data based on min and max values along the components axis (axis=2)

    Parameters
    ----------
    data : torch.Tensor, shape (batch_size, n_nodes, n_components, n_dims)
        Input data tensor
    max : torch.Tensor, shape (n_nodes, 1, n_dims)
    min : torch.Tensor, shape (n_nodes, 1, n_dims)
        Min and max values for normalization

    """
    norm_data = (data - min) / (max - min + 1e-8)
    return norm_data

def std_normalize(data, mean, std):
    """
    Normalize the data based on mean and standard deviation along the components axis (axis=2)

    Parameters
    ----------
    data : torch.Tensor, shape (batch_size, n_nodes, n_components, n_dims)
        Input data tensor
    mean : torch.Tensor, shape (n_nodes, 1, n_dims)  
    std : torch.Tensor, shape (n_nodes, 1, n_dims)
        Mean and standard deviation for normalization 
    """
    norm_data = (data - mean) / (std + 1e-8)
    return norm_data