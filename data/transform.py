"""
This module contains 
- pipeline class `DataTransformer` to apply domain transformations and normalization to the data.
- domain transofrm functions and 
- normalization functions.
"""
import sys
import os

# DATA_DIR = os.path.join((os.path.abspath(__file__)))
# sys.path.insert(0, DATA_DIR) if DATA_DIR not in sys.path else None

from scipy.signal import butter, filtfilt
import numpy as np
import torch

# local imports
from .config import DataConfig


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
            The type of domain transformation to be applied (e.g., 'time', 'freq').
        data_config : DataConfig
            Data configuration object containing sampling frequency.
        
        """
        self.domain_config = domain_config
        self.domain = domain_config['type'] 
        self.fs = domain_config['fs']

    def transform(self, time_data):
        """
        Apply the domain transform to the data.
        
        Parameters
        ----------
        time_data : torch.Tensor, shape (batch_size, n_nodes, n_timesteps, n_dims)
            Input data tensor of time signals

        Returns
        -------
        time_data : torch.Tensor, shape (batch_size, n_nodes, n_timesteps, n_dims)
            Filtered time data after applying high-pass filter if specified.
        freq_mag : torch.Tensor, shape (batch_size, n_nodes, n_bins, n_dims)
            Frequency magnitude after applying FFT.
        freq_bins : torch.Tensor, shape (batch_size, n_nodes, n_bins, n_dims)
            Frequency bins corresponding to the frequency magnitude.

        Notes
        -----
        - If `domain` is 'time', only `time_data` is returned.
        - If `domain` is 'freq', both `freq_mag` and `freq_bins` are returned as a tuple.
        
        """

        if self.domain == 'time':
            if self.domain_config['cutoff_freq'] > 0:
                time_data = high_pass_filter(time_data, self.domain_config['cutoff_freq'], self.fs)
            return time_data

        elif self.domain == 'freq':
            if self.domain_config['cutoff_freq'] > 0:
                time_data = high_pass_filter(time_data, self.domain_config['cutoff_freq'], self.fs)
   
            freq_mag, freq_bin = to_freq_domain(time_data, self.fs)
            return freq_mag, freq_bin
        
    
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

    def normalize(self, data):
        """
        Apply normalization to the data.
        
        Parameters
        ----------
        data : torch.Tensor
            Input data tensor of shape (batch_size, n_nodes, n_components, n_dims).
        """
        if self.data_stats is None:
            min_val = data.min(dim=2, keepdim=True).values  
            max_val = data.max(dim=2, keepdim=True).values  # Shape: (n_samples, n_nodes, 1, n_dims)
            data_stats = {
                'mean': torch.mean(data, dim=(0, 2), keepdim=True),
                'std': torch.std(data, dim=(0, 2), keepdim=True),
                'min': torch.min(min_val, dim=0, keepdim=True).values,
                'max': torch.max(max_val, dim=0, keepdim=True).values
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
    data : torch.Tensor, shape (batch_size, n_nodes, n_timesteps, n_dims)
    cutoff_freq : float
        Cutoff frequency for the high-pass filter.
    fs : np.array, shape (n_nodes, n_dims)
        Sampling frequency for each node and dimension

    Returns
    -------
    filtered_data : torch.Tensor, shape (batch_size, n_nodes, n_timesteps, n_dims)
        Filtered data.
    """
    # move to CPU and convert to numpy
    device = data.device
    data_np = data.detach().cpu().numpy()

    batch_size, n_nodes, n_timesteps, n_dims = data_np.shape

    filtered_data = np.zeros_like(data_np)

    # apply filter to each node and dim
    for node in range(n_nodes):
        for dim in range(n_dims):
            b, a = butter(4, cutoff_freq / (0.5 * fs[node, dim]), btype='high')
            
            # apply filter to each node for every sample
            for sample in range(batch_size):
                filtered_data[sample, node, :, dim] = filtfilt(b, a, data_np[sample, node, :, dim])

    # convert back to torch and original device
    return torch.from_numpy(filtered_data).to(device)

def to_freq_domain(data, fs):
    """
    Convert data to frequency domain using FFT.

    Parameters
    ----------
    data : torch.tensor, shape (batch_size, n_nodes, n_timesteps, n_dims)
        Input time data array.

    fs : np.array, shape (n_nodes, n_dims)
        Sampling frequency for each node and dimension

    Returns
    -------
    freq_mag : torch.tensor, shape (batch_size, n_nodes, n_bins, n_dims)
        Real-valued frequency magnitude.
    freq_bins : torch.tensor, shape (batch_size, n_nodes, n_bins, n_dims)
        Frequency bins, broadcasted to match freq_mag.
    """
    # move to CPU and convert to numpy
    device = data.device
    data_np = data.detach().cpu().numpy()

    batch_size, n_nodes, n_timesteps, n_dims = data_np.shape

    freq_mag = []
    bins = []

    # apply FFT to each dimension of each node for every sample
    for node in range(n_nodes):
        freq_mag_node = []
        bins_node = []

        for dim in range(n_dims):
            bins_dim = np.fft.rfftfreq(n_timesteps, d=1/fs[node, dim]) # shape (n_bins,)
            freq_mag_dim = np.abs(np.fft.rfft(data_np[:, node, :, dim], axis=1, norm='forward')) # real valued FFT magnitude, shape (batch_size, n_bins)

            # remove Nyquist frequency if n_timesteps is even
            if n_timesteps % 2 == 0:
                freq_mag_dim = freq_mag_dim[:, :-1]
                bins_dim = bins_dim[:-1]

            # store results for each dimension
            freq_mag_node.append(freq_mag_dim)
            bins_node.append(bins_dim)

        # stack results across dimensions
        freq_mag_node = np.stack(freq_mag_node, axis=-1) # shape (batch_size, n_bins, n_dims)
        bins_node = np.stack(bins_node, axis=-1) # shape (n_bins, n_dims)

        # broadcast bins to match freq_mag_node shape
        bins_node_exp = np.broadcast_to(bins_node, freq_mag_node.shape)

        # store results for current node
        freq_mag.append(freq_mag_node)
        bins.append(bins_node_exp)

    # stack results across nodes
    freq_mag = np.stack(freq_mag, axis=1) # shape (batch_size, n_nodes, n_bins, n_dims)
    bins = np.stack(bins, axis=1)
    
    # convert back to torch and original device
    freq_mag_tensor = torch.from_numpy(freq_mag).to(device)
    freq_bins_tensor = torch.from_numpy(bins).to(device)

    return freq_mag_tensor, freq_bins_tensor

  
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