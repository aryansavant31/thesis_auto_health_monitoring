import numpy as np
import torch

def get_freq_amp(freq_mag):
    """
    Converts frequency magnitude to frequency amplitude.

    Note
    ----
    The `freq_mag` is assumed to be half sided spectrum i.e. the number of bins are _(N/2)_, where **N** is number of timesteps in the time domain signal.
    
    Parameters
    ----------
    freq_mag : torch.tensor, shape (batch_size, n_nodes, n_bins, n_dims)
        Frequency magnitude data.

    Returns
    -------
    freq_amp : torch.tensor, shape (batch_size, n_nodes, n_bins, n_dims)
    """
    n_bins = freq_mag.shape[2]
    amplitude = torch.abs(freq_mag) / n_bins  # normalize by number of frequency bins
    return amplitude

def get_freq_psd(freq_mag, fs):
    """
    Converts frequency magnitude to power spectral density (PSD).

    Note
    ----
    The `freq_mag` is assumed to be half sided spectrum i.e. the number of bins are _(N/2)_, where **N** is number of timesteps in the time domain signal.
    
    Parameters
    ----------
    freq_mag : torch.tensor, shape (batch_size, n_nodes, n_bins, n_dims)
        Frequency magnitude data.
    fs : list
        Sampling frequency list of the data (length should match number of dimensions).
    
    Returns
    -------
    psd: torch.tensor, shape (batch_size, n_nodes, n_bins, n_dims)
        Power spectral density data
    """
    n_bins = freq_mag.shape[2]
    fs_tensor = torch.tensor(fs, dtype=freq_mag.dtype, device=freq_mag.device).view(1, 1, 1, -1)

    psd = (torch.abs(freq_mag) ** 2) / (n_bins * fs_tensor)  # normalize by number of frequency bins and sampling frequency
    return psd

def first_n_modes(freq_bins, psd, n_modes):
    """
    Extracts the first n modes of each sample from the power spectral density (PSD) data

    Parameters
    ----------
    freq_bins : torch.tensor, shape (batch_size, n_nodes, n_bins, n_dims)
        Frequency bins corresponding to the PSD data.
    psd : torch.tensor, shape (batch_size, n_nodes, n_bins, n_dims)
        Power spectral density data.
    n_modes : int
        Number of modes to extract.

    Returns
    -------
    top_psd : torch.tensor, shape (batch_size, n_nodes, n_modes, n_dims)
        The top n modes of the PSD data for each sample
    top_freqs : torch.tensor, shape (batch_size, n_nodes, n_modes, n_dims)
        The frequencies corresponding to the top n modes.
    
    """
    top_psd, top_idx = torch.topk(psd, n_modes, dim=2, largest=True, sorted=True)
    top_freq_bins = torch.gather(freq_bins, dim=2, index=top_idx) 

    return top_psd, top_freq_bins