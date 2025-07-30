import numpy as np

def get_freq_amp(freq_mag):
    """
    Converts frequency magnitude to frequency amplitude.

    Note
    ----
    The `freq_mag` is assumed to be half sided spectrum i.e. the number of bins are _(N/2)_, where **N** is number of timesteps in the time domain signal.
    
    Parameters
    ----------
    freq_mag : np.ndarray, shape (n_samples, n_freq_bins, n_dims)
        Frequency magnitude data.
    """
    amplitude = np.abs(freq_mag) / freq_mag.shape[1]  # normalize by number of frequency bins
    return amplitude

def get_freq_psd(freq_mag, fs):
    """
    Converts frequency magnitude to power spectral density (PSD).

    Note
    ----
    The `freq_mag` is assumed to be half sided spectrum i.e. the number of bins are _(N/2)_, where **N** is number of timesteps in the time domain signal.
    
    Parameters
    ----------
    freq_mag : np.ndarray, shape (n_samples, n_freq_bins)
        Frequency magnitude data.
    fs : float
        Sampling frequency.
    
    Returns
    -------
    psd: np.ndarray, shape (n_samples, n_freq_bins)
        Power spectral density data
    """
    psd = (np.abs(freq_mag) ** 2) / (freq_mag.shape[1] * fs)  # normalize by number of frequency bins and sampling frequency
    return psd

def first_n_modes(freq_bins, psd, n_modes):
    """
    Extracts the first n modes of each sample from the power spectral density (PSD) data

    Parameters
    ----------
    freq_bins : np.ndarray, shape (n_freq_bins,)
        Frequency bins corresponding to the PSD data.
    psd : np.ndarray, shape (n_samples, n_freq_bins)
        Power spectral density data.
    n_modes : int
        Number of modes to extract.

    Returns
    -------
    top_psd : np.ndarray, shape (n_samples, n_modes)
        The top n modes of the PSD data for each sample
    top_freqs : np.ndarray, shape (n_samples, n_modes)
        The frequencies corresponding to the top n modes.
    
    """
    n_samples, n_freq_bins = psd.shape
    top_psd = np.zeros((n_samples, n_modes))
    top_freqs = np.zeros((n_samples, n_modes))

    for i in range(n_samples):
        top_idx = np.argsort(psd[i, :])[-n_modes:] # indices of the n largest values (in ascending order)
        top_psd[i, :] = psd[i, top_idx]
        top_freqs[i, :] = freq_bins[top_idx]

    return top_psd, top_freqs