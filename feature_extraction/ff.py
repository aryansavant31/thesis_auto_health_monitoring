import numpy as np
import torch
from scipy.stats import kurtosis
from scipy.signal import find_peaks
from scipy.special import erfinv

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


# lucas features

def meanF(amplitudes):
    """
    Calculate mean of frequency amplitudes.
    
    Parameters
    ----------
    amplitudes : torch.Tensor
        Input tensor with shape (batch_size, n_nodes, n_bins, n_dims)
        
    Returns
    -------
    torch.Tensor
        Mean values with shape (batch_size, n_nodes, 1, n_dims)
    """
    return torch.mean(amplitudes, dim=2, keepdim=True)


def varianceF(amplitudes):
    """
    Calculate variance of frequency amplitudes.
    
    Parameters
    ----------
    amplitudes : torch.Tensor
        Input tensor with shape (batch_size, n_nodes, n_bins, n_dims)
        
    Returns
    -------
    torch.Tensor
        Variance values with shape (batch_size, n_nodes, 1, n_dims)
    """
    return torch.var(amplitudes, dim=2, keepdim=True, unbiased=False)


def skewnessF(amplitudes):
    """
    Calculate skewness of frequency amplitudes.
    
    Parameters
    ----------
    amplitudes : torch.Tensor
        Input tensor with shape (batch_size, n_nodes, n_bins, n_dims)
        
    Returns
    -------
    torch.Tensor
        Skewness values with shape (batch_size, n_nodes, 1, n_dims)
    """
    mean_val = meanF(amplitudes)
    var_val = varianceF(amplitudes)
    
    centered = amplitudes - mean_val
    third_moment = torch.mean(centered**3, dim=2, keepdim=True)
    
    # avoid division by zero
    var_val = torch.clamp(var_val, min=1e-8)
    skew = third_moment / (var_val**(3/2))
    
    return skew


def kurtosisF(amplitudes):
    """
    Calculate kurtosis of frequency amplitudes.
    
    Parameters
    ----------
    amplitudes : torch.Tensor
        Input tensor with shape (batch_size, n_nodes, n_bins, n_dims)
        
    Returns
    -------
    torch.Tensor
        Kurtosis values with shape (batch_size, n_nodes, 1, n_dims)
    """
    mean_val = meanF(amplitudes)
    var_val = varianceF(amplitudes)
    
    centered = amplitudes - mean_val
    fourth_moment = torch.mean(centered**4, dim=2, keepdim=True)
    
    # Avoid division by zero
    var_val = torch.clamp(var_val, min=1e-8)
    kurt = fourth_moment / (var_val**2) - 3
    
    return kurt


def central_freq(freq, amplitudes):
    """
    Calculate central frequency.
    
    Parameters
    ----------
    freq : torch.Tensor
        Frequency tensor with shape (batch_size, n_nodes, n_bins, n_dims)
    amplitudes : torch.Tensor
        Amplitude tensor with shape (batch_size, n_nodes, n_bins, n_dims)
        
    Returns
    -------
    torch.Tensor
        Central frequency with shape (batch_size, n_nodes, 1, n_dims)
    """
    numerator = torch.sum(freq * amplitudes, dim=2, keepdim=True)
    denominator = torch.sum(amplitudes, dim=2, keepdim=True)
    
    # Avoid division by zero
    denominator = torch.clamp(denominator, min=1e-8)
    
    return numerator / denominator


def stdF(freq, amplitudes):
    """
    Calculate standard deviation of frequency.
    
    Parameters
    ----------
    freq : torch.Tensor
        Frequency tensor with shape (batch_size, n_nodes, n_bins, n_dims)
    amplitudes : torch.Tensor
        Amplitude tensor with shape (batch_size, n_nodes, n_bins, n_dims)
        
    Returns
    -------
    torch.Tensor
        Standard deviation with shape (batch_size, n_nodes, 1, n_dims)
    """
    central_f = central_freq(freq, amplitudes)
    numerator = torch.sum((freq - central_f)**2 * amplitudes, dim=2, keepdim=True)
    denominator = torch.sum(amplitudes, dim=2, keepdim=True)
    
    # Avoid division by zero
    denominator = torch.clamp(denominator, min=1e-8)
    
    return torch.sqrt(numerator / denominator)


def rmsF(freq, amplitudes):
    """
    Calculate root mean square frequency.
    
    Parameters
    ----------
    freq : torch.Tensor
        Frequency tensor with shape (batch_size, n_nodes, n_bins, n_dims)
    amplitudes : torch.Tensor
        Amplitude tensor with shape (batch_size, n_nodes, n_bins, n_dims)
        
    Returns
    -------
    torch.Tensor
        RMS frequency with shape (batch_size, n_nodes, 1, n_dims)
    """
    numerator = torch.sum(freq**2 * amplitudes, dim=2, keepdim=True)
    denominator = torch.sum(amplitudes, dim=2, keepdim=True)
    
    # Avoid division by zero
    denominator = torch.clamp(denominator, min=1e-8)
    
    return torch.sqrt(numerator / denominator)


def spectral_spread(freq, amplitudes):
    """
    Calculate spectral spread.
    
    Parameters
    ----------
    freq : torch.Tensor
        Frequency tensor with shape (batch_size, n_nodes, n_bins, n_dims)
    amplitudes : torch.Tensor
        Amplitude tensor with shape (batch_size, n_nodes, n_bins, n_dims)
        
    Returns
    -------
    torch.Tensor
        Spectral spread with shape (batch_size, n_nodes, 1, n_dims)
    """
    central_f = central_freq(freq, amplitudes)
    numerator = torch.sum((freq - central_f)**2 * amplitudes, dim=2, keepdim=True)
    denominator = torch.sum(amplitudes, dim=2, keepdim=True)
    
    # Avoid division by zero
    denominator = torch.clamp(denominator, min=1e-8)
    
    return torch.sqrt(numerator / denominator)


def spectral_entropy(amplitudes):
    """
    Calculate spectral entropy.
    
    Parameters
    ----------
    amplitudes : torch.Tensor
        Amplitude tensor with shape (batch_size, n_nodes, n_bins, n_dims)
        
    Returns
    -------
    torch.Tensor
        Spectral entropy with shape (batch_size, n_nodes, 1, n_dims)
    """
    total_power = torch.sum(amplitudes, dim=2, keepdim=True)
    total_power = torch.clamp(total_power, min=1e-8)
    
    prob = amplitudes / total_power
    prob = torch.clamp(prob, min=1e-8)
    
    entropy = -torch.sum(prob * torch.log2(prob), dim=2, keepdim=True)
    
    return entropy


def kp_value_batch(power, alpha=0.01):
    """
    Find peaks for batch processing.
    
    Parameters
    ----------
    power : torch.Tensor
        Power tensor with shape (batch_size, n_nodes, n_bins, n_dims)
    alpha : float
        Error level for threshold calculation
        
    Returns
    -------
    list
        List of peak indices for each batch/node/dim combination
    """
    threshold = 2 * (erfinv(1 - alpha)) ** 2
    batch_size, n_nodes, n_bins, n_dims = power.shape
    
    peak_indices = []
    for b in range(batch_size):
        batch_peaks = []
        for n in range(n_nodes):
            node_peaks = []
            for d in range(n_dims):
                signal = power[b, n, :, d].cpu().numpy()
                peaks, _ = find_peaks(signal, height=threshold)
                node_peaks.append(peaks)
            batch_peaks.append(node_peaks)
        peak_indices.append(batch_peaks)
    
    return peak_indices


def total_power(amplitudes):
    """
    Calculate total power using peak detection.
    
    Parameters
    ----------
    amplitudes : torch.Tensor
        Amplitude tensor with shape (batch_size, n_nodes, n_bins, n_dims)
        
    Returns
    -------
    torch.Tensor
        Total power with shape (batch_size, n_nodes, 1, n_dims)
    """
    power = 2 * (amplitudes ** 2)
    peak_indices = kp_value_batch(power)
    
    batch_size, n_nodes, n_bins, n_dims = power.shape
    result = torch.zeros(batch_size, n_nodes, 1, n_dims, device=power.device)
    
    for b in range(batch_size):
        for n in range(n_nodes):
            for d in range(n_dims):
                peaks = peak_indices[b][n][d]
                if len(peaks) > 0:
                    result[b, n, 0, d] = torch.sum(power[b, n, peaks, d])
    
    return result


def median_freq(freq, amplitudes):
    """
    Calculate median frequency using peak detection.
    
    Parameters
    ----------
    freq : torch.Tensor
        Frequency tensor with shape (batch_size, n_nodes, n_bins, n_dims)
    amplitudes : torch.Tensor
        Amplitude tensor with shape (batch_size, n_nodes, n_bins, n_dims)
        
    Returns
    -------
    torch.Tensor
        Median frequency with shape (batch_size, n_nodes, 1, n_dims)
    """
    power = 2 * (amplitudes ** 2)
    peak_indices = kp_value_batch(power)
    total_pow = total_power(amplitudes)
    
    batch_size, n_nodes, n_bins, n_dims = power.shape
    result = torch.zeros(batch_size, n_nodes, 1, n_dims, device=freq.device)
    
    for b in range(batch_size):
        for n in range(n_nodes):
            for d in range(n_dims):
                peaks = peak_indices[b][n][d]
                target = total_pow[b, n, 0, d] / 2
                
                cumsum = 0
                for peak_idx in peaks:
                    cumsum += power[b, n, peak_idx, d]
                    if cumsum >= target:
                        result[b, n, 0, d] = freq[b, n, peak_idx, d]
                        break
    
    return result


def pkF(amplitudes):
    """
    Calculate peak frequency.
    
    Parameters
    ----------
    amplitudes : torch.Tensor
        Amplitude tensor with shape (batch_size, n_nodes, n_bins, n_dims)
        
    Returns
    -------
    torch.Tensor
        Peak frequency with shape (batch_size, n_nodes, 1, n_dims)
    """
    power = amplitudes ** 2
    return torch.max(power, dim=2, keepdim=True)[0]


def first_spectral_moment(freq, amplitudes):
    """
    Calculate first spectral moment using peak detection.
    
    Parameters
    ----------
    freq : torch.Tensor
        Frequency tensor with shape (batch_size, n_nodes, n_bins, n_dims)
    amplitudes : torch.Tensor
        Amplitude tensor with shape (batch_size, n_nodes, n_bins, n_dims)
        
    Returns
    -------
    torch.Tensor
        First spectral moment with shape (batch_size, n_nodes, 1, n_dims)
    """
    power = 2 * (amplitudes ** 2)
    peak_indices = kp_value_batch(power)
    total_pow = total_power(amplitudes)
    
    batch_size, n_nodes, n_bins, n_dims = power.shape
    result = torch.zeros(batch_size, n_nodes, 1, n_dims, device=freq.device)
    
    for b in range(batch_size):
        for n in range(n_nodes):
            for d in range(n_dims):
                peaks = peak_indices[b][n][d]
                if len(peaks) > 0 and total_pow[b, n, 0, d] > 0:
                    moment = torch.sum(freq[b, n, peaks, d] * power[b, n, peaks, d])
                    result[b, n, 0, d] = moment / total_pow[b, n, 0, d]
    
    return result


def second_spectral_moment(freq, amplitudes):
    """
    Calculate second spectral moment using peak detection.
    
    Parameters
    ----------
    freq : torch.Tensor
        Frequency tensor with shape (batch_size, n_nodes, n_bins, n_dims)
    amplitudes : torch.Tensor
        Amplitude tensor with shape (batch_size, n_nodes, n_bins, n_dims)
        
    Returns
    -------
    torch.Tensor
        Second spectral moment with shape (batch_size, n_nodes, 1, n_dims)
    """
    power = 2 * (amplitudes ** 2)
    peak_indices = kp_value_batch(power)
    total_pow = total_power(amplitudes)
    
    batch_size, n_nodes, n_bins, n_dims = power.shape
    result = torch.zeros(batch_size, n_nodes, 1, n_dims, device=freq.device)
    
    for b in range(batch_size):
        for n in range(n_nodes):
            for d in range(n_dims):
                peaks = peak_indices[b][n][d]
                if len(peaks) > 0 and total_pow[b, n, 0, d] > 0:
                    moment = torch.sum((freq[b, n, peaks, d] ** 2) * power[b, n, peaks, d])
                    result[b, n, 0, d] = moment / total_pow[b, n, 0, d]
    
    return result


def third_spectral_moment(freq, amplitudes):
    """
    Calculate third spectral moment using peak detection.
    
    Parameters
    ----------
    freq : torch.Tensor
        Frequency tensor with shape (batch_size, n_nodes, n_bins, n_dims)
    amplitudes : torch.Tensor
        Amplitude tensor with shape (batch_size, n_nodes, n_bins, n_dims)
        
    Returns
    -------
    torch.Tensor
        Third spectral moment with shape (batch_size, n_nodes, 1, n_dims)
    """
    power = 2 * (amplitudes ** 2)
    peak_indices = kp_value_batch(power)
    total_pow = total_power(amplitudes)
    
    batch_size, n_nodes, n_bins, n_dims = power.shape
    result = torch.zeros(batch_size, n_nodes, 1, n_dims, device=freq.device)
    
    for b in range(batch_size):
        for n in range(n_nodes):
            for d in range(n_dims):
                peaks = peak_indices[b][n][d]
                if len(peaks) > 0 and total_pow[b, n, 0, d] > 0:
                    moment = torch.sum((freq[b, n, peaks, d] ** 3) * power[b, n, peaks, d])
                    result[b, n, 0, d] = moment / total_pow[b, n, 0, d]
    
    return result


def fourth_spectral_moment(freq, amplitudes):
    """
    Calculate fourth spectral moment using peak detection.
    
    Parameters
    ----------
    freq : torch.Tensor
        Frequency tensor with shape (batch_size, n_nodes, n_bins, n_dims)
    amplitudes : torch.Tensor
        Amplitude tensor with shape (batch_size, n_nodes, n_bins, n_dims)
        
    Returns
    -------
    torch.Tensor
        Fourth spectral moment with shape (batch_size, n_nodes, 1, n_dims)
    """
    power = 2 * (amplitudes ** 2)
    peak_indices = kp_value_batch(power)
    total_pow = total_power(amplitudes)
    
    batch_size, n_nodes, n_bins, n_dims = power.shape
    result = torch.zeros(batch_size, n_nodes, 1, n_dims, device=freq.device)
    
    for b in range(batch_size):
        for n in range(n_nodes):
            for d in range(n_dims):
                peaks = peak_indices[b][n][d]
                if len(peaks) > 0 and total_pow[b, n, 0, d] > 0:
                    moment = torch.sum((freq[b, n, peaks, d] ** 4) * power[b, n, peaks, d])
                    result[b, n, 0, d] = moment / total_pow[b, n, 0, d]
    
    return result


def vcf(freq, amplitudes):
    """
    Calculate variance of central frequency.
    
    Parameters
    ----------
    freq : torch.Tensor
        Frequency tensor with shape (batch_size, n_nodes, n_bins, n_dims)
    amplitudes : torch.Tensor
        Amplitude tensor with shape (batch_size, n_nodes, n_bins, n_dims)
        
    Returns
    -------
    torch.Tensor
        VCF with shape (batch_size, n_nodes, 1, n_dims)
    """
    total_pow = total_power(amplitudes)
    
    # Handle zero power case
    mask = total_pow > 0
    result = torch.zeros_like(total_pow)
    
    a1 = second_spectral_moment(freq, amplitudes) / torch.clamp(total_pow, min=1e-8)
    a2 = first_spectral_moment(freq, amplitudes) / torch.clamp(total_pow, min=1e-8)
    a2_squared = a2 ** 2
    
    result = torch.where(mask, a1 - a2_squared, torch.zeros_like(a1))
    
    return result


def frequency_ratio(amplitudes):
    """
    Calculate frequency ratio (low to high frequency power).
    
    Parameters
    ----------
    amplitudes : torch.Tensor
        Amplitude tensor with shape (batch_size, n_nodes, n_bins, n_dims)
        
    Returns
    -------
    torch.Tensor
        Frequency ratio with shape (batch_size, n_nodes, 1, n_dims)
    """
    power = 2 * (amplitudes ** 2)
    n_bins = power.shape[2]
    mid_point = n_bins // 2
    
    low_freq_power = torch.sum(power[:, :, :mid_point, :], dim=2, keepdim=True)
    high_freq_power = torch.sum(power[:, :, mid_point+1:, :], dim=2, keepdim=True)
    
    # Handle division by zero
    high_freq_power = torch.clamp(high_freq_power, min=1e-8)
    
    return low_freq_power / high_freq_power


def hsc(freq, amplitudes):
    """
    Calculate harmonic spectral centroid using peak detection.
    
    Parameters
    ----------
    freq : torch.Tensor
        Frequency tensor with shape (batch_size, n_nodes, n_bins, n_dims)
    amplitudes : torch.Tensor
        Amplitude tensor with shape (batch_size, n_nodes, n_bins, n_dims)
        
    Returns
    -------
    torch.Tensor
        HSC with shape (batch_size, n_nodes, 1, n_dims)
    """
    power = 2 * (amplitudes ** 2)
    peak_indices = kp_value_batch(power)
    
    batch_size, n_nodes, n_bins, n_dims = power.shape
    result = torch.zeros(batch_size, n_nodes, 1, n_dims, device=freq.device)
    
    for b in range(batch_size):
        for n in range(n_nodes):
            for d in range(n_dims):
                peaks = peak_indices[b][n][d]
                if len(peaks) > 0:
                    numerator = torch.sum(freq[b, n, peaks, d] * amplitudes[b, n, peaks, d])
                    denominator = torch.sum(amplitudes[b, n, peaks, d])
                    if denominator > 0:
                        result[b, n, 0, d] = numerator / denominator
    
    return result


def spectral_flux(amplitudes):
    """
    Calculate spectral flux.
    
    Parameters
    ----------
    amplitudes : torch.Tensor
        Amplitude tensor with shape (batch_size, n_nodes, n_bins, n_dims)
        
    Returns
    -------
    torch.Tensor
        Spectral flux with shape (batch_size, n_nodes, 1, n_dims)
    """
    power = 2 * (amplitudes ** 2)
    
    # Calculate differences between consecutive bins
    diff = power[:, :, 1:, :] - power[:, :, :-1, :]
    flux = torch.sqrt(torch.sum(torch.abs(diff)**2, dim=2, keepdim=True))
    
    return flux


def rolloff_frequency_90(freq, amplitudes):
    """
    Calculate 90% rolloff frequency.
    
    Parameters
    ----------
    freq : torch.Tensor
        Frequency tensor with shape (batch_size, n_nodes, n_bins, n_dims)
    amplitudes : torch.Tensor
        Amplitude tensor with shape (batch_size, n_nodes, n_bins, n_dims)
        
    Returns
    -------
    torch.Tensor
        90% rolloff frequency with shape (batch_size, n_nodes, 1, n_dims)
    """
    power = 2 * (amplitudes ** 2)
    total_pow = total_power(amplitudes)
    target = total_pow * 0.9
    
    cumsum = torch.cumsum(power, dim=2)
    mask = cumsum >= target
    
    # Find first index where cumsum exceeds target
    rolloff_indices = torch.argmax(mask.float(), dim=2, keepdim=True)
    
    # Gather frequencies at rolloff indices
    result = torch.gather(freq, 2, rolloff_indices)
    
    return result


def rolloff_frequency_85(freq, amplitudes):
    """
    Calculate 85% rolloff frequency.
    
    Parameters
    ----------
    freq : torch.Tensor
        Frequency tensor with shape (batch_size, n_nodes, n_bins, n_dims)
    amplitudes : torch.Tensor
        Amplitude tensor with shape (batch_size, n_nodes, n_bins, n_dims)
        
    Returns
    -------
    torch.Tensor
        85% rolloff frequency with shape (batch_size, n_nodes, 1, n_dims)
    """
    power = 2 * (amplitudes ** 2)
    total_pow = total_power(amplitudes)
    target = total_pow * 0.85
    
    cumsum = torch.cumsum(power, dim=2)
    mask = cumsum >= target
    
    rolloff_indices = torch.argmax(mask.float(), dim=2, keepdim=True)
    result = torch.gather(freq, 2, rolloff_indices)
    
    return result


def rolloff_frequency_75(freq, amplitudes):
    """
    Calculate 75% rolloff frequency.
    
    Parameters
    ----------
    freq : torch.Tensor
        Frequency tensor with shape (batch_size, n_nodes, n_bins, n_dims)
    amplitudes : torch.Tensor
        Amplitude tensor with shape (batch_size, n_nodes, n_bins, n_dims)
        
    Returns
    -------
    torch.Tensor
        75% rolloff frequency with shape (batch_size, n_nodes, 1, n_dims)
    """
    power = 2 * (amplitudes ** 2)
    total_pow = total_power(amplitudes)
    target = total_pow * 0.75
    
    cumsum = torch.cumsum(power, dim=2)
    mask = cumsum >= target
    
    rolloff_indices = torch.argmax(mask.float(), dim=2, keepdim=True)
    result = torch.gather(freq, 2, rolloff_indices)
    
    return result


def rolloff_frequency_95(freq, amplitudes):
    """
    Calculate 95% rolloff frequency.
    
    Parameters
    ----------
    freq : torch.Tensor
        Frequency tensor with shape (batch_size, n_nodes, n_bins, n_dims)
    amplitudes : torch.Tensor
        Amplitude tensor with shape (batch_size, n_nodes, n_bins, n_dims)
        
    Returns
    -------
    torch.Tensor
        95% rolloff frequency with shape (batch_size, n_nodes, 1, n_dims)
    """
    power = 2 * (amplitudes ** 2)
    total_pow = total_power(amplitudes)
    target = total_pow * 0.95
    
    cumsum = torch.cumsum(power, dim=2)
    mask = cumsum >= target
    
    rolloff_indices = torch.argmax(mask.float(), dim=2, keepdim=True)
    result = torch.gather(freq, 2, rolloff_indices)
    
    return result


def upper_limit_harmonicity(freq, amplitudes):
    """
    Calculate upper limit of harmonicity using peak detection.
    
    Parameters
    ----------
    freq : torch.Tensor
        Frequency tensor with shape (batch_size, n_nodes, n_bins, n_dims)
    amplitudes : torch.Tensor
        Amplitude tensor with shape (batch_size, n_nodes, n_bins, n_dims)
        
    Returns
    -------
    torch.Tensor
        Upper limit of harmonicity with shape (batch_size, n_nodes, 1, n_dims)
    """
    power = 2 * (amplitudes ** 2)
    peak_indices = kp_value_batch(power)
    
    batch_size, n_nodes, n_bins, n_dims = power.shape
    result = torch.zeros(batch_size, n_nodes, 1, n_dims, device=freq.device)
    
    for b in range(batch_size):
        for n in range(n_nodes):
            for d in range(n_dims):
                peaks = peak_indices[b][n][d]
                if len(peaks) > 0:
                    # Get the highest frequency peak
                    highest_peak_idx = peaks[-1]
                    result[b, n, 0, d] = freq[b, n, highest_peak_idx, d]
    
    return result