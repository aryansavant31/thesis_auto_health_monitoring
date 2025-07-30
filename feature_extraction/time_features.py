import numpy as np


def mean(amplitudes):
    """
    Compute mean along the time axis for each sample and dimension.
    
    Parameters
    ----------
    amplitudes : np.ndarray, shape (n_samples, n_timesteps, n_dims)
        Input signal amplitudes.
    
    Returns
    -------
    np.ndarray, shape (n_samples, 1, n_dims)
        Mean values computed along the time axis.
    """
    return np.mean(amplitudes, axis=1, keepdims=True)

def variance(amplitudes):
    """
    Compute variance along the time axis for each sample and dimension.
    
    Parameters
    ----------
    amplitudes : np.ndarray, shape (n_samples, n_timesteps, n_dims)
        Input signal amplitudes.
    
    Returns
    -------
    np.ndarray, shape (n_samples, 1, n_dims)
        Variance values computed along the time axis.
    """
    return np.var(amplitudes, axis=1, keepdims=True, ddof=0)

def std(amplitudes):
    """
    Compute standard deviation along the time axis for each sample and dimension.
    
    Parameters
    ----------
    amplitudes : np.ndarray, shape (n_samples, n_timesteps, n_dims)
        Input signal amplitudes.
    
    Returns
    -------
    np.ndarray, shape (n_samples, 1, n_dims)
        Standard deviation values computed along the time axis.
    """
    return np.std(amplitudes, axis=1, keepdims=True, ddof=0)

def rms(amplitudes):
    """
    Compute root mean square along the time axis for each sample and dimension.
    
    Parameters
    ----------
    amplitudes : np.ndarray, shape (n_samples, n_timesteps, n_dims)
        Input signal amplitudes.
    
    Returns
    -------
    np.ndarray, shape (n_samples, 1, n_dims)
        RMS values computed along the time axis.
    """
    return np.sqrt(np.mean(np.square(amplitudes), axis=1, keepdims=True))

def max(amplitudes):
    """
    Compute maximum value along the time axis for each sample and dimension.
    
    Parameters
    ----------
    amplitudes : np.ndarray, shape (n_samples, n_timesteps, n_dims)
        Input signal amplitudes.
    
    Returns
    -------
    np.ndarray, shape (n_samples, 1, n_dims)
        Maximum values computed along the time axis.
    """
    return np.max(amplitudes, axis=1, keepdims=True)

def kurtosis(amplitudes):
    """
    Compute kurtosis along the time axis for each sample and dimension.
    
    Parameters
    ----------
    amplitudes : np.ndarray, shape (n_samples, n_timesteps, n_dims)
        Input signal amplitudes.
    
    Returns
    -------
    np.ndarray, shape (n_samples, 1, n_dims)
        Kurtosis values computed along the time axis.
    """
    n_samples, n_timesteps, n_dims = amplitudes.shape
    mean_vals = np.mean(amplitudes, axis=1, keepdims=True)
    
    fourth_moment = np.mean((amplitudes - mean_vals) ** 4, axis=1, keepdims=True)
    second_moment = np.mean((amplitudes - mean_vals) ** 2, axis=1, keepdims=True)
    
    return n_timesteps * fourth_moment / (second_moment ** 2)

def skewness(amplitudes):
    """
    Compute skewness along the time axis for each sample and dimension.
    
    Parameters
    ----------
    amplitudes : np.ndarray, shape (n_samples, n_timesteps, n_dims)
        Input signal amplitudes.
    
    Returns
    -------
    np.ndarray, shape (n_samples, 1, n_dims)
        Skewness values computed along the time axis.
    """
    n_samples, n_timesteps, n_dims = amplitudes.shape
    mean_vals = np.mean(amplitudes, axis=1, keepdims=True)
    std_vals = np.std(amplitudes, axis=1, keepdims=True, ddof=0)
    
    third_moment = np.mean((amplitudes - mean_vals) ** 3, axis=1, keepdims=True)
    
    return third_moment / (n_timesteps * std_vals ** 3)

def eo(amplitudes):
    """
    Compute EO feature along the time axis for each sample and dimension.
    
    Parameters
    ----------
    amplitudes : np.ndarray, shape (n_samples, n_timesteps, n_dims)
        Input signal amplitudes.
    
    Returns
    -------
    np.ndarray, shape (n_samples, 1, n_dims)
        EO feature values computed along the time axis.
    """
    n_samples, n_timesteps, n_dims = amplitudes.shape
    result = np.zeros((n_samples, 1, n_dims))
    
    for s in range(n_samples):
        for d in range(n_dims):
            amp = amplitudes[s, :, d]
            diff_square = np.diff(amp ** 2)
            b = np.mean(diff_square)
            
            numerator = np.sum((diff_square - b) ** 4)
            denominator = np.sum((diff_square - b) ** 2) ** 2
            
            result[s, 0, d] = (n_timesteps ** 2 * numerator) / denominator
    
    return result


def mean_abs(amplitudes):
    """
    Compute mean of absolute values along the time axis for each sample and dimension.
    
    Parameters
    ----------
    amplitudes : np.ndarray, shape (n_samples, n_timesteps, n_dims)
        Input signal amplitudes.
    
    Returns
    -------
    np.ndarray, shape (n_samples, 1, n_dims)
        Mean absolute values computed along the time axis.
    """
    return np.mean(np.abs(amplitudes), axis=1, keepdims=True)


def srav(amplitudes):
    """
    Compute SRAV (Square Root of Absolute Value) feature along the time axis.
    
    Parameters
    ----------
    amplitudes : np.ndarray, shape (n_samples, n_timesteps, n_dims)
        Input signal amplitudes.
    
    Returns
    -------
    np.ndarray, shape (n_samples, 1, n_dims)
        SRAV values computed along the time axis.
    """
    sqrt_abs = np.sqrt(np.abs(amplitudes))
    mean_sqrt_abs = np.mean(sqrt_abs, axis=1, keepdims=True)
    return mean_sqrt_abs ** 2


def shape_factor(amplitudes):
    """
    Compute shape factor (RMS/Mean_abs) for each sample and dimension.
    
    Parameters
    ----------
    amplitudes : np.ndarray, shape (n_samples, n_timesteps, n_dims)
        Input signal amplitudes.
    
    Returns
    -------
    np.ndarray, shape (n_samples, 1, n_dims)
        Shape factor values computed along the time axis.
    """
    rms_vals = rms(amplitudes)
    mean_abs_vals = mean_abs(amplitudes)
    return rms_vals / mean_abs_vals


def impulse_factor(amplitudes):
    """
    Compute impulse factor (Max/Mean_abs) for each sample and dimension.
    
    Parameters
    ----------
    amplitudes : np.ndarray, shape (n_samples, n_timesteps, n_dims)
        Input signal amplitudes.
    
    Returns
    -------
    np.ndarray, shape (n_samples, 1, n_dims)
        Impulse factor values computed along the time axis.
    """
    max_vals = max(amplitudes)
    mean_abs_vals = mean_abs(amplitudes)
    return max_vals / mean_abs_vals


def crest_factor(amplitudes):
    """
    Compute crest factor (Max/RMS) for each sample and dimension.
    
    Parameters
    ----------
    amplitudes : np.ndarray, shape (n_samples, n_timesteps, n_dims)
        Input signal amplitudes.
    
    Returns
    -------
    np.ndarray, shape (n_samples, 1, n_dims)
        Crest factor values computed along the time axis.
    """
    max_vals = max(amplitudes)
    rms_vals = rms(amplitudes)
    return max_vals / rms_vals


def clearance_factor(amplitudes):
    """
    Compute clearance factor for each sample and dimension.
    
    Parameters
    ----------
    amplitudes : np.ndarray, shape (n_samples, n_timesteps, n_dims)
        Input signal amplitudes.
    
    Returns
    -------
    np.ndarray, shape (n_samples, 1, n_dims)
        Clearance factor values computed along the time axis.
    """
    max_vals = max(amplitudes)
    srav_vals = srav(amplitudes)
    return max_vals / srav_vals


def log_log_ratio(amplitudes):
    """
    Compute log-log ratio feature for each sample and dimension.
    
    Parameters
    ----------
    amplitudes : np.ndarray, shape (n_samples, n_timesteps, n_dims)
        Input signal amplitudes.
    
    Returns
    -------
    np.ndarray, shape (n_samples, 1, n_dims)
        Log-log ratio values computed along the time axis.
    """
    std_vals = std(amplitudes)
    log_sum = np.sum(np.log(np.abs(amplitudes) + 1), axis=1, keepdims=True)
    return log_sum / np.log(std_vals)


def std_deviation_index(amplitudes):
    """
    Compute standard deviation index (STD/Mean_abs) for each sample and dimension.
    
    Parameters
    ----------
    amplitudes : np.ndarray, shape (n_samples, n_timesteps, n_dims)
        Input signal amplitudes.
    
    Returns
    -------
    np.ndarray, shape (n_samples, 1, n_dims)
        Standard deviation index values computed along the time axis.
    """
    std_vals = std(amplitudes)
    mean_abs_vals = mean_abs(amplitudes)
    return std_vals / mean_abs_vals


def fifth_moment(amplitudes):
    """
    Compute fifth moment for each sample and dimension.
    
    Parameters
    ----------
    amplitudes : np.ndarray, shape (n_samples, n_timesteps, n_dims)
        Input signal amplitudes.
    
    Returns
    -------
    np.ndarray, shape (n_samples, 1, n_dims)
        Fifth moment values computed along the time axis.
    """
    mean_vals = np.mean(amplitudes, axis=1, keepdims=True)
    return np.mean((amplitudes - mean_vals) ** 5, axis=1, keepdims=True)


def fifth_moment_normalized(amplitudes):
    """
    Compute normalized fifth moment for each sample and dimension.
    
    Parameters
    ----------
    amplitudes : np.ndarray, shape (n_samples, n_timesteps, n_dims)
        Input signal amplitudes.
    
    Returns
    -------
    np.ndarray, shape (n_samples, 1, n_dims)
        Normalized fifth moment values computed along the time axis.
    """
    fifth_moment = fifth_moment(amplitudes)
    std_vals = std(amplitudes)
    return fifth_moment / (std_vals ** 5)


def sixth_moment(amplitudes):
    """
    Compute sixth moment for each sample and dimension.
    
    Parameters
    ----------
    amplitudes : np.ndarray, shape (n_samples, n_timesteps, n_dims)
        Input signal amplitudes.
    
    Returns
    -------
    np.ndarray, shape (n_samples, 1, n_dims)
        Sixth moment values computed along the time axis.
    """
    mean_vals = np.mean(amplitudes, axis=1, keepdims=True)
    return np.mean((amplitudes - mean_vals) ** 6, axis=1, keepdims=True)


def pulse_index(amplitudes):
    """
    Compute pulse index (Max/Mean) for each sample and dimension.
    
    Parameters
    ----------
    amplitudes : np.ndarray, shape (n_samples, n_timesteps, n_dims)
        Input signal amplitudes.
    
    Returns
    -------
    np.ndarray, shape (n_samples, 1, n_dims)
        Pulse index values computed along the time axis.
    """
    max_vals = max(amplitudes)
    mean_vals = mean(amplitudes)
    return max_vals / mean_vals


def margin_index(amplitudes):
    """
    Compute margin index (Max/SRAV) for each sample and dimension.
    
    Parameters
    ----------
    amplitudes : np.ndarray, shape (n_samples, n_timesteps, n_dims)
        Input signal amplitudes.
    
    Returns
    -------
    np.ndarray, shape (n_samples, 1, n_dims)
        Margin index values computed along the time axis.
    """
    max_vals = max(amplitudes)
    srav_vals = srav(amplitudes)
    return max_vals / srav_vals


def mean_deviation_ratio(amplitudes):
    """
    Compute mean deviation ratio (Mean/STD) for each sample and dimension.
    
    Parameters
    ----------
    amplitudes : np.ndarray, shape (n_samples, n_timesteps, n_dims)
        Input signal amplitudes.
    
    Returns
    -------
    np.ndarray, shape (n_samples, 1, n_dims)
        Mean deviation ratio values computed along the time axis.
    """
    mean_vals = mean(amplitudes)
    std_vals = std(amplitudes)
    return mean_vals / std_vals


def difference_variance(amplitudes):
    """
    Compute difference variance feature for each sample and dimension.
    
    Parameters
    ----------
    amplitudes : np.ndarray, shape (n_samples, n_timesteps, n_dims)
        Input signal amplitudes.
    
    Returns
    -------
    np.ndarray, shape (n_samples, 1, n_dims)
        Difference variance values computed along the time axis.
    """
    diff = np.diff(amplitudes, axis=1)
    return np.sum(diff ** 2, axis=1, keepdims=True) / (amplitudes.shape[1] - 2)


def min(amplitudes):
    """
    Compute minimum value along the time axis for each sample and dimension.
    
    Parameters
    ----------
    amplitudes : np.ndarray, shape (n_samples, n_timesteps, n_dims)
        Input signal amplitudes.
    
    Returns
    -------
    np.ndarray, shape (n_samples, 1, n_dims)
        Minimum values computed along the time axis.
    """
    return np.min(amplitudes, axis=1, keepdims=True)


def peak_value(amplitudes):
    """
    Compute peak value ((Max - Min)/2) for each sample and dimension.
    
    Parameters
    ----------
    amplitudes : np.ndarray, shape (n_samples, n_timesteps, n_dims)
        Input signal amplitudes.
    
    Returns
    -------
    np.ndarray, shape (n_samples, 1, n_dims)
        Peak values computed along the time axis.
    """
    max_vals = max(amplitudes)
    min_vals = min(amplitudes)
    return (max_vals - min_vals) / 2


def peak_to_peak(amplitudes):
    """
    Compute peak-to-peak value (Max - Min) for each sample and dimension.
    
    Parameters
    ----------
    amplitudes : np.ndarray, shape (n_samples, n_timesteps, n_dims)
        Input signal amplitudes.
    
    Returns
    -------
    np.ndarray, shape (n_samples, 1, n_dims)
        Peak-to-peak values computed along the time axis.
    """
    max_vals = max(amplitudes)
    min_vals = min(amplitudes)
    return max_vals - min_vals


def hist_lower_bound(amplitudes):
    """
    Compute histogram lower bound for each sample and dimension.
    
    Parameters
    ----------
    amplitudes : np.ndarray, shape (n_samples, n_timesteps, n_dims)
        Input signal amplitudes.
    
    Returns
    -------
    np.ndarray, shape (n_samples, 1, n_dims)
        Histogram lower bound values computed along the time axis.
    """
    n_timesteps = amplitudes.shape[1]
    min_vals = min(amplitudes)
    max_vals = max(amplitudes)
    
    c = 0.5 * (max_vals - min_vals) / (n_timesteps - 1)
    return min_vals - c


def hist_upper_bound(amplitudes):
    """
    Compute histogram upper bound for each sample and dimension.
    
    Parameters
    ----------
    amplitudes : np.ndarray, shape (n_samples, n_timesteps, n_dims)
        Input signal amplitudes.
    
    Returns
    -------
    np.ndarray, shape (n_samples, 1, n_dims)
        Histogram upper bound values computed along the time axis.
    """
    n_timesteps = amplitudes.shape[1]
    min_vals = min(amplitudes)
    max_vals = max(amplitudes)
    
    c = 0.5 * (max_vals - min_vals) / (n_timesteps - 1)
    return max_vals + c


def latitude_factor(amplitudes):
    """
    Compute latitude factor for each sample and dimension.
    
    Parameters
    ----------
    amplitudes : np.ndarray, shape (n_samples, n_timesteps, n_dims)
        Input signal amplitudes.
    
    Returns
    -------
    np.ndarray, shape (n_samples, 1, n_dims)
        Latitude factor values computed along the time axis.
    """
    max_vals = max(amplitudes)
    sqrt_mean = np.mean(np.sqrt(np.abs(amplitudes)), axis=1, keepdims=True) ** 2
    return max_vals / sqrt_mean


def normalized_std(amplitudes):
    """
    Compute normalized standard deviation (STD/RMS) for each sample and dimension.
    
    Parameters
    ----------
    amplitudes : np.ndarray, shape (n_samples, n_timesteps, n_dims)
        Input signal amplitudes.
    
    Returns
    -------
    np.ndarray, shape (n_samples, 1, n_dims)
        Normalized standard deviation values computed along the time axis.
    """
    std_vals = std(amplitudes)
    rms_vals = rms(amplitudes)
    return std_vals / rms_vals


def waveform_indicator(amplitudes):
    """
    Compute waveform indicator (RMS/Mean) for each sample and dimension.
    
    Parameters
    ----------
    amplitudes : np.ndarray, shape (n_samples, n_timesteps, n_dims)
        Input signal amplitudes.
    
    Returns
    -------
    np.ndarray, shape (n_samples, 1, n_dims)
        Waveform indicator values computed along the time axis.
    """
    rms_vals = rms(amplitudes)
    mean_vals = mean(amplitudes)
    return rms_vals / mean_vals


def wilson_amplitude(amplitudes, threshold=0.02):
    """
    Compute Wilson amplitude feature for each sample and dimension.
    
    Parameters
    ----------
    amplitudes : np.ndarray, shape (n_samples, n_timesteps, n_dims)
        Input signal amplitudes.
    threshold : float, optional
        Threshold value for the feature, by default 0.02.
    
    Returns
    -------
    np.ndarray, shape (n_samples, 1, n_dims)
        Wilson amplitude values computed along the time axis.
    """
    diff = np.abs(np.diff(amplitudes, axis=1))
    return np.sum(np.heaviside(diff - threshold, 1), axis=1, keepdims=True)


def zero_crossing_rate(amplitudes):
    """
    Compute zero crossing rate for each sample and dimension.
    
    Parameters
    ----------
    amplitudes : np.ndarray, shape (n_samples, n_timesteps, n_dims)
        Input signal amplitudes.
    
    Returns
    -------
    np.ndarray, shape (n_samples, 1, n_dims)
        Zero crossing rate values computed along the time axis.
    """
    signs = np.sign(amplitudes)
    sign_changes = np.abs(np.diff(signs, axis=1))
    return np.mean(sign_changes, axis=1, keepdims=True)


def waveform_length(amplitudes):
    """
    Compute waveform length for each sample and dimension.
    
    Parameters
    ----------
    amplitudes : np.ndarray, shape (n_samples, n_timesteps, n_dims)
        Input signal amplitudes.
    
    Returns
    -------
    np.ndarray, shape (n_samples, 1, n_dims)
        Waveform length values computed along the time axis.
    """
    return np.sum(np.abs(np.diff(amplitudes, axis=1)), axis=1, keepdims=True)


def energy(amplitudes):
    """
    Compute energy (sum of squared absolute values) for each sample and dimension.
    
    Parameters
    ----------
    amplitudes : np.ndarray, shape (n_samples, n_timesteps, n_dims)
        Input signal amplitudes.
    
    Returns
    -------
    np.ndarray, shape (n_samples, 1, n_dims)
        Energy values computed along the time axis.
    """
    return np.sum(np.abs(amplitudes) ** 2, axis=1, keepdims=True)


def mean_abs_modif1(amplitudes):
    """
    Compute modified mean absolute value 1 for each sample and dimension.
    
    Parameters
    ----------
    amplitudes : np.ndarray, shape (n_samples, n_timesteps, n_dims)
        Input signal amplitudes.
    
    Returns
    -------
    np.ndarray, shape (n_samples, 1, n_dims)
        Modified mean absolute value 1 computed along the time axis.
    """
    n_samples, n_timesteps, n_dims = amplitudes.shape
    result = np.zeros((n_samples, 1, n_dims))
    
    for s in range(n_samples):
        for d in range(n_dims):
            amp = amplitudes[s, :, d]
            n = n_timesteps
            
            # First quarter with 0.5 weight
            m = np.sum(0.5 * np.abs(amp[:n//4]))
            
            # Last quarter with 0.5 weight
            m += np.sum(0.5 * np.abs(amp[3*n//4 + 1:]))
            
            # Middle half with full weight
            m += np.sum(np.abs(amp[n//4:3*n//4 + 1]))
            
            result[s, 0, d] = m / n
    
    return result


def mean_abs_modif2(amplitudes):
    """
    Compute modified mean absolute value 2 for each sample and dimension.
    
    Parameters
    ----------
    amplitudes : np.ndarray, shape (n_samples, n_timesteps, n_dims)
        Input signal amplitudes.
    
    Returns
    -------
    np.ndarray, shape (n_samples, 1, n_dims)
        Modified mean absolute value 2 computed along the time axis.
    """
    n_samples, n_timesteps, n_dims = amplitudes.shape
    result = np.zeros((n_samples, 1, n_dims))
    
    for s in range(n_samples):
        for d in range(n_dims):
            amp = amplitudes[s, :, d]
            n = n_timesteps
            
            # First quarter with linear weight
            indices = np.arange(0, n//4)
            m = np.sum((4 * indices / n) * np.abs(amp[:n//4]))
            
            # Last quarter with linear weight
            indices = np.arange(3*n//4 + 1, n)
            m += np.sum((4 * (indices - n) / n) * np.abs(amp[3*n//4 + 1:]))
            
            # Middle half with full weight
            m += np.sum(np.abs(amp[n//4:3*n//4 + 1]))
            
            result[s, 0, d] = m / n
    
    return result

