import torch

def mean(amplitudes):
    """
    Compute mean along the time axis for each batch, node, and dimension.
    
    Parameters
    ----------
    amplitudes : torch.Tensor, shape (batch_size, n_nodes, n_timesteps, n_dims)
        Input signal amplitudes.
    
    Returns
    -------
    torch.Tensor, shape (batch_size, n_nodes, 1, n_dims)
        Mean values computed along the time axis.
    """
    return torch.mean(amplitudes, dim=2, keepdim=True)


def variance(amplitudes):
    """
    Compute variance along the time axis for each batch, node, and dimension.
    
    Parameters
    ----------
    amplitudes : torch.Tensor, shape (batch_size, n_nodes, n_timesteps, n_dims)
        Input signal amplitudes.
    
    Returns
    -------
    torch.Tensor, shape (batch_size, n_nodes, 1, n_dims)
        Variance values computed along the time axis.
    """
    return torch.var(amplitudes, dim=2, keepdim=True, unbiased=False)


def std(amplitudes):
    """
    Compute standard deviation along the time axis for each batch, node, and dimension.
    
    Parameters
    ----------
    amplitudes : torch.Tensor, shape (batch_size, n_nodes, n_timesteps, n_dims)
        Input signal amplitudes.
    
    Returns
    -------
    torch.Tensor, shape (batch_size, n_nodes, 1, n_dims)
        Standard deviation values computed along the time axis.
    """
    return torch.std(amplitudes, dim=2, keepdim=True, unbiased=False)


def rms(amplitudes):
    """
    Compute root mean square along the time axis for each batch, node, and dimension.
    
    Parameters
    ----------
    amplitudes : torch.Tensor, shape (batch_size, n_nodes, n_timesteps, n_dims)
        Input signal amplitudes.
    
    Returns
    -------
    torch.Tensor, shape (batch_size, n_nodes, 1, n_dims)
        RMS values computed along the time axis.
    """
    return torch.sqrt(torch.mean(torch.square(amplitudes), dim=2, keepdim=True))


def max(amplitudes):
    """
    Compute maximum value along the time axis for each batch, node, and dimension.
    
    Parameters
    ----------
    amplitudes : torch.Tensor, shape (batch_size, n_nodes, n_timesteps, n_dims)
        Input signal amplitudes.
    
    Returns
    -------
    torch.Tensor, shape (batch_size, n_nodes, 1, n_dims)
        Maximum values computed along the time axis.
    """
    return torch.max(amplitudes, dim=2, keepdim=True)[0]


def kurtosis(amplitudes):
    """
    Compute kurtosis along the time axis for each batch, node, and dimension.
    
    Parameters
    ----------
    amplitudes : torch.Tensor, shape (batch_size, n_nodes, n_timesteps, n_dims)
        Input signal amplitudes.
    
    Returns
    -------
    torch.Tensor, shape (batch_size, n_nodes, 1, n_dims)
        Kurtosis values computed along the time axis.
    """
    n_timesteps = amplitudes.shape[2]
    mean_vals = torch.mean(amplitudes, dim=2, keepdim=True)
    
    fourth_moment = torch.mean((amplitudes - mean_vals) ** 4, dim=2, keepdim=True)
    second_moment = torch.mean((amplitudes - mean_vals) ** 2, dim=2, keepdim=True)
    
    return n_timesteps * fourth_moment / (second_moment ** 2)


def skewness(amplitudes):
    """
    Compute skewness along the time axis for each batch, node, and dimension.
    
    Parameters
    ----------
    amplitudes : torch.Tensor, shape (batch_size, n_nodes, n_timesteps, n_dims)
        Input signal amplitudes.
    
    Returns
    -------
    torch.Tensor, shape (batch_size, n_nodes, 1, n_dims)
        Skewness values computed along the time axis.
    """
    n_timesteps = amplitudes.shape[2]
    mean_vals = torch.mean(amplitudes, dim=2, keepdim=True)
    std_vals = torch.std(amplitudes, dim=2, keepdim=True, unbiased=False)
    
    third_moment = torch.mean((amplitudes - mean_vals) ** 3, dim=2, keepdim=True)
    
    return third_moment / (n_timesteps * std_vals ** 3)


def eo(amplitudes):
    """
    Compute EO feature along the time axis for each batch, node, and dimension.
    
    Parameters
    ----------
    amplitudes : torch.Tensor, shape (batch_size, n_nodes, n_timesteps, n_dims)
        Input signal amplitudes.
    
    Returns
    -------
    torch.Tensor, shape (batch_size, n_nodes, 1, n_dims)
        EO feature values computed along the time axis.
    """
    batch_size, n_nodes, n_timesteps, n_dims = amplitudes.shape
    
    # Compute squared amplitudes and their differences
    amp_squared = amplitudes ** 2
    diff_square = torch.diff(amp_squared, dim=2)
    
    # Compute mean of differences
    b = torch.mean(diff_square, dim=2, keepdim=True)
    
    # Compute centered differences
    centered_diff = diff_square - b
    
    # Compute numerator and denominator
    numerator = torch.sum(centered_diff ** 4, dim=2, keepdim=True)
    denominator = torch.sum(centered_diff ** 2, dim=2, keepdim=True) ** 2
    
    result = (n_timesteps ** 2 * numerator) / denominator
    
    return result


def mean_abs(amplitudes):
    """
    Compute mean of absolute values along the time axis for each batch, node, and dimension.
    
    Parameters
    ----------
    amplitudes : torch.Tensor, shape (batch_size, n_nodes, n_timesteps, n_dims)
        Input signal amplitudes.
    
    Returns
    -------
    torch.Tensor, shape (batch_size, n_nodes, 1, n_dims)
        Mean absolute values computed along the time axis.
    """
    return torch.mean(torch.abs(amplitudes), dim=2, keepdim=True)


def sq_root_abs(amplitudes):
    """
    Compute SRAV (Square Root of Absolute Value) feature along the time axis.
    
    Parameters
    ----------
    amplitudes : torch.Tensor, shape (batch_size, n_nodes, n_timesteps, n_dims)
        Input signal amplitudes.
    
    Returns
    -------
    torch.Tensor, shape (batch_size, n_nodes, 1, n_dims)
        SRAV values computed along the time axis.
    """
    sqrt_abs = torch.sqrt(torch.abs(amplitudes))
    mean_sqrt_abs = torch.mean(sqrt_abs, dim=2, keepdim=True)
    return mean_sqrt_abs ** 2


def shape_factor(amplitudes):
    """
    Compute shape factor (RMS/Mean_abs) for each batch, node, and dimension.
    
    Parameters
    ----------
    amplitudes : torch.Tensor, shape (batch_size, n_nodes, n_timesteps, n_dims)
        Input signal amplitudes.
    
    Returns
    -------
    torch.Tensor, shape (batch_size, n_nodes, 1, n_dims)
        Shape factor values computed along the time axis.
    """
    rms_vals = rms(amplitudes)
    mean_abs_vals = mean_abs(amplitudes)
    return rms_vals / mean_abs_vals


def impulse_factor(amplitudes):
    """
    Compute impulse factor (Max/Mean_abs) for each batch, node, and dimension.
    
    Parameters
    ----------
    amplitudes : torch.Tensor, shape (batch_size, n_nodes, n_timesteps, n_dims)
        Input signal amplitudes.
    
    Returns
    -------
    torch.Tensor, shape (batch_size, n_nodes, 1, n_dims)
        Impulse factor values computed along the time axis.
    """
    max_vals = max(amplitudes)
    mean_abs_vals = mean_abs(amplitudes)
    return max_vals / mean_abs_vals


def crest_factor(amplitudes):
    """
    Compute crest factor (Max/RMS) for each batch, node, and dimension.
    
    Parameters
    ----------
    amplitudes : torch.Tensor, shape (batch_size, n_nodes, n_timesteps, n_dims)
        Input signal amplitudes.
    
    Returns
    -------
    torch.Tensor, shape (batch_size, n_nodes, 1, n_dims)
        Crest factor values computed along the time axis.
    """
    max_vals = max(amplitudes)
    rms_vals = rms(amplitudes)
    return max_vals / rms_vals


def clearance_factor(amplitudes):
    """
    Compute clearance factor for each batch, node, and dimension.
    
    Parameters
    ----------
    amplitudes : torch.Tensor, shape (batch_size, n_nodes, n_timesteps, n_dims)
        Input signal amplitudes.
    
    Returns
    -------
    torch.Tensor, shape (batch_size, n_nodes, 1, n_dims)
        Clearance factor values computed along the time axis.
    """
    max_vals = max(amplitudes)
    srav_vals = sq_root_abs(amplitudes)
    return max_vals / srav_vals


def log_log_ratio(amplitudes):
    """
    Compute log-log ratio feature for each batch, node, and dimension.
    
    Parameters
    ----------
    amplitudes : torch.Tensor, shape (batch_size, n_nodes, n_timesteps, n_dims)
        Input signal amplitudes.
    
    Returns
    -------
    torch.Tensor, shape (batch_size, n_nodes, 1, n_dims)
        Log-log ratio values computed along the time axis.
    """
    std_vals = std(amplitudes)
    log_sum = torch.sum(torch.log(torch.abs(amplitudes) + 1), dim=2, keepdim=True)
    return log_sum / torch.log(std_vals)


def std_deviation_index(amplitudes):
    """
    Compute standard deviation index (STD/Mean_abs) for each batch, node, and dimension.
    
    Parameters
    ----------
    amplitudes : torch.Tensor, shape (batch_size, n_nodes, n_timesteps, n_dims)
        Input signal amplitudes.
    
    Returns
    -------
    torch.Tensor, shape (batch_size, n_nodes, 1, n_dims)
        Standard deviation index values computed along the time axis.
    """
    std_vals = std(amplitudes)
    mean_abs_vals = mean_abs(amplitudes)
    return std_vals / mean_abs_vals


def fifth_moment(amplitudes):
    """
    Compute fifth moment for each batch, node, and dimension.
    
    Parameters
    ----------
    amplitudes : torch.Tensor, shape (batch_size, n_nodes, n_timesteps, n_dims)
        Input signal amplitudes.
    
    Returns
    -------
    torch.Tensor, shape (batch_size, n_nodes, 1, n_dims)
        Fifth moment values computed along the time axis.
    """
    mean_vals = torch.mean(amplitudes, dim=2, keepdim=True)
    return torch.mean((amplitudes - mean_vals) ** 5, dim=2, keepdim=True)


def fifth_moment_normalized(amplitudes):
    """
    Compute normalized fifth moment for each batch, node, and dimension.
    
    Parameters
    ----------
    amplitudes : torch.Tensor, shape (batch_size, n_nodes, n_timesteps, n_dims)
        Input signal amplitudes.
    
    Returns
    -------
    torch.Tensor, shape (batch_size, n_nodes, 1, n_dims)
        Normalized fifth moment values computed along the time axis.
    """
    fifth_moment_vals = fifth_moment(amplitudes)
    std_vals = std(amplitudes)
    return fifth_moment_vals / (std_vals ** 5)


def sixth_moment(amplitudes):
    """
    Compute sixth moment for each batch, node, and dimension.
    
    Parameters
    ----------
    amplitudes : torch.Tensor, shape (batch_size, n_nodes, n_timesteps, n_dims)
        Input signal amplitudes.
    
    Returns
    -------
    torch.Tensor, shape (batch_size, n_nodes, 1, n_dims)
        Sixth moment values computed along the time axis.
    """
    mean_vals = torch.mean(amplitudes, dim=2, keepdim=True)
    return torch.mean((amplitudes - mean_vals) ** 6, dim=2, keepdim=True)


def pulse_index(amplitudes):
    """
    Compute pulse index (Max/Mean) for each batch, node, and dimension.
    
    Parameters
    ----------
    amplitudes : torch.Tensor, shape (batch_size, n_nodes, n_timesteps, n_dims)
        Input signal amplitudes.
    
    Returns
    -------
    torch.Tensor, shape (batch_size, n_nodes, 1, n_dims)
        Pulse index values computed along the time axis.
    """
    max_vals = max(amplitudes)
    mean_vals = mean(amplitudes)
    return max_vals / mean_vals


def margin_index(amplitudes):
    """
    Compute margin index (Max/SRAV) for each batch, node, and dimension.
    
    Parameters
    ----------
    amplitudes : torch.Tensor, shape (batch_size, n_nodes, n_timesteps, n_dims)
        Input signal amplitudes.
    
    Returns
    -------
    torch.Tensor, shape (batch_size, n_nodes, 1, n_dims)
        Margin index values computed along the time axis.
    """
    max_vals = max(amplitudes)
    srav_vals = sq_root_abs(amplitudes)
    return max_vals / srav_vals


def mean_deviation_ratio(amplitudes):
    """
    Compute mean deviation ratio (Mean/STD) for each batch, node, and dimension.
    
    Parameters
    ----------
    amplitudes : torch.Tensor, shape (batch_size, n_nodes, n_timesteps, n_dims)
        Input signal amplitudes.
    
    Returns
    -------
    torch.Tensor, shape (batch_size, n_nodes, 1, n_dims)
        Mean deviation ratio values computed along the time axis.
    """
    mean_vals = mean(amplitudes)
    std_vals = std(amplitudes)
    return mean_vals / std_vals


def difference_variance(amplitudes):
    """
    Compute difference variance feature for each batch, node, and dimension.
    
    Parameters
    ----------
    amplitudes : torch.Tensor, shape (batch_size, n_nodes, n_timesteps, n_dims)
        Input signal amplitudes.
    
    Returns
    -------
    torch.Tensor, shape (batch_size, n_nodes, 1, n_dims)
        Difference variance values computed along the time axis.
    """
    diff = torch.diff(amplitudes, dim=2)
    return torch.sum(diff ** 2, dim=2, keepdim=True) / (amplitudes.shape[2] - 2)


def min(amplitudes):
    """
    Compute minimum value along the time axis for each batch, node, and dimension.
    
    Parameters
    ----------
    amplitudes : torch.Tensor, shape (batch_size, n_nodes, n_timesteps, n_dims)
        Input signal amplitudes.
    
    Returns
    -------
    torch.Tensor, shape (batch_size, n_nodes, 1, n_dims)
        Minimum values computed along the time axis.
    """
    return torch.min(amplitudes, dim=2, keepdim=True)[0]


def peak_value(amplitudes):
    """
    Compute peak value ((Max - Min)/2) for each batch, node, and dimension.
    
    Parameters
    ----------
    amplitudes : torch.Tensor, shape (batch_size, n_nodes, n_timesteps, n_dims)
        Input signal amplitudes.
    
    Returns
    -------
    torch.Tensor, shape (batch_size, n_nodes, 1, n_dims)
        Peak values computed along the time axis.
    """
    max_vals = max(amplitudes)
    min_vals = min(amplitudes)
    return (max_vals - min_vals) / 2


def peak_to_peak(amplitudes):
    """
    Compute peak-to-peak value (Max - Min) for each batch, node, and dimension.
    
    Parameters
    ----------
    amplitudes : torch.Tensor, shape (batch_size, n_nodes, n_timesteps, n_dims)
        Input signal amplitudes.
    
    Returns
    -------
    torch.Tensor, shape (batch_size, n_nodes, 1, n_dims)
        Peak-to-peak values computed along the time axis.
    """
    max_vals = max(amplitudes)
    min_vals = min(amplitudes)
    return max_vals - min_vals


def hist_lower_bound(amplitudes):
    """
    Compute histogram lower bound for each batch, node, and dimension.
    
    Parameters
    ----------
    amplitudes : torch.Tensor, shape (batch_size, n_nodes, n_timesteps, n_dims)
        Input signal amplitudes.
    
    Returns
    -------
    torch.Tensor, shape (batch_size, n_nodes, 1, n_dims)
        Histogram lower bound values computed along the time axis.
    """
    n_timesteps = amplitudes.shape[2]
    min_vals = min(amplitudes)
    max_vals = max(amplitudes)
    
    c = 0.5 * (max_vals - min_vals) / (n_timesteps - 1)
    return min_vals - c


def hist_upper_bound(amplitudes):
    """
    Compute histogram upper bound for each batch, node, and dimension.
    
    Parameters
    ----------
    amplitudes : torch.Tensor, shape (batch_size, n_nodes, n_timesteps, n_dims)
        Input signal amplitudes.
    
    Returns
    -------
    torch.Tensor, shape (batch_size, n_nodes, 1, n_dims)
        Histogram upper bound values computed along the time axis.
    """
    n_timesteps = amplitudes.shape[2]
    min_vals = min(amplitudes)
    max_vals = max(amplitudes)
    
    c = 0.5 * (max_vals - min_vals) / (n_timesteps - 1)
    return max_vals + c


def latitude_factor(amplitudes):
    """
    Compute latitude factor for each batch, node, and dimension.
    
    Parameters
    ----------
    amplitudes : torch.Tensor, shape (batch_size, n_nodes, n_timesteps, n_dims)
        Input signal amplitudes.
    
    Returns
    -------
    torch.Tensor, shape (batch_size, n_nodes, 1, n_dims)
        Latitude factor values computed along the time axis.
    """
    max_vals = max(amplitudes)
    sqrt_mean = torch.mean(torch.sqrt(torch.abs(amplitudes)), dim=2, keepdim=True) ** 2
    return max_vals / sqrt_mean


def normalized_std(amplitudes):
    """
    Compute normalized standard deviation (STD/RMS) for each batch, node, and dimension.
    
    Parameters
    ----------
    amplitudes : torch.Tensor, shape (batch_size, n_nodes, n_timesteps, n_dims)
        Input signal amplitudes.
    
    Returns
    -------
    torch.Tensor, shape (batch_size, n_nodes, 1, n_dims)
        Normalized standard deviation values computed along the time axis.
    """
    std_vals = std(amplitudes)
    rms_vals = rms(amplitudes)
    return std_vals / rms_vals


def waveform_indicator(amplitudes):
    """
    Compute waveform indicator (RMS/Mean) for each batch, node, and dimension.
    
    Parameters
    ----------
    amplitudes : torch.Tensor, shape (batch_size, n_nodes, n_timesteps, n_dims)
        Input signal amplitudes.
    
    Returns
    -------
    torch.Tensor, shape (batch_size, n_nodes, 1, n_dims)
        Waveform indicator values computed along the time axis.
    """
    rms_vals = rms(amplitudes)
    mean_vals = mean(amplitudes)
    return rms_vals / mean_vals


def wilson_amplitude(amplitudes, threshold=0.02):
    """
    Compute Wilson amplitude feature for each batch, node, and dimension.
    
    Parameters
    ----------
    amplitudes : torch.Tensor, shape (batch_size, n_nodes, n_timesteps, n_dims)
        Input signal amplitudes.
    threshold : float, optional
        Threshold value for the feature, by default 0.02.
    
    Returns
    -------
    torch.Tensor, shape (batch_size, n_nodes, 1, n_dims)
        Wilson amplitude values computed along the time axis.
    """
    diff = torch.abs(torch.diff(amplitudes, dim=2))
    return torch.sum((diff > threshold).float(), dim=2, keepdim=True)


def zero_crossing_rate(amplitudes):
    """
    Compute zero crossing rate for each batch, node, and dimension.
    
    Parameters
    ----------
    amplitudes : torch.Tensor, shape (batch_size, n_nodes, n_timesteps, n_dims)
        Input signal amplitudes.
    
    Returns
    -------
    torch.Tensor, shape (batch_size, n_nodes, 1, n_dims)
        Zero crossing rate values computed along the time axis.
    """
    signs = torch.sign(amplitudes)
    sign_changes = torch.abs(torch.diff(signs, dim=2))
    return torch.mean(sign_changes, dim=2, keepdim=True)


def waveform_length(amplitudes):
    """
    Compute waveform length for each batch, node, and dimension.
    
    Parameters
    ----------
    amplitudes : torch.Tensor, shape (batch_size, n_nodes, n_timesteps, n_dims)
        Input signal amplitudes.
    
    Returns
    -------
    torch.Tensor, shape (batch_size, n_nodes, 1, n_dims)
        Waveform length values computed along the time axis.
    """
    return torch.sum(torch.abs(torch.diff(amplitudes, dim=2)), dim=2, keepdim=True)


def energy(amplitudes):
    """
    Compute energy (sum of squared absolute values) for each batch, node, and dimension.
    
    Parameters
    ----------
    amplitudes : torch.Tensor, shape (batch_size, n_nodes, n_timesteps, n_dims)
        Input signal amplitudes.
    
    Returns
    -------
    torch.Tensor, shape (batch_size, n_nodes, 1, n_dims)
        Energy values computed along the time axis.
    """
    return torch.sum(torch.abs(amplitudes) ** 2, dim=2, keepdim=True)


def mean_abs_modif1(amplitudes):
    """
    Compute modified mean absolute value 1 for each batch, node, and dimension.
    
    Parameters
    ----------
    amplitudes : torch.Tensor, shape (batch_size, n_nodes, n_timesteps, n_dims)
        Input signal amplitudes.
    
    Returns
    -------
    torch.Tensor, shape (batch_size, n_nodes, 1, n_dims)
        Modified mean absolute value 1 computed along the time axis.
    """
    batch_size, n_nodes, n_timesteps, n_dims = amplitudes.shape
    n = n_timesteps
    
    # Create weight tensor
    weights = torch.ones_like(amplitudes)
    
    # First quarter with 0.5 weight
    weights[:, :, :n//4, :] = 0.5
    
    # Last quarter with 0.5 weight  
    weights[:, :, 3*n//4 + 1:, :] = 0.5
    
    # Compute weighted mean
    weighted_sum = torch.sum(weights * torch.abs(amplitudes), dim=2, keepdim=True)
    return weighted_sum / n


def mean_abs_modif2(amplitudes):
    """
    Compute modified mean absolute value 2 for each batch, node, and dimension.
    
    Parameters
    ----------
    amplitudes : torch.Tensor, shape (batch_size, n_nodes, n_timesteps, n_dims)
        Input signal amplitudes.
    
    Returns
    -------
    torch.Tensor, shape (batch_size, n_nodes, 1, n_dims)
        Modified mean absolute value 2 computed along the time axis.
    """
    batch_size, n_nodes, n_timesteps, n_dims = amplitudes.shape
    n = n_timesteps
    
    # Create weight tensor
    weights = torch.ones_like(amplitudes)
    
    # First quarter with linear weight
    indices = torch.arange(0, n//4, device=amplitudes.device, dtype=amplitudes.dtype)
    weights[:, :, :n//4, :] = (4 * indices / n).view(1, 1, -1, 1)
    
    # Last quarter with linear weight
    indices = torch.arange(3*n//4 + 1, n, device=amplitudes.device, dtype=amplitudes.dtype)
    weights[:, :, 3*n//4 + 1:, :] = (4 * (indices - n) / n).view(1, 1, -1, 1)
    
    # Compute weighted mean
    weighted_sum = torch.sum(weights * torch.abs(amplitudes), dim=2, keepdim=True)
    return weighted_sum / n