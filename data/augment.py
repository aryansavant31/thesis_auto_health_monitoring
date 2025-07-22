import numpy as np

def add_gaussian_noise(signal, mean=0.0, std=0.01):
    """
    Adds Gaussian noise to the input signal.

    Parameters
    ----------
    signal : np.ndarray
        Input signal of shape (n_samples, n_timesteps)
    mean : float
        Mean of the Gaussian noise
    std : float
        Standard deviation of the Gaussian noise

    Returns
    -------
    noisy_signal : np.ndarray
        Signal with added Gaussian noise
    """
    noise = np.random.normal(mean, std, signal.shape)
    return signal + noise