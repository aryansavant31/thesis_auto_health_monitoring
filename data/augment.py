import numpy as np

def add_gaussian_noise(signal, mean=0.0, std=0.01):
    """
    Adds Gaussian noise to the input signal.

    Parameters
    ----------
    signal : np.ndarray, shape (n_samples, n_timesteps)
        Input signal
    mean : float
        Mean of the Gaussian noise
    std : float
        Standard deviation of the Gaussian noise

    Returns
    -------
    noisy_signal : np.ndarray, shape (n_samples, n_timesteps)
        Signal with added Gaussian noise
    """
    noise = np.random.normal(mean, std, signal.shape)
    return signal + noise

def add_sine_waves(signal, freqs=[10.0], std_factors=[5.0], fs=1000.0):
    """
    Applies frequency modulation to the input signal.

    Parameters
    ----------
    signal : np.ndarray, shape (n_samples, n_timesteps)
        Input signal
    freqs : list
        List of frequencies of the modulation
    std_factors : list
        List of factors to multiply the signal std for each frequency
    fs : float
        Sampling frequency of the signal
    
    Returns 
    -------
    modulated_signal : np.ndarray, shape (n_samples, n_timesteps)
        Frequency modulated signal
    """
    n_samples, n_timesteps = signal.shape
    time = np.arange(n_timesteps) / fs

    signal_std = np.std(signal)

    # init modulated signal
    modulated_signal = signal.copy()

    # add sine waves for each frequency and amplitude pair
    for freq, std_factor in zip(freqs, std_factors):
        amp = signal_std * std_factor * np.sqrt(2)
        sine_wave = amp * np.sin(2 * np.pi * freq * time)
        sine_wave = np.tile(sine_wave, (n_samples, 1))  # match the shape of signal
        modulated_signal += sine_wave
    
    return modulated_signal

def add_glitches(signal, prob=0.01, std_factor=5.0):
    """
    Introduces random glitches into the input signal.
    Glitch range: [-glitch_amp, glitch_amp]

    Parameters
    ----------
    signal : np.ndarray, shape (n_samples, n_timesteps)
        Input signal
    prob : float
        Probability of a glitch occurring at any timestep
    amp : float
        Amplitude of the glitch

    Returns
    -------
    glitched_signal : np.ndarray, shape (n_samples, n_timesteps)
        Signal with random glitches introduced
    """
    n_samples, n_timesteps = signal.shape

    # Calculate std of the original signal
    signal_std = np.std(signal)
    glitch_amp = signal_std * std_factor

    # create random masks for glitches
    glitch_mask = np.random.rand(n_samples, n_timesteps) < prob # anything more that glitch_prob will be 0
    glitches = glitch_amp * (2 * np.random.rand(n_samples, n_timesteps) - 1)  # random values in [-glitch_amp, glitch_amp]
    
    return signal + (glitch_mask * glitches)