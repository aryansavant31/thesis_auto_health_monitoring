def get_freq_feat_config(feat_type, **kwargs):
    """
    All the frequency feature extraction configurations are defined here.

    Parameters
    ----------
    feat_type : str
        The type of feature extraction to be used (e.g., 'first_n_modes', 'PCA').

    **kwargs : dict
        2 or more dimensional features
        - `first_n_modes` : **n_modes** (_int_) (will get 'mode values' and its 'frequency')
        - `full_spectrum` : **parameters** (_list_) (will get 'psd', 'mag', 'amp', 'freq')
    """
    config = {}
    config['type'] = feat_type

    # 2 or more dimensional features

    if feat_type == 'first_n_modes':
        config['n_modes'] = kwargs.get('n_modes', 5)  # default to 5 modes if not specified
    
    elif feat_type == 'full_spectrum':
        config['parameters'] = kwargs.get('parameters', ['psd', 'mag', 'amp', 'freq'])  # default to all parameters if not specified

    # 1 dimensional features

    return config

def get_time_feat_config(feat_type, **kwargs):
     """
     All the time feature extraction configurations are defined here.

    Parameters
    ----------
    feat_type : str
        The type of time feature extraction
    
    **kwargs : dict
     """
     pass

def get_reduc_config(reduc_type, **kwargs):
    """
    Parameters
    ----------
    reduc_type : str
        The type of feature reduction to be used (e.g., 'PCA').
    **kwargs : dict
        For all options of `reduc_type`:
        - `PCA`: **n_components** (_int_) (number of components to keep)
    """
    config = {}
    config['type'] = reduc_type

    if reduc_type == 'PCA':
        config['n_components'] = kwargs.get('n_components', 5)  # default to 5 components if not specified

    return config