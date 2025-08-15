import os, sys
# SETTINGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
# sys.path.insert(0, SETTINGS_DIR) if SETTINGS_DIR not in sys.path else None

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT_DIR) if ROOT_DIR not in sys.path else None

# global imports
from data.config import DataConfig


def get_freq_feat_config(feat_type, data_config:DataConfig, **kwargs):
    """
    All the frequency feature extraction configurations are defined here.

    Parameters
    ----------
    feat_type : str
        The type of feature extraction to be used (e.g., 'first_n_modes', 'PCA').
    data_config : DataConfig
        Data configuration object containing sampling frequency.

    **kwargs : dict
        - `from_ranks`: **n** (_int_) (number of top features to extract),
            **perf_v** (_int_) (performance version), **rank_v** (_str_) (rank version, e.g., '[a=0.5]'),

        2 or more dimensional features
        - `first_n_modes` : **n_modes** (_int_) (will get 'mode values' and its 'frequency')
        - `full_spectrum` : **parameters** (_list_) (will get 'psd', 'mag', 'amp', 'freq')
    """
    from feature_extraction.settings.manager import get_feature_list
    config = {}
    config['type'] = feat_type
    config['fs'] = data_config.fs

    # rank based features
    if feat_type == 'from_ranks':
        config['feat_list'] = get_feature_list(
            n = kwargs.get('n', 5),
            perf_v = kwargs.get('perf_v', 1),
            rank_v = kwargs.get('rank_v', '[a=0.5]'),
            domain='freq',
            data_config=data_config
        )

    # 2 or more dimensional features
    elif feat_type == 'first_n_modes':
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
    data_config : DataConfig
        Data configuration object

    **kwargs : dict
        - `from_ranks`: **n** (_int_) (number of top features to extract), 
            **perf_v** (_int_) (performance version), **rank_v** (_str_) (rank version, e.g., '[a=0.5]'), 
            **data_config** (_DataConfig_) (data configuration object) 
    """
    from feature_extraction.settings.manager import get_feature_list
    
    config = {}
    config['type'] = feat_type

    # rank based features
    if feat_type == 'from_ranks':
        config['feat_list'] = get_feature_list(
            n = kwargs.get('n', 5),
            perf_v = kwargs.get('perf_v', 1),
            rank_v = kwargs.get('rank_v', '[a=0.5]'),
            domain='time',
            data_config=kwargs.get('data_config', None)
        )

    return config

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

