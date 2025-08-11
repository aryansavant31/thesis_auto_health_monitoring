import os, sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT_DIR) if ROOT_DIR not in sys.path else None

SETTINGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SETTINGS_DIR) if SETTINGS_DIR not in sys.path else None


# global imports
from data.config import DataConfig, get_domain_config
from feature_extraction.settings.feature_config import get_freq_feat_config, get_time_feat_config, get_reduc_config

class AnomalyDetectorTrainConfig:
    def __init__(self, data_config:DataConfig):
        """
        Parameters
        ----------
        data_config : DataConfig
            The data configuration object (also stored as an attribute).
            
        1: Training Attributes
        -----------------------
        model_num : int
            Anomaly detector model number.
        is_log : bool
            Whether to log the training process.
        
        - **_Dataset Parameters_**
        batch_size : int
            Size of the training batch.
        train_rt : float
            Ratio of training data.
        test_rt : float
            Ratio of testing data.
        
        2: Model Attributes
        -----------------------
        anom_config : dict
            Configuration for the anomaly detection model.

        - **_Run Parameters_**
        domain_config : dict
            Configuration for the data domain.
        raw_data_norm : None
            Normalization type for raw data ('min_max', 'std' or None).
        feat_configs : list
            List of feature configurations.
        reduc_config : dict
            Configuration for dimensionality reduction.
        feat_norm : None
            Normalization type for features.
        """
        self.data_config = data_config

    # 1: Training parameters
        self.model_num = 1
        self.is_log = True

        # dataset parameters
        self.batch_size  = 50
        self.train_rt    = 0.8
        self.test_rt     = 0.2
        self.num_workers = 10

    # 2: Model parameters
        self.anom_config = get_anom_config('IF', n_estimators=1000, contam=0.3)

        # run parameters
        self.domain_config = get_domain_config('time', data_config=self.data_config)
        self.raw_data_norm = 'min_max' 
        self.feat_configs = [
            # get_time_feat_config('from_ranks', n=5, perf_v=1, rank_v='[a=0.5]', data_config=self.data_config), 
        ]  
        self.reduc_config = None # or None
        self.feat_norm = None


def get_anom_config(anom_type, **kwargs):
    """
    Parameters
    ----------
    anom_type : str
        The type of fault detection algorithm to be used. 
        - `SVM`: Support Vector Machine
        - `IF`: Isolation Forest)
    **kwargs : dict
        For all options of `anom_type`:
        - `SVM`: **kernel**, **nu**, **gamma**
        - `IF`: **n_estimators**, **seed**, **contam**, **n_jobs**

    """
    anom_config = {}
    anom_config['type'] = anom_type

    if anom_type == 'SVM':
        anom_config['kernel'] = kwargs.get('kernel', 'rbf')
        anom_config['gamma'] = kwargs.get('gamma', 'scale')
        anom_config['nu'] = kwargs.get('nu', 0.5)

    elif anom_type == 'IF':
        anom_config['n_estimators'] = kwargs.get('n_estimators', 100)
        anom_config['seed'] = kwargs.get('seed', 42)
        anom_config['contam'] = kwargs.get('contam', 'auto')
        anom_config['n_jobs'] = kwargs.get('n_jobs', -1)

        # hyperparameters for isolation forest

        # anom_config['hparams'] = {
        #     HP_NUM_TREES.name: anom_config['n_estimators'],
        #     HP_CONTAM.name: anom_config['contam'],
        # }

    return anom_config  

if __name__ == "__main__":
    
    from fault_detection.settings.manager import SelectFaultDetectionModel
    model_selector = SelectFaultDetectionModel(run_type='train')
    model_selector.select_model_and_params()