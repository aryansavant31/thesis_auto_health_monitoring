import os, sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT_DIR) if ROOT_DIR not in sys.path else None

# SETTINGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
# sys.path.insert(0, SETTINGS_DIR) if SETTINGS_DIR not in sys.path else None


# global imports
from data.config import DataConfig, get_domain_config
#from feature_extraction.settings.feature_config import get_freq_feat_config, get_time_feat_config, get_reduc_config

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
        self.model_num = 7
        self.is_log = True

        # dataset parameters
        self.batch_size  = 10
        self.train_rt    = 0.8
        self.test_rt     = 0.2
        self.num_workers = 1

    # 2: Model parameters
        self.anom_config = get_anom_config('IF')

        # run parameters
        self.domain_config = get_domain_config('freq', data_config=self.data_config)
        self.raw_data_norm = 'min_max' 
        self.feat_configs = [
            # get_time_feat_config('from_ranks', n=5, perf_v=1, rank_v='[a=0.5]', data_config=self.data_config), 
        ]  
        self.reduc_config = None # or None
        self.feat_norm = None

    # 3: Hyperparameters and plots
        self.hparams = self.get_hparams()

        self.test_plots = {
            'confusion_matrix'      : [True, {}],
            'roc_curve'             : [False, {}],
            'anomaly_score_dist-1'  : [True, {'is_pred':True}],
            'anomaly_score_dist-2'  : [True, {'is_pred':False}],
            'pair_plot'             : [True, {}],
        }

    def get_hparams(self):
        """
        Sets the hyperparameters for the anomaly detection model.
        """
        init_hparams = {
            'batch_size': self.batch_size,
            'train_rt': self.train_rt,
            'test_rt': self.test_rt,

            'domain': self.domain_config['type'],
            'raw_data_norm': self.raw_data_norm,
            'feats': f"[{', '.join([feat_config['type'] for feat_config in self.feat_configs])}]",
            'reduc': self.reduc_config['type'] if self.reduc_config else 'None',
            'feat_norm': self.feat_norm
        }
        hparams = {**init_hparams, **self.anom_config}

        for key, value in hparams.items():
            if isinstance(value, list):
                hparams[key] = ', '.join(map(str, value))
            elif isinstance(value, (int, float)):
                hparams[key] = str(value)
            elif value is None:
                hparams[key] = 'None'

        return hparams
        
        # [TODO] all hyperparams must be string (convert any hp which is int or list into str)

def get_anom_config(anom_type, **kwargs):
    """
    Parameters
    ----------
    anom_type : str
        The type of fault detection algorithm to be used. 
        - `1SVM`: OneClass Support Vector Machine
        - `IF`: Isolation Forest)
    **kwargs : dict
        For all options of `anom_type`:
        - `1SVM`: **kernel**, **nu**, **gamma**
        - `IF`: **n_estimators**, **seed**, **contam**, **n_jobs**

    """
    anom_config = {}
    anom_config['anom_type'] = anom_type

    if anom_type == '1SVM':
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