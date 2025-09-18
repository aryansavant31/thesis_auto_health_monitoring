import os, sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT_DIR) if ROOT_DIR not in sys.path else None

# SETTINGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
# sys.path.insert(0, SETTINGS_DIR) if SETTINGS_DIR not in sys.path else None


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
        self.batch_size  = 1
        self.train_rt    = 0.8
        self.test_rt     = 0.2
        self.num_workers = 1

    # 2: Model parameters
        self.anom_type = 'IF'
        self.n_estimators = 1000
        self.contam = 0.01
        self.set_anom_config()

        # uncertainity parameters
        self.ok_percentage = 1

        # run parameters
        self.domain_config = get_domain_config('time')
        self.raw_data_norm = None
        self.feat_configs = [
            #get_freq_feat_config('first_n_modes', n_modes=10)
        ]  
        self.reduc_config = None
        self.feat_norm = None

    # 3: Hyperparameters and plots
        self.hparams = self.get_hparams()

        self.train_plots = {
            'confusion_matrix_simple'              : [True, {}],
            'confusion_matrix_advance'      : [False, {}],
            'roc_curve'                     : [False, {}],
            'anomaly_score_dist_simple-1'   : [False, {'is_pred':True, 'is_log_x': False, 'num':1}],
            'anomaly_score_dist_simple-2'   : [False, {'is_pred':True, 'is_log_x': True, 'bins':80, 'num':2}],
            # 'anomaly_score_dist_simple-2'   : [True, {'is_pred':False}],
            'anomaly_score_dist_advance-1'    : [False, {'num': 1, 'is_log_x': False}],
            'anomaly_score_dist_advance-2'    : [False, {'num': 2, 'is_log_x': True, 'bins':80}],
            # 'anomaly_score_dist_advance-2'    : [False, {'percentile_ok': 95, 'percentile_nok': 95, 'num': 2}],
            'pair_plot'                     : [False, {}],
        }

        self.test_plots = {
            'confusion_matrix_simple'              : [True, {}],
            'confusion_matrix_advance'      : [False, {}],
            'roc_curve'                     : [False, {}],
            'anomaly_score_dist_simple-1'   : [False, {'is_pred':True, 'is_log_x': False, 'num':1}],
            'anomaly_score_dist_simple-2'   : [False, {'is_pred':True, 'is_log_x': True, 'bins':80, 'num':2}],
            # 'anomaly_score_dist_simple-2'   : [True, {'is_pred':False}],
            'anomaly_score_dist_advance-1'    : [False, {'num': 1, 'is_log_x': False}],
            'anomaly_score_dist_advance-2'    : [False, {'num': 2, 'is_log_x': True, 'bins':80}],
            #'anomaly_score_dist_advance-2'    : [True, {'percentile_ok': 95, 'percentile_nok': 95, 'num': 2}],
            'pair_plot'                     : [False, {}],
        }

    def set_anom_config(self):
        if self.anom_type == 'IF':
            self.anom_config = get_anom_config('IF', n_estimators=self.n_estimators, contam=self.contam)

    def get_hparams(self):
        """
        Sets the hyperparameters for the anomaly detection model.
        """
        domain_str = self._get_config_str([self.domain_config])
        feat_str = self._get_config_str(self.feat_configs)
        reduc_str = self._get_config_str([self.reduc_config]) if self.reduc_config else 'None'

        init_hparams = {
            'model_num': self.model_num,
            'batch_size': self.batch_size,
            'train_rt': self.train_rt,
            'test_rt': self.test_rt,
            'window_length': self.data_config.window_length,
            'stride': self.data_config.stride,

            'ok_percentage': self.ok_percentage,
            'domain': domain_str,
            'raw_data_norm': self.raw_data_norm,
            'feats': f"[{feat_str}]",
            'reduc': reduc_str,
            'feat_norm': self.feat_norm
        }
        hparams = {**init_hparams, **self.anom_config}

        for key, value in hparams.items():
            if isinstance(value, list):
                hparams[key] = ', '.join(map(str, value))
            elif isinstance(value, (int, float, dict)):
                if key != 'model_num':  # keep model_num as int
                    hparams[key] = str(value)
            elif value is None:
                hparams[key] = 'None'

        return hparams
        
    def _get_config_str(self, configs:list):
        """
        Get a neat string that has the type of config and its parameters.
        Eg: "PCA(comps=3)"
        """
        config_strings = []

        for config in configs:
            additional_keys = ', '.join([f"{key}={value}" for key, value in config.items() if key not in ['fs', 'type', 'feat_list']])
            if additional_keys:
                config_strings.append(f"{config['type']}({additional_keys})")
            else:
                config_strings.append(f"{config['type']}")

        return ', '.join(config_strings)

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

        `1SVM`: 
                - **kernel** ('linear', 'poly', 'rbf' *_(default)_*, 'sigmoid'), 
                - **gamma** ('scale' *_(default)_*, 'auto', float), 
                - **nu** (float, default=0.5), 
                - **is_shrinking** (bool, default=True), 
                - **tol** (float, default=1e-3), 
                - **cache_size** (float, default=200), 
                - **is_verbose** (bool, default=False), 
                - **max_iter** (int, default=-1), 
                - **coef0** (float, default=0.0, for 'poly' and 'sigmoid' kernels), 
                - **degree** (int, default=3, for 'poly' kernel)

        `IF`: 
                - **n_estimators** (int, default=100), 
                - **seed** (int, default=42), 
                - **contam** ('auto' *_(default)_*, float), 
                - **n_jobs** (int, default=-1), 
                - **verbose** (int, default=0), 
                - **max_samples** ('auto' *_(default)_*, int, float), 
                - **bootstrap** (bool, default=False), 
                - **warm_start** (bool, default=False), 
                - **max_features** (float, default=1.0)
        
    """
    anom_config = {}
    anom_config['anom_type'] = anom_type

    if anom_type == '1SVM':
        anom_config['1SVM/kernel'] = kwargs.get('kernel', 'rbf')
        anom_config['1SVM/gamma'] = kwargs.get('gamma', 'scale')
        anom_config['1SVM/nu'] = kwargs.get('nu', 0.5)
        anom_config['1SVM/is_shrinking'] = kwargs.get('is_shrinking', True)
        anom_config['1SVM/tol'] = kwargs.get('tol', 1e-3)
        anom_config['1SVM/cache_size'] = kwargs.get('cache_size', 200)
        anom_config['1SVM/is_verbose'] = kwargs.get('is_verbose', False)
        anom_config['1SVM/max_iter'] = kwargs.get('max_iter', -1)
        anom_config['1SVM/coef0'] = kwargs.get('coef0', 0.0)  # for 'poly' and 'sigmoid' kernels
        anom_config['1SVM/degree'] = kwargs.get('degree', 3)  # for 'poly' kernel

    elif anom_type == 'IF':
        anom_config['IF/n_estimators'] = kwargs.get('n_estimators', 100)
        anom_config['IF/seed'] = kwargs.get('seed', 42)
        anom_config['IF/contam'] = kwargs.get('contam', 'auto')
        anom_config['IF/n_jobs'] = kwargs.get('n_jobs', -1)
        anom_config['IF/verbose'] = kwargs.get('verbose', 0)
        anom_config['IF/max_samples'] = kwargs.get('max_samples', 'auto')
        anom_config['IF/bootstrap'] = kwargs.get('bootstrap', False)
        anom_config['IF/warm_start'] = kwargs.get('warm_start', False)
        anom_config['IF/max_features'] = kwargs.get('max_features', 1.0)

        # hyperparameters for isolation forest

        # anom_config['hparams'] = {
        #     HP_NUM_TREES.name: anom_config['n_estimators'],
        #     HP_CONTAM.name: anom_config['contam'],
        # }

    return anom_config  

class AnomalyDetectorTrainSweep:
    def __init__(self, data_config:DataConfig):
        """
        Train Sweep 1: Obj is to evalaute consistency of results when using same models and data. (Does model interpret same data differently)
        Train Sweep 2: Obj is to evaluate different time feature performance
        """
        self.data_config = data_config

        self.train_sweep_num = 3

    # 1: Training parameters
        # dataset parameters
        self.batch_size  = [1]
        self.train_rt    = [0.8]
        self.test_rt     = [0.2]
        self.num_workers = [1]

    # 2: Model parameters
        self.anom_type = ['IF']
        self.n_estimators = [2000, 2000, 2000, 2000]
        self.contam = [0.01]

        # run parameters
        self.domain_config = [get_domain_config('freq')]
        self.raw_data_norm = [None]
        self.feat_configs = [
            [],
            [get_freq_feat_config('first_n_modes', n_modes=10)]
            #[get_time_feat_config('mean')], 
            # [get_time_feat_config('kurtosis')], 
            # [get_freq_feat_config('stdF')], 
            # [get_time_feat_config('max')],
            # [get_time_feat_config('peak_to_peak')], 
            #[get_time_feat_config('skewness')]

        ]
        self.reduc_config = [None]
        self.feat_norm = [None]


if __name__ == "__main__":
    
    from fault_detection.settings.manager import SelectFaultDetectionModel
    model_selector = SelectFaultDetectionModel(run_type='train')
    model_selector.select_model_and_params()