import os, sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT_DIR) if ROOT_DIR not in sys.path else None

# SETTINGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
# sys.path.insert(0, SETTINGS_DIR) if SETTINGS_DIR not in sys.path else None

# global imports
from data.config import DataConfig

class AnomalyDetectorInferConfig:
    def __init__(self, data_config:DataConfig, run_type, selected_model_path=None):
        """
        Parameters
        ----------
        data_config : DataConfig
            The data configuration object (also stored as an attribute).

        Miscellaneous Attributes
        -----------------------
        log_config : object
            Train config object of selected trained model.
        ckpt_path : str
            Path to the selected anomaly detector model
        """
        from fault_detection.settings.manager import load_log_config, get_selected_model_path

        self.data_config = data_config
        self.run_type = run_type
        self.selected_model_path = selected_model_path

        if self.selected_model_path is None:
            self.selected_model_path = get_selected_model_path()

        self.log_config = load_log_config(self.selected_model_path)

        self.is_log = True
        self.version = 1
        
        self.num_workers = 1
        self.batch_size = 1

        self.nok_percentage = 0.99
        self.cutoff_freq = 0
        self.update_infer_configs()

        self.infer_hparams = self.get_infer_hparams()

        self.test_plots = {
            'confusion_matrix_simple'              : [True, {}],
            'confusion_matrix_advance'      : [True, {}],
            'roc_curve'                     : [False, {}],
            'anomaly_score_dist_simple-1'   : [True, {'is_pred':True, 'is_log_x': False, 'num':1}],
            'anomaly_score_dist_simple-2'   : [True, {'is_pred':True, 'is_log_x': True, 'bins':80, 'num':2}],
            # 'anomaly_score_dist_simple-2'   : [False, {'is_pred':False, 'is_log_x': False}],
            'anomaly_score_dist_advance-1'    : [True, {'num': 1, 'is_log_x': False}],
            'anomaly_score_dist_advance-2'    : [True, {'num': 2, 'is_log_x': True, 'bins':80}],
            # 'anomaly_score_dist_advance-2'    : [True, {'num': 2}],
            'pair_plot'                     : [True, {}],
        }

    def update_infer_configs(self):
        """
        Update the model parameters with the current infer settings.
        """
        self.log_config.domain_config.update({'cutoff_freq' : self.cutoff_freq})
        self.domain_config = self.log_config.domain_config

    def get_infer_hparams(self):

        domain_str = self._get_config_str([self.domain_config])
        hparams = {
            f'domain/{self.run_type}': domain_str,
            f'{self.run_type}_version': self.version,
            f'batch_size/{self.run_type}': self.batch_size,
            f'nok_percentage/{self.run_type}': self.nok_percentage,
        }

        for key, value in hparams.items():
            if isinstance(value, list):
                hparams[key] = ', '.join(map(str, value))
            elif isinstance(value, (int, float, dict)):
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
    
class AnomalyDetectorInferSweep:
    def __init__(self, data_config: DataConfig):
        """
        Infer sweep 1 - feature robustness agaisnt noise
        """
        from fault_detection.settings.manager import get_selected_model_path

        self.data_config = data_config
        self.infer_sweep_num = 2

        self.selected_model_path = get_selected_model_path(is_multi=True)

        self.batch_size = [1]
        self.cutoff_freq = [0]
        
        
if __name__ == "__main__":
    from fault_detection.settings.manager import SelectFaultDetectionModel
    user_text = "To view/select trained fault detection models, type (a)\nTo view custom tested models, type (b)\nTo view predicted models, type (c)\nEnter input: "
    user_input = input(user_text).strip("'\"")
    if user_input.lower() == 'a':
        run_type = 'train'
    elif user_input.lower() == 'b':
        run_type = 'custom_test'
    elif user_input.lower() == 'c':
        run_type = 'predict'
    else:
        raise ValueError("Invalid input. Please enter 'a', 'b', or 'c'.")
    model_selector = SelectFaultDetectionModel(run_type=run_type)
    model_selector.select_model_and_params()
        