import os, sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT_DIR) if ROOT_DIR not in sys.path else None

# SETTINGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
# sys.path.insert(0, SETTINGS_DIR) if SETTINGS_DIR not in sys.path else None

# global imports
from data.config import DataConfig

class AnomalyDetectorInferConfig:
    def __init__(self, data_config:DataConfig):
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
        self.log_config = load_log_config()
        self.ckpt_path = get_selected_model_path()

        self.is_log = True
        self.version = 1
        
        self.num_workers = 1
        self.batch_size = 1

        self.log_config.domain_config.update({'cutoff_freq' : 100})
        self.domain_config = self.log_config.domain_config
        print("domain config", self.domain_config)

        self.new_hparams = self.get_new_hparams()

        self.test_plots = {
            'confusion_matrix'      : [True, {}],
            'roc_curve'             : [False, {}],
            'anomaly_score_dist-1'  : [True, {'is_pred':True}],
            'anomaly_score_dist-2'  : [True, {'is_pred':False}],
            'pair_plot'             : [True, {}],
        }

        
    def get_new_hparams(self):

        domain_str = self._get_config_str([self.domain_config])
        new_hparams = {
            'domain': domain_str,
        }

        for key, value in new_hparams.items():
            if isinstance(value, list):
                new_hparams[key] = ', '.join(map(str, value))
            elif isinstance(value, (int, float)):
                new_hparams[key] = str(value)
            elif value is None:
                new_hparams[key] = 'None'

        return new_hparams
    
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
        