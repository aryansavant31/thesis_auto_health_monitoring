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
        self.version = 2
        
        self.num_workers = 1
        self.batch_size = 1

        self.test_plots = {
            'confusion_matrix'      : [True, {}],
            'roc_curve'             : [False, {}],
            'anomaly_score_dist-1'  : [True, {'is_pred':True}],
            'anomaly_score_dist-2'  : [True, {'is_pred':False}],
            'pair_plot'             : [True, {}],
        }
        
        
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
        