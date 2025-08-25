import os, sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT_DIR) if ROOT_DIR not in sys.path else None
# from manager import SelectTopologyEstimatorModel, load_selected_config, get_selected_ckpt_path

# TOPOLOGY_ESTIMATION_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(os.path.dirname(TOPOLOGY_ESTIMATION_DIR))

# global imports
from data.config import DataConfig, get_domain_config
from feature_extraction.settings.feature_config import get_freq_feat_config, get_time_feat_config, get_reduc_config

# local imports
from topology_estimation.settings.train_config import get_spf_config

class NRIInferConfig:
    def __init__(self, data_config:DataConfig):
        """
        Miscellaneous Attributes
        -----------------------
        log_config : object
            Train config object of selected trained model.
        ckpt_path : str
            Path to the selected NRI model
        """
        from topology_estimation.settings.manager import load_selected_config, get_selected_ckpt_path
        
        self.data_config = data_config
        self.log_config = load_selected_config('nri')
        self.ckpt_path = get_selected_ckpt_path('nri')

        self.is_log = True
        self.version = 2
        
        self.num_workers = 1
        self.batch_size = 1

    # Encoder run parameters
        self.temp = 0.1
        self.is_hard = False

    # Decoder run parameters
        self.skip_first_edge_type = False
        self.pred_steps = 1
        self.is_burn_in = False
        self.burn_in_steps = 1
        self.is_dynamic_graph = False

    # Sparsifier parameters
        self.spf_config = get_spf_config('no_spf', is_expert=True)
        
        self.spf_domain_config   = get_domain_config('time', data_config=self.data_config)
        self.spf_raw_data_norm = None 
        self.spf_feat_configs = [
            # get_time_feat_config('first_n_modes', data_config=self.data_config),
        ]    
        self.spf_feat_norm = None
        self.spf_reduc_config = None


class DecoderInferConfig:
    def __init__(self, data_config:DataConfig):
        """
        Miscellaneous Attributes
        -----------------------
        log_config : object
            Train config object of selected trained model.
        ckpt_path : str
            Path to the selected Decoder model
        """
        from topology_estimation.settings.manager import load_selected_config, get_selected_ckpt_path

        self.data_config = data_config
        self.log_config = load_selected_config('decoder')
        self.ckpt_path = get_selected_ckpt_path('decoder')

        self.is_log = True
        self.version = 2
        
        self.num_workers = 1
        self.batch_size = 1

    # Decoder run parameters
        self.skip_first_edge_type = False
        self.pred_steps = 1
        self.is_burn_in = False
        self.burn_in_steps = 1
        self.is_dynamic_graph = False
        
        # if dynamic graph is true
        self.temp = 1.0        # temperature for Gumble Softmax
        self.is_hard = True 

    # Sparsifier parameters 
        self.spf_config = get_spf_config('no_spf', is_expert=True)
        
        self.spf_domain_config   = get_domain_config('time', data_config=self.data_config)
        self.spf_raw_data_norm = None 
        self.spf_feat_configs = [
            # get_time_feat_config('first_n_modes', data_config=self.data_config),
        ]    
        self.spf_feat_norm = None
        self.spf_reduc_config = None





if __name__ == "__main__":
    from manager import SelectTopologyEstimatorModel

    user_text_1 = "To view/select nri models, type (a)\nTo view/select decoder models, type (b)\nEnter input: "
    user_input_1 = input(user_text_1).strip("'\"")
    if user_input_1.lower() == 'a':
        framework = 'nri'
    elif user_input_1.lower() == 'b':
        framework = 'decoder'
    else:
        raise ValueError("Invalid input. Please enter 'a' or 'b'.")
    

    user_text_2 = f"To view/select trained {framework} models, type (a)\nTo view custom tested {framework} models, type (b)\nTo view predicted {framework} models, type (c)\nEnter input: "
    user_input_2 = input(user_text_2).strip("'\"")
    if user_input_2.lower() == 'a':
        run_type = 'train'
    elif user_input_2.lower() == 'b':
        run_type = 'custom_test'
    elif user_input_2.lower() == 'c':
        run_type = 'predict'
    else:
        raise ValueError("Invalid input. Please enter 'a', 'b', or 'c'.")
    model_selector = SelectTopologyEstimatorModel(framework=framework, run_type=run_type)
    model_selector.select_ckpt_and_params()