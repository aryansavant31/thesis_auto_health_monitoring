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
    framework = 'nri'

    def __init__(self, data_config:DataConfig, run_type, selected_model_path=None):
        """
        Miscellaneous Attributes
        -----------------------
        log_config : object
            Train config object of selected trained model.
        selected_model_path : str
            Path to the selected NRI model
        """
        from topology_estimation.settings.manager import load_log_config, get_selected_model_path
        
        self.data_config = data_config
        self.run_type = run_type
        self.selected_model_path = selected_model_path

        if self.selected_model_path is None:
            self.selected_model_path = get_selected_model_path(framework=NRIInferConfig.framework)

        self.log_config = load_log_config(framework=NRIInferConfig.framework, model_path=self.selected_model_path)

        self.is_log = True
        self.version = 1
        
        self.num_workers = 1
        self.batch_size = 1

    # Encoder run parameters
        # input processor parameters
        self.enc_cutoff_freq = 100
        self.log_config.enc_domain_config.update({'cutoff_freq' : self.enc_cutoff_freq})
        self.enc_domain_config = self.log_config.enc_domain_config

        self.temp = 0.1
        self.is_hard = False

    # Decoder 
        # input processor parameters
        self.dec_cutoff_freq = 100
        self.log_config.dec_domain_config.update({'cutoff_freq' : self.dec_cutoff_freq})
        self.dec_domain_config = self.log_config.dec_domain_config

        # run parameters
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

    # Hyperparameters for logging
        self.infer_hyperparams = self.get_infer_hyperarams()


    def get_infer_hyperarams(self):
        domain_dec_str = get_config_str([self.dec_domain_config])
        domain_enc_str = get_config_str([self.enc_domain_config])
        spf_domain_str = get_config_str([self.spf_domain_config])
        spf_feat_str = get_config_str(self.spf_feat_configs)
        spf_reduc_str = get_config_str([self.spf_reduc_config]) if self.spf_reduc_config is not None else 'None'

        hyperparams = {
            f'{self.run_type}_version': self.version,
            f'batch_size_{self.run_type}': self.batch_size,
            f'dec/domain_{self.run_type}': domain_dec_str,
            f'dec/skip_first_edge_{self.run_type}': self.skip_first_edge_type,
            f'dec/pred_steps_{self.run_type}': self.pred_steps,
            f'dec/is_burn_in_{self.run_type}': self.is_burn_in,
            f'dec/burn_in_steps_{self.run_type}': self.burn_in_steps,
            f'dec/is_dynamic_graph_{self.run_type}': self.is_dynamic_graph,

            f'enc/domain_{self.run_type}': domain_enc_str,
            f'enc/temp_{self.run_type}': self.temp,
            f'enc/is_hard_{self.run_type}': self.is_hard,

            # sparsifier parameters
            f'spf/config_{self.run_type}': f"{self.spf_config['type']} (expert={self.spf_config['is_expert']})" if self.spf_config['type'] != 'no_spf' else 'no_spf',
            f'spf/domain_{self.run_type}': spf_domain_str,
            f'spf/raw_data_norm_{self.run_type}': self.spf_raw_data_norm,
            f'spf/feats_{self.run_type}': f"[{spf_feat_str}]",
            f'spf/reduc_{self.run_type}': spf_reduc_str,
            f'spf/feat_norm_{self.run_type}': self.spf_feat_norm
        }

        for key, value in hyperparams.items():
            if isinstance(value, list):
                hyperparams[key] = ', '.join(map(str, value))
            elif isinstance(value, (int, float, dict, dict)):
                hyperparams[key] = str(value)
            elif value is None:
                hyperparams[key] = 'None'

        return hyperparams

class NRIInferSweep:
    def __init__(self, data_config:DataConfig):
        pass

class DecoderInferConfig:
    framework = 'decoder'

    def __init__(self, data_config:DataConfig, run_type, selected_model_path=None):
        """
        Miscellaneous Attributes
        -----------------------
        log_config : object
            Train config object of selected trained model.
        ckpt_path : str
            Path to the selected Decoder model
        """
        from topology_estimation.settings.manager import load_log_config, get_selected_model_path

        self.data_config = data_config
        self.run_type = run_type
        self.selected_model_path = selected_model_path

        if self.selected_model_path is None:
            self.selected_model_path = get_selected_model_path(framework=DecoderInferConfig.framework)

        self.log_config = load_log_config(framework=DecoderInferConfig.framework, model_path=self.selected_model_path)


        self.is_log = False
        self.version = 1
        
        self.num_workers = 1
        self.batch_size = 50

    # Input processor parameters
        self.dec_cutoff_freq = 100
        self.log_config.dec_domain_config.update({'cutoff_freq' : self.dec_cutoff_freq})
        self.dec_domain_config = self.log_config.dec_domain_config

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

    # Hyperparameters for logging
        self.infer_hyperparams = self.get_infer_hyperparams()


    def get_infer_hyperparams(self):
        domain_dec_str = get_config_str([self.dec_domain_config])
        spf_domain_str = get_config_str([self.spf_domain_config])
        spf_feat_str = get_config_str(self.spf_feat_configs)
        spf_reduc_str = get_config_str([self.spf_reduc_config]) if self.spf_reduc_config is not None else 'None'

        hyperparams = {
            f'{self.run_type}_version': self.version,
            f'batch_size_{self.run_type}': self.batch_size,
            f'dec/domain_{self.run_type}': domain_dec_str,
            f'dec/skip_first_edge_{self.run_type}': self.skip_first_edge_type,
            f'dec/pred_steps_{self.run_type}': self.pred_steps,
            f'dec/is_burn_in_{self.run_type}': self.is_burn_in,
            f'dec/burn_in_steps_{self.run_type}': self.burn_in_steps,
            f'dec/is_dynamic_graph_{self.run_type}': self.is_dynamic_graph,
            f'enc/temp_{self.run_type}': self.temp,
            f'enc/is_hard_{self.run_type}': self.is_hard,

            # sparsifier parameters
            f'spf/config_{self.run_type}': f"{self.spf_config['type']} (expert={self.spf_config['is_expert']})" if self.spf_config['type'] != 'no_spf' else 'no_spf',
            f'spf/domain_{self.run_type}': spf_domain_str,
            f'spf/raw_data_norm_{self.run_type}': self.spf_raw_data_norm,
            f'spf/feats_{self.run_type}': f"[{spf_feat_str}]",
            f'spf/reduc_{self.run_type}': spf_reduc_str,
            f'spf/feat_norm_{self.run_type}': self.spf_feat_norm
        }

        for key, value in hyperparams.items():
            if isinstance(value, list):
                hyperparams[key] = ', '.join(map(str, value))
            elif isinstance(value, (int, float, dict, dict)):
                hyperparams[key] = str(value)
            elif value is None:
                hyperparams[key] = 'None'

        return hyperparams


def get_config_str(configs:list):
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
    from topology_estimation.settings.manager import SelectTopologyEstimatorModel

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