import os
import sys
from manager import SelectTopologyEstimatorModel, load_selected_config, get_selected_ckpt_path

TOPOLOGY_ESTIMATION_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(TOPOLOGY_ESTIMATION_DIR))

from feature_extraction.settings import get_freq_fex_config


class NRIPredictConfig:
    def __init__(self):
        self.log_config = load_selected_config()
        self.ckpt_path = get_selected_ckpt_path()

        self.set_predict_params()
        self.set_run_params()
  
    def set_predict_params(self):
        self.version = 2
        self.batch_size = 50
        self.amt_rt = 0.8

    def set_custom_test_params(self):
        pass

    def set_run_params(self):
       # input graph paramters
        self.sparsif_type     = 'path'
        self.domain_sparsif   = self.log_config.domain_sparsif_config  # options: time, frequency
        self.fex_configs_sparsif = self.log_config.fex_configs_sparsif  # first feature extraction config type

        # [TODO]: Add domain config, raw_norm and fex_norm, reduc_config for sparsifier, encoder and decoder (see fault detection config)

        self.domain_encoder   = self.log_config.domain_encoder_config
        self.norm_type_encoder = self.log_config.norm_type_encoder
        self.fex_configs_encoder = self.log_config.fex_configs_encoder

        # Gumble Softmax Parameters
        self.temp = 1.0       # temperature for Gumble Softmax
        self.is_hard = True      # if True, use hard Gumble Softmax

        # decoder run parameters
        self.domain_decoder = self.log_config.domain_decoder_config   # options: time, frequency
        self.norm_type_decoder = self.log_config.norm_type_decoder
        self.fex_configs_decoder = self.log_config.fex_configs_decoder

        self.skip_first_edge_type = True 
        # TASK: add rest of the decoder run params

if __name__ == "__main__":
    user_text = "To view/select trained topology (edge) estimation models, type (a)\nTo view custom tested models, type (b)\nTo view predicted models, type (c)\nEnter input: "
    user_input = input(user_text).strip("'\"")
    if user_input.lower() == 'a':
        run_type = 'train'
    elif user_input.lower() == 'b':
        run_type = 'custom_test'
    elif user_input.lower() == 'c':
        run_type = 'predict'
    else:
        raise ValueError("Invalid input. Please enter 'a', 'b', or 'c'.")
    model_selector = SelectTopologyEstimatorModel(framework='directed_graph', run_type=run_type)
    model_selector.select_ckpt_and_params()