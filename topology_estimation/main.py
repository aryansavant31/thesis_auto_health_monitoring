"""
main class for topology estimation with both training and inference code
"""
from config import TopologyEstimatorConfig
from encoder_blocks import Encoder
from decoder_blocks import Decoder
from topology_estimator import TopologyEstimator

def main(run_type, block_type='tp', model_num=None):
    par = TopologyEstimatorConfig()
    # Load data

    # ---- Train the model ----
    
    if run_type == 'train':
        
        # Initialize encoder and decoder
        encoder = Encoder(par.encoder_pipeline, 
                        par.n_edge_types, 
                        par.is_residual_connection, 
                        n_timesteps, 
                        n_dims, 
                        par.edge_emb_configs_enc, 
                        par.node_emb_configs_enc, 
                        par.dropout_prob_mlp_enc, 
                        par.batch_norm_mlp_enc, 
                        par.attention_output_size)
        
        decoder = Decoder()

        # Initialize topology estimator
        topology_estimator = TopologyEstimator(encoder, decoder)
    

    # ----- Run a trained model ------

    elif run_type == 'run':
        # load model as per model_num

        # run the loaded model

        pass

