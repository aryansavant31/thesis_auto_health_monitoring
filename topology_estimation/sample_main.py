import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from utils.tensorboard_logger import StdoutToTensorBoard

# This will resolve log_dir relative to main.py
version = 'v28'
connsol_log = StdoutToTensorBoard(log_dir=f"logs/trials/nri/version_{version}")

print("Starting run from main script...")

from data.config import DataConfig
from data.prep import load_spring_particle_data

data_config = DataConfig()
data_config.set_train_valid_dataset()

# get node and edge dataset path from which data will be loaded
node_ds_paths, edge_ds_paths = data_config.get_dataset_paths()

# load datalaoders
train_loader, valid_loader, test_loader = load_spring_particle_data(node_ds_paths, edge_ds_paths)

dataiter = iter(train_loader)
data = next(dataiter)

n_nodes = data[0].shape[1]
n_timesteps = data[0].shape[2]
n_dims = data[0].shape[3]

print(f"Number of nodes: {n_nodes}")
print(f"Number of timesteps: {n_timesteps}")  
print(f"Number of dimensions: {n_dims}")

import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

# Generate off-diagonal fully connected graph
off_diag = np.ones([5, 5]) - np.eye(5)

rec_rel = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
send_rel = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
rec_rel = torch.FloatTensor(rec_rel)
send_rel = torch.FloatTensor(send_rel)

rec_rel = rec_rel.to(device)
send_rel = send_rel.to(device)

from topology_estimation.config import TopologyEstimatorConfig
from topology_estimation.encoder import Encoder
from torchinfo import summary

tp_config = TopologyEstimatorConfig()
tp_config.set_encoder_params()

encoder = Encoder(n_timesteps=n_timesteps, 
                  n_dims=n_dims,
                  pipeline=tp_config.encoder_pipeline, 
                  n_edge_types=tp_config.n_edge_types, 
                  is_residual_connection=tp_config.is_residual_connection,
                  edge_emd_configs=tp_config.edge_emb_configs_enc, 
                  node_emd_configs=tp_config.node_emb_configs_enc, 
                  drop_out_prob=tp_config.dropout_prob_enc,
                  batch_norm=tp_config.batch_norm_enc, 
                  attention_output_size=tp_config.attention_output_size)

encoder.set_input_graph(rec_rel, send_rel)
enocder = encoder.to(device)

# print(summary(encoder, (64, 5, n_timesteps, n_dims)))
print(encoder)

from topology_estimation.decoder import Decoder

tp_config.set_decoder_params()

decoder = Decoder(n_dim=n_dims,
                  msg_out_size=tp_config.msg_out_size,
                  n_edge_types=tp_config.n_edge_types,
                  skip_first=tp_config.skip_first_edge_type,
                  edge_mlp_config=tp_config.edge_mlp_config_dec,
                  recurrent_emd_type=tp_config.recurrent_emd_type,
                  out_mlp_config=tp_config.out_mlp_config_dec,
                  do_prob=tp_config.dropout_prob_dec,
                  is_batch_norm=tp_config.is_batch_norm_dec)


# generate random edge matrix
edge_matrix = torch.rand((64, 20, 2))
edge_matrix = edge_matrix.to(device)

decoder.set_input_graph(rec_rel, send_rel)
decoder.set_edge_matrix(edge_matrix)
decoder.set_run_params()

decoder = decoder.to(device)

# print(summary(decoder, (64, 5, n_timesteps, n_dims)))
print(decoder)

from topology_estimation.nri import NRI


nri_model = NRI(encoder, decoder)
nri_model.set_run_params()
nri_model.set_input_graph(rec_rel, send_rel)

print(nri_model)

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import RichProgressBar

tp_config.set_training_params()
nri_model.set_training_params()

if tp_config.is_log:
    logger = TensorBoardLogger('logs/trials', name=None, version=version)
else:
    logger = None

trainer = Trainer(
    max_epochs=tp_config.max_epochs,
    logger=logger,
    enable_progress_bar=True,
    log_every_n_steps=1,)

trainer.fit(model=nri_model, train_dataloaders=train_loader)