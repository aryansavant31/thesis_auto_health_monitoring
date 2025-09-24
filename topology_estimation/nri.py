import os, sys
# ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.insert(0, ROOT_DIR) if ROOT_DIR not in sys.path else None

# TP_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.insert(0, TP_DIR) if TP_DIR not in sys.path else None

# import other imports
from pytorch_lightning import LightningModule
import torch
import torch.nn.functional as F
from torch.optim import Adam, SGD
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd
import pickle

# local imports
from .utils.loss import kl_categorical, kl_categorical_uniform, nll_gaussian
from .utils.schedulers import BetaScheduler, TempScheduler
from .encoder import Encoder
from .decoder import Decoder


class NRI(LightningModule):
    def __init__(self):
        super(NRI, self).__init__()
        self.encoder_model_params = None
        self.decoder_model_params = None

    def set_hyperparams(self, hyperparams):
        self.hyperparams = hyperparams

    def set_training_params(
        self, lr_enc=0.001, lr_dec=0.001, 
        is_beta_annealing=True,
        final_beta=1.0, warmup_frac_beta=0.3,
        optimizer='adam', add_const_kld=False,
        loss_type_enc='kld', loss_type_dec='nll', prior=None,
        is_enc_warmup=True, warmup_acc_cutoff=0.75, sustain_enc_warmup=True,
        final_gamma=0.5, warmup_frac_gamma=0.3,
        dec_loss_stabilize_steps=20, dec_loss_bound_update_interval=10, dec_loss_window_size=30,
    ):
        self.lr_enc = lr_enc
        self.lr_dec = lr_dec
        self.is_beta_annealing = is_beta_annealing
        self.final_beta = final_beta
        self.warmup_frac_beta = warmup_frac_beta
        self.optimizer = optimizer
        self.add_const_kld = add_const_kld
        self.prior = prior
        self.loss_type_encoder = loss_type_enc
        self.loss_type_decoder = loss_type_dec

        # warmup parameters
        self.is_enc_warmup = is_enc_warmup
        self.warmup_acc_cutoff = warmup_acc_cutoff
        self.sustain_enc_warmup = sustain_enc_warmup
        self.final_gamma = final_gamma
        self.warmup_frac_gamma = warmup_frac_gamma

        self.dec_loss_stabilize_steps = dec_loss_stabilize_steps 
        self.dec_loss_bound_update_interval = dec_loss_bound_update_interval
        self.dec_loss_window_size = dec_loss_window_size

        print(f"\nTraining parameters set to: \nlr_enc={self.lr_enc}, \nlr_dec={self.lr_dec}, " 
              f"\nfinal_beta={self.final_beta}, \nwarmup_frac={self.warmup_frac_beta}, "
              f"\noptimizer={self.optimizer}, \nloss_type_encoder={self.loss_type_encoder}, \nloss_type_decoder={self.loss_type_decoder}, \nprior={self.prior}, \nadd_const_kld={self.add_const_kld}"
              f"\nis_enc_warmup: {self.is_enc_warmup}, \nwarmup_acc_cutoff: {self.warmup_acc_cutoff}, \nsustain_enc_warmup: {self.sustain_enc_warmup}, "
              f"\nfinal_gamma: {self.final_gamma}, \nwarmup_frac_gamma: {self.warmup_frac_gamma}, "
              f"\ndec_loss_stabilize_steps: {self.dec_loss_stabilize_steps}, \ndec_loss_bound_update_interval: {self.dec_loss_bound_update_interval}, \ndec_loss_window_size: {self.dec_loss_window_size}\n")
        
    def set_input_example_for_graph(self, n_nodes):
        self.example_input_array = torch.rand((1, n_nodes, self.encoder.n_comps, self.encoder.n_dims))

    def set_input_graph(self, rec_rel, send_rel):
        """
        Set the relationship matrices defining the input graph structure.
        
        Parameters
        ----------
        rec_rel : torch.Tensor, shape (n_edges, n_nodes)
            Receiver relationship matrix.
        
        send_rel : torch.Tensor, shape (n_edges, n_nodes)
            Sender relationship matrix.
        """
        self.encoder.set_input_graph(rec_rel, send_rel)
        self.decoder.set_input_graph(rec_rel, send_rel)

    def set_run_params(self, dec_run_params, data_config,
                       init_temp=1.0, min_temp=0.3, decay_temp=0.001, is_hard=True,
                       dynamic_rel=False):
        """
        Parameters
        ----------
        dec_run_params : dict
            Parameters for the decoder run. For details, see the docstring of `Decoder.set_run_params()`.
        temp : float
            Temperature for Gumble Softmax.
        is_hard : bool
            If True, use hard Gumble Softmax.
        """
        self.init_temp = init_temp
        self.min_temp = min_temp
        self.decay_temp = decay_temp
        self.is_hard = is_hard
        self.dynamic_rel = dynamic_rel

        self.encoder.set_run_params(data_config=data_config)
        self.decoder.set_run_params(**dec_run_params, data_config=data_config)
        
    def build_model(self):
        """
        Build the NRI model by constructing the encoder and decoder.
        """
        # build encoder
        self.encoder = Encoder()
        for key, value in self.encoder_model_params.items():
            setattr(self.encoder, key, value)
        self.encoder.build_model()
        self.encoder.raw_data_normalizer = self.encoder_raw_data_normalizer
        self.encoder.feat_normalizer = self.encoder_feat_normalizer

        # build decoder
        self.decoder = Decoder()
        for key, value in self.decoder_model_params.items():
            setattr(self.decoder, key, value)
        self.decoder.build_model()
        self.decoder.raw_data_normalizer = self.decoder_raw_data_normalizer
        self.decoder.feat_normalizer = self.decoder_feat_normalizer

    def fit_normalizers(self, train_loader):
        """
        Fit the normalizers of the encoder and decoder using training data.

        Parameters
        ----------
        train_loader : DataLoader
            DataLoader for the training dataset.
        """
        self.encoder.fit_normalizers(train_loader)
        self.decoder.fit_normalizers(train_loader)

    def print_model_info(self):
        """
        Print the model information

        Note
        ----
        Ensure to run `build_model()` before running this method.
        """
        print(5 * '-', 'NRI Model Summary', 5 * '-')
        print(2 * '-', 'Encoder Summary')
        print(self.encoder)
        print(2 * '-', 'Decoder Summary')
        print(self.decoder)

    def edge_matrix_to_rel(self, edge_matrix, n_nodes, threshold=0.5):
        """
        Construct rec_rel and send_rel matrices from edge_matrix averaged over batch.

        Parameters
        ----------
        edge_matrix : torch.Tensor, shape (batch_size, n_edges, n_edge_types)
            Edge matrix from encoder (values between 0 and 1).
        n_nodes : int
            Number of nodes.
        threshold : float
            Threshold for deciding edge existence.

        Returns
        -------
        rec_rel : torch.Tensor, shape (n_edges, n_nodes)
        send_rel : torch.Tensor, shape (n_edges, n_nodes)
        """
        # Average over batch
        edge_vals = edge_matrix[:, :, 0].mean(dim=0)  # shape: (n_edges,)
        n_edges = edge_matrix.shape[1]
        rec_rel = torch.zeros(n_edges, n_nodes, device=edge_matrix.device)
        send_rel = torch.zeros(n_edges, n_nodes, device=edge_matrix.device)

        edge_idx = 0
        for sender in range(n_nodes):
            for receiver in range(n_nodes):
                if sender != receiver:
                    if edge_vals[edge_idx] <= threshold:  # edge exists
                        rec_rel[edge_idx, receiver] = 1.0
                        send_rel[edge_idx, sender] = 1.0
                    edge_idx += 1
        return rec_rel, send_rel

    def forward(self, data):
        """
        Run the forward pass of the encoder and decoder.

        Note
        ----
        Ensure to run `set_input_graph()` and `set_decoder_run_params()` before running this method.

        Parameters
        ----------
        data : torch.Tensor, shape (batch_size, n_nodes, n_timesteps, n_dims)
            Input data tensor containing the entire trajectory data of all nodes.
        
        Returns
        -------
        edge_pred : torch.Tensor, shape (batch_size, n_edges, n_edge_types)
            Predicted edge probabilities.
        x_pred : torch.Tensor, shape (batch_size, n_nodes, n_components-1, n_dim)
            Predicted node data
        x_var : torch.Tensor, shape (batch_size, n_nodes, n_components-1, n_dim)
            Variance of the predicted node data.
        """
        # Encoder
        logits = self.encoder(data)
        temp = self.temp_scheduler.step()
        edge_matrix = F.gumbel_softmax(logits, tau=temp, hard=self.is_hard)
        edge_pred = F.softmax(logits, dim=-1)

        # print("Edge pred", edge_pred[0,:,:]) # DEBUG
        # print("logits", logits[0,:,:]) # DEBUG
        # print("Edge matrix from gumble", edge_matrix[0,:,:]) # DEBUG

        # Decoder
        # make rec_rel and send_rel from edge_matrix
        if self.dynamic_rel:
            n_nodes = data.size(1)
            rec_rel, send_rel = self.edge_matrix_to_rel(edge_pred, n_nodes, threshold=0.5)
            self.decoder.set_input_graph(rec_rel, send_rel)

        self.decoder.set_edge_matrix(edge_matrix)
        x_pred, x_var = self.decoder(data)

        return edge_pred, edge_matrix, x_pred, x_var, temp
    
# ================== PYTORCH LIGHTNING TRAINER METHODS ================== #

    def on_load_checkpoint(self, checkpoint):
        # model params
        self.encoder_model_params = checkpoint['encoder_model_params']
        self.decoder_model_params = checkpoint['decoder_model_params']
        self.encoder_raw_data_normalizer = checkpoint['encoder_raw_data_normalizer']
        self.encoder_feat_normalizer = checkpoint['encoder_feat_normalizer']
        self.decoder_raw_data_normalizer = checkpoint['decoder_raw_data_normalizer']
        self.decoder_feat_normalizer = checkpoint['decoder_feat_normalizer']
        self.hyperparams = checkpoint["hyperparams"]

        # train params
        self.lr_enc = checkpoint['lr_enc']
        self.lr_dec = checkpoint['lr_dec']
        self.is_beta_annealing = checkpoint['is_beta_annealing']
        self.final_beta = checkpoint['final_beta']
        self.warmup_frac_beta = checkpoint['warmup_frac_beta']
        self.optimizer = checkpoint['optimizer']
        self.prior = checkpoint['prior']
        self.loss_type_encoder = checkpoint['loss_type_encoder']
        self.loss_type_decoder = checkpoint['loss_type_decoder']
        self.add_const_kld = checkpoint['add_const_kld']
        self.is_enc_warmup = checkpoint['is_enc_warmup']
        self.warmup_acc_cutoff = checkpoint['warmup_acc_cutoff']
        self.sustain_enc_warmup = checkpoint['sustain_enc_warmup']
        self.final_gamma = checkpoint['final_gamma']
        self.warmup_frac_gamma = checkpoint['warmup_frac_gamma']
        self.dec_loss_stabilize_steps = checkpoint['dec_loss_stabilize_steps']
        self.dec_loss_bound_update_interval = checkpoint['dec_loss_bound_update_interval']
        self.dec_loss_window_size = checkpoint['dec_loss_window_size']

        # load previous training data
        self.train_losses = checkpoint['train_losses']
        self.train_accuracies = checkpoint['train_accuracies']
        self.train_entropys = checkpoint['train_entropys']
        self.train_gains = checkpoint['train_gains']

        self.custom_step = checkpoint['custom_step']
        self.custom_current_epoch = checkpoint['custom_current_epoch']
        self.beta_scheduler = checkpoint['beta_scheduler']
        self.gamma_scheduler = checkpoint['gamma_scheduler']
        self.temp_scheduler = checkpoint['temp_scheduler']

        self.is_decoder_stabilized = checkpoint['is_decoder_stabilized']
        self.decoder_stabilization_counter = checkpoint['decoder_stabilization_counter']
        self.enc_warmup_end_step = checkpoint.get('enc_warmup_end_step', None)
        self.dec_stabilize_step = checkpoint.get('dec_stabilize_step', None)
        self.decoder_loss_upper_bound = checkpoint['decoder_loss_upper_bound']
        self.decoder_loss_lower_bound = checkpoint['decoder_loss_lower_bound']
        self.recent_decoder_losses = checkpoint['recent_decoder_losses']

        # rebuild model
        self.build_model()

    def on_save_checkpoint(self, checkpoint):
        # model params
        checkpoint['encoder_model_params'] = self.encoder_model_params
        checkpoint['decoder_model_params'] = self.decoder_model_params
        checkpoint['encoder_raw_data_normalizer'] = self.encoder.raw_data_normalizer
        checkpoint['encoder_feat_normalizer'] = self.encoder.feat_normalizer
        checkpoint['decoder_raw_data_normalizer'] = self.decoder.raw_data_normalizer
        checkpoint['decoder_feat_normalizer'] = self.decoder.feat_normalizer
        checkpoint["hyperparams"] = self.hyperparams

        # train params
        checkpoint['lr_enc'] = self.lr_enc
        checkpoint['lr_dec'] = self.lr_dec
        checkpoint['is_beta_annealing'] = self.is_beta_annealing
        checkpoint['final_beta'] = self.final_beta
        checkpoint['warmup_frac_beta'] = self.warmup_frac_beta
        checkpoint['optimizer'] = self.optimizer
        checkpoint['prior'] = self.prior
        checkpoint['loss_type_encoder'] = self.loss_type_encoder
        checkpoint['loss_type_decoder'] = self.loss_type_decoder
        checkpoint['add_const_kld'] = self.add_const_kld
        checkpoint['is_enc_warmup'] = self.is_enc_warmup
        checkpoint['warmup_acc_cutoff'] = self.warmup_acc_cutoff
        checkpoint['sustain_enc_warmup'] = self.sustain_enc_warmup
        checkpoint['final_gamma'] = self.final_gamma
        checkpoint['warmup_frac_gamma'] = self.warmup_frac_gamma
        checkpoint['dec_loss_stabilize_steps'] = self.dec_loss_stabilize_steps
        checkpoint['dec_loss_bound_update_interval'] = self.dec_loss_bound_update_interval
        checkpoint['dec_loss_window_size'] = self.dec_loss_window_size

        # save training data
        checkpoint['train_losses'] = self.train_losses
        checkpoint['train_accuracies'] = self.train_accuracies
        checkpoint['train_entropys'] = self.train_entropys
        checkpoint['train_gains'] = self.train_gains

        checkpoint['custom_step'] = self.custom_step
        checkpoint['custom_current_epoch'] = self.custom_current_epoch
        checkpoint['beta_scheduler'] = self.beta_scheduler
        checkpoint['gamma_scheduler'] = self.gamma_scheduler
        checkpoint['temp_scheduler'] = self.temp_scheduler

        checkpoint['is_decoder_stabilized'] = self.is_decoder_stabilized
        checkpoint['decoder_stabilization_counter'] = self.decoder_stabilization_counter
        checkpoint['enc_warmup_end_step'] = getattr(self, 'enc_warmup_end_step', None)
        checkpoint['dec_stabilize_step'] = getattr(self, 'dec_stabilize_step', None)
        checkpoint['decoder_loss_upper_bound'] = self.decoder_loss_upper_bound
        checkpoint['decoder_loss_lower_bound'] = self.decoder_loss_lower_bound
        checkpoint['recent_decoder_losses'] = self.recent_decoder_losses

    
    def configure_optimizers(self):
        # get params from encoder and decoder
        encoder_params = list(self.encoder.parameters())
        decoder_params = list(self.decoder.parameters())
 
        # select optimizer
        if self.optimizer == 'adam':
            return Adam([
                {'params': encoder_params, 'lr': self.lr_enc},
                {'params': decoder_params, 'lr': self.lr_dec}
            ])
        elif self.optimizer == 'sgd':
            return SGD(encoder_params + decoder_params, lr=self.lr)
        
# ====== Trainer.fit() methods  =======
        
    def on_train_start(self):
        """
        Called at the start of training.
        """
        super().on_train_start()

        if getattr(self, "custom_step", 0) == 0:
            self.custom_step = -1  # will be incremented to 0 in first training step
            self.custom_current_epoch = -1  # will be incremented to 0 in first epoch start

            # initialze scehdulers
            self.beta_scheduler = BetaScheduler(
                total_steps=self.custom_max_epochs * len(self.trainer.train_dataloader), 
                final_beta=self.final_beta,
                warmup_frac=self.warmup_frac_beta
            )

            self.gamma_scheduler = BetaScheduler(
                total_steps=self.custom_max_epochs * len(self.trainer.train_dataloader), 
                final_beta=self.final_gamma,
                warmup_frac=self.warmup_frac_gamma
            )

            self.temp_scheduler = TempScheduler(
                init_tau=self.init_temp,
                min_tau=self.min_temp,
                decay=self.decay_temp
            )

            self.is_decoder_stabilized = False
            self.decoder_stabilization_counter = 0
            
            self.decoder_loss_upper_bound = None
            self.decoder_loss_lower_bound = None
            self.recent_decoder_losses = []

            self.train_losses = {
                'nri/train_losses': [],
                'enc/train_losses': [],
                'enc/train_warmup_losses': [],
                'dec/train_losses': [],
                'nri/val_losses': [],
                'enc/val_losses': [],
                'enc/val_warmup_losses': [],
                'dec/val_losses': [],
            }
            self.train_accuracies = {
                'enc/train_edge_accuracy': [],
                'enc/val_edge_accuracy': [],
            }
            self.train_entropys = {
                'enc/train_entropy': [],
                'enc/val_entropy': [],
            }
            self.train_gains = {
                'enc_loss/beta': [],
                'enc_warmup_loss/gamma': [],
                'dec_loss/alpha': [],
                'nri_loss/delta': [],
                'dec/temp': [],
            }
            self.start_time = time.time()
        else:
            print(f"\nResuming training from global step {self.custom_step} at epoch {self.custom_current_epoch}.")

        # variables always to load
        self.model_id = os.path.basename(self.logger.log_dir) if self.logger else 'nri_model'
        self.tb_tag = self.model_id.split('-')[0].strip('[]').replace('_(', "  (").replace('+', " + ") if self.logger else 'nri_model'
        self.run_type = "train"
        self.custom_max_epochs = self.trainer.max_epochs + self.custom_current_epoch + 1

        self.found_rep1 = False # for decoder output plot

        self.n_steps_per_epoch = len(self.trainer.train_dataloader)
        self.encoder.init_input_processors()
        self.decoder.init_input_processors()

        self.start_time = time.time()

        
    def on_train_batch_start(self, batch, batch_idx):
        super().on_train_batch_start(batch, batch_idx)

        self.custom_step += 1

    def on_train_epoch_start(self):
        super().on_train_epoch_start()

        self.custom_current_epoch += 1

    def _encoder_warmup_loss(self, relations, edge_pred, edge_accuracy):
        """
        Calculate the encoder warmup loss using cross-entropy.

        Note
        ----
        This method assumes that the `relations` tensor is available in the current batch.
        """
        if relations is not None:
            enc_target = relations.argmax(dim=-1).view(-1)      # shape: (batch_size * n_edges,)
            enc_pred = edge_pred.view(-1, edge_pred.size(-1))   # (batch_size * n_edges, n_types)
            ce_loss_encoder = F.cross_entropy(enc_pred, enc_target)

            if self.custom_step > 0:
                if edge_accuracy > self.warmup_acc_cutoff:
                    if self.is_enc_warmup:
                        self.enc_warmup_end_step = self.custom_step
                        warmup_text = f"Warmup may re-enable if edge accuracy drops below cutoff {self.warmup_acc_cutoff}." if self.sustain_enc_warmup else "Encoder warmup disabled for the rest of training."
                        print(f"\nEncoder warmup completed at step {self.enc_warmup_end_step}. {warmup_text}\n")

                    self.is_enc_warmup = False
                    ce_loss_encoder = 0.0
                    
                else:
                    if not self.is_enc_warmup:
                        print(f"\nEncoder warmup re-enabled at step {self.custom_step} as edge accuracy dropped below cutoff {self.warmup_acc_cutoff}.")
                    self.is_enc_warmup = True
        else:
            print("Warning: relations not provided for encoder cross-entropy loss calculation. So encoder warmup is disabled.") if self.custom_step == 0 else None
            self.is_enc_warmup = False
            ce_loss_encoder = 0.0

        return ce_loss_encoder
        
    def _forward_pass(self, batch, batch_idx):
        """
        Perform a forward pass through the model.
        """
        data, relations, _, rep_num = batch
        
        num_nodes = data.size(1)
        target = self.decoder.process_input_data(data)[:, :, 1:, :] # get target for decoder based on its transform

        # Forward pass
        edge_pred, edge_matrix, x_pred, x_var, temp = self.forward(data)

        # Edge accuracy calculation
        if relations is not None:
            edge_accuracy = (edge_pred.argmax(dim=-1) == relations.argmax(dim=-1)).float().mean()
            edge_accuracy_per_sample = (edge_pred.argmax(dim=-1) == relations.argmax(dim=-1)).float().mean(dim=1)
        else:
            edge_accuracy = None

        # Loss calculation
        # encoder loss
        if self.loss_type_encoder == 'kld':
            if self.prior is not None:
                mean_kl_per_edge, mean_kl_per_sample = kl_categorical(edge_pred, self.prior, num_nodes)
            else:
                mean_kl_per_edge, mean_kl_per_sample = kl_categorical_uniform(
                    edge_pred, num_nodes, add_const=self.add_const_kld
                )
            loss_encoder = mean_kl_per_sample

        beta = self.beta_scheduler(self.custom_step) if self.is_beta_annealing else 1.0

        # encoder warmup loss
        if self.is_enc_warmup or self.sustain_enc_warmup:  
            if self.is_decoder_stabilized:      
                ce_loss_encoder = self._encoder_warmup_loss(relations, edge_pred, edge_accuracy)
                gamma = self.gamma_scheduler(self.custom_step)
            else:
                ce_loss_encoder = torch.tensor(0.0, device=self.device)
                gamma = 0
        else:
            ce_loss_encoder = torch.tensor(0.0, device=self.device)
            gamma = 0

        # entropy calculation
        entropy_per_edge = -torch.sum(edge_pred * torch.log(edge_pred + 1e-12), dim=-1)  # shape (batch_size, n_edges)
        mean_entropy_per_edge = entropy_per_edge.mean()  # shape (scalar)
        # mean computed over n_edges * batch_size

        # decoder loss
        if self.loss_type_decoder == 'nll':
            loss_decoder = nll_gaussian(x_pred, target, x_var)
        elif self.loss_type_decoder == 'mse':
            loss_decoder = F.mse_loss(x_pred, target)


        if self.is_enc_warmup:

            # check if decoder is stabilized (stabilized just for once) (only used for training)
            if not self.is_decoder_stabilized:
            
                # add on to recent decoder losses till 
                self.recent_decoder_losses.append(loss_decoder.item())
                if len(self.recent_decoder_losses) > self.dec_loss_window_size:
                    self.recent_decoder_losses.pop(0)

                # update bounds every bound_update_interval steps
                if self.custom_step % self.dec_loss_bound_update_interval == 0 and len(self.recent_decoder_losses) == self.dec_loss_window_size:
                    self.decoder_loss_upper_bound = max(self.recent_decoder_losses)
                    self.decoder_loss_lower_bound = min(self.recent_decoder_losses)

                if self.decoder_loss_upper_bound is not None and self.decoder_loss_lower_bound is not None:

                    if self.decoder_loss_lower_bound <= loss_decoder.item() <= self.decoder_loss_upper_bound:
                        self.decoder_stabilization_counter += 1
                        print(f"Step {self.custom_step}: Decoder stabilization counter: {self.decoder_stabilization_counter}/{self.dec_loss_stabilize_steps}")

                        if self.decoder_stabilization_counter >= self.dec_loss_stabilize_steps:
                            self.is_decoder_stabilized = True
                            self.dec_stabilize_step = self.custom_step
                            print(f"\nDecoder stabilized at step {self.dec_stabilize_step}. Starting encoder warmup.\n")
                    else:
                        self.decoder_stabilization_counter = 0

        # alpha calculation
       

        # total loss calulation
        loss = (beta*loss_encoder) + (gamma*ce_loss_encoder) + loss_decoder

        # make dict
        log_data = {
            'loss': loss,
            'beta': beta if self.is_beta_annealing else 1.0,
            'gamma': gamma if self.is_enc_warmup else 0.0,
            'temp': temp,
            'loss_encoder': loss_encoder,
            'ce_loss_encoder': ce_loss_encoder,
            'entropy_per_edge': mean_entropy_per_edge,
            'loss_decoder': loss_decoder,
            'edge_accuracy': edge_accuracy,
            'edge_accuracy_per_sample': edge_accuracy_per_sample
        }

        # prepare data for decoder output plot
        target_rep_num = 1001.0001

        # find data with rep_num = 1001.0001
        if target_rep_num in rep_num:
            self.found_rep1 = True
            target_idx = rep_num.tolist().index(target_rep_num)
            x_pred_rep1 = x_pred[target_idx:target_idx+1]
            x_var_rep1 = x_var[target_idx:target_idx+1]
            target_rep1 = target[target_idx:target_idx+1]
            rep_num_rep1 = rep_num[target_idx:target_idx+1]

            self.decoder_plot_data_rep1 = {
            'x_pred': x_pred_rep1,
            'x_var': x_var_rep1,
            'target': target_rep1,
            'rep_num': rep_num_rep1,
            }
            print(f"\nFound rep_num = {target_rep_num} in batch {batch_idx} of epoch {self.current_epoch}. Decoder output plot will be made for this data.")

        elif not self.found_rep1:
            self.decoder_plot_data_rep1 = None 

        decoder_plot_data = {
            'x_pred': x_pred,
            'x_var': x_var,
            'target': target,
            'rep_num': rep_num,
        }

        return log_data, decoder_plot_data, edge_pred
    
   
    def training_step(self, batch, batch_idx):
        """
        Training step for the topology estimator.

        Parameters
        ----------
        batch : tuple
            A tuple containing the data batch and the relationship batch.
            - data_batch : tuple
                Contains the data tensor and the relations tensor.
            - rel_batch : tuple
                Contains the receiver and sender relationship matrices.
        """
        # if self.custom_current_epoch == 0 and batch_idx == 0:
        #     self.start_time = time.time()

        log_data, self.decoder_plot_data_train, _ = self._forward_pass(batch, batch_idx)
        self.train_log_data = log_data

        # Log the losses
        log_dict = {
            'nri/train_loss': log_data['loss'],
            'enc/train_loss': log_data['loss_encoder'],
            'enc/train_warmup_loss': log_data['ce_loss_encoder'],
            'enc/train_entropy': log_data['entropy_per_edge'],
            'dec/train_loss': log_data['loss_decoder'],
        }

        if log_data['edge_accuracy'] is not None:
            log_dict['enc/train_edge_accuracy'] = log_data['edge_accuracy']

        # log every n steps
        self.n = 5

        if (self.custom_step) % self.n == 0:
            self.log_dict(
                log_dict,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True
            )
        # else:
        #     print(f"Step {self.custom_step}, Batch {batch_idx}/{len(self.trainer.train_dataloader)}")
            
        return log_data['loss']
    
    def on_train_batch_end(self, outputs, batch, batch_idx):

        if (self.custom_step) % self.n == 0:
            # log losses
            self.train_losses['nri/train_losses'].append(self.train_log_data['loss'].item())
            self.train_losses['enc/train_losses'].append(self.train_log_data['loss_encoder'].item())
            self.train_losses['dec/train_losses'].append(self.train_log_data['loss_decoder'].item())
            
            if self.train_log_data['edge_accuracy'] is not None:
                self.train_accuracies['enc/train_edge_accuracy'].append(self.train_log_data['edge_accuracy'].item())

            if self.is_enc_warmup:
                self.train_losses['enc/train_warmup_losses'].append(self.train_log_data['ce_loss_encoder'].item())
            else:
                self.train_losses['enc/train_warmup_losses'].append(0.0)

            # log entropies
            self.train_entropys['enc/train_entropy'].append(self.train_log_data['entropy_per_edge'].item())

            # log gains
            self.train_gains['enc_loss/beta'].append(self.train_log_data['beta'])
            self.train_gains['enc_warmup_loss/gamma'].append(self.train_log_data['gamma'])
            self.train_gains['dec/temp'].append(self.train_log_data['temp'])

            # gains with value = 1
            self.train_gains['dec_loss/alpha'].append(1)
            self.train_gains['nri_loss/delta'].append(1)

            print(f"\nStep {self.custom_step}, Epoch {self.custom_current_epoch+1}/{self.custom_max_epochs}, Batch {batch_idx}/{len(self.trainer.train_dataloader)}")
            print(
                f"temp: {self.train_log_data['temp']:,.4f}, "
                f"beta: {self.train_log_data['beta']:,.4f}, "
                f"gamma: {self.train_log_data['gamma']:,.4f}, \n"
                f"enc_train_warmup_loss: {self.train_log_data['ce_loss_encoder']:,.4f}, \n"
                f"nri_train_loss: {self.train_log_data['loss']:,.4f}, "
                f"enc_train_loss: {self.train_log_data['loss_encoder']:,.4f}, "
                f"enc_train_entropy: {self.train_log_data['entropy_per_edge']:,.4f}, "
                f"dec_train_loss: {self.train_log_data['loss_decoder']:,.4f}, "
                f"train_edge_accuracy: {self.train_log_data['edge_accuracy']:,.4f}, " # if self.train_log_data['edge_accuracy'] is not None else ""
            )

    
    def validation_step(self, batch, batch_idx):
        """
        Validation step for the topology estimator.

        Parameters
        ----------
        batch : tuple
            A tuple containing the data batch and the relationship batch.
            - data_batch : tuple
                Contains the data tensor and the relations tensor.
            - rel_batch : tuple
                Contains the receiver and sender relationship matrices.
        """
        log_data, self.decoder_plot_data_val, _ = self._forward_pass(batch, batch_idx)

        # Log the losses
        log_dict = {
            'nri/val_loss': log_data['loss'],
            'enc/val_loss': log_data['loss_encoder'],
            'enc/val_warmup_loss': log_data['ce_loss_encoder'],
            'enc/val_entropy': log_data['entropy_per_edge'],
            'dec/val_loss': log_data['loss_decoder'],
        }

        if log_data['edge_accuracy'] is not None:
            log_dict['enc/val_edge_accuracy'] = log_data['edge_accuracy']

        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True
        )

        return log_data['loss']
    

    def on_train_epoch_end(self):
        """
        Called at the end of each training epoch. Updates the training losses and accuracies.
        """
        # train
        # self.train_losses['nri/train_losses'].append(self.trainer.callback_metrics['nri/train_loss'].item())
        # self.train_losses['enc/train_losses'].append(self.trainer.callback_metrics['enc/train_loss'].item())
        # self.train_losses['dec/train_losses'].append(self.trainer.callback_metrics['dec/train_loss'].item())

        print(f"\nEpoch {self.custom_current_epoch+1}/{self.custom_max_epochs} completed, Global Step: {self.custom_step}")
        print(
            f"nri_train_loss: {self.train_losses['nri/train_losses'][-1]:,.4f}, " 
            f"enc_train_loss: {self.train_losses['enc/train_losses'][-1]:,.4f}, "
            f"enc_train_warmup_loss: {self.trainer.callback_metrics['enc/train_warmup_loss'].item():,.4f}, "
            f"enc_train_entropy: {self.trainer.callback_metrics['enc/train_entropy'].item():,.4f}, "
            f"dec_train_loss: {self.train_losses['dec/train_losses'][-1]:,.4f}, "
            f"train_edge_accuracy: {self.train_accuracies['enc/train_edge_accuracy'][-1]:.4f}"
            )

        # make decoder output plot for training data
        self.decoder_output_plot(**self.decoder_plot_data_train, type='train', is_end=False) if self.custom_current_epoch+1 < self.custom_max_epochs else None

        # validation
        if self.trainer.val_dataloaders:
            self.train_losses['nri/val_losses'].append(self.trainer.callback_metrics['nri/val_loss'].item())
            self.train_losses['enc/val_losses'].append(self.trainer.callback_metrics['enc/val_loss'].item())
            self.train_losses['dec/val_losses'].append(self.trainer.callback_metrics['dec/val_loss'].item())

            if 'enc/val_edge_accuracy' in self.trainer.callback_metrics:
                self.train_accuracies['enc/val_edge_accuracy'].append(self.trainer.callback_metrics['enc/val_edge_accuracy'].item())

            if self.is_enc_warmup:
                self.train_losses['enc/val_warmup_losses'].append(self.trainer.callback_metrics['enc/val_warmup_loss'].item())

            # make decoder output plot for val data
            self.decoder_output_plot(**self.decoder_plot_data_val, type='val', is_end=False) if self.custom_current_epoch+1 < self.custom_max_epochs else None

            print(
                f"nri_val_loss: {self.train_losses['nri/val_losses'][-1]:,.4f}, " 
                f"enc_val_loss: {self.train_losses['enc/val_losses'][-1]:,.4f}, "
                f"enc_val_warmup_loss: {self.trainer.callback_metrics['enc/val_warmup_loss'].item():,.4f}, "
                f"enc_val_entropy: {self.trainer.callback_metrics['enc/val_entropy'].item():,.4f}, "
                f"dec_val_loss: {self.train_losses['dec/val_losses'][-1]:,.4f}, "
                f"val_edge_accuracy: {self.train_accuracies['enc/val_edge_accuracy'][-1]:.4f}"
                "\n\n" + 75*'-' + '\n'
                )
            

        # update hparams
        self.training_time = time.time() - self.start_time
        self.hyperparams.update({
            'model_id': self.model_id,
            'n_steps': self.custom_step,
            'model_num': float(self.logger.log_dir.split('_')[-1]) if self.logger else 0,

            # log train data
            'training_time': self.training_time,
            'nri/train_loss': self.train_losses['nri/train_losses'][-1],
            'enc/train_loss': self.train_losses['enc/train_losses'][-1],
            'enc/train_entropy': self.trainer.callback_metrics['enc/train_entropy'].item(),
            'dec/train_loss': self.train_losses['dec/train_losses'][-1],
            'enc/train_edge_accuracy': self.train_accuracies['enc/train_edge_accuracy'][-1] if self.train_accuracies['enc/train_edge_accuracy'] else -1,
            
            # log validation data
            'nri/val_loss': self.train_losses['nri/val_losses'][-1],
            'enc/val_loss': self.train_losses['enc/val_losses'][-1],
            'enc/val_entropy': self.trainer.callback_metrics['enc/val_entropy'].item(),
            'dec/val_loss': self.train_losses['dec/val_losses'][-1],
            'enc/val_edge_accuracy': self.train_accuracies['enc/val_edge_accuracy'][-1] if self.train_accuracies['enc/val_edge_accuracy'] else -1,
        })

    # def on_validation_epoch_end(self):
    #     """
    #     Called at the end of the validation epoch. Updates the validation losses and accuracies.
    #     """
        

    #     # make decoder output plot
    #     self.decoder_output_plot(**self.decoder_plot_data_val)

    #     # print stats after each epoch
    #     print(
    #         f"nri/val_loss: {self.train_losses['nri/val_losses'][-1]:.4f}, " 
    #         f"enc/val_loss: {self.train_losses['enc/val_losses'][-1]:.4f}, "
    #         f"dec/val_loss: {self.train_losses['dec/val_losses'][-1]:.4f}, "
    #         f"enc/val_edge_accuracy: {self.train_accuracies['enc/val_edge_accuracy'][-1]:.4f}"
    #         )
        
    # def on_after_backward(self):
    #     for name, param in self.encoder.named_parameters():
    #         if param.grad is not None:
    #             print(f"{name} grad norm: {param.grad.norm()}")

    def on_train_end(self):
        """
        Called at the end of training.
        """
        print(f"\nTraining completed in {self.training_time:.2f} seconds or {self.training_time / 60:.2f} minutes or {self.training_time / 60 / 60} hours.")
        print(f"Total training steps: {self.custom_step}")

        if self.logger:

            # save the train_accuracies, train_losses as csv files
            with open(os.path.join(self.logger.log_dir, 'train_accuracies.pkl'), 'wb') as f:
                pickle.dump(self.train_accuracies, f)
            with open(os.path.join(self.logger.log_dir, 'train_losses.pkl'), 'wb') as f:
                pickle.dump(self.train_losses, f)
            with open(os.path.join(self.logger.log_dir, 'train_entropys.pkl'), 'wb') as f:
                pickle.dump(self.train_entropys, f)

            print(f"\nTraining completed for model '{self.model_id}'. Trained model saved at {os.path.join(self.logger.log_dir, 'checkpoints')}")
        else:
            print(f"\nTraining completed for model '{self.model_id}'. Logging is disabled, so no checkpoints are saved.")

        print('\n' + 75*'-')

        # plot training losses and decoder output
        self.training_loss_plot()
        self.edge_accuracy_plot()
        self.edge_entropy_plot()
        self.decoder_output_plot(**self.decoder_plot_data_train, type='train')
        self.decoder_output_plot(**self.decoder_plot_data_val, type='val') if self.trainer.val_dataloaders else None


# ====== Trainer.test() methods  ======

    def on_test_start(self):
        """
        Called at the start of testing.
        """
        super().on_test_start()

        # initialze scehdulers
        # self.beta_scheduler = BetaScheduler(
        #     total_steps=self.custom_max_epochs * len(self.trainer.test_dataloaders), 
        #     final_beta=self.final_beta,
        #     warmup_frac=self.warmup_frac_beta
        # )
        self.custom_step = -1  # will be incremented to 0 in first test step

        self.found_rep1 = False # for decoder output plot

        self.is_beta_annealing = False  # no beta annealing during testing
        self.is_enc_warmup = False   # no encoder warmup during testing

        self.temp_scheduler = TempScheduler(
            init_tau=self.init_temp,
            min_tau=self.min_temp,
            decay=self.decay_temp
        )

        self.encoder.init_input_processors()
        self.decoder.init_input_processors()

        # Log model information
        self.model_id = self.hyperparams.get('model_id', 'nri_model')
        self.tb_tag = self.model_id.split('-')[0].strip('[]').replace('_(', "  (").replace('+', " + ") if self.logger else 'nri_model'
        self.run_type = os.path.basename(self.logger.log_dir) if self.logger else 'test'

        self.start_time = time.time()

    def on_test_batch_start(self, batch, batch_idx, dataloader_idx = 0):
        super().on_test_batch_start(batch, batch_idx, dataloader_idx)
        self.custom_step += 1

  
    def test_step(self, batch, batch_idx):
        """
        Test step for the topology estimator.

        Parameters
        ----------
        batch : tuple
            A tuple containing the data batch and the relationship batch.
            - data_batch : tuple
                Contains the data tensor and the relations tensor.
            - rel_batch : tuple
                Contains the receiver and sender relationship matrices.
        """
        log_data, self.decoder_plot_data_test, self.edge_preds = self._forward_pass(batch, batch_idx)
        data, _, _, self.rep_nums = batch
        num_nodes = data.size(1)

        # convert edge predictions to adjacency matrix
        self.adj_matrices = self.edge_pred_to_adjacency_matrix(self.edge_preds, num_nodes)

        # Log the losses and metrics
        log_dict = {
            'nri/test_loss': log_data['loss'],
            'enc/test_loss': log_data['loss_encoder'],
            'enc/test_entropy': log_data['entropy_per_edge'],
            'dec/test_loss': log_data['loss_decoder'],
        }

        if log_data['edge_accuracy'] is not None:
            log_dict['enc/test_edge_accuracy'] = log_data['edge_accuracy']

        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True
        )

        return log_data['loss']
    
    def on_test_epoch_end(self):
        """
        Called at the end of the test epoch. Updates the hyperparameters with test losses and accuracies.
        """
        self.infer_time = time.time() - self.start_time
        print(f"\nTesting completed in {self.infer_time:.2f} seconds or {self.infer_time / 60:.2f} minutes or {self.infer_time / 60 / 60} hours.")

        # update hyperparams
        self.hyperparams.update({
            'infer_time': self.infer_time,
            'nri/test_loss': self.trainer.callback_metrics['nri/test_loss'].item(),
            'enc/test_loss': self.trainer.callback_metrics['enc/test_loss'].item(),
            'enc/test_entropy': self.trainer.callback_metrics['enc/test_entropy'].item(),
            'dec/test_loss': self.trainer.callback_metrics['dec/test_loss'].item(),
        })

        if 'enc/test_edge_accuracy' in self.trainer.callback_metrics:
            self.hyperparams['enc/test_edge_accuracy'] = self.trainer.callback_metrics['enc/test_edge_accuracy'].item()


        # print stats after testing
        print(
            f"\nnri_test_loss: {self.hyperparams['nri/test_loss']:,.4f}, " 
            f"enc_test_loss: {self.hyperparams['enc/test_loss']:,.4f}, "
            f"dec_test_loss: {self.hyperparams['dec/test_loss']:,.4f}, "
            f"test_edge_accuracy: {self.hyperparams.get('enc/test_edge_accuracy', -1):.4f}"
            )
        
        print("\nEdge predictions are as follows (showing probabilities for each edge type):")
        for edge_pred, rep in zip(self.edge_preds, self.rep_nums):
            print(f"\nRep {rep.item():,.3f}:")
            print(edge_pred.cpu().numpy())

        print("\nAdjacency matrix from edge pred is as follows:")
        for adj_matrix, rep in zip(self.adj_matrices, self.rep_nums):
            print(f"\nRep {rep.item():,.3f}:")
            print(adj_matrix.cpu().numpy())


        if self.logger:
            self.logger.log_hyperparams(self.hyperparams)
            print(f"\nTest metrics and hyperparameters logged for tensorboard at {self.logger.log_dir}")
        else:
            print("\nTest metrics and hyperparameters not logged as logging is disabled.")
        
        print('\n' + 75*'-')

        # make decoder output plot
        if self.decoder_plot_data_rep1:
            self.decoder_output_plot(**self.decoder_plot_data_rep1, type=self.run_type)
        else:
            print(f"\nNo target rep num. Hence decoder output is made for first sample in the last test batch.")
            self.decoder_output_plot(**self.decoder_plot_data_test, type=self.run_type)


# ====== Trainer.predict() method  ====== 

    def on_predict_start(self):
        """
        Called at the start of prediction.
        """
        super().on_predict_start()

        self.custom_step = -1  # will be incremented to 0 in first predict step

        self.found_rep1 = False # for decoder output plot

        self.is_beta_annealing = False  # no beta annealing during prediction
        self.is_enc_warmup = False   # no encoder warmup during prediction

        self.temp_scheduler = TempScheduler(
            init_tau=self.init_temp,
            min_tau=self.min_temp,
            decay=self.decay_temp
        )

        self.encoder.init_input_processors()
        self.decoder.init_input_processors()

        # Log model information
        self.model_id = self.hyperparams.get('model_id', 'nri_model')
        self.tb_tag = self.model_id.split('-')[0].strip('[]').replace('_(', "  (").replace('+', " + ") if self.logger else 'nri_model'
        self.run_type = os.path.basename(self.logger.log_dir) if self.logger else 'predict'
        
        self.start_time = time.time()

    def on_predict_batch_start(self, batch, batch_idx, dataloader_idx = 0):
        super().on_predict_batch_start(batch, batch_idx, dataloader_idx)
        self.custom_step += 1

    def predict_step(self, batch, batch_idx):
        """
        Prediction step for the topology estimator.

        Parameters
        ----------
        batch : tuple
            A tuple containing the data batch and the relationship batch.
            - data_batch : tuple
                Contains the data tensor and the relations tensor.
            - rel_batch : tuple
                Contains the receiver and sender relationship matrices.

        Returns
        -------
        dict
            A dictionary containing the predicted edge probabilities, node data, and variance.
        """
        log_data, self.decoder_plot_data_predict, edge_pred = self._forward_pass(batch, batch_idx)

        data_batch, _ = batch
        data, _, _, _ = data_batch
        num_nodes = data.size(1)

        # convert edge predictions to adjacency matrix
        adj_matrix = self.edge_pred_to_adjacency_matrix(edge_pred, num_nodes)

        infer_time = time.time() - self.start_time
        print(f"\nPrediction completed in {infer_time:.2f} seconds or {infer_time / 60:.2f} minutes or {infer_time / 60 / 60} hours.")

        node_labels = [f"{node_name}" for node_name in self.decoder.data_config.signal_types['group'].keys()]
        adj_df = pd.DataFrame(adj_matrix.cpu().numpy(), index=node_labels, columns=node_labels)

        print(f"\n Adjacency matrix (shape {adj_matrix.shape})")
        print(adj_df)

        print(f"\nDecoder residual: {log_data['loss_decoder'].item():,.4f}")

        if self.logger:
            adj_mat_text = "##Predicted Adjacency Matrix\n"
            adj_mat_text += adj_df.to_markdown() + '\n'
            self.logger.experiment.add_text(f"{self.model_id} + {self.run_type}", adj_mat_text, global_step=self.custom_step)

        print('\n' + 75*'-')

        # make decoder output plot
        self.decoder_output_plot(**self.decoder_plot_data_predict, type=self.run_type)

        return {
            'nri/edge_pred': edge_pred,
            'nri/adj_matrix': adj_matrix,
            'nri/residual': log_data['loss_decoder']
        }
    
    def edge_pred_to_adjacency_matrix(self, edge_pred, n_nodes):
        """
        Convert edge predictions to an adjacency matrix.

        Parameters
        ----------
        edge_pred : torch.Tensor, shape (batch_size, n_edges, n_edge_types)
            Containing edge predictions from encoder.
        n_nodes : int
            Number of nodes.

        Returns
        -------
        adj_matrix : torch.Tensor, shape (batch_size, n_nodes, n_nodes, n_edge_types)
            Tensor representing the adjacency matrix.
        """
        batch_size, n_edges, n_edge_types = edge_pred.shape

        # Initialize an empty adjacency matrix
        adj_matrix = torch.zeros(batch_size, n_nodes, n_nodes, n_edge_types, device=edge_pred.device)

        # Fill the adjacency matrix
        edge_idx = 0
        for from_node in range(n_nodes):
            for to_node in range(n_nodes):
                if from_node != to_node:  # Skip self-loops
                    adj_matrix[:, from_node, to_node, :] = edge_pred[:, edge_idx, :]
                    edge_idx += 1

        return adj_matrix


# ================== Visualization Methods =======================

    def training_loss_plot(self):
        """
        Plot all the losses against the epochs.
        """
        if not self.train_losses:
            raise ValueError("No training losses found. Please run the training step first.")
        
        print("\n" + 12*"<" + " TRAINING LOSS PLOT (TRAIN + VAL) " + 12*">")
        print(f"\nCreating training loss plot for {self.model_id}...")

        train_steps = [step * self.n for step in range(1, len(self.train_losses[f'nri/train_losses']) + 1)]
        val_steps = [epoch * self.n_steps_per_epoch for epoch in range(1, len(self.train_losses[f'nri/val_losses']) + 1)]

        # update font settings for plots
        plt.rcParams.update({
            "text.usetex": False,   # No external LaTeX
            "font.family": "serif",
            "mathtext.fontset": "cm",  # Computer Modern math
        })

        # create a figure with 3 subplots in a vertical grid
        i = 4 if self.train_losses[f'enc/train_warmup_losses'] != [] else 3
        fig, axes = plt.subplots(i, 2, figsize=(18, i*3), dpi=100, sharex=True) #13

        j = i
        if self.train_losses[f'enc/train_warmup_losses'] != []:
            train_warmup_steps = [step * self.n for step in range(1, len(self.train_losses[f'enc/train_warmup_losses']) + 1)]
            val_warmup_steps = [epoch * self.n_steps_per_epoch for epoch in range(1, len(self.train_losses[f'enc/val_warmup_losses']) + 1)] if self.train_losses[f'enc/val_warmup_losses'] != [] else []

            # plot encoder warmup losses
            axes[i-j, 0].plot(train_warmup_steps, self.train_losses[f'enc/train_warmup_losses'], label='train loss', color='purple')
            axes[i-j, 0].plot(val_warmup_steps, self.train_losses[f'enc/val_warmup_losses'], label='val loss', color='magenta', linestyle='--', marker='o', markersize=4)
            axes[i-j, 0].set_title('Encoder Warmup Losses (CE)')
            axes[i-j, 0].set_ylabel('Loss')
            axes[i-j, 0].legend()
            axes[i-j, 0].grid(True)

            # plot encoder warmup gamma
            axes[i-j, 1].plot(train_steps, self.train_gains['enc_warmup_loss/gamma'], label='gamma', color='brown')
            axes[i-j, 1].set_title('Encoder Warmup Loss Gain (Gamma)')
            axes[i-j, 1].set_ylabel('Gamma')
            axes[i-j, 1].grid(True)
            axes[i-j, 1].legend()

            j -= 1

        # plot encoder losses
        axes[i-j, 0].plot(train_steps, self.train_losses[f'enc/train_losses'], label='train loss', color='olive')
        axes[i-j, 0].plot(val_steps, self.train_losses[f'enc/val_losses'], label='val loss', color='green', linestyle='--', marker='o', markersize=4)
        axes[i-j, 0].set_title(f'Encoder Losses ({self.loss_type_encoder.upper()})')
        axes[i-j, 0].set_ylabel('Loss')
        axes[i-j, 0].legend()
        axes[i-j, 0].grid(True)

        # plot encoder beta
        axes[i-j, 1].plot(train_steps, self.train_gains['enc_loss/beta'], label='beta', color='teal')
        axes[i-j, 1].set_title('Encoder Loss Gain (Beta)')
        axes[i-j, 1].set_ylabel('Beta')
        axes[i-j, 1].legend()
        axes[i-j, 1].grid(True)

        j -= 1

        # plot decoder losses
        axes[i-j, 0].plot(train_steps, self.train_losses[f'dec/train_losses'], label='train loss', color='blue')
        axes[i-j, 0].plot(val_steps, self.train_losses[f'dec/val_losses'], label='val loss', color='cyan', linestyle='--', marker='o', markersize=4)

        if hasattr(self, 'dec_stabilize_step'):
            if self.dec_stabilize_step is not None:
                axes[i-j, 0].axvline(
                    x=self.dec_stabilize_step, color='black', linestyle=':', linewidth=1.8,
                    label='encoder assist start step'
                )

        if hasattr(self, 'enc_warmup_end_step'):
            if self.enc_warmup_end_step is not None:
                axes[i-j, 0].axvline(
                    x=self.enc_warmup_end_step, color='red', linestyle=':', linewidth=1.8,
                    label='encoder assist end step'
                )

        axes[i-j, 0].set_title(f'Decoder Losses ({self.loss_type_decoder.upper()})')
        axes[i-j, 0].set_ylabel('Loss (log)')
        axes[i-j, 0].set_yscale('log')
        axes[i-j, 0].legend()
        axes[i-j, 0].grid(True)

        # plot decoder gain
        axes[i-j, 1].plot(train_steps, self.train_gains['dec_loss/alpha'], label='alpha', color='navy')
        axes[i-j, 1].set_title('Decoder Loss Gain (Alpha)')
        axes[i-j, 1].set_ylabel('Alpha')
        axes[i-j, 1].legend()
        axes[i-j, 1].grid(True)

        j -= 1

        # plot nri losses
        axes[i-j, 0].plot(train_steps, self.train_losses[f'nri/train_losses'], label='train loss', color='red')
        axes[i-j, 0].plot(val_steps, self.train_losses[f'nri/val_losses'], label='val loss', color='orange', linestyle='--', marker='o', markersize=4)
        axes[i-j, 0].set_title('NRI Losses (Encoder + Decoder)')
        axes[i-j, 0].set_ylabel('Loss (log)')
        axes[i-j, 0].set_xlabel('Steps')
        axes[i-j, 0].set_yscale('log')
        axes[i-j, 0].legend()
        axes[i-j, 0].grid(True)

        # plot nri gain
        axes[i-j, 1].plot(train_steps, self.train_gains['nri_loss/delta'], label='delta', color='darkred')
        axes[i-j, 1].set_title('NRI Loss Gain (Delta)')
        axes[i-j, 1].set_ylabel('Delta')
        axes[i-j, 1].set_xlabel('Steps')
        axes[i-j, 1].legend()
        axes[i-j, 1].grid(True)

        fig.suptitle(f"Train and Validation Losses : [{self.model_id} / train]", fontsize=15) # 15

        # save loss plot if logger is avaialble
        if self.logger:
            fig.savefig(os.path.join(self.logger.log_dir, f'training_loss_plot_({self.model_id}).png'), dpi=500)
            self.logger.experiment.add_figure(f"{self.tb_tag}/{self.model_id}/{self.run_type}/training_loss_plot", fig, global_step=self.custom_step, close=True)
            print(f"\nTraining loss (train + val) plot logged at {self.logger.log_dir}\n")
        else:
            print("\nTraining loss plot not logged as logging is disabled.\n")

    
    def edge_accuracy_plot(self):
        """
        Plot the encoder edge prediction accuracy against the steps.
        """
        if not self.train_accuracies:
            raise ValueError("No edge accuracy found. Please run the training step first.")
        
        print("\n" + 12*"<" + " ENCODER EDGE ACCURACY PLOT (TRAIN + VAL) " + 12*">")
        print(f"\nCreating encoder edge accuracy plot for {self.model_id}...")

        train_steps = [step * self.n for step in range(1, len(self.train_accuracies[f'enc/train_edge_accuracy']) + 1)]
        val_steps = [epoch * self.n_steps_per_epoch for epoch in range(1, len(self.train_accuracies[f'enc/val_edge_accuracy']) + 1)]

        plt.figure(figsize=(8, 4), dpi=100)

        # update font settings for plots
        plt.rcParams.update({
            "text.usetex": False,   # No external LaTeX
            "font.family": "serif",
            "mathtext.fontset": "cm",  # Computer Modern math
        })

        plt.plot(train_steps, self.train_accuracies[f'enc/train_edge_accuracy'], label='train accuracy', color='blue')
        plt.plot(val_steps, self.train_accuracies[f'enc/val_edge_accuracy'], label='val accuracy', color='orange', linestyle='--', marker='o', markersize=4)
        if hasattr(self, 'dec_stabilize_step'):
            if self.dec_stabilize_step is not None:
                plt.axvline(
                    x=self.dec_stabilize_step, color='black', linestyle=':', linewidth=1.8,
                    label='encoder assist start step'
                )
        if hasattr(self, 'enc_warmup_end_step'):
            if self.enc_warmup_end_step is not None:
                plt.axvline(
                    x=self.enc_warmup_end_step, color='red', linestyle=':', linewidth=1.8,
                    label='encoder assist end step'
                )

        plt.title(f'Encoder Edge Prediction Accuracy : [{self.model_id}]', fontsize=10, pad=20)
        plt.xlabel('Steps')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(True)
        
        # save accuracy plot if logger is avaialble
        if self.logger:
            plt.savefig(os.path.join(self.logger.log_dir, f'edge_accuracy_plot_({self.model_id}).png'), dpi=500)
            self.logger.experiment.add_figure(f"{self.tb_tag}/{self.model_id}/{self.run_type}/edge_accuracy_plot", plt.gcf(), global_step=self.custom_step, close=True)
            print(f"\nEncoder edge accuracy (train + val) plot logged at {self.logger.log_dir}\n")
        else:
            print("\nEncoder edge accuracy plot not logged as logging is disabled.\n")


    def edge_entropy_plot(self):
        """
        Plot the encoder edge prediction entropy against the steps.
        """
        if not self.train_entropys:
            raise ValueError("No edge entropy found. Please run the training step first.")
        
        print("\n" + 12*"<" + " ENCODER EDGE ENTROPY PLOT (TRAIN + VAL) " + 12*">")
        print(f"\nCreating encoder edge entropy plot for {self.model_id}...")

        train_steps = [step * self.n for step in range(1, len(self.train_entropys[f'enc/train_entropy']) + 1)]
        #val_steps = [epoch * self.n_steps_per_epoch for epoch in range(1, len(self.train_entropys[f'enc/val_entropy']) + 1)]

        plt.figure(figsize=(8, 4), dpi=100)

        # update font settings for plots
        plt.rcParams.update({
            "text.usetex": False,   # No external LaTeX
            "font.family": "serif",
            "mathtext.fontset": "cm",  # Computer Modern math
        })

        plt.plot(train_steps, self.train_entropys[f'enc/train_entropy'], label='train entropy', color='blue')
        # plt.plot(val_steps, self.train_entropys[f'enc/val_entropy'], label='val entropy', color='orange', linestyle='--')

        if hasattr(self, 'dec_stabilize_step'):
            if self.dec_stabilize_step is not None:
                plt.axvline(
                    x=self.dec_stabilize_step, color='black', linestyle=':', linewidth=1.8,
                    label='encoder assist start step'
                )
        if hasattr(self, 'enc_warmup_end_step'):
            if self.enc_warmup_end_step is not None:
                plt.axvline(
                    x=self.enc_warmup_end_step, color='red', linestyle=':', linewidth=1.8,
                    label='encoder assist end step'
                )

        plt.title(f'Encoder Edge Prediction Entropy : [{self.model_id}]', fontsize=10, pad=20)
        plt.xlabel('Steps')
        plt.ylabel('Entropy')
        # plt.ylim(0, 1)
        plt.legend()
        plt.grid(True)
        
        # save entropy plot if logger is avaialble
        if self.logger:
            plt.savefig(os.path.join(self.logger.log_dir, f'edge_entropy_plot_({self.model_id}).png'), dpi=500)
            self.logger.experiment.add_figure(f"{self.tb_tag}/{self.model_id}/{self.run_type}/edge_entropy_plot", plt.gcf(), global_step=self.custom_step, close=True)
            print(f"\nEncoder edge entropy (train + val) plot logged at {self.logger.log_dir}\n")
        else:
            print("\nEncoder edge entropy plot not logged as logging is disabled.\n")
            
    def decoder_output_plot(self, x_pred, x_var, target, rep_num, type:str, is_end=True, sample_idx=0):
        """
        Plot the decoder output for a given sample.

        Parameters
        ----------
        x_pred : torch.Tensor, shape (batch_size, n_nodes, n_components-1, n_dim)
            Predicted node data from the decoder.
        x_var : torch.Tensor, shape (batch_size, n_nodes, n_components-1, n_dim)
            Variance of the predicted node data.
        target : torch.Tensor, shape (batch_size, n_nodes, n_components-1, n_dim)
            Target node data for the decoder.
        rep_num : int
            rep label
        type : str
            Type of data (e.g., 'train', 'val', 'test', 'predict').
        is_end : bool, optional
            If True, indicates that this is the final plot after training/testing. Default is True.
        sample_idx : int, optional
            Index of the sample to plot. Default is 0.
        """

        if is_end:
            print("\n" + 12*"<" + f" DECODER OUTPUT PLOT ({type.upper()}) " + 12*">") if is_end else None
            print(f"\nCreating decoder output plot for rep '{rep_num[sample_idx]:,.4f}' for {self.model_id}...")

        # convert tensors to numpy arrays for plotting
        x_pred = x_pred.detach().cpu().numpy()
        x_var = x_var.detach().cpu().numpy()
        target = target.detach().cpu().numpy()

        batch_size, n_nodes, n_comps, n_dims = x_pred.shape

        node_names = [f"{node_name}" for node_name in self.decoder.data_config.signal_types['group'].keys()]

        # update font settings for plots
        plt.rcParams.update({
            "text.usetex": False,   # No external LaTeX
            "font.family": "serif",
            "mathtext.fontset": "cm",  # Computer Modern math
        })
        
        # create figure with subplots for each node and dimension
        fig, axes = plt.subplots(n_nodes, n_dims, figsize=(n_dims * 5, n_nodes * 3), sharex=False, sharey=False, dpi=75)
        if n_nodes == 1:
            axes = np.expand_dims(axes, axis=0)  # ensure axes is 2D for consistent indexing
        if n_dims == 1:
            axes = np.expand_dims(axes, axis=1)

        fig.suptitle(f"Decoder Output for Rep {rep_num[sample_idx]:,.4f} : [{self.model_id} / {type}]", fontsize=16)

        for node in range(n_nodes):
            dim_names = self.decoder.data_config.signal_types['group'][node_names[node]]

            for dim in range(n_dims):
                ax = axes[node, dim]

                # extract data for the current node and dim
                timesteps = np.arange(n_comps)
                gt = target[sample_idx, node, :, dim]  # ground truth
                pred = x_pred[sample_idx, node, :, dim]  # predictions
                conf_band_upper = pred + 1.96 * np.sqrt(x_var[sample_idx, node, :, dim])  # upper confidence band
                conf_band_lower = pred - 1.96 * np.sqrt(x_var[sample_idx, node, :, dim])  # lower confidence band

                # plot ground truth, predictions, and confidence band
                ax.plot(timesteps, gt, label="ground truth", color="blue", linestyle="--")
                ax.plot(timesteps, pred, label="prediction", color="red", alpha=0.7)

                prediction_start_step = n_comps - self.decoder.final_pred_steps if self.decoder.is_burn_in else 0
                ax.axvline(x=prediction_start_step - 1, color='green', linestyle=':', label='start of prediction', linewidth=1.8) if prediction_start_step > 0 else None

                if self.decoder.show_conf_band:
                    ax.fill_between(timesteps, conf_band_lower, conf_band_upper, color="orange", alpha=0.3, label="confidence band")

                # Add labels and legend
                #if node == n_nodes - 1:
                ax.set_xlabel("timesteps")
                ax.set_ylabel(f"{dim_names[dim]}")
                         
                if node == 0 and dim == n_dims - 1:
                    ax.legend(loc="upper right")

                # add node name as title for each row
                ax.set_title(f"{node_names[node]} ({dim_names[dim]})", fontsize=11)

                ax.grid(True)

        # adjust subplot spacing to prevent label overlap
        plt.subplots_adjust(left=0.15, bottom=0.1, right=0.95, top=0.9, wspace=0.3, hspace=0.6)

        # save the plot if logger is available
        if self.logger:
            self.logger.experiment.add_figure(f"{self.tb_tag}/{self.model_id}/{self.run_type}/decoder_output_plot_{type}", fig, global_step=self.custom_step, close=True)

            if is_end:
                fig.savefig(os.path.join(self.logger.log_dir, f'dec_output_{type}_({self.model_id}).png'), dpi=500)
                print(f"\nDecoder output plot for rep '{rep_num[sample_idx]}' logged at {self.logger.log_dir}\n")
        else:
            if is_end:
                print("\nDecoder output plot not logged as logging is disabled.\n")