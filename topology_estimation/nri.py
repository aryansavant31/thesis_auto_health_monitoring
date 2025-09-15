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
        loss_type_enc='kld', loss_type_dec='nll', prior=None
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

        print(f"\nTraining parameters set to: \nlr_enc={self.lr_enc}, \nlr_dec={self.lr_dec}, " 
              f"\nfinal_beta={self.final_beta}, \nwarmup_frac={self.warmup_frac_beta}, "
              f"\noptimizer={self.optimizer}, \nloss_type_encoder={self.loss_type_encoder}, \nloss_type_decoder={self.loss_type_decoder}, \nprior={self.prior}, \nadd_const_kld={self.add_const_kld}")
        
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

    def set_run_params(self, dec_run_params, data_config, data_stats, 
                       init_temp=1.0, min_temp=0.3, decay_temp=0.001, is_hard=True):
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

        self.encoder.set_run_params(data_config=data_config, data_stats=data_stats)
        self.decoder.set_run_params(**dec_run_params, data_config=data_config, data_stats=data_stats)
        
    def build_model(self):
        """
        Build the NRI model by constructing the encoder and decoder.
        """
        # build encoder
        self.encoder = Encoder()
        for key, value in self.encoder_model_params.items():
            setattr(self.encoder, key, value)
        self.encoder.build_model()

        # build decoder
        self.decoder = Decoder()
        for key, value in self.decoder_model_params.items():
            setattr(self.decoder, key, value)
        self.decoder.build_model()

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

        print("Edge pred", edge_pred[0,:,:]) # DEBUG
        print("logits", logits[0,:,:]) # DEBUG
        print("Edge matrix from gumble", edge_matrix[0,:,:]) # DEBUG

        # Decoder
        self.decoder.set_edge_matrix(edge_matrix)
        x_pred, x_var = self.decoder(data)

        return edge_pred, edge_matrix, x_pred, x_var, temp
    
# ================== PYTORCH LIGHTNING TRAINER METHODS ================== #

    def on_load_checkpoint(self, checkpoint):
        # model params
        self.encoder_model_params = checkpoint['encoder_model_params']
        self.decoder_model_params = checkpoint['decoder_model_params']
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

        # rebuild model
        self.build_model()

    def on_save_checkpoint(self, checkpoint):
        # model params
        checkpoint['encoder_model_params'] = self.encoder_model_params
        checkpoint['decoder_model_params'] = self.decoder_model_params
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

        # initialze scehdulers
        self.beta_scheduler = BetaScheduler(
            total_steps=self.trainer.max_epochs * len(self.trainer.train_dataloader), 
            final_beta=self.final_beta,
            warmup_frac=self.warmup_frac_beta
        )
        self.temp_scheduler = TempScheduler(
            init_tau=self.init_temp,
            min_tau=self.min_temp,
            decay=self.decay_temp
        )

        self.model_id = os.path.basename(self.logger.log_dir) if self.logger else 'nri_model'
        self.tb_tag = self.model_id.split('-')[0].strip('[]').replace('_(', "  (").replace('+', " + ") if self.logger else 'nri_model'
        self.run_type = "train"
        
        self.encoder.init_input_processors()
        self.decoder.init_input_processors()

        self.train_losses = {
            'nri/train_losses': [],
            'enc/train_losses': [],
            'dec/train_losses': [],
            'nri/val_losses': [],
            'enc/val_losses': [],
            'dec/val_losses': [],
        }
        self.train_accuracies = {
            'enc/train_edge_accuracy': [],
            'enc/val_edge_accuracy': [],
        }

        self.start_time = time.time()
        
    def _forward_pass(self, batch, batch_idx):
        """
        Perform a forward pass through the model.
        """
        data, relations, _, rep_num = batch
        
        num_nodes = data.size(1)
        target = self.decoder.process_input_data(data)[:, :, 1:, :] # get target for decoder based on its transform

        # Forward pass
        edge_pred, edge_matrix, x_pred, x_var, temp = self.forward(data)

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

        elif self.loss_type_encoder == 'ce':
            enc_target = relations.argmax(dim=-1)  # shape: (batch_size, n_edges)
            enc_pred = edge_pred.view(-1, edge_pred.size(-1))  # (batch_size * n_edges, n_types)
            enc_target = enc_target.view(-1)  # (batch_size * n_edges,)
            loss_encoder = F.cross_entropy(enc_pred, enc_target)

        # entropy calculation
        entropy_per_edge = -torch.sum(edge_pred * torch.log(edge_pred + 1e-12), dim=-1)  # shape (batch_size, n_edges)
        mean_entropy_per_edge = entropy_per_edge.mean()  # shape (scalar)
        # mean computed over n_edges * batch_size

        # decoder loss
        if self.loss_type_decoder == 'nll':
            loss_decoder = nll_gaussian(x_pred, target, x_var)
        elif self.loss_type_decoder == 'mse':
            loss_decoder = F.mse_loss(x_pred, target)

        # total loss
        beta = self.beta_scheduler(self.global_step) if self.is_beta_annealing else 1.0
        # lambda_ent = max(0.0, 0.1 * (1 - self.global_step / (0.5 * self.trainer.max_epochs * len(self.trainer.train_dataloader))))

        loss = (loss_encoder * beta) + loss_decoder # + (0.8 * mean_entropy_per_edge)

        if relations is not None:
            edge_accuracy = (edge_pred.argmax(dim=-1) == relations.argmax(dim=-1)).float().mean()
        else:
            edge_accuracy = None
        
        # make dict
        log_data = {
            'loss': loss,
            'beta': beta if self.is_beta_annealing else 1.0,
            'temp': temp,
            'loss_encoder': loss_encoder,
            'entropy_per_edge': mean_entropy_per_edge,
            'loss_decoder': loss_decoder,
            'edge_accuracy': edge_accuracy,
        }

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
        # if self.current_epoch == 0 and batch_idx == 0:
        #     self.start_time = time.time()

        log_data, self.decoder_plot_data_train, _ = self._forward_pass(batch, batch_idx)

        # Log the losses
        log_dict = {
            'nri/train_loss': log_data['loss'],
            'enc/train_loss': log_data['loss_encoder'],
            'enc/train_entropy': log_data['entropy_per_edge'],
            'dec/train_loss': log_data['loss_decoder'],
        }

        if log_data['edge_accuracy'] is not None:
            log_dict['enc/train_edge_accuracy'] = log_data['edge_accuracy']

        self.n = 5  # log every n batches

        if (self.global_step) % self.n == 0:

            print(f"Step {self.global_step}, Epoch {self.current_epoch+1}/{self.trainer.max_epochs}, Batch {batch_idx}/{len(self.trainer.train_dataloader)}")
            print(
                f"temp: {log_data['temp']:,.4f}, "
                f"beta: {log_data['beta']:,.4f}, \n"
                f"nri_train_loss: {log_data['loss']:,.4f}, "
                f"enc_train_loss: {log_data['loss_encoder']:,.4f}, "
                f"enc_train_entropy: {log_data['entropy_per_edge']:,.4f}, "
                f"dec_train_loss: {log_data['loss_decoder']:,.4f}, "
                f"enc_train_edge_accuracy: {log_data['edge_accuracy']:,.4f}" if log_data['edge_accuracy'] is not None else ""
            )
            print("")

            self.log_dict(
                log_dict,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True
            )
        # else:
        #     print(f"Step {self.global_step}, Batch {batch_idx}/{len(self.trainer.train_dataloader)}")
            
        return log_data['loss']
    
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
        self.train_losses['nri/train_losses'].append(self.trainer.callback_metrics['nri/train_loss'].item())
        self.train_losses['enc/train_losses'].append(self.trainer.callback_metrics['enc/train_loss'].item())
        self.train_losses['dec/train_losses'].append(self.trainer.callback_metrics['dec/train_loss'].item())

        if 'enc/train_edge_accuracy' in self.trainer.callback_metrics:
            self.train_accuracies['enc/train_edge_accuracy'].append(self.trainer.callback_metrics['enc/train_edge_accuracy'].item())

        print(f"\nEpoch {self.current_epoch+1}/{self.trainer.max_epochs} completed, Global Step: {self.global_step}")
        print(
            f"nri_train_loss: {self.train_losses['nri/train_losses'][-1]:,.4f}, " 
            f"enc_train_loss: {self.train_losses['enc/train_losses'][-1]:,.4f}, "
            f"enc_train_entropy: {self.trainer.callback_metrics['enc/train_entropy'].item():,.4f}, "
            f"dec_train_loss: {self.train_losses['dec/train_losses'][-1]:,.4f}, "
            f"enc_train_edge_accuracy: {self.train_accuracies['enc/train_edge_accuracy'][-1]:.4f}"
            )

        # make decoder output plot for training data
        self.decoder_output_plot(**self.decoder_plot_data_train, type='train', is_end=False) if self.current_epoch+1 < self.trainer.max_epochs else None

        # validation
        if self.trainer.val_dataloaders:
            self.train_losses['nri/val_losses'].append(self.trainer.callback_metrics['nri/val_loss'].item())
            self.train_losses['enc/val_losses'].append(self.trainer.callback_metrics['enc/val_loss'].item())
            self.train_losses['dec/val_losses'].append(self.trainer.callback_metrics['dec/val_loss'].item())

            if 'enc/val_edge_accuracy' in self.trainer.callback_metrics:
                self.train_accuracies['enc/val_edge_accuracy'].append(self.trainer.callback_metrics['enc/val_edge_accuracy'].item())

            # make decoder output plot for val data
            self.decoder_output_plot(**self.decoder_plot_data_val, type='val', is_end=False) if self.current_epoch+1 < self.trainer.max_epochs else None

            print(
                f"nri_val_loss: {self.train_losses['nri/val_losses'][-1]:,.4f}, " 
                f"enc_val_loss: {self.train_losses['enc/val_losses'][-1]:,.4f}, "
                f"enc_val_entropy: {self.trainer.callback_metrics['enc/val_entropy'].item():,.4f}, "
                f"dec_val_loss: {self.train_losses['dec/val_losses'][-1]:,.4f}, "
                f"enc_val_edge_accuracy: {self.train_accuracies['enc/val_edge_accuracy'][-1]:.4f}"
                "\n\n" + 75*'-' + '\n'
                )
            

        # update hparams
        self.training_time = time.time() - self.start_time
        self.hyperparams.update({
            'model_id': self.model_id,
            'n_steps': self.global_step,
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
        print(f"Total training steps: {self.global_step}")

        if self.logger:
            print(f"\nTraining completed for model '{self.model_id}'. Trained model saved at {os.path.join(self.logger.log_dir, 'checkpoints')}")
        else:
            print(f"\nTraining completed for model '{self.model_id}'. Logging is disabled, so no checkpoints are saved.")

        print('\n' + 75*'-')

        # plot training losses and decoder output
        self.training_loss_plot()
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
        #     total_steps=self.trainer.max_epochs * len(self.trainer.test_dataloaders), 
        #     final_beta=self.final_beta,
        #     warmup_frac=self.warmup_frac_beta
        # )
        self.is_beta_annealing = False  # no beta annealing during testing

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
            f"enc_test_edge_accuracy: {self.hyperparams.get('enc/test_edge_accuracy', -1):.4f}"
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
        self.decoder_output_plot(**self.decoder_plot_data_test, type=self.run_type)


# ====== Trainer.predict() method  ====== 

    def on_predict_start(self):
        """
        Called at the start of prediction.
        """
        super().on_predict_start()

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
            self.logger.experiment.add_text(f"{self.model_id} + {self.run_type}", adj_mat_text, global_step=self.global_step)

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

        epochs = range(1, len(self.train_losses[f'enc/train_losses']) + 1)

        # update font settings for plots
        plt.rcParams.update({
            "text.usetex": False,   # No external LaTeX
            "font.family": "serif",
            "mathtext.fontset": "cm",  # Computer Modern math
        })

        # create a figure with 3 subplots in a vertical grid
        fig, axes = plt.subplots(3, 1, figsize=(8, 8), dpi=100, sharex=True)

        # plot encoder losses
        axes[0].plot(epochs, self.train_losses[f'enc/train_losses'], label='train loss', color='olive')
        axes[0].plot(epochs, self.train_losses[f'enc/val_losses'], label='val loss', color='green', linestyle='--')
        axes[0].set_title('Encoder Losses')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)

        # plot decoder losses
        axes[1].plot(epochs, self.train_losses[f'dec/train_losses'], label='train loss', color='blue')
        axes[1].plot(epochs, self.train_losses[f'dec/val_losses'], label='val loss', color='cyan', linestyle='--')
        axes[1].set_title('Decoder Losses')
        axes[1].set_ylabel('Loss')
        # axes[1].set_yscale('log')
        axes[1].legend()
        axes[1].grid(True)

        # plot nri losses
        axes[2].plot(epochs, self.train_losses[f'nri/train_losses'], label='train loss', color='red')
        axes[2].plot(epochs, self.train_losses[f'nri/val_losses'], label='val loss', color='orange', linestyle='--')
        axes[2].set_title('NRI Losses (Encoder + Decoder)')
        axes[2].set_ylabel('Loss')
        axes[2].set_xlabel('Epochs')
        # axes[2].set_yscale('log')
        axes[2].legend()
        axes[2].grid(True)

        fig.suptitle(f"Train and Validation Losses : [{self.model_id}]", fontsize=12)

        # save loss plot if logger is avaialble
        if self.logger:
            fig.savefig(os.path.join(self.logger.log_dir, f'training_loss_plot_({self.model_id}).png'), dpi=500)
            self.logger.experiment.add_figure(f"{self.tb_tag}/{self.model_id}/{self.run_type}/training_loss_plot", fig, global_step=self.global_step, close=True)
            print(f"\nTraining loss (train + val) plot logged at {self.logger.log_dir}\n")
        else:
            print("\nTraining loss plot not logged as logging is disabled.\n")

            
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
            print(f"\nCreating decoder output plot for rep '{rep_num[sample_idx]:,.3f}' for {self.model_id}...")

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
        fig, axes = plt.subplots(n_nodes, n_dims, figsize=(n_dims * 4, n_nodes * 3), sharex=False, sharey=False, dpi=75)
        if n_nodes == 1:
            axes = np.expand_dims(axes, axis=0)  # ensure axes is 2D for consistent indexing
        if n_dims == 1:
            axes = np.expand_dims(axes, axis=1)

        fig.suptitle(f"Decoder Output for Rep {rep_num[sample_idx]:,.3f} : [{self.model_id} / {type}]", fontsize=16)

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
                if self.decoder.show_conf_band:
                    ax.fill_between(timesteps, conf_band_lower, conf_band_upper, color="orange", alpha=0.3, label="confidence band")

                # Add labels and legend
                #if node == n_nodes - 1:
                ax.set_xlabel("components")
                ax.set_ylabel(f"{dim_names[dim]} (SI units)")
                         
                if node == 0 and dim == n_dims - 1:
                    ax.legend(loc="upper right")

                # add node name as title for each row
                ax.set_title(f"{node_names[node]}", fontsize=11)

                ax.grid(True)

        # adjust subplot spacing to prevent label overlap
        plt.subplots_adjust(left=0.15, bottom=0.1, right=0.95, top=0.9, wspace=0.3, hspace=0.6)

        # save the plot if logger is available
        if self.logger:
            self.logger.experiment.add_figure(f"{self.tb_tag}/{self.model_id}/{self.run_type}/decoder_output_plot_{type}", fig, global_step=self.global_step, close=True)

            if is_end:
                fig.savefig(os.path.join(self.logger.log_dir, f'dec_output_{type}_({self.model_id}).png'), dpi=500)
                print(f"\nDecoder output plot for rep '{rep_num[sample_idx]}' logged at {self.logger.log_dir}\n")
        else:
            if is_end:
                print("\nDecoder output plot not logged as logging is disabled.\n")