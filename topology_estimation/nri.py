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

# local imports
from .utils.loss import kl_categorical, kl_categorical_uniform, nll_gaussian
from .encoder import Encoder
from .decoder import Decoder


class NRI(LightningModule):
    def __init__(self, encoder_params, decoder_params, hparams):
        super(NRI, self).__init__()
        self.save_hyperparameters()  # This will log encoder_params and decoder_params
        self.encoder = Encoder(**encoder_params)
        self.decoder = Decoder(**decoder_params)
        self.hparams = hparams

    def print_model_info(self):
        """
        Print the model information
        """
        print(5 * '-', 'NRI Model Summary', 5 * '-')
        print(2 * '-', 'Encoder Summary')
        print(self.encoder)
        print(2 * '-', 'Decoder Summary')
        print(self.decoder)

    def set_training_params(self, lr=0.001, optimizer='adam', loss_type_enc='kld',
                             loss_type_dec='nll', prior=None):
        self.lr = lr
        self.optimizer = optimizer
        self.prior = prior
        self.loss_type_encoder = loss_type_enc
        self.loss_type_decoder = loss_type_dec

        self.train_losses = {
            'nri/train_losses': [],
            'encoder/train_losses': [],
            'decoder/train_losses': [],
            'nri/val_losses': [],
            'encoder/val_losses': [],
            'decoder/val_losses': [],
        }
        self.train_accuracies = {
            'encoder/train_edge_accuracy': [],
            'encoder/val_edge_accuracy': [],
        }
        
    def set_input_example_for_graph(self, n_nodes):
        self.save_hyperparameters()
        self.example_input_array = torch.rand((1, n_nodes, self.encoder.n_components, self.encoder.n_dims))

    def set_input_graph(self, rec_rel, send_rel):
        """
        Set the relationship matrices defining the input graph structure.
        
        Parameters
        ----------
        rec_rel : torch.Tensor, shape (batch_size, n_edges, n_nodes)
            Receiver relationship matrix.
        
        send_rel : torch.Tensor, shape (batch_size, n_edges, n_nodes)
            Sender relationship matrix.
        """
        self.encoder.set_input_graph(rec_rel, send_rel)
        self.decoder.set_input_graph(rec_rel, send_rel)

    def set_run_params(self, enc_run_params, dec_run_params, data_stats, temp, is_hard):
        """
        Parameters
        ----------
        enc_run_params : dict
            Parameters for the encoder run. For details, see the docstring of `Encoder.set_run_params()`.
        dec_run_params : dict
            Parameters for the decoder run. For details, see the docstring of `Decoder.set_run_params()`.
        temp : float
            Temperature for Gumble Softmax.
        is_hard : bool
            If True, use hard Gumble Softmax.
        """
        self.save_hyperparameters()
        self.temp = temp
        self.is_hard = is_hard

        self.encoder.set_run_params(**enc_run_params, data_stats=data_stats)
        self.decoder.set_run_params(**dec_run_params, data_stats=data_stats)
        

    def forward(self, data, batch_idx, current_epoch=0):
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
        logits = self.encoder(data, batch_idx, current_epoch=current_epoch)
        edge_matrix = F.gumbel_softmax(logits, tau=self.temp, hard=self.is_hard)
        edge_pred = F.softmax(logits, dim=-1)

        # Decoder
        self.decoder.set_edge_matrix(edge_matrix)
        x_pred, x_var = self.decoder(data, batch_idx, current_epoch=current_epoch)

        return edge_pred, x_pred, x_var
    
    def configure_optimizers(self):
        # get params from encoder and decoder
        encoder_params = list(self.encoder.parameters())
        decoder_params = list(self.decoder.parameters())
 
        # select optimizer
        if self.optimizer == 'adam':
            return Adam(encoder_params + decoder_params, lr=self.lr)
        elif self.optimizer == 'sgd':
            return SGD(encoder_params + decoder_params, lr=self.lr)
        
    def _forward_pass(self, batch, batch_idx):
        """
        Perform a forward pass through the model.
        """
        data_batch, rel_batch = batch

        data, relations, _, rep_num = data_batch
        rec_rel, send_rel = rel_batch
        
        num_nodes = data.size(1)
        target = self.decoder.process_input_data(data, batch_idx=batch_idx, current_epoch=self.current_epoch)[:, :, 1:, :] # get target for decoder based on its transform

        # Forward pass
        self.set_input_graph(rec_rel, send_rel)
        edge_pred, x_pred, x_var = self.forward(data, batch_idx, current_epoch=self.current_epoch)

        # Loss calculation
        # encoder loss
        if self.loss_type_encoder == 'kld':
            if self.prior:
                loss_encoder = kl_categorical(edge_pred, self.prior, num_nodes)
            else:
                loss_encoder = kl_categorical_uniform(edge_pred, num_nodes)

        # decoder loss
        if self.loss_type_decoder == 'nll':
            loss_decoder = nll_gaussian(x_pred, target, x_var)

        # total loss
        loss = loss_encoder + loss_decoder

        if relations is not None:
            edge_accuracy = (edge_pred.argmax(dim=-1) == relations).float().mean()
        else:
            edge_accuracy = None
        
        # make dict
        log_data = {
            'loss': loss,
            'loss_encoder': loss_encoder,
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
    
    def edge_pred_to_adjacency_matrix(edge_pred, n_nodes):
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
        if self.current_epoch == 0 and batch_idx == 0:
            self.start_time = time.time()
            print(f"train start time: {self.start_time}")

        log_data, _, _ = self._forward_pass(batch, batch_idx)

        # Log the losses
        log_dict = {
            'nri/train_loss': log_data['loss'],
            'encoder/train_loss': log_data['loss_encoder'],
            'decoder/train_loss': log_data['loss_decoder'],
        }

        if log_data['edge_accuracy'] is not None:
            log_dict['encoder/train_edge_accuracy'] = log_data['edge_accuracy']

        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )
            
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
            'encoder/val_loss': log_data['loss_encoder'],
            'decoder/val_loss': log_data['loss_decoder'],
        }

        if log_data['edge_accuracy'] is not None:
            log_dict['encoder/val_edge_accuracy'] = log_data['edge_accuracy']

        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )

        return log_data['loss']

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
        log_data, self.decoder_plot_data_test, _ = self._forward_pass(batch, batch_idx)

        # Log the losses and metrics
        log_dict = {
            'nri/test_loss': log_data['loss'],
            'encoder/test_loss': log_data['loss_encoder'],
            'decoder/test_loss': log_data['loss_decoder'],
        }

        if log_data['edge_accuracy'] is not None:
            log_dict['encoder/test_edge_accuracy'] = log_data['edge_accuracy']

        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )

        return log_data['loss']

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

        print(f"\n Adjacency matrix (shape {adj_matrix.shape})")
        print(adj_matrix)

        print(f"\nDecoder residuals: {log_data['loss_decoder'].item():.4f}")
        print('\n' + 75*'-')

        self.model_id = self.hparams.get('model_id', 'NRI_Model')

        # make decoder output plot
        self.decoder_output_plot(**self.decoder_plot_data_predict)

        return {
            'edge_pred': edge_pred,
            'adj_matrix': adj_matrix,
            'residuals': log_data['loss_decoder']
        }
    
    def on_train_epoch_end(self):
        """
        Called at the end of each training epoch. Updates the training losses and accuracies.
        """
        self.train_losses['nri/train_losses'].append(self.trainer.callback_metrics['nri/train_loss'].item())
        self.train_losses['encoder/train_losses'].append(self.trainer.callback_metrics['encoder/train_loss'].item())
        self.train_losses['decoder/train_losses'].append(self.trainer.callback_metrics['decoder/train_loss'].item())

        if 'encoder/train_edge_accuracy' in self.trainer.callback_metrics:
            self.train_accuracies['encoder/train_edge_accuracy'].append(self.trainer.callback_metrics['encoder/train_edge_accuracy'].item())

        # print stats after each epoch
        print(
            f"\nEpoch {self.current_epoch}/{self.trainer.max_epochs}:" 
            f"\nnri/train_loss: {self.train_losses['nri/train_losses'][-1]:.4f}, " 
            f"encoder/train_loss: {self.train_losses['encoder/train_losses'][-1]:.4f}, "
            f"decoder/train_loss: {self.train_losses['decoder/train_losses'][-1]:.4f}, "
            f"encoder/train_edge_accuracy: {self.train_accuracies['encoder/train_edge_accuracy'][-1]:.4f}"
            )

    def on_validation_epoch_end(self):
        """
        Called at the end of the validation epoch. Updates the validation losses and accuracies.
        """
        self.train_losses['nri/val_losses'].append(self.trainer.callback_metrics['nri/val_loss'].item())
        self.train_losses['encoder/val_losses'].append(self.trainer.callback_metrics['encoder/val_loss'].item())
        self.train_losses['decoder/val_losses'].append(self.trainer.callback_metrics['decoder/val_loss'].item())

        if 'encoder/val_edge_accuracy' in self.trainer.callback_metrics:
            self.train_accuracies['encoder/val_edge_accuracy'].append(self.trainer.callback_metrics['encoder/val_edge_accuracy'].item())

        # make decoder output plot
        self.decoder_output_plot(**self.decoder_plot_data_val)

        # print stats after each epoch
        print(
            f"nri/val_loss: {self.train_losses['nri/val_losses'][-1]:.4f}, " 
            f"encoder/val_loss: {self.train_losses['encoder/val_losses'][-1]:.4f}, "
            f"decoder/val_loss: {self.train_losses['decoder/val_losses'][-1]:.4f}, "
            f"encoder/val_edge_accuracy: {self.train_accuracies['encoder/val_edge_accuracy'][-1]:.4f}"
            )
        
    def on_test_epoch_end(self):
        """
        Called at the end of the test epoch. Updates the hyperparameters with test losses and accuracies.
        """
        # Log model information
        self.model_id = self.hparams.get('model_id', 'NRI_Model')
        self.run_type = os.path.basename(self.logger.log_dir) if self.logger else 'test'

        # update hparams
        self.hparams.update({
            'nri/test_loss': self.trainer.callback_metrics['nri/test_loss'].item(),
            'encoder/test_loss': self.trainer.callback_metrics['encoder/test_loss'].item(),
            'decoder/test_loss': self.trainer.callback_metrics['decoder/test_loss'].item(),
        })

        if 'encoder/test_edge_accuracy' in self.trainer.callback_metrics:
            self.hparams['encoder/test_edge_accuracy'] = self.trainer.callback_metrics['encoder/test_edge_accuracy'].item()

        # make decoder output plot
        self.decoder_output_plot(**self.decoder_plot_data_test)

        # print stats after each epoch
        print(
            f"\nnri/test_loss: {self.hparams['nri/test_loss']:.4f}, " 
            f"encoder/test_loss: {self.hparams['encoder/test_loss']:.4f}, "
            f"decoder/test_loss: {self.hparams['decoder/test_loss']:.4f}, "
            f"encoder/test_edge_accuracy: {self.hparams.get('encoder/test_edge_accuracy', -1):.4f}"
            )
        
        if self.logger:
            self.logger.log_hyperparams(self.hparams, {})
            print(f"\nTest metrics and hyperparameters logged for tensorboard at {self.logger.log_dir}")
        else:
            print("\nTest metrics and hyperparameters not logged as logging is disabled.")
        
        print('\n' + 75*'-')

    def on_train_end(self):
        training_time = time.time() - self.start_time
        print(f"\nTraining completed in {training_time:.2f} seconds or {training_time / 60:.2f} minutes or {training_time / 60 / 60} hours.")

        # Log model information
        self.model_id = os.path.basename(self.logger.log_dir) if self.logger else 'NRI_model'
        self.run_type = 'train'

        # update hparams
        self.hparams.update({
            'model_id': self.model_id,

            # log train data
            'training_time': training_time,
            'nri/train_loss': self.train_losses['nri/train_losses'][-1],
            'encoder/train_loss': self.train_losses['encoder/train_losses'][-1],
            'decoder/train_loss': self.train_losses['decoder/train_losses'][-1],
            'encoder/train_edge_accuracy': self.train_accuracies['encoder/train_edge_accuracy'][-1] if self.train_accuracies['encoder/train_edge_accuracy'] else -1,
            
            # log validation data
            'nri/val_loss': self.train_losses['nri/val_losses'][-1],
            'encoder/val_loss': self.train_losses['encoder/val_losses'][-1],
            'decoder/val_loss': self.train_losses['decoder/val_losses'][-1],
            'encoder/val_edge_accuracy': self.train_accuracies['encoder/val_edge_accuracy'][-1] if self.train_accuracies['encoder/val_edge_accuracy'] else -1,
        })

        # plot training losses
        self.training_loss_plot()

        if self.logger:
            print(f"\nTraining completed for model '{self.model_id}'. Trained model saved at {os.path.join(self.logger.log_dir, 'checkpoints')}")
        else:
            print(f"\nTraining completed for model '{self.model_id}'. Logging is disabled, so no checkpoints are saved.")

        print('\n' + 75*'-')

# ================== Visualization Methods =======================

    def training_loss_plot(self):
        """
        Plot all the losses against the epochs.
        """
        if not self.train_losses:
            raise ValueError("No training losses found. Please run the training step first.")
        
        print("\n" + 12*"<" + " TRAINING LOSS PLOT (TRAIN + VAL) " + 12*">")
        print(f"\nCreating training loss plot for {self.model_id}...")

        epochs = range(1, len(self.train_losses[f'encoder/train_losses']) + 1)

        # create a figure with 3 subplots in a vertical grid
        fig, axes = plt.subplots(3, 1, figsize=(8, 12), dpi=100, sharex=True)

        # plot encoder losses
        axes[0].plot(epochs, self.train_losses[f'encoder/train_losses'], label='train loss', color='olive')
        axes[0].plot(epochs, self.train_losses[f'encoder/val_losses'], label='val loss', color='green', linestyle='--')
        axes[0].set_title('Encoder Losses')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)

        # plot decoder losses
        axes[1].plot(epochs, self.train_losses[f'decoder/train_losses'], label='train loss', color='blue')
        axes[1].plot(epochs, self.train_losses[f'decoder/val_losses'], label='val loss', color='cyan', linestyle='--')
        axes[1].set_title('Decoder Losses')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)

        # plot nri losses
        axes[2].plot(epochs, self.train_losses[f'nri/train_losses'], label='train loss', color='red')
        axes[2].plot(epochs, self.train_losses[f'nri/val_losses'], label='val loss', color='orange', linestyle='--')
        axes[2].set_title('NRI Losses (Encoder + Decoder)')
        axes[2].set_ylabel('Loss')
        axes[2].set_xlabel('Epochs')
        axes[2].legend()
        axes[2].grid(True)

        fig.subtitle(f"Train and Validation Losses ({self.model_id})", fontsize=16)

        # save loss plot if logger is avaialble
        if self.logger:
            fig.savefig(os.path.join(self.logger.log_dir, f'training_loss_plot_({self.model_id})'), dpi=500)
            self.logger.experiment.add_figure(f"{self.model_id}/training_loss_plot", fig, global_step=self.global_step, close=True)
            print(f"\nTraining loss (train + val) plot logged at {self.logger.log_dir}\n")
        else:
            print("\nTraining loss plot not logged as logging is disabled.\n")
            
    def decoder_output_plot(self, x_pred, x_var, target, rep_num, sample_idx=0):
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
        sample_idx : int, optional
            Index of the sample to plot. Default is 0.
        """
        print("\n" + 12*"<" + " DECODER OUTPUT PLOT " + 12*">")
        print(f"\nCreating decoder output plot for rep '{rep_num[sample_idx]}' for {self.model_id}...")

        # convert tensors to numpy arrays for plotting
        x_pred = x_pred.detach().cpu().numpy()
        x_var = x_var.detach().cpu().numpy()
        target = target.detach().cpu().numpy()

        batch_size, n_nodes, n_comps, n_dims = x_pred.shape

        node_names = [f'Node {i+1}' for i in range(n_nodes)]
        dim_names = [f'Dim {i+1}' for i in range(n_dims)]

        # create figure with subplots for each node and dimension
        fig, axes = plt.subplots(n_nodes, n_dims, figsize=(n_dims * 4, n_nodes * 3), sharex=True, sharey=True)
        if n_nodes == 1:
            axes = np.expand_dims(axes, axis=0)  # ensure axes is 2D for consistent indexing
        if n_dims == 1:
            axes = np.expand_dims(axes, axis=1)

        fig.suptitle(f"Decoder Output for Rep {rep_num[sample_idx]} ({self.model_id})", fontsize=16)

        for node in range(n_nodes):
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
                ax.plot(timesteps, pred, label="prediction", color="orange")
                ax.fill_between(timesteps, conf_band_lower, conf_band_upper, color="orange", alpha=0.3, label="confidence band")

                # Add labels and legend
                if node == n_nodes - 1:
                    ax.set_xlabel(dim_names[dim])
                if dim == 0:
                    ax.set_ylabel(node_names[node])
                if node == 0 and dim == n_dims - 1:
                    ax.legend(loc="upper right")

                ax.grid(True)

        # save the plot if logger is available
        if self.logger:
            fig.savefig(os.path.join(self.logger.log_dir, f'decoder_output_plot_({rep_num[sample_idx]})_({self.model_id})'), dpi=500)
            self.logger.experiment.add_figure(f"{self.model_id}/decoder_output_plot", fig, global_step=self.global_step, close=True)
            print(f"\nDecoder output plot for rep '{rep_num[sample_idx]}' logged at {self.logger.log_dir}\n")
        else:
            print("\nDecoder output plot not logged as logging is disabled.\n")
                


