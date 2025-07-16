from pytorch_lightning import LightningModule
import torch
import torch.nn.functional as F
from torch.optim import Adam, SGD
import matplotlib
matplotlib.use('Agg')  # <-- Add this at the very top, before importing pyplot
import matplotlib.pyplot as plt
from .utils.loss import kl_categorical, kl_categorical_uniform, nll_gaussian

class NRI(LightningModule):
    def __init__(self, encoder, decoder):
        super(NRI, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def set_training_params(self, lr=0.001, optimizer='adam', loss_type_encoder='kld',
                             loss_type_decoder='nll', prior=None):
        self.lr = lr
        self.optimizer = optimizer
        self.prior = prior
        self.loss_type_encoder = loss_type_encoder
        self.loss_type_decoder = loss_type_decoder

        self.train_losses_per_epoch = []

    def set_input_example_for_graph(self, n_nodes):
        self.example_input_array = torch.rand((1, n_nodes, self.encoder.n_datapoints, self.encoder.n_dims))

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

    def set_run_params(self, skip_first_edge_type=False,pred_steps=1,
                is_burn_in=False, burn_in_steps=1, is_dynamic_graph=False,
                encoder=None, temp=0.5, is_hard=False):
        """
        Parameters
        ----------
        dynamic_graph : bool
            If True, the edge types are estimated dynamically at each step.
            - Example:
                when step number (eg 42) is beyond the burn in step (40 in my eg), 
                if dynamics graph is true,  new latent graph will be estimated for data from 
                42 - 40 = 2nd timestep till 42th timestep. 
                So basically, for burn-in step sized trajectory (in my case, trajectory size = 40), 
                the graph will be estimated from encoder. 
                So if graph is dynamic, it means the graph can change from timestep 'burnin_step' (40) onwards.
        """
        self.temp = temp
        self.is_hard = is_hard

        self.decoder.set_run_params(skip_first_edge_type=skip_first_edge_type, pred_steps=pred_steps, is_burn_in=is_burn_in, burn_in_steps=burn_in_steps, 
                                    is_dynamic_graph=is_dynamic_graph, encoder=encoder,
                                    temp=temp, is_hard=is_hard)

    def forward(self, data):
        """
        Run the forward pass of the encoder and decoder.

        Note
        ----
        Ensure to run `set_input_graph()` and `set_decoder_run_params()` before running this method.

        Parameters
        ----------
        data : torch.Tensor, shape (batch_size, n_nodes, n_datapoints, n_dims)
            Input data tensor containing the entire trajectory data of all nodes.
        
        Returns
        -------
        edge_pred : torch.Tensor, shape (batch_size, n_edges, n_edge_types)
            Predicted edge probabilities.
        x_pred : torch.Tensor, shape (batch_size, n_nodes, n_datapoints-1, n_dim)
            Predicted node data
        x_var : torch.Tensor, shape (batch_size, n_nodes, n_datapoints-1, n_dim)
            Variance of the predicted node data.
        """
        # Encoder
        logits = self.encoder(data)
        edge_matrix = F.gumbel_softmax(logits, tau=self.temp, hard=self.is_hard)
        edge_pred = F.softmax(logits, dim=-1)

        # Decoder
        self.decoder.set_edge_matrix(edge_matrix)
        x_pred, x_var = self.decoder(data)

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
        
    def training_step(self, batch, batch_idx):
        """
        Training step for the topology estimator.

        Parameters
        ----------
        batch : tuple
            A tuple containing the node data and the edge matrix label
            - data : torch.Tensor, shape (batch_size, n_nodes, n_datapoints, n_dims)
            - relations : torch.Tensor, shape (batch_size, n_edges)
        """
        data, relations = batch

        num_nodes = data.size(1)
        target = data[:, :, 1:, :]

        edge_pred, x_pred, x_var = self.forward(data)

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
            edge_accuracy = 'None'

        self.log_dict(
            {
                'train_loss': loss,
                'train_loss_encoder': loss_encoder,
                'train_loss_decoder': loss_decoder,
                'train_edge_accuracy': edge_accuracy,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )
            
        return loss
    
    def on_train_epoch_end(self):
        avg_loss = self.trainer.callback_metrics['train_loss'].item()
        self.train_losses_per_epoch.append(avg_loss)

    def on_train_end(self):
        if self.logger:
            fig, ax = plt.subplots()
            ax.plot(range(1, len(self.train_losses_per_epoch) + 1), self.train_losses_per_epoch)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Average Training Loss")
            ax.set_title("Training Loss vs Epoch")
            
            self.logger.experiment.add_figure("Loss vs Epoch", fig, global_step=self.global_step)
            plt.close(fig)
        else:
            print("No logger set, so no plots made.")
        
    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def predict_step(self, batch, batch_idx):
        pass



