import lightning  as L
import torch
import torch.nn.functional as F
from torch.optim import Adam, SGD
from utils.loss import kl_categorical, kl_categorical_uniform, nll_gaussian


class TopologyEstimator(L.LightningModule):
    def __init__(self, encoder, decoder):
        super(TopologyEstimator, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def set_training_params(self, lr=0.001, optimizer='adam', loss_type_encoder='kld',
                             loss_type_decoder='nnl', prior=None):
        self.lr = lr
        self.optimizer = optimizer
        self.prior = prior
        self.loss_type_encoder = loss_type_encoder
        self.loss_type_decoder = loss_type_decoder

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

    def set_run_params(self, pred_steps=1,
                is_burn_in=False, burn_in_steps=1, is_dynamic_graph=False,
                encoder=None, temp=None, is_hard=False):
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

        self.decoder.set_run_params(pred_steps=pred_steps, is_burn_in=is_burn_in, burn_in_steps=burn_in_steps, 
                                    is_dynamic_graph=is_dynamic_graph, encoder=self.encoder,
                                    temp=temp, is_hard=is_hard)

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
        x_pred : torch.Tensor, shape (batch_size, n_nodes, n_timesteps-1, n_dim)
            Predicted node data
        x_var : torch.Tensor, shape (batch_size, n_nodes, n_timesteps-1, n_dim)
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
            - data : torch.Tensor, shape (batch_size, n_nodes, n_timesteps, n_dims)
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
        if self.loss_type_decoder == 'nnl':
            loss_decoder = nll_gaussian(x_pred, target, x_var)

        # total loss
        loss = loss_encoder + loss_decoder
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_loss_encoder', loss_encoder, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_loss_decoder', loss_decoder, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        if relations is not None:
            edge_accuracy = (edge_pred.argmax(dim=-1) == relations).float().mean()
            self.log('train_edge_accuracy', edge_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            
        return loss
    
    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def predict_step(self, batch, batch_idx):
        pass



