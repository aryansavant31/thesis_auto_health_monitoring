import os, sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR) if ROOT_DIR not in sys.path else None

TP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TP_DIR) if TP_DIR not in sys.path else None

# other imports
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.optim import Adam, SGD
from pytorch_lightning import LightningModule
import time
import matplotlib.pyplot as plt
import numpy as np

# global imports
from data.config import DataConfig
from data.transform import DomainTransformer, DataNormalizer
from feature_extraction.extractor import FrequencyFeatureExtractor, TimeFeatureExtractor, FeatureReducer

# local imports
from utils.models import MLP, GRU
from utils.loss import nll_gaussian

class Decoder(LightningModule):

    def __init__(self):
        """
        Initialize the Decoder model.

        Attributes to set:
        -----------------
        - **_Model Parameters_**
        n_dims : int
            Number of dimensions of each node's feature vector.
        msg_out_size : int
            Size of the message output from the edge MLPs.
        n_edge_types : int
            Number of different edge types in the graph.
        """
        super(Decoder, self).__init__()
        # model parameters
        self.n_dims = None  # input parameter
        self.msg_out_size = None # pipeline parameters ....
        self.n_edge_types = None
        self.edge_mlp_config = None # embedding parameters ....
        self.recur_emb_type = None
        self.out_mlp_config = None
        self.do_prob = None
        self.is_batch_norm = None

        # input processor parameters
        self.domain_config = None
        self.raw_data_norm = None
        self.feat_norm = None
        self.feat_configs = None
        self.reduc_config = None

    def set_hyperparams(self, hyperparams):
        self.hyperparams = hyperparams

    def set_input_graph(self, rec_rel, send_rel, make_edge_matrix=False, **kwargs):
        """
        Set the relationship matrices defining the input graph structure to encoder.
        
        Parameters
        ----------
        rec_rel: torch.Tensor, shape (n_edges, n_nodes)
            Reciever matrix
            
        send_rel: torch.Tensor, shape (n_edges, n_nodes)
            Sender matrix
        """
        self.rec_rel = rec_rel
        self.send_rel = send_rel

        # create edge matrix from relation matrices if specified
        if make_edge_matrix:
            edge_matrix = 0.5*(rec_rel + send_rel).sum(dim=-1, keepdim=True)
            edge_matrix = edge_matrix.unsqueeze(0).repeat(kwargs['batch_size'], 1, 1) # shape: (batch_size, n_edges, n_edge_types)
            self.set_edge_matrix(edge_matrix)
            print(f"\nEdge matrix is created from relation matrices and set to decoder.")

    def set_edge_matrix(self, edge_matrix):
        """
        Set the edge matrix to decoder.
        
        Parameters
        ----------
        edge_matrix : torch.Tensor, shape (batch_size, n_edges, n_edge_types)
            Edge matrix containing the probability of edge types for all possible edges 
            for given number of nodes.
        """
        self.edge_matrix = edge_matrix

    def set_run_params(
        self, data_config:DataConfig, data_stats, 
        skip_first_edge_type=False, pred_steps=1,
        is_burn_in=False, burn_in_steps=1, is_dynamic_graph=False,
        encoder=None, temp=None, is_hard=False, show_conf_band=True
    ):
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
        self.is_hard = is_hard
        self.data_stats = data_stats
        self.data_config = data_config
        self.skip_first_edge_type = skip_first_edge_type  # skip edge type = 0
        self.pred_steps = pred_steps
        self.is_burn_in = is_burn_in
        self.burn_in_steps = burn_in_steps
        self.is_dynamic_graph = is_dynamic_graph
        self.encoder = encoder
        self.temp = temp
        self.show_conf_band = show_conf_band
   
        self.domain = self.domain_config['type']
        self.feat_names = self._get_feature_names() if self.feat_configs else None

    def _get_feature_names(self):
        """
        Get the names of the features that will be used in the anomaly detection model.
        """
        # non_rank_feats = [feat_config['type'] for feat_config in self._feat_configs if feat_config['type'] != 'from_ranks']
        rank_feats = next((feat_config['feat_list'] for feat_config in self.feat_configs if feat_config['type'] == 'from_ranks'), None)

        return rank_feats

    def set_training_params(self, lr=0.001, optimizer='adam', loss_type='nll', momentum=0.9):
        self.lr = lr
        self.optimizer = optimizer
        self.loss_type = loss_type
        self.momentum = momentum

        print(f"\nTraining parameters set to: \nlr={self.lr}, \noptimizer={self.optimizer}, \nloss_type={self.loss_type}")

    def build_model(self):
        # Make MLPs for each edge type
        self.edge_mlp_fn = nn.ModuleList(MLP(2*self.msg_out_size, 
                                self.edge_mlp_config,
                                self.do_prob, 
                                self.is_batch_norm) for _ in range(self.n_edge_types))
        
        # Make recurrent embedding function
        if self.recur_emb_type == 'gru':
            self.recurrent_emb_fn = GRU(self.n_dims,
                                        self.msg_out_size)
        elif self.recur_emb_type == 'mlp':
            self.mlp_emb_fn = nn.Linear(self.msg_out_size, self.msg_out_size)
        
        # Make MLP to predict mean of prediction
        self.mean_mlp = MLP(self.msg_out_size,
                           self.out_mlp_config,
                           self.do_prob,
                           self.is_batch_norm)
        
        # Make MLP to predict variance of prediction
        self.var_mlp = MLP(self.msg_out_size,
                           self.out_mlp_config,
                           self.do_prob,
                           self.is_batch_norm)
        
        self.output_layer_size = self.out_mlp_config[-1][0]  
        self.mean_output_layer = nn.Linear(self.output_layer_size, self.n_dims) 
        self.var_output_layer = nn.Linear(self.output_layer_size, self.n_dims) 

    
    def init_input_processors(self):
        print(f"\nInitializing input processors for decoder model...") 

        domain_str = self._get_config_str([self.domain_config])
        feat_str = self._get_config_str(self.feat_configs) if self.feat_configs else 'None'
        reduc_str = self._get_config_str([self.reduc_config]) if self.reduc_config else 'None'

        self.domain_transformer = DomainTransformer(domain_config=self.domain_config, data_config=self.data_config)
        #if self.domain == 'time':
        print(f"\n>> Domain transformer initialized: {domain_str}") 
        # elif self.domain == 'freq':
        #     print(f"\n>> Domain transformer initialized for 'frequency' domain") 

        # initialize raw data normalizers
        if self.raw_data_norm:
            if not self.feat_configs or self.domain == 'time': # have raw data normalization for both doamin if no feature extraction. But if feature extraction, only allow time domain
                self.raw_data_normalizer = DataNormalizer(norm_type=self.raw_data_norm, data_stats=self.data_stats)
                print(f"\n>> Raw data normalizer initialized with '{self.raw_data_norm}' normalization") 
            else:
                self.raw_data_normalizer = None
                print(f"\n>> Raw data normalization skipped due to data domain being '{self.domain}' ({self.domain} data cannot be normalized before feature extraction.")
        else:
            self.raw_data_normalizer = None
            print("\n>> No raw data normalization is applied") 

        # initialize feature normalizer
        if self.feat_norm:
            if self.feat_configs:
                self.feat_normalizer = DataNormalizer(norm_type=self.feat_norm)
                print(f"\n>> Feature normalizer initialized with '{self.feat_norm}' normalization")
            else:
                self.feat_normalizer = None
                print(f"\n>> Feature normalization skipped as no feature extraction is applied.") 
        else:
            self.feat_normalizer = None
            print("\n>> No feature normalization is applied") 

        # define feature objects
        if self.domain == 'time':
            if self.feat_configs:
                self.time_fex = TimeFeatureExtractor(self.feat_configs)
                print(f"\n>> Time feature extractor initialized with features: {feat_str}") 
            else:
                self.time_fex = None
                print("\n>> No time feature extraction is applied") 

        elif self.domain == 'freq':
            if self.feat_configs:
                self.freq_fex = FrequencyFeatureExtractor(self.feat_configs, data_config=self.data_config)
                print(f"\n>> Frequency feature extractor initialized with features: {feat_str}") 
            else:
                self.freq_fex = None
                print("\n>> No frequency feature extraction is applied") 
        
        # define feature reducer
        if self.reduc_config:
            self.feat_reducer = FeatureReducer(reduc_config=self.reduc_config)
            print(f"\n>> Feature reducer initialized with '{reduc_str}' reduction") 
        else:
            self.feat_reducer = None
            print("\n>> No feature reduction is applied") 

        print('\n' + 75*'-')

    def _get_config_str(self, configs:list):
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
    
    def pairwise_op(self, node_emb, rec_rel, send_rel):
        """
        Returns
        -------
        edge_feature : torch.Tensor, (batch_size, num_edges, 2*msg_out_size)
        """
        receivers = torch.matmul(rec_rel, node_emb)
        senders = torch.matmul(send_rel, node_emb)
        return torch.cat([senders, receivers], dim=-1)
    
    def get_start_idx(self):
        if self.skip_first_edge_type:
            start_idx = 1
            norm = float(self.n_edge_types) - 1.
        else:
            start_idx = 0
            norm = float(self.n_edge_types)

        return start_idx, norm
    
    def single_step_forward(self, inputs, rec_rel, send_rel,
                            rel_type, hidden, step):
        """
        Parameters
        ----------
        inputs : torch.Tensor, shape (batch_size, n_nodes, n_dims)

        hidden : torch.Tensor, shape (batch_size, n_nodes, msg_out_size)

        Returns
        -------
        pred : torch.Tensor, shape (batch_size, n_nodes, n_dims)
            Predicted next step for each node.

        var : torch.Tensor, shape (batch_size, n_nodes, n_dims)
            Predicted variance of the next step for each node.
            
        hidden : torch.Tensor, shape (batch_size, n_nodes, msg_out_size)
            Hidden state for the next step, used for recurrent embedding.
        """
        n_nodes = inputs.size(1)
        
        # node2edge
        pre_msg = self.pairwise_op(hidden, rec_rel, send_rel) # pre_msg is edge feature e_ij

        all_msgs = torch.zeros(pre_msg.size(0), pre_msg.size(1), self.msg_out_size, device=inputs.device)

        # skip first edge type if specified
        start_idx, norm = self.get_start_idx()

        # Run separate MLP for every edge type
        for i in range(start_idx, len(self.edge_mlp_fn)):
            msg = self.edge_mlp_fn[i](pre_msg)
            msg = msg * rel_type[:, :, i:i + 1]
            all_msgs += msg / norm

        agg_msgs = torch.matmul(all_msgs.transpose(-2, -1), rec_rel).transpose(-2, -1)  #### MSG (MeSsage aGgregation) (#Correct)
        agg_msgs = agg_msgs.contiguous() / n_nodes  # Average
        # agg_msgs has shape (batch_size, n_nodes, msg_out_size)

        # Recurrent embedding function
        if self.recur_emb_type in ['gru']:
            hidden = self.recurrent_emb_fn(inputs, agg_msgs, hidden)         #### h_tilde_j^t+1   
        
        elif self.recur_emb_type in ['mlp']:
            hidden = self.mlp_emb_fn(agg_msgs)
            hidden = F.relu(hidden)

        # Predict mean delta of signal
        x_m = self.mean_mlp(hidden)  
        mean = self.mean_output_layer(x_m)       #### fout(h_tilde_j^t+1)

        # Predict variance of delta of signal
        x_v = self.var_mlp(hidden)
        var = F.softplus(self.var_output_layer(x_v)) + 1e-6  # softplus to ensure variance is positive

        # Add mean delta to get next step prediction
        pred = inputs + mean            #### mu_j^t+1

        return pred, var, hidden
    
    def get_edge_matrix(data, encoder, rec_rel, send_rel, temp, is_hard):
        logits = encoder(data, rec_rel, send_rel)
        edge_matrix = F.gumbel_softmax(logits, tau=temp, hard=is_hard)

        return edge_matrix
    
    def process_input_data(self, time_data):
        """
        Parameters
        ----------
        time_data : torch.Tensor, shape (batch_size, n_nodes, n_timesteps, n_dims)
            Input node data

        Returns
        -------
        data : torch.Tensor, shape (batch_size, n_nodes, n_components, n_dims)
        """

        # domain transform data (mandatory)
        if self.domain == 'time':
            data = self.domain_transformer.transform(time_data)
        elif self.domain == 'freq':
            data, freq_bins = self.domain_transformer.transform(time_data)

        # normalize raw data (optional)
        if self.raw_data_normalizer:
            data = self.raw_data_normalizer.normalize(data)

        # extract features from data (optional)
        if self.domain == 'time':
            if self.time_fex:
                data = self.time_fex.extract(data)
        elif self.domain == 'freq':
            if self.freq_fex:
                data = self.freq_fex.extract(data, freq_bins)

        # normalize features (optional : if feat_norm is provided)
        if self.feat_normalizer:
            data = self.feat_normalizer.normalize(data)

        # reduce features (optional : if reduc_config is provided)
        if self.feat_reducer:
            data = self.feat_reducer.reduce(data)

        return data
    
    
    def forward(self, data):
        """
        Run the forward pass of the decoder.

        Note
        ----
        Ensure to run `set_input_graph()`, `set_edge_matrix()`, and `set_run_params()` before running this method.

        Parameters
        ----------
        data : torch.Tensor, shape (batch_size, n_nodes, n_timesteps, n_dims)
            Input data tensor containing the entire trajectory data of all nodes.
        
        Returns
        -------
        preds : torch.Tensor, shape (batch_size, n_nodes, n_components-1, n_dims)
        vars : torch.Tensor, shape (batch_size, n_nodes, n_components-1, n_dims)
        """
        # Put all varaibles to device of data
        self.rec_rel = self.rec_rel.to(data.device)
        self.send_rel = self.send_rel.to(data.device)
        self.edge_matrix = self.edge_matrix.to(data.device) 
        
        # process data
        data = self.process_input_data(data)

        # data has shape [batch_size, n_nodes, n_components, n_dims]
        inputs = data.transpose(1, 2).contiguous()
        # inputs has shape [batch_size, n_components, n_nodes, n_dims]

        n_components = inputs.size(1)

        hidden = torch.zeros(inputs.size(0), inputs.size(2), self.msg_out_size, device=inputs.device)
        
        if inputs.is_cuda:
            hidden = hidden.cuda()

        pred_all = []
        var_all = []

        for step in range(0, n_components - 1):

            if self.is_burn_in:
                if step <= self.burn_in_steps:
                    ins = inputs[:, step, :, :]  # here, ins = ground truth
                else:
                    ins = pred_all[step - 1]    # here, ins = last prediction wrt to current step
            else:
                assert (self.pred_steps <= n_components) # if pred_Step is 100 and n_components is 50, this will return error
                
                if not step % self.pred_steps: # if step is multiple of pred_steps, use ground truth as input
                    ins = inputs[:, step, :, :]
                else:                           # else use last prediction as input
                    ins = pred_all[step - 1]

            if self.is_dynamic_graph and step >= self.burn_in_steps: 
                
                # NOTE: Assumes burn_in_steps = args.timesteps
                self.edge_matrix = self.get_edge_matrix(ins, self.encoder, self.rec_rel, self.send_rel, self.temp, self.is_hard)

            pred, var, hidden = self.single_step_forward(ins, self.rec_rel, self.send_rel,
                                                    self.edge_matrix, hidden, step)
            pred_all.append(pred)
            var_all.append(var)

        preds = torch.stack(pred_all, dim=1)
        vars = torch.stack(var_all, dim=1)
        preds = preds.transpose(1, 2).contiguous()
        vars = vars.transpose(1, 2).contiguous()

        return preds, vars
    
# ================== PYTORCH LIGHTNING TRAINER METHODS ================== #

    def on_load_checkpoint(self, checkpoint):
        # model params
        self.n_dims = checkpoint["n_dims"]
        self.msg_out_size = checkpoint["msg_out_size"]
        self.n_edge_types = checkpoint["n_edge_types"]
        self.edge_mlp_config = checkpoint["edge_mlp_config"]
        self.recur_emb_type = checkpoint["recur_emb_type"]
        self.out_mlp_config = checkpoint["out_mlp_config"]
        self.do_prob = checkpoint["do_prob"]
        self.is_batch_norm = checkpoint["is_batch_norm"]
        self.hyperparams = checkpoint["hyperparams"]

        # input processor params
        self.domain_config = checkpoint["domain_config"]
        self.raw_data_norm = checkpoint["raw_data_norm"]
        self.feat_norm = checkpoint["feat_norm"]
        self.feat_configs = checkpoint["feat_configs"]
        self.reduc_config = checkpoint["reduc_config"]

        # train params
        self.lr = checkpoint["lr"]
        self.optimizer = checkpoint["optimizer"]
        self.loss_type = checkpoint["loss_type"]
        self.momentum = checkpoint["momentum"]

        # rebuild the model with the restored attributes
        self.build_model()

    def on_save_checkpoint(self, checkpoint):
        # model params
        checkpoint["n_dims"] = self.n_dims
        checkpoint["msg_out_size"] = self.msg_out_size
        checkpoint["n_edge_types"] = self.n_edge_types
        checkpoint["edge_mlp_config"] = self.edge_mlp_config
        checkpoint["recur_emb_type"] = self.recur_emb_type
        checkpoint["out_mlp_config"] = self.out_mlp_config
        checkpoint["do_prob"] = self.do_prob
        checkpoint["is_batch_norm"] = self.is_batch_norm
        checkpoint["hyperparams"] = self.hyperparams

        # input processor params
        checkpoint["domain_config"] = self.domain_config
        checkpoint["raw_data_norm"] = self.raw_data_norm
        checkpoint["feat_norm"] = self.feat_norm
        checkpoint["feat_configs"] = self.feat_configs
        checkpoint["reduc_config"] = self.reduc_config

        # train params
        checkpoint["lr"] = self.lr
        checkpoint["optimizer"] = self.optimizer
        checkpoint["loss_type"] = self.loss_type
        checkpoint["momentum"] = self.momentum

    def configure_optimizers(self):
        if self.optimizer == 'adam':
            return Adam(self.parameters(), lr=self.lr)
        elif self.optimizer == 'sgd':
            return SGD(self.parameters(), lr=self.lr, momentum=self.momentum)
        
# ====== Trainer.fit() methods  ======
        
    def on_fit_start(self):
        """
        Called at the start of training.
        """
        super().on_fit_start()
        
        self.model_id = os.path.basename(self.logger.log_dir) if self.logger else 'decoder'
        self.tb_tag = self.model_id.split('-')[0].strip('[]').replace('_(', "  (").replace('+', " + ") if self.logger else 'decoder'
        self.run_type = "train"

        self.init_input_processors()

        self.train_losses = {
            'train_losses': [],
            'val_losses': [],
        }
        self.start_time = time.time()

    
    def _forward_pass(self, batch, batch_idx):
        """
        Perform a forward pass through the model.
        """
        data, relations, _, rep_num = batch

        # edge_matrix = 0.5*(rec_rel + send_rel).sum(dim=2, keepdim=True) 
        
        target = self.process_input_data(data)[:, :, 1:, :] # get target for decoder based on its transform

        # Forward pass
        x_pred, x_var = self.forward(data)

        # Loss calculation
        if self.loss_type == 'nll':
            loss = nll_gaussian(x_pred, target, x_var)
        if self.loss_type == 'mse':
            loss = F.mse_loss(x_pred, target)

        decoder_plot_data = {
            'x_pred': x_pred,
            'x_var': x_var,
            'target': target,
            'rep_num': rep_num,
        }

        return loss, decoder_plot_data
    

    def training_step(self, batch, batch_idx):
        """
        Training step for the decoder.

        Parameters
        ----------
        batch : tuple
            A tuple containing the node data and the edge matrix label
            - data : torch.Tensor, shape (batch_size, n_nodes, n_timesteps, n_dims)
            - relations : torch.Tensor, shape (batch_size, n_edges)
        """

        # if self.current_epoch == 0 and batch_idx == 0:
        #     self.start_time = time.time()

        loss, self.decoder_plot_data_train = self._forward_pass(batch, batch_idx)

        self.log_dict(
            {'train_loss': loss},
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True
        )     
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step for the decoder.

        Parameters
        ----------
        batch : tuple
            A tuple containing the data batch and the relationship batch.
            - data_batch : tuple
                Contains the data tensor and the relations tensor.
            - rel_batch : tuple
                Contains the receiver and sender relationship matrices.
        """
        loss, self.decoder_plot_data_val = self._forward_pass(batch, batch_idx)

        self.log_dict(
            {'val_loss': loss},
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True
        )
        return loss
    
    def on_train_epoch_end(self):
        """
        Called at the end of each training epoch. Updates the training losses.
        """
        self.train_losses['train_losses'].append(self.trainer.callback_metrics['train_loss'].item())
        print(f"\nEpoch {self.current_epoch+1}/{self.trainer.max_epochs} completed, Global Step: {self.global_step}")
        train_loss_str = f"train_loss: {self.train_losses['train_losses'][-1]:,.4f}"

        # make decoder output plot for training data
        self.decoder_output_plot(**self.decoder_plot_data_train, type='train', is_end=False) if self.current_epoch+1 < self.trainer.max_epochs else None

        if self.trainer.val_dataloaders:
            self.train_losses['val_losses'].append(self.trainer.callback_metrics['val_loss'].item())
            val_loss_str = f", val_loss: {self.train_losses['val_losses'][-1]:,.4f}"

            # make decoder output plot for val data
            self.decoder_output_plot(**self.decoder_plot_data_val, type='val', is_end=False) if self.current_epoch+1 < self.trainer.max_epochs else None
        else:
            val_loss_str = ""
        
        print(f"{train_loss_str}{val_loss_str}")

        # update hparams
        self.training_time = time.time() - self.start_time
        self.hyperparams.update({
            'model_id': self.model_id,
            'model_num': float(self.logger.log_dir.split('_')[-1]) if self.logger else 0,
            'training_time': self.training_time,
            'n_steps': self.global_step,
            'train_loss': self.train_losses['train_losses'][-1],
            'val_loss': self.train_losses['val_losses'][-1] if self.trainer.val_dataloaders else None,
        })

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
        self.init_input_processors()

        # Log model information
        self.model_id = self.hyperparams.get('model_id', 'decoder')
        self.tb_tag = self.model_id.split('-')[0].strip('[]').replace('_(', "  (").replace('+', " + ") if self.logger else 'decoder'
        self.run_type = os.path.basename(self.logger.log_dir) if self.logger else 'test'

        self.start_time = time.time()

    def test_step(self, batch, batch_idx):
        """
        Test step for the decoder.

        Parameters
        ----------
        batch : tuple
            A tuple containing the data batch and the relationship batch.
            - data_batch : tuple
                Contains the data tensor and the relations tensor.
            - rel_batch : tuple
                Contains the receiver and sender relationship matrices.
        """
        loss, self.decoder_plot_data_test = self._forward_pass(batch, batch_idx)

        self.log_dict(
            {'test_loss': loss},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )
        return loss
    
    def on_test_epoch_end(self):
        """
        Called at the end of the test epoch. Updates the hyperparameters with test losses.
        """
        self.infer_time = time.time() - self.start_time
        print(f"\nTesting completed in {self.infer_time:.2f} seconds or {self.infer_time / 60:.2f} minutes or {self.infer_time / 60 / 60} hours.")

        # update hparams
        self.hyperparams.update({
            'test_loss': self.trainer.callback_metrics['test_loss'].item(),
            'infer_time': self.infer_time,
            })

        # print stats after each epoch
        print(f"\ntest_loss: {self.hyperparams['test_loss']:,.4f}")
        
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
        super().on_predict_start()
        self.init_input_processors()

        # Log model information
        self.model_id = self.hyperparams.get('model_id', 'decoder')
        self.tb_tag = self.model_id.split('-')[0].strip('[]').replace('_(', "  (").replace('+', " + ") if self.logger else 'decoder'
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
        """
        loss, self.decoder_plot_data_predict = self._forward_pass(batch, batch_idx)

        infer_time = time.time() - self.start_time
        print(f"\nPrediction completed in {infer_time:.2f} seconds or {infer_time / 60:.2f} minutes or {infer_time / 60 / 60} hours.")
        
        print(f"\nDecoder residual: {loss.item():,.4f}")
        print('\n' + 75*'-')

        # make decoder output plot
        self.decoder_output_plot(**self.decoder_plot_data_predict, type=self.run_type)

        return {
            'dec/residual': loss.item(),
            'dec/x_pred': self.decoder_plot_data_predict['x_pred'],
        }
    
    
# ================== Visualization Methods =======================

    def training_loss_plot(self):
        """
        Plot all the losses against the epochs.
        """
        if not self.train_losses:
            raise ValueError("No training losses found. Please run the training step first.")
        
        print("\n" + 12*"<" + " TRAINING LOSS PLOT (TRAIN + VAL) " + 12*">")
        print(f"\nCreating training loss plot for {self.model_id}...")

        epochs = range(1, len(self.train_losses[f'train_losses']) + 1)

        # update font settings for plots
        plt.rcParams.update({
            "text.usetex": False,   # No external LaTeX
            "font.family": "serif",
            "mathtext.fontset": "cm",  # Computer Modern math
        })

        # create a figure with 3 subplots in a vertical grid
        plt.figure(figsize=(8, 6), dpi=100)

        # plot nri losses
        plt.plot(epochs, self.train_losses[f'train_losses'], label='train loss', color='blue')
        plt.plot(epochs, self.train_losses[f'val_losses'], label='val loss', color='orange', linestyle='--')
        plt.title(F"Train and Validation Losses : [{self.model_id}]")
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        # plt.yscale('log')
        plt.legend()
        plt.grid(True)

        # save loss plot if logger is avaialble
        if self.logger:
            fig = plt.gcf()
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

        node_names = [f"{node_name}" for node_name in self.data_config.signal_types['group'].keys()]

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
            dim_names = self.data_config.signal_types['group'][node_names[node]]

            for dim in range(n_dims):
                ax = axes[node, dim]

                # extract data for the current node and dim
                timesteps = np.arange(n_comps)
                gt = target[sample_idx, node, :, dim]  # ground truth
                pred = x_pred[sample_idx, node, :, dim]  # predictions
                std_dev = np.sqrt(x_var[sample_idx, node, :, dim])  # standard deviation

                conf_band_upper = pred + (pred * 1.96 * std_dev)  # upper confidence band
                conf_band_lower = pred - (pred * 1.96 * std_dev)  # lower confidence band

                # plot ground truth, predictions, and confidence band
                ax.plot(timesteps, gt, label="ground truth", color="blue", linestyle="--")
                ax.plot(timesteps, pred, label="prediction", color="red", alpha=0.7)
                if self.show_conf_band:
                    ax.fill_between(timesteps, conf_band_lower, conf_band_upper, color="orange", alpha=0.3, label="relative confidence")

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