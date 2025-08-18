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

# global imports
from data.transform import DomainTransformer, DataNormalizer
from feature_extraction.extractor import FrequencyFeatureExtractor, TimeFeatureExtractor, FeatureReducer

# local imports
from utils.models import MLP, GRU
from utils.loss import nll_gaussian

class Decoder(LightningModule):

    def __init__(self, n_dims, 
                 msg_out_size, n_edge_types,
                 edge_mlp_config, recur_emb_type, out_mlp_config, do_prob, is_batch_norm 
                 ):
        super(Decoder, self).__init__()
        # input parameters
        self.n_dims = n_dims

        # pipeline parameters
        self.msg_out_size = msg_out_size
        self.n_edge_types = n_edge_types

        # embedding parameters
        self.edge_mlp_config = edge_mlp_config
        self.recurremt_emb_type = recur_emb_type
        self.out_mlp_config = out_mlp_config
        self.do_prob = do_prob
        self.is_batch_norm = is_batch_norm
        
        # Make MLPs for each edge type
        self.edge_mlp_fn = nn.ModuleList(MLP(2*self.msg_out_size, 
                                self.edge_mlp_config,
                                self.do_prob, 
                                self.is_batch_norm) for _ in range(self.n_edge_types))
        
        # Make recurrent embedding function
        if self.recurremt_emb_type == 'gru':
            self.recurrent_emb_fn = GRU(self.n_dims,
                                        self.msg_out_size)
        
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
        
        self.output_layer_size = out_mlp_config[-1][0]  
        self.mean_output_layer = nn.Linear(self.output_layer_size, n_dims) 
        self.var_output_layer = nn.Linear(self.output_layer_size, n_dims) 
    
    def set_input_graph(self, rec_rel, send_rel):
        """
        Set the relationship matrices defining the input graph structure to encoder.
        
        Parameters
        ----------
        rec_rel: torch.Tensor, shape (batch_size, n_edges, n_nodes)
            Reciever matrix
            
        send_rel: torch.Tensor, shape (batch_size, n_edges, n_nodes)
            Sender matrix
        """
        self.rec_rel = rec_rel
        self.send_rel = send_rel

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

    def set_run_params(self, data_stats, domain_config, raw_data_norm=None, feat_norm=None, feat_configs=[], reduc_config=None, 
                        skip_first_edge_type=False, pred_steps=1,
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
        self._domain_config = domain_config
        self._raw_data_norm = raw_data_norm
        self._feat_norm = feat_norm
        self._feat_configs = feat_configs
        self._reduc_config = reduc_config
        self._is_hard = is_hard

        self.data_stats = data_stats
        self.skip_first_edge_type = skip_first_edge_type  # skip edge type = 0
        self.pred_steps = pred_steps
        self.is_burn_in = is_burn_in
        self.burn_in_steps = burn_in_steps
        self.is_dynamic_graph = is_dynamic_graph
        self.encoder = encoder
        self.temp = temp
   
        self._domain = domain_config['type']
        self._feat_names = self._get_feature_names() if self._feat_configs else None

    def set_training_params(self, lr=0.001, optimizer='adam', loss_type='nll'):
        self.lr = lr
        self.optimizer = optimizer
        self.loss_type = loss_type

        self.train_losses_per_epoch = []
        
    def _get_feature_names(self):
        """
        Get the names of the features that will be used in the anomaly detection model.
        """
        non_rank_feats = [feat_config['type'] for feat_config in self._feat_configs if feat_config['type'] != 'from_ranks']
        rank_feats = next((feat_config['feat_list'] for feat_config in self._feat_configs if feat_config['type'] == 'from_ranks'), [])

        return non_rank_feats + rank_feats
    
    def init_input_processors(self):
        print(f"\nInitializing input processors for anomaly detection model...") 

        self.domain_transformer = DomainTransformer(domain_config=self._domain_config)
        if self._domain == 'time':
            print(f"\n>> Domain transformer initialized for 'time' domain") 
        elif self._domain == 'freq':
            print(f"\n>> Domain transformer initialized for 'frequency' domain") 

        # initialize data normalizers
        if self._raw_data_norm:
            self.raw_data_normalizer = DataNormalizer(norm_type=self._raw_data_norm, data_stats=self.data_stats)
            print(f"\n>> Raw data normalizer initialized with '{self._raw_data_norm}' normalization") 
        else:
            self.raw_data_normalizer = None
            print("\n>> No raw data normalization is applied") 

        if self._feat_norm:
            self.feat_normalizer = DataNormalizer(norm_type=self._feat_norm)
            print(f"\n>> Feature normalizer initialized with '{self._feat_norm}' normalization") 
        else:
            self.feat_normalizer = None
            print("\n>> No feature normalization is applied") 

        # define feature objects
        if self._domain == 'time':
            if self._feat_configs:
                self.time_fex = TimeFeatureExtractor(self._feat_configs)
                print(f"\n>> Time feature extractor initialized with features: {', '.join([feat_config['type'] for feat_config in self._feat_configs])}") 
            else:
                self.time_fex = None
                print("\n>> No time feature extraction is applied") 

        elif self._domain == 'freq':
            if self._feat_configs:
                self.freq_fex = FrequencyFeatureExtractor(self._feat_configs)
                print(f"\n>> Frequency feature extractor initialized with features: {', '.join([feat_config['type'] for feat_config in self._feat_configs])}") 
            else:
                self.freq_fex = None
                print("\n>> No frequency feature extraction is applied") 
        
        # define feature reducer
        if self._reduc_config:
            self.feat_reducer = FeatureReducer(reduc_config=self._reduc_config)
            print(f"\n>> Feature reducer initialized with '{self._reduc_config['type']}' reduction") 
        else:
            self.feat_reducer = None
            print("\n>> No feature reduction is applied") 

        print('\n' + 75*'-') 
    
    def pairwise_op(self, node_emb, rec_rel, send_rel):
        receivers = torch.bmm(rec_rel, node_emb)
        senders = torch.bmm(send_rel, node_emb)
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
                            rel_type, hidden):
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
        # node2edge
        pre_msg = self.pairwise_op(hidden, rec_rel, send_rel) 

        all_msgs = torch.zeros(pre_msg.size(0), pre_msg.size(1), self.msg_out_size, device=inputs.device)

        # skip first edge type if specified
        start_idx, norm = self.get_start_idx()

        # Run separate MLP for every edge type
        for i in range(start_idx, len(self.edge_mlp_fn)):
            msg = self.edge_mlp_fn[i](pre_msg)
            msg = msg * rel_type[:, :, i:i + 1]
            all_msgs += msg / norm

        agg_msgs = torch.bmm(all_msgs.transpose(1, 2), rec_rel).transpose(1, 2)    #### MSG (MeSsage aGgregation)
        agg_msgs = agg_msgs.contiguous() / inputs.size(2)  # Average
        # agg_msgs has shape (batch_size, n_nodes, msg_out_size)

        # Recurrent embedding function
        hidden = self.recurrent_emb_fn(inputs, agg_msgs, hidden)         #### h_tilde_j^t+1   

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
    
    def process_input_data(self, time_data, batch_idx=0, current_epoch=0):
        """
        Parameters
        ----------
        time_data : torch.Tensor, shape (batch_size, n_nodes, n_timesteps, n_dims)
            Input node data

        Returns
        -------
        data : torch.Tensor, shape (batch_size, n_nodes, n_components, n_dims)
        """
        if current_epoch == 0 and batch_idx == 0:
            self.init_input_processors()

        # domain transform data (mandatory)
        if self._domain == 'time':
            data = self.domain_transformer.transform(time_data)
        elif self._domain == 'freq':
            data, freq_bins = self.domain_transformer.transform(time_data)

        # normalize raw data (optional)
        if self.raw_data_normalizer:
            if self._domain == 'time':
                data = self.raw_data_normalizer.normalize(data)
            elif self._domain == 'freq':
                print("\nFrequency data cannot be normalized before feature extraction. Hence skipping raw data normalization (for all iterations).") if current_epoch == 0 and batch_idx == 0 else None

        # extract features from data (optional)
        is_fex = False
        if self._domain == 'time':
            if self.time_fex:
                data = self.time_fex.extract(data)
                is_fex = True
        elif self._domain == 'freq':
            if self.freq_fex:
                data = self.freq_fex.extract(data, freq_bins)
                is_fex = True

        # normalize features (optional : if feat_norm is provided)
        if self.feat_normalizer:
            if is_fex:
                data = self.feat_normalizer.normalize(data)
            else:
                print("\nNo features extracted, so feature normalization is skipped (for all iterations).") if current_epoch == 0 and batch_idx == 0 else None

        # reduce features (optional : if reduc_config is provided)
        if self.feat_reducer:
            data = self.feat_reducer.reduce(data)

        return data
    
    def forward(self, data, batch_idx, current_epoch=0):
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

        # process data
        data = self.process_input_data(data, batch_idx=batch_idx, current_epoch=current_epoch)

        # data has shape [batch_size, n_nodes, n_components, n_dims]
        inputs = data.transpose(1, 2).contiguous()
        # inputs has shape [batch_size, n_components, n_nodes, n_dims]

        n_components = inputs.size(1)

        hidden = torch.zeros(inputs.size(0), inputs.size(2), self.msg_out_size, device=inputs.device)
        
        if inputs.is_cuda:
            hidden = hidden.cuda()

        pred_all = []
        var_all = []

        for step in range(0, inputs.size(1) - 1):

            if self.is_burn_in:
                if step <= self.burn_in_steps:
                    ins = inputs[:, step, :, :]  # here, ins = ground truth
                else:
                    ins = pred_all[step - 1]    # here, ins = last prediction wrt to current step
            else:
                assert (self.pred_steps <= n_components) # if pred_Step is 100 and timesteps is 50, this will return error
                # Use ground truth trajectory input vs. last prediction
                if not step % self.pred_steps:
                    ins = inputs[:, step, :, :]
                else:
                    ins = pred_all[step - 1]

            if self.is_dynamic_graph and step >= self.burn_in_steps: 
                
                # NOTE: Assumes burn_in_steps = args.timesteps
                self.edge_matrix = self.get_edge_matrix(ins, self.encoder, self.rec_rel, self.send_rel, self.temp, self.is_hard)

            pred, var, hidden = self.single_step_forward(ins, self.rec_rel, self.send_rel,
                                                    self.edge_matrix, hidden)
            pred_all.append(pred)
            var_all.append(var)

        preds = torch.stack(pred_all, dim=1)
        vars = torch.stack(var_all, dim=1)
        preds = preds.transpose(1, 2).contiguous()
        vars = vars.transpose(1, 2).contiguous()

        return preds, vars
    
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

        if self.current_epoch == 0 and batch_idx == 0:
            self.start_time = time.time()
            print(f"train start time: {self.start_time}")

        data_batch, rel_batch = batch

        data, relations, _ = data_batch
        rec_rel, send_rel = rel_batch

        edge_matrix = 0.5*(rec_rel + send_rel).sum(dim=2, keepdim=True) 
        
        target = self.process_input_data(data, batch_idx=batch_idx, current_epoch=self.current_epoch)[:, :, 1:, :] # get target for decoder based on its transform

        # Forward pass
        self.set_input_graph(rec_rel, send_rel)
        self.set_edge_matrix(edge_matrix)
        x_pred, x_var = self.forward(data, batch_idx, current_epoch=self.current_epoch)

        # Loss calculation
        if self.loss_type == 'nll':
            loss = nll_gaussian(x_pred, target, x_var)

        self.log_dict(
            {
                'train_loss': loss,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )
            
        return loss

    def configure_optimizers(self):
        if self.optimizer == 'adam':
            return Adam(self.parameters(), lr=self.lr)
        elif self.optimizer == 'sgd':
            return SGD(self.parameters(), lr=self.lr)