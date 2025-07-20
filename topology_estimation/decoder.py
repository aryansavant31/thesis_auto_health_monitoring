import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from .utils.models import MLP, GRU
from pytorch_lightning import LightningModule
from data.transform import DataTransformer
from feature_extraction import FeatureExtractor

class Decoder(LightningModule):

    def __init__(self, n_dim, 
                 msg_out_size, n_edge_types,
                 edge_mlp_config, recurrent_emd_type, out_mlp_config, do_prob, is_batch_norm 
                 ):
        super(Decoder, self).__init__()
        # input parameters
        self.n_dim = n_dim

        # pipeline parameters
        self.msg_out_size = msg_out_size
        self.n_edge_types = n_edge_types

        # embedding parameters
        self.edge_mlp_config = edge_mlp_config
        self.recurremt_emb_type = recurrent_emd_type
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
            self.recurrent_emb_fn = GRU(self.n_dim,
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
        self.mean_output_layer = nn.Linear(self.output_layer_size, n_dim) 
        self.var_output_layer = nn.Linear(self.output_layer_size, n_dim) 
    
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

    def set_run_params(self, data_stats, domain='time', norm_type='std', fex_configs=[], 
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
        self.skip_first_edge_type = skip_first_edge_type  # skip edge type = 0
        self.pred_steps = pred_steps
        self.is_burn_in = is_burn_in
        self.burn_in_steps = burn_in_steps
        self.is_dynamic_graph = is_dynamic_graph
        self.encoder = encoder
        self.temp = temp
        self.is_hard = is_hard
        self.fex_configs = fex_configs
        
        self.transform = DataTransformer(domain=domain, norm_type=norm_type, data_stats=data_stats)
        self.feature_extractor = FeatureExtractor(fex_configs=fex_configs)
        
    
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
        inputs : torch.Tensor, shape (batch_size, n_nodes, n_dim)

        hidden : torch.Tensor, shape (batch_size, n_nodes, msg_out_size)
        Returns
        -------
        pred : torch.Tensor, shape (batch_size, n_nodes, n_dim)
            Predicted next step for each node.

        var : torch.Tensor, shape (batch_size, n_nodes, n_dim)
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
    
    def process_input_data(self, data):
        """
        Transform the data
            - domain change
            - normalization
        Feature extraction
        """
        # transform data
        data = self.transform(data)

        # extract features from data if fex_configs are provided
        if self.fex_configs:
            data = self.feature_extractor(data)

        return data
    
    def forward(self, data):
        """
        Run the forward pass of the decoder.

        Note
        ----
        Ensure to run `set_input_graph()`, `set_edge_matrix()`, and `set_run_params()` before running this method.

        Parameters
        ----------
        data : torch.Tensor, shape (batch_size, n_nodes, n_timesteps, n_dim)
            Input data tensor containing the entire trajectory data of all nodes.
        
        Returns
        -------
        preds : torch.Tensor, shape (batch_size, n_nodes, n_components-1, n_dim)
        vars : torch.Tensor, shape (batch_size, n_nodes, n_components-1, n_dim)
        """
        # process data
        data = self.process_input_data(data)

        # data has shape [batch_size, n_nodes, n_components, n_dim]
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
