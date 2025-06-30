import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from models import MLP, LSTM, RNN, GRU

class Decoder(nn.Module):

    def __init__(self, n_dim, msg_out_size, n_edge_types, edge_mlp_config, out_mlp_config, 
                 do_prob, is_batch_norm, recurrent_emd_type, n_layers_recurrent, skip_first=False):
        super(Decoder, self).__init__()
        # input parameters
        self.n_dim = n_dim

        # pipeline parameters
        self.msg_out_size = msg_out_size
        self.n_edge_types = n_edge_types
        self.skip_first_edge_type = skip_first  # skip edge type = 0

        # embedding parameters
        self.edge_mlp_config = edge_mlp_config
        self.recurremt_emb_type = recurrent_emd_type
        self.n_layers_recurrent = n_layers_recurrent
        self.out_mlp_config = out_mlp_config
        
        
        # Make MLPs for each edge type
        self.edge_mlp_fn = [MLP(2*self.msg_out_size, 
                                self.edge_mlp_config,
                                do_prob, 
                                is_batch_norm,
                                is_gnn=True) for _ in range(self.n_edge_types)]
        
        # Make recurrent embedding function
        if self.recurremt_emb_type == 'lstm':
            self.recurrent_emb_fn = LSTM(self.n_dim, 
                                         self.n_layers_recurrent,
                                         self.msg_out_size)
        elif self.recurremt_emb_type == 'rnn':
            self.recurrent_emb_fn = RNN(self.n_dim, 
                                        self.n_layers_recurrent,
                                        self.msg_out_size)
        elif self.recurremt_emb_type == 'gru':
            self.recurrent_emb_fn = GRU(self.n_dim, 
                                        self.n_layers_recurrent,
                                        self.msg_out_size)
        
        # Make output MLP
        self.out_mlp = MLP(self.msg_out_size,
                           self.out_mlp_config,
                           do_prob,
                           is_batch_norm,
                           is_gnn=True)
        
        self.output_layer_size = out_mlp_config[-1][0]  
        self.output_layer = nn.Linear(self.output_layer_size, n_dim)  
                           
        
    def pairwise_op(self, node_emb, rec_rel, send_rel):
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
                            rel_type, hidden):

        # node2edge
        pre_msg = self.pairwise_op(hidden, rec_rel, send_rel) 

        all_msgs = Variable(torch.zeros(pre_msg.size(0), pre_msg.size(1),
                                        self.msg_out_size))

        # skip first edge type if specified
        start_idx, norm = self.get_start_idx()

        # Run separate MLP for every edge type
        for i in range(start_idx, len(self.edge_mlp_fn)):
            msg = self.edge_mlp_fn[i](pre_msg)
            msg = msg * rel_type[:, :, i:i + 1]
            all_msgs += msg / norm

        agg_msgs = all_msgs.transpose(-2, -1).matmul(rec_rel).transpose(-2,     #### MSG (MeSsage aGgregation)
                                                                        -1)
        agg_msgs = agg_msgs.contiguous() / inputs.size(2)  # Average

        # Recurrent embedding function
        hidden = self.recurrent_emb_fn(inputs, agg_msgs)         #### h_tilde_j^t+1   

        # Output MLP
        x = self.out_mlp(hidden)  
        pred = self.output_layer(x)       #### fout(h_tilde_j^t+1)

        # Predict position/velocity difference
        pred = inputs + pred            #### mu_j^t+1

        return pred, hidden
    
    def get_edge_type(data, encoder, rec_rel, send_rel, temp):
        logits = encoder(data, rec_rel, send_rel)
        edge_type = gumbel_softmax(logits, tau=temp, hard=True)

        return edge_type

    def forward(self, data, edge_type, rec_rel, send_rel, pred_steps=1,
                burn_in=False, burn_in_steps=1, dynamic_graph=False,
                encoder=None, temp=None):

        inputs = data.transpose(1, 2).contiguous()

        time_steps = inputs.size(1)

        # inputs has shape
        # [batch_size, num_timesteps, num_atoms, num_dims]

        # rel_type has shape:
        # [batch_size, num_atoms*(num_atoms-1), num_edge_types]

        hidden = Variable(
            torch.zeros(inputs.size(0), inputs.size(2), self.msg_out_shape))
        if inputs.is_cuda:
            hidden = hidden.cuda()

        pred_all = []

        for step in range(0, inputs.size(1) - 1):

            if burn_in:
                if step <= burn_in_steps:
                    ins = inputs[:, step, :, :]  # here, ins = ground truth
                else:
                    ins = pred_all[step - 1]    # here, ins = last prediction wrt to current step
            else:
                assert (pred_steps <= time_steps) # if pred_Step is 100 and timesteps is 50, this will return error
                # Use ground truth trajectory input vs. last prediction
                if not step % pred_steps:
                    ins = inputs[:, step, :, :]
                else:
                    ins = pred_all[step - 1]

            if dynamic_graph and step >= burn_in_steps: 
                # when step number (eg 42) is beyond the burn in step (40 in my eg), 
                # if dynamics graph is true, this code will estiamte a new latent graph for data from 
                # 42 - 40 = 2 timestep till 42th timestep. 
                # So basically, for burn-in step sized trajectory (in my case, trajectory size = 40), 
                # the graph will be estimated from encoder. 
                # So if graph is dynamic, it means the graph can change from timestep 'burnin_step' (40) onwards.

                # NOTE: Assumes burn_in_steps = args.timesteps
                edge_type = self.get_edge_type(ins, encoder, rec_rel, send_rel, temp)

            pred, hidden = self.single_step_forward(ins, rec_rel, send_rel,
                                                    edge_type, hidden)
            pred_all.append(pred)

        preds = torch.stack(pred_all, dim=1)

        return preds.transpose(1, 2).contiguous()
