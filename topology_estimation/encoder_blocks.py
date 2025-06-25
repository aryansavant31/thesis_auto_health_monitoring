"""
Will contain the encoder and decoder models 
"""
import torch
import torch.nn as nn
from common.models import MLP


class Encoder(nn.Module):
    def __init__(self, par, n_timesteps, n_dims):
        super(Encoder, self).__init__()
        self.par = par
        self.pipeline= self.par.encoder_pipeline
        self.n_timesteps = n_timesteps
        self.n_dims = n_dims

        self.init_embedding_functions()
        self.init_attention_layers()

        # define output layer
        final_emd_type = self.pipeline[-1][1]
        final_input_size = self.par.edge_emb_configs[final_emd_type][-1][0]
        self.output_layer = nn.Linear(final_input_size, self.par.n_edge_types)


    def init_attention_layers(self):
        self.attention_layer_dict = {}

        for layer_num, layer in enumerate(self.pipeline):
            layer_type = layer[0].split('/')[1].split('.')[0]

            if layer_type == 'aggregate':
                if layer[1] == 'weighted_sum':
                    prev_edge_emb_type = self.pipeline[layer_num-1][1]
                    input_size = self.par.edge_emb_configs[prev_edge_emb_type][-1][0]
                    self.attention_layer_dict[layer[0]] = nn.Linear(input_size, self.par.attention_output_size)
    
    def init_embedding_functions(self):
        self.emb_fn_dict = {}

        emd_fn_rank = 0
        node_emd_fn_rank = 0
        edge_emd_fn_rank = 0

        # Loop through pipeline to initialize the embedding functions

        for layer_num, layer in enumerate(self.pipeline):
            
            layer_type = layer[0].split('/')[1].split('.')[0]


            # Edge embedding functions

            if layer_type == 'edge_emd':
                emd_fn_rank += 1
                edge_emd_fn_rank += 1
                node_emd_fn_rank = 0    # reset node emd fn rank

                # Check if it is the first edge embedding function
                if emd_fn_rank == 1:
                    if layer[1] == 'mlp':
                        edge_main_input_size = self.n_timesteps * self.n_dims
                    elif layer[1] == 'cnn':
                        edge_main_input_size = self.n_dims
                else:
                    if layer[1] == 'mlp':
                        # Check if edge embedding is repeated more than once
                        if edge_emd_fn_rank < 2:
                            edge_main_input_size = self.par.node_emb_configs[layer[1]][-1][0]
                        else:
                            edge_main_input_size = self.par.edge_emb_configs[layer[1]][-1][0]

                    elif layer[1] == 'cnn':
                        edge_main_input_size = 1  # input dim to CNN type edge_emb after emb fn rank > 1

                # Check input sizes

                # input size for intermediate edge embedding fn
                if layer_num < len(self.pipeline)-1:
                    if self.pipeline[layer_num-1][1] == 'concat':
                        edge_emd_input_size = 2 * edge_main_input_size
                    
                    else:
                        edge_emd_input_size = edge_main_input_size

                # input size for last layer (which is always edge embedding)
                else:
                    if self.par.residual_connection:
                        if self.pipeline[layer_num-1][1] == 'concat':
                            edge_emd_input_size = 3 * edge_main_input_size
                        else:
                            edge_emd_input_size = 2 * edge_main_input_size
                    else:
                        if self.pipeline[layer_num-1][1] == 'concat':
                            edge_emd_input_size = 2 * edge_main_input_size
                        else:
                            edge_emd_input_size = edge_main_input_size
                    

                # Initialize edge embedding fn
                if layer[1] == 'mlp':
                    self.emb_fn_dict[layer[0]] = MLP(edge_emd_input_size, 
                                                    self.par.edge_emb_configs['mlp'],
                                                    do_prob=self.par.dropout_prob_mlp,
                                                    is_batch_norm=self.par.batch_norm_mlp,
                                                    is_gnn=True
                                                    )
                elif layer[1] == 'cnn':
                    self.emb_fn_dict[layer[0]] = 'CNN' # placeholder

            # Node embedding functions

            elif layer_type == 'node_emd':
                emd_fn_rank += 1
                node_emd_fn_rank += 1
                edge_emd_fn_rank = 0
                
                # Check if it is the first node embedding function
                if emd_fn_rank == 1:
                    if layer[1] == 'mlp':
                        node_emd_input_size = self.n_timesteps * self.n_dims
                    elif layer[1] == 'cnn':
                        node_emd_input_size = self.n_dims

                else:

                    prev_layer_type = self.pipeline[layer_num-1][0].split('/')[1].split('.')[0]
                    
                    if layer[1] == 'mlp':
                        # Check if edge embedding is repeated more than once
                        if node_emd_fn_rank < 2:
                            node_main_input_size = self.par.edge_emb_configs[layer[1]][-1][0]
                        else:
                            node_main_input_size = self.par.node_emb_configs[layer[1]][-1][0]
                    elif layer[1] == 'cnn':
                        node_main_input_size = 1 # input dim to CNN type node emb after emb fn rank > 1

                    # Check input sizes
                    if prev_layer_type == 'agg' or self.pipeline[layer_num-1][1] != 'concat':
                        node_emd_input_size = node_main_input_size       
                    else:
                        node_emd_input_size = 2* node_main_input_size

                # Initialize node embedding fn
                if layer[1] == 'mlp':
                    self.emb_fn_dict[layer[0]] = MLP(node_emd_input_size, 
                                                    self.par.node_emb_configs['mlp'],
                                                    do_prob=self.par.dropout_prob_mlp,
                                                    is_batch_norm=self.par.batch_norm_mlp,
                                                    is_gnn=True
                                                    )
                elif layer[1] == 'cnn':
                    self.emb_fn_dict[layer[0]] = 'CNN'
    

    # -------- Message passing funtions -----------------

    def pairwise_op(self, node_emb, rec_rel, send_rel, pairwise_op):
        """
        Returns
        -------
        edge_feature : torch.Tensor
            Shape (batch_size, num_edges, num_features)
        """
        receiver_emd = torch.matmul(rec_rel, node_emb)
        sender_emd = torch.matmul(send_rel, node_emb)

        if pairwise_op == 'sum':
            edge_feature = receiver_emd + sender_emd
        elif pairwise_op == 'concat':
            edge_feature = torch.cat((receiver_emd, sender_emd), dim=-1)
        elif pairwise_op == 'mean':
            edge_feature = (receiver_emd + sender_emd) / 2

        return edge_feature
    
    def aggregate(self, edge_emb, rel_rec, agg_type, layer_type):
        if agg_type == 'sum':
            node_feature = torch.matmul(rel_rec.t(), edge_emb)
        elif agg_type == 'mean':
            node_feature = torch.matmul(rel_rec.t(), edge_emb) / rel_rec.sum(dim=0, keepdim=True)
        # elif agg_type == 'weighted_sum':  

        #     alpha = self.get_attention_weights(edge_emb)
        #     node_feature = torch.matmul(rel_rec.t(), edge_emb * alpha)

        return node_feature
    
    # def get_attention_weights(self, edge_emb, layer_type):
    #     """
    #     Returns attention weights for the edges.
    #     """
    #     if layer_type not in self.attention_layer_dict:
    #         raise ValueError(f"Attention layer {layer_type} not found in attention layer dictionary.")

    #     attention_layer = self.attention_layer_dict[layer_type]
    #     alpha = attention_layer(edge_emb)
    #     alpha = torch.softmax(alpha, dim=1)

    def combine(self):
        pass

    def reshape_for_node_emd(self, x, rank, emd_fn_type, batch_size, n_nodes):
        if rank == 1:
            if emd_fn_type == 'mlp':
                x = x.view(batch_size, n_nodes, self.n_timesteps * self.n_dims)
            elif emd_fn_type == 'cnn':
                x = x.view(batch_size * n_nodes, self.n_dims, self.n_timesteps)
        elif rank == 2:
            if emd_fn_type == 'mlp':
                x = x.view(batch_size, n_nodes, x.size(-1))
            elif emd_fn_type == 'cnn':
                x = x.view(batch_size * n_nodes, 1, x.size(-1))

        return x

    def forward(self, x, rec_rel, send_rel):
        """
        Parameters
        ----------
        x: torch.Tensor, shape (batch_size, n_nodes, n_timesteps, n_dims)
            Input node data
            
        rec_rel: torch.Tensor, shape (n_edges, n_nodes)
            Reciever matrix
            
        send_rel: torch.Tensor, shape (n_edges, n_nodes)
            Sender matrix

        """
        # change input shape to (batch_size, n_nodes, n_timesteps * n_dims)
        # x = input.view(input.size(0), input.size(1), -1)
        emd_fn_rank = 0
        batch_size = x.size(0)
        n_nodes = x.size(1)

        for layer_num, layer in enumerate(self.pipeline):
            layer_type = layer[0].split('/')[1].split('.')[0]

            # node embedding
            if layer_type == 'node_emd':
                emd_fn_rank += 1
                emb_fn = self.emb_fn_dict[layer[0]]

                x = self.reshape_for_node_emd(x, emd_fn_rank, layer[1], batch_size, n_nodes)  
                x = emb_fn(x)
                    
            # pairwise operation    
            elif layer_type == 'pairwise_op':
                x = self.pairwise_op(x, rec_rel, send_rel, layer[1])

            # aggregation
            elif layer_type == 'aggregate':
                x = self.aggregate(x, rec_rel, layer[1], layer[0])

            # combine
            elif layer_type == 'combine':
                pass

            # edge embedding
            elif layer_type == 'edge_emd':
                emd_fn_rank += 1
                emb_fn = self.emb_fn_dict[layer[0]]

                if layer_num < len(self.pipeline) - 1:
                    x = emb_fn(x)

                    if layer[0].split('.')[-1] == '@':
                        x_skip = x

                else:
                    if self.par.residual_connection:
                        x = torch.cat((x, x_skip), dim=-1)
                    x = emb_fn(x)

        return self.output_layer(x)

                

        

