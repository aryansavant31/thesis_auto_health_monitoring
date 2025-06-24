"""
Will contain the encoder and decoder models 
"""
import torch
import torch.nn as nn
from common.models import MLP


class Encoder(nn.Module):
    def __init__(self, par, input_size):
        super(Encoder, self).__init__()
        self.par = par
        self.pipeline= self.par.encoder_pipeline
        self.input_size = input_size

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

        # Initialize the first node embedding fn
        if self.pipeline[0][1] == 'mlp':
            self.emb_fn_dict[self.pipeline[0][0]] = MLP(self.input_size, 
                                                        self.par.node_end_configs['mlp'],
                                                        self.par.output_size,
                                                        do_prob=self.par.dropout_prob_mlp,
                                                        batch_norm=self.par.batch_norm_mlp
                                                        )
        elif self.pipeline[0][1] == 'cnn':
            self.emb_fn_dict[self.pipeline[0][0]] = 'CNN' # placeholder

        # Loop through pipeline to initialize the embedding functions

        for layer_num, layer in enumerate(self.pipeline):
            if layer_num == 0:
                continue # skip first layer

            layer_type = layer[0].split('/')[1].split('.')[0]

            # Edge embedding functions

            if layer_type == 'edge_emd':
                # Check input sizes

                # input size for intermediate edge embedding fn
                if layer_num < len(self.pipeline)-1:
                    if self.pipeline[layer_num-1][1] == 'concat':
                        edge_emd_input_size = 2 * self.par.node_emb_configs[layer[1]][-1][0]
                    
                    else:
                        edge_emd_input_size = self.par.node_emb_configs[layer[1]][-1][0]

                # input size for last layer (which is always edge embedding)
                else:
                    if self.par.residual_connection:
                        if self.pipeline[layer_num-1][1] == 'concat':
                            edge_emd_input_size = 3 * self.par.edge_emb_configs[layer[1]][-1][0]
                        else:
                            edge_emd_input_size = 2 * self.par.edge_emb_configs[layer[1]][-1][0]
                    else:
                        if self.pipeline[layer_num-1][1] == 'concat':
                            edge_emd_input_size = 2 * self.par.edge_emb_configs[layer[1]][-1][0]
                        else:
                            edge_emd_input_size = self.par.edge_emb_configs[layer[1]][-1][0]
                    

                # Initialize edge embedding fn
                if layer[1] == 'mlp':
                    self.emb_fn_dict[layer[0]] = MLP(edge_emd_input_size, 
                                                    self.par.edge_emb_configs['mlp'],
                                                    do_prob=self.par.dropout_prob_mlp,
                                                    batch_norm=self.par.batch_norm_mlp
                                                    )
                elif layer[1] == 'cnn':
                    self.emb_fn_dict[layer[0]] = 'CNN' # placeholder

            # Node embedding functions

            elif layer_type == 'node_emd':
                prev_layer_type = self.pipeline[layer_num-1][0].split('/')[1].split('.')[0] 

                if prev_layer_type == 'agg' or self.pipeline[layer_num-1][1] != 'concat':
                    node_emd_input_size = self.par.edge_emb_configs[layer[1]][-1][0]       
                else:
                    node_emd_input_size = 2* self.par.edge_emb_configs[layer[1]][-1][0]

                # Initialize node embedding fn
                if layer[1] == 'mlp':
                    self.emb_fn_dict[layer[0]] = MLP(node_emd_input_size, 
                                                    self.par.node_emb_configs['mlp'],
                                                    do_prob=self.par.dropout_prob_mlp,
                                                    batch_norm=self.par.batch_norm_mlp
                                                    )
                elif layer[1] == 'cnn':
                    self.emb_fn_dict[layer[0]] = 'CNN'
    
    # ---- Message passing funtions -------

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

    def forward(self, x, rec_rel, send_rel):
        """
        """

        for layer_num, layer in enumerate(self.pipeline):
            layer_type = layer[0].split('/')[1].split('.')[0]

            # node embedding
            if layer_type == 'node_emd':
                emb_fn = self.emb_fn_dict[layer[0]]
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

                

        

