"""
Will contain the encoder and decoder models 
"""
import torch
import torch.nn as nn
from .utils.models import MLP
from pytorch_lightning import LightningModule
from data.transform import DataTransformer
from feature_extraction.feature_extractor import FeatureExtractor

class MessagePassingLayers():
    def __init__(self):
        super(MessagePassingLayers, self).__init__()
    
    def pairwise_op(self, node_emb, rec_rel, send_rel, pairwise_op, emd_fn_rank, batch_size, n_nodes, next_emb_fn_type):
        """
        It forms edge features from node embeddings using pairwise operations.

        Parameters
        ----------
        node_emb : torch.Tensor
            Shape depends on rank and prev layer type:
            - If rank == 1: (batch_size, n_nodes, n_components, n_dims)
            - If rank > 1:
                - If prev_emb_fn_type == 'mlp': (batch_size, n_nodes, n_hid_out)
                - If prev_emb_fn_type == 'cnn': (batch_size * n_nodes, n_chn_out)

        rec_rel : torch.Tensor, shape (batch_size, n_edges, n_nodes)
            Receiver matrix, used to get node embeddings on the receiving end of edges.

        send_rel : torch.Tensor, shape (batch_size, n_edges, n_nodes)
            Sender matrix, used to get node embeddings on the sending end of edges.

        pairwise_op : str
            Type of pairwise operation to perform on node embeddings.

        emd_fn_rank : int
            Rank of the layer in the pipeline, used to determine the shape of the output.

        batch_size : int
            Number of samples in the batch.

        n_nodes : int
            Number of nodes per sample.

        next_emb_fn_type : str
            Type of the next embedding function in the pipeline, used to determine the shape of the output.

        Returns
        -------
        edge_feature : torch.Tensor
            Shape depends on next_emb_fn_type:
            - If next_emb_fn_type = 'mlp': (batch_size, num_edges, num_features)
            - If next_emb_fn_type = 'cnn': (batch_size * num_edges, dim_size, num_features)
        """
        # reshape input (for both 1st time and subsequent times (cnn or mlp output))
        node_emb = node_emb.view(batch_size, n_nodes, -1)

        receiver_emds = torch.bmm(rec_rel, node_emb)
        sender_emds = torch.bmm(send_rel, node_emb)

        # optimize receiver and sender emd shape for next edge emb type
        receiver_emds, sender_emds = self.optimize_shape_for_pairwise_op(
            receiver_emds, sender_emds, next_emb_fn_type, batch_size, emd_fn_rank)
            
        if pairwise_op == 'sum':
            edge_feature = receiver_emds + sender_emds

        elif pairwise_op == 'concat':
            # change dim to concat depending on next emb fn type
            if next_emb_fn_type == 'mlp':
                dim = 2
            elif next_emb_fn_type == 'cnn':
                dim = 1
            edge_feature = torch.cat((receiver_emds, sender_emds), dim=dim)

        elif pairwise_op == 'mean':
            edge_feature = (receiver_emds + sender_emds) / 2

        return edge_feature
    
    def aggregate(self, edge_emb, rel_rec, agg_type, layer_type):
        """
        Aggregates edge embeddings or messages to form main message for each node
        
        Parameters
        ----------
        edge_emb : torch.Tensor
            Shape depends on prev edge_emd_fn type:
            - If prev edge_emd_fn_type == 'mlp': (batch_size, n_edges, n_features)
            - If prev edge_emd_fn_type == 'cnn': (batch_size * n_edges, n_chn_out)

        rel_rec : torch.Tensor, shape (n_edges, n_nodes)
            Receiver matrix, used to get node features using the reciver type edges

        agg_type : str
            Type of aggregation to perform on edge embeddings/message

        
        Returns
        -------
        node_feature : torch.Tensor, shape (batch_size, n_nodes, n_features)

        """
        if agg_type == 'sum':
            node_feature = torch.bmm(rel_rec.transpose(1, 2), edge_emb)
        elif agg_type == 'mean':
            n_edges_per_node = rel_rec.sum(dim=1, keepdim=True).transpose(1, 2) # shape (batch_size, n_nodes, 1)
            edge_feature_sum = torch.bmm(rel_rec.transpose(1, 2), edge_emb) # shape (batch_size, n_nodes, n_features)   
            node_feature = edge_feature_sum / n_edges_per_node
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

    # -------- Helper functions -----------------

    def optimize_shape_for_node_emd(self, x, rank, emd_fn_type, batch_size, n_nodes):
        """
        Reshape the input for node embedding function based on its type and rank in pipeline.

        Parameters
        ----------
        x : torch.Tensor
            Shape depends on rank and node_emd_fn_type:
            - If rank == 1: (batch_size, n_nodes, n_components, n_dims)
            - If rank > 1:
                - If prev_layer is 'aggregate': (batch_size, n_nodes, n_features)
                - If prev_layer is 'node_emd_fn':
                    - If prev_node_emd_fn_type == 'mlp': (batch_size, n_nodes, n_hid_out)
                    - If prev_node_emd_fn_type == 'cnn': (batch_size * n_nodes, n_chn_out)
        
        rank : int
            Rank of the node embedding function in the pipeline.
       
        emd_fn_type : str
            Type of the node embedding function
        
        batch_size : int
        
        n_nodes : int
            Number of nodes per sample

        Returns
        -------
        x : torch.Tensor
            Reshaped tensor ready for input in node embedding function. Shapes are as follows:
            - If rank == 1:
                - If node_emd_fn_type == 'mlp': (batch_size, n_nodes, n_components * n_dims)
                - If node_emd_fn_type == 'cnn': (batch_size * n_nodes, n_dims, n_components
            - If rank > 1:
                - If node_emd_fn_type == 'mlp': (batch_size, n_nodes, n_features)
                - If node_emd_fn_type == 'cnn': (batch_size * n_nodes, 1, n_features)

        """
        if rank == 1: # this rank differntiation is only requried b/c of CNN's dim
            if emd_fn_type == 'mlp':
                x = x.view(batch_size, n_nodes, self.n_components * self.n_dims)
            elif emd_fn_type == 'cnn':
                x = x.view(batch_size * n_nodes, self.n_dims, self.n_components)
        elif rank == 2:
            if emd_fn_type == 'mlp':
                x = x.view(batch_size, n_nodes, x.size(-1))
            elif emd_fn_type == 'cnn':
                x = x.view(batch_size * n_nodes, 1, x.size(-1))

        return x
    
    def optimize_shape_for_edge_emd(self, x, emd_fn_type, batch_size, n_edges):
        """
        Reshape the input for edge embedding function. 
        This function is only used when prev layer is not pairwise operation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor. Shape depends on prev_edge_emd_fn_type:
            - If prev_edge_emd_fn_type == 'mlp': (batch_size, n_edges, n_features)
            - If prev_edge_emd_fn_type == 'cnn': (batch_size * n_edges, n_chn_out)
        
        emd_fn_type : str
            Type of the edge embedding function

        batch_size : int

        n_edges : int
            Number of edges per sample

        Returns
        -------
        x : torch.Tensor
            Reshaped tensor ready for input in edge embedding function. Shapes are as follows:
            - If edge_emd_fn_type == 'mlp': (batch_size, n_edges, n_features)
            - If edge_emd_fn_type == 'cnn': (batch_size * n_edges, 1, n_features)
        """

        if emd_fn_type == 'mlp':
            # reshape to (batch_size, n_edges, n_feat)
            x = x.view(batch_size, n_edges, -1)
        elif emd_fn_type == 'cnn':
            # reshape to (batch_size * n_edges, 1, n_feat)
            x = x.view(batch_size * n_edges, 1, -1)

        return x
    
    def optimize_shape_for_pairwise_op(self, receiver_emds, sender_emds, next_emb_fn_type, batch_size, emd_fn_rank):
        """
        Reshape the receiver and sender embeddings if edge emd after pairwise in cnn

        Parameters
        ----------
        receiver_emds : torch.Tensor, shape (batch_size, n_edges, n_features)
            Node embeddings on reciver end on each edge

        sender_emds : torch.Tensor, shape (batch_size, n_edges, n_features)
            Node embeddings on sender end on each edge

        next_emb_fn_type : str
            Type of the next embedding function in the pipeline, used to determine the shape of the output

        batch_size : int

        emd_fn_rank : int
            Rank of the pairwise op layer in the pipeline, used to determine the shape of the output

        Returns
        -------
        receiver_emds : torch.Tensor
        sender_emds : torch.Tensor
            Rehsaped tensors. Shapes are as follows:
            - If next_edge_emb_fn_type = 'mlp': (batch_size, n_edges, n_features)
            - If next_edge_emb_fn_type = 'cnn': (batch_size * n_edges, dim_size, n_features)
        """

        if next_emb_fn_type == 'cnn':
            # check dim for cnn for given rank of pariwise operation
            if emd_fn_rank == 1:  # if this is the first pairwise operation
                dim_size = self.n_dims
            else:
                dim_size = 1
            # reshape receiver and sender emb for CNN input
            # (batch_size * n_edges, dim_size, n_feat (n_timestep if rank = 1, else n_hid))
            receiver_emds = receiver_emds.view(batch_size * receiver_emds.size(1), dim_size, -1)
            sender_emds = sender_emds.view(batch_size * sender_emds.size(1), dim_size, -1)

            return receiver_emds, sender_emds
        
        elif next_emb_fn_type == 'mlp':
            # no reshape requried if mlp
            return receiver_emds, sender_emds



class Encoder(LightningModule, MessagePassingLayers):
    def __init__(self, n_components, n_dims, 
                 pipeline, n_edge_types, is_residual_connection, 
                 edge_emd_configs, node_emd_configs, drop_out_prob, batch_norm, attention_output_size):
        
        super(Encoder, self).__init__()
        MessagePassingLayers.__init__(self)

        # input parameters
        self.n_components = n_components
        self.n_dims = n_dims

        # pipeline parameters
        self.pipeline = pipeline
        self.n_edge_types = n_edge_types
        self.is_residual_connection = is_residual_connection

        # embedding configurations
        self.edge_emb_configs = edge_emd_configs
        self.node_emb_configs = node_emd_configs
        self.dropout_prob = drop_out_prob
        self.batch_norm = batch_norm

        # attention parameters
        self.attention_output_size = attention_output_size

        self.init_embedding_functions()
        self.init_attention_layers()

        # define output layer
        final_emd_type = self.pipeline[-1][1]
        final_input_size = self.edge_emb_configs[final_emd_type][-1][0]
        self.output_layer = nn.Linear(final_input_size, self.n_edge_types)


    def init_attention_layers(self):
        self.attention_layer_dict = nn.ModuleDict()

        for layer_num, layer in enumerate(self.pipeline):
            layer_type = layer[0].split('/')[1].split('.')[0]

            if layer_type == 'aggregate':
                if layer[1] == 'weighted_sum':
                    prev_edge_emb_type = self.pipeline[layer_num-1][1]
                    input_size = self.edge_emb_configs[prev_edge_emb_type][-1][0]
                    self.attention_layer_dict[layer[0]] = nn.Linear(input_size, self.attention_output_size)
    
    def init_embedding_functions(self):
        self.emb_fn_dict = nn.ModuleDict()

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
                        edge_main_input_size = self.n_components * self.n_dims
                    elif layer[1] == 'cnn':
                        edge_main_input_size = self.n_dims
                else:
                    if layer[1] == 'mlp':
                        # Check if edge embedding is repeated more than once
                        if edge_emd_fn_rank < 2:
                            prev_node_emd_fn_type = self.pipeline[layer_num-2][1]
                            edge_main_input_size = self.node_emb_configs[prev_node_emd_fn_type][-1][0]
                        else:
                            prev_edge_emd_fn_type = self.pipeline[layer_num-1][1]
                            edge_main_input_size = self.edge_emb_configs[prev_edge_emd_fn_type][-1][0]

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
                    if self.is_residual_connection:
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
                    self.emb_fn_dict[layer[0].replace(".", "")] = MLP(edge_emd_input_size, 
                                                    self.edge_emb_configs['mlp'],
                                                    do_prob=self.dropout_prob['mlp'],
                                                    is_batch_norm=self.batch_norm['mlp'],
                                                    )
                elif layer[1] == 'cnn':
                    self.emb_fn_dict[layer[0].replace(".", "")] = 'CNN' # placeholder

            # Node embedding functions

            elif layer_type == 'node_emd':
                emd_fn_rank += 1
                node_emd_fn_rank += 1
                edge_emd_fn_rank = 0
                
                # Check if it is the first node embedding function
                if emd_fn_rank == 1:
                    if layer[1] == 'mlp':
                        node_emd_input_size = self.n_components * self.n_dims
                    elif layer[1] == 'cnn':
                        node_emd_input_size = self.n_dims

                else:

                    prev_layer_type = self.pipeline[layer_num-1][0].split('/')[1].split('.')[0]
                    
                    if layer[1] == 'mlp':
                        # Check if node embedding is repeated more than once
                        if node_emd_fn_rank < 2:
                            prev_edge_emd_fn_type = self.pipeline[layer_num-2][1]
                            node_main_input_size = self.edge_emb_configs[prev_edge_emd_fn_type][-1][0]
                        else:
                            prev_node_emd_fn_type = self.pipeline[layer_num-1][1]
                            node_main_input_size = self.node_emb_configs[prev_node_emd_fn_type][-1][0]
                            
                    elif layer[1] == 'cnn':
                        node_main_input_size = 1 # input dim to CNN type node emb after emb fn rank > 1

                    # Check input sizes
                    if prev_layer_type == 'agg' or self.pipeline[layer_num-1][1] != 'concat':
                        node_emd_input_size = node_main_input_size       
                    else:
                        node_emd_input_size = 2* node_main_input_size

                # Initialize node embedding fn
                if layer[1] == 'mlp':
                    self.emb_fn_dict[layer[0].replace(".", "")] = MLP(node_emd_input_size, 
                                                    self.node_emb_configs['mlp'],
                                                    do_prob=self.dropout_prob['mlp'],
                                                    is_batch_norm=self.batch_norm['mlp'],
                                                    )
                elif layer[1] == 'cnn':
                    self.emb_fn_dict[layer[0].replace(".", "")] = 'CNN'

    def set_input_graph(self, rec_rel, send_rel):
        """
        Set the relationship matrices defining the input graph structure.
        
        Parameters
        ----------
        rec_rel: torch.Tensor, shape (batch_size, n_edges, n_nodes)
            Reciever matrix
            
        send_rel: torch.Tensor, shape (batch_size, n_edges, n_nodes)
            Sender matrix
        """
        self.rec_rel = rec_rel
        self.send_rel = send_rel

    def set_run_params(self, data_stats, domain='time', norm_type=None, fex_configs=[]):
        """
        Set the run parameters for the encoder.
        """
        self.fex_configs = fex_configs

        self.transform = DataTransformer(domain=domain, norm_type=norm_type, data_stats=data_stats)
        self.feature_extractor = FeatureExtractor(fex_configs=fex_configs)

    def process_input_data(self, data):
        """
        Transform the data
            - domain change
            - normalization
        Feature extraction

        Parameters
        ----------
        data: torch.Tensor, shape (batch_size, n_nodes, n_timesteps, n_dims)
            Input node data

        Returns
        -------
        data: torch.Tensor, shape (batch_size, n_nodes, n_components, n_dims)
        """
        # transform data
        data = self.transform(data)

        # extract features from data if fex_configs are provided
        if self.fex_configs:
            data = self.feature_extractor(data)

        return data

    def forward(self, data):
        """
        Forward pass through the encoder pipeline to compute edge logits.

        Note
        ----
        Ensure to run `set_input_graph()` before running this method.

        Parameters
        ----------
        data: torch.Tensor, shape (batch_size, n_nodes, n_timesteps, n_dims)
            Input node data
            
        Returns
        -------
        edge_matrix: torch.Tensor, shape (batch_size, n_edges, n_edge_types) 

        """
        emd_fn_rank = 0   # used to find the first embedding function (for node or edge)
        batch_size = data.size(0)
        n_nodes = data.size(1)
        n_edges = self.rec_rel.size(1)

        # process the input data
        x = self.process_input_data(data)

        # extract features from data
        # [TODO] Add the feature extraction logic here if needed

        for layer_num, layer in enumerate(self.pipeline):
            layer_type = layer[0].split('/')[1].split('.')[0]

            # node embedding
            if layer_type == 'node_emd':
                emd_fn_rank += 1
                emb_fn = self.emb_fn_dict[layer[0].replace(".", "")]

                x = self.optimize_shape_for_node_emd(x, emd_fn_rank, layer[1], batch_size, n_nodes)  
                x = emb_fn(x)
                    
            # pairwise operation    
            elif layer_type == 'pairwise_op':
                emd_fn_rank += 1
                next_emd_fn_type = self.pipeline[layer_num+1][1]
                x = self.pairwise_op(x, self.rec_rel, self.send_rel, layer[1], emd_fn_rank, batch_size, n_nodes, next_emd_fn_type)

            # aggregation
            elif layer_type == 'aggregate':
                x = self.aggregate(x, self.rec_rel, layer[1], layer[0])

            # combine
            elif layer_type == 'combine':
                pass

            # edge embedding
            elif layer_type == 'edge_emd':
                emb_fn = self.emb_fn_dict[layer[0].replace(".", "")]
                x = self.optimize_shape_for_edge_emd(x, layer[1], batch_size, n_edges)

                if layer_num < len(self.pipeline) - 1:
                    x = emb_fn(x)

                    if layer[0].split('.')[-1] == '@':
                        x_skip = x             
                else:
                    # skip connection for last edge embedding layer
                    if self.is_residual_connection:
                        x = torch.cat((x, x_skip), dim=-1)
                    x = emb_fn(x)

        edge_matrix = self.output_layer(x)
        return edge_matrix
                

        

