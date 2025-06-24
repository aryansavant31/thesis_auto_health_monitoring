import torch
import torch.nn as nn
from common.models import MLP

class Node2Edge(nn.Modules):
    def __init__(self, par):
        super(Node2Edge, self).__init__()
        self.par = par
        self.edge_emb_fn = MLP()

    def pairwise_op(self, node_emb, rec_rel, send_rel):
        """
        Returns
        -------
        edge_feature : torch.Tensor
            Shape (batch_size, num_edges, num_features)
        """
        receiver_emd = torch.matmul(rec_rel, node_emb)
        sender_emd = torch.matmul(send_rel, node_emb)

        if self.par.pairwise_op == 'sum':
            edge_feature = receiver_emd + sender_emd
        elif self.par.pairwise_op == 'concat':
            edge_feature = torch.cat((receiver_emd, sender_emd), dim=-1)
        elif self.par.pairwise_op == 'mean':
            edge_feature = (receiver_emd + sender_emd) / 2

        return edge_feature
    
    def get_embedding(self, feature):
        input_size = feature.shape[-1]



    def node2edge(self):
        pass
        
    def edge2node():
        pass
