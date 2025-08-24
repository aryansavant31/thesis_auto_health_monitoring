import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# class FullyConnectedGraph:
#     def __init__(self, n_nodes, batch_size):
#         """
#         Initialize a fully connected graph with no self loops

#         Parameters
#         ----------
#         n_nodes : int
#             Number of nodes in the graph.
#         """
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.n_nodes = n_nodes
#         self.batch_size = batch_size
    
#     def get_relation_matrices(self):
#         """
#         Generate relation matrices for the fully connected graph.

#         Returns
#         -------
#         rec_rel : torch.Tensor, shape (batch_size, n_edges, n_nodes)
#             Receiver relation matrix that indicate which edges are on reciever end of nodes.
#         send_rel : torch.Tensor, shape (batch_size, n_edges, n_nodes)   
#             Sender relation matrix that indicate which edges are senders of nodes.
#         """
#         # create the relation matrices (edge order: e12, e13, e14, ..., e21, e23, e24, ...)
#         n_edges = self.n_nodes * (self.n_nodes - 1)  # excluding self-loops
#         rec_rel = np.zeros((n_edges, self.n_nodes))
#         send_rel = np.zeros((rec_rel.shape))

#         edge_idx = 0
#         for sender in range(self.n_nodes):
#             for receiver in range(self.n_nodes):
#                 if sender != receiver:  # Exclude self-loops
#                     rec_rel[edge_idx, receiver] = 1  # Receiver node
#                     send_rel[edge_idx, sender] = 1  # Sender node
#                     edge_idx += 1

#         # make batch_size number of relation matrices
#         rec_rel = np.tile(rec_rel, (self.batch_size, 1, 1)).reshape(self.batch_size, n_edges, self.n_nodes)
#         send_rel = np.tile(send_rel, (self.batch_size, 1, 1)).reshape(self.batch_size, n_edges, self.n_nodes)

#         rec_rel = torch.FloatTensor(rec_rel).to(self.device)
#         send_rel = torch.FloatTensor(send_rel).to(self.device)

#         return rec_rel, send_rel


class RelationMatrixMaker:
    """
    It will contains methods to make sparsified graph
    The methods would include:
    - expert knowledge infusion
    - undirected graph learning (using the MakeUndirectedGraph class)
    """
    def __init__(self, spf_config):
        self.spf_config = spf_config

    def get_relation_matrix(self, data_loader: DataLoader):
        n_nodes = next(iter(data_loader))[0].shape[1]
        y_edges = next(iter(data_loader))[1][0]

        rec_rel, send_rel = self.make_relation_matrix(y_edges, n_nodes)

        print(f"\nLoading Relation Matrices...")
        print(f"\nReciever relation matrix:")
        print(rec_rel, f"\nshape: {rec_rel.shape}")

        print(f"\nSender relation matrix:")
        print(send_rel, f"\nshape: {send_rel.shape}")

        print("\n" + 75*'-')

        return rec_rel, send_rel

    def make_relation_matrix(self, y_edges, n_nodes):
        """
        Generate relation matrices for the sparsified graph.

        Parameters
        ----------
        y_edges : torch.Tensor, shape (batch_size, n_edges)
            Edge label indicating which edges are present in the graph.
        n_nodes : int
            Number of nodes in the graph.

        Returns
        -------
        rec_rel : torch.Tensor, shape (n_edges, n_nodes)
            Receiver relation matrix that indicate which edges are on reciever end of nodes.
        send_rel : torch.Tensor, shape (n_edges, n_nodes)   
            Sender relation matrix that indicate which edges are senders of nodes.
        """
        n_edges = n_nodes * (n_nodes - 1)  # excluding self-loops

        rec_rel = torch.zeros((n_edges, n_nodes), device=y_edges.device)
        send_rel = torch.zeros((n_edges, n_nodes), device=y_edges.device)

        # if self.spf_config['type'] != 'no_spf':
        #     spf_edges = None # Placeholder for future implementation of undirected graph learning

        edge_idx = 0
        for sender in range(n_nodes):
            for receiver in range(n_nodes):
                if sender != receiver:
                    # # add sparsified graph edges if enabled
                    # if self.spf_config['type'] != 'no_spf':
                    #     rec_rel[edge_idx, receiver] = spf_edges[edge_idx] 
                    #     send_rel[edge_idx, sender] = spf_edges[edge_idx]
                    
                    # make fully connected graph
                    rec_rel[edge_idx, receiver] = 1
                    send_rel[edge_idx, sender] = 1
                    
                    # infuse expert knowledge if enabled
                    if self.spf_config['is_expert']:
                        try:
                            mask = y_edges[edge_idx] != -1
                            rec_rel[edge_idx, receiver] = torch.where(mask, y_edges[edge_idx], rec_rel[edge_idx, receiver])
                            send_rel[edge_idx, sender] = torch.where(mask, y_edges[edge_idx], send_rel[edge_idx, sender])

                        except ValueError:
                            print(f"ValueError: 'is_expert' is True but y_edges is found to be null or has incorrect shape. \
                                   Please check the input data.")
                            raise

                    edge_idx += 1

        return rec_rel, send_rel
        

class UndirectedGraph:
    """
    - similarity matrix generation (the formuals will be in utils folder)
    - edge matrix construction
    """
    pass