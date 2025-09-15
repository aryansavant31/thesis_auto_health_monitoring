import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

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
        self.n_nodes = next(iter(data_loader))[0].shape[1]
        y_edges = next(iter(data_loader))[1][0]

        print(f"\nLoading Relation Matrices...")

        self.rec_rel, self.send_rel = self.make_relation_matrix(y_edges, self.n_nodes)

        print(f"\nRelation Matrices loaded successfully.")

        print(self.get_relation_matrices_summary())
        print("\n" + 75*'-')

        return self.rec_rel, self.send_rel

    def get_relation_matrices_summary(self):
        """
        Returns a summary of the relation matrices including their shapes and contents.
        """
        text = "\n## Relation Matrices Summary \n"

        # Ensure full printing
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)

        text = "\n## Relation Matrices Summary \n"

        # Adjacency matrix 
        adj_mat = torch.matmul(self.send_rel.t(), self.rec_rel)

        node_labels = [f"n{i+1}" for i in range(self.n_nodes)]
        adj_df = pd.DataFrame(adj_mat.cpu().numpy(), index=node_labels, columns=node_labels)
        
        text += f"\n**Adjacency matrix for input** => shape: {adj_df.shape}\n"
        text += adj_df.to_string() + "\n\n"

        # rec_rel and send_rel edge labels
        n_edges = self.rec_rel.shape[0]
        edge_labels = []
        for sender in range(self.n_nodes):
            for receiver in range(self.n_nodes):
                if sender != receiver:
                    edge_labels.append(f"e{sender+1}{receiver+1}")

        # Receiver relation matrix 
        rec_df = pd.DataFrame(self.rec_rel.cpu().numpy(), index=edge_labels, columns=node_labels)
        text += f"\n**Receiver relation matrix** => shape: {rec_df.shape}\n"
        text += rec_df.to_string() + "\n\n"

        # Sender relation matrix 
        send_df = pd.DataFrame(self.send_rel.cpu().numpy(), index=edge_labels, columns=node_labels)
        text += f"\n**Sender relation matrix:** => shape: {send_df.shape}\n"
        text += send_df.to_string() + "\n"

        # reset pandas options to default
        pd.reset_option('display.max_rows')
        pd.reset_option('display.max_columns')

        return text

        # adj_mat = torch.matmul(self.send_rel.t(), self.rec_rel)
        # text += f"\n**Adjacency matrix for input**\n"
        # text += np.array2string(adj_mat.cpu().numpy(), max_line_width=120, precision=2, suppress_small=True, separator=', ')
        # text += f", shape: {adj_mat.shape}\n"

        # text += f"\n**Reciever relation matrix:**\n"
        # text += np.array2string(self.rec_rel.cpu().numpy(), max_line_width=120, precision=2, suppress_small=True, separator=', ')
        # text += f", shape: {self.rec_rel.shape}\n"

        # text += f"\n**Sender relation matrix**\n"
        # text += np.array2string(self.send_rel.cpu().numpy(), max_line_width=120, precision=2, suppress_small=True, separator=', ')
        # text += f", shape: {self.send_rel.shape}\n"


        # return text

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
                            # If y_edges[edge_idx, 0] == 1, set to 0 (no edge); if 0, keep as 1 (edge exists)
                            if y_edges[edge_idx, 0] == 1:
                                rec_rel[edge_idx, receiver] = 0
                                send_rel[edge_idx, sender] = 0
                        except Exception as e:
                            print(f"Error in expert knowledge infusion: {e}")
                            raise

                    edge_idx += 1

        return rec_rel, send_rel
        

class UndirectedGraph:
    """
    - similarity matrix generation (the formuals will be in utils folder)
    - edge matrix construction
    """
    pass