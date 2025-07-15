import numpy as np
import torch

class FullyConnectedGraph:
    def __init__(self, n_nodes, batch_size):
        """
        Initialize a fully connected graph with no self loops

        Parameters
        ----------
        n_nodes : int
            Number of nodes in the graph.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_nodes = n_nodes
        self.batch_size = batch_size
        self.fc_adj_matrix = np.ones((n_nodes, n_nodes)) - np.eye(n_nodes)

    def encode_onehot(self, labels):
        classes = set(labels)
        classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                        enumerate(classes)}
        labels_onehot = np.array(list(map(classes_dict.get, labels)),
                                dtype=np.int32)
        return labels_onehot
    
    def get_relation_matrices(self):
        """
        Generate relation matrices for the fully connected graph.

        Returns
        -------
        rec_rel : torch.Tensor, shape (batch_size, n_edges, n_nodes)
            Receiver relation matrix that indicate which edges are on reciever end of nodes.
        send_rel : torch.Tensor, shape (batch_size, n_edges, n_nodes)   
            Sender relation matrix that indicate which edges are senders of nodes.
        """
        rec_rel = np.array(self.encode_onehot(np.where(self.fc_adj_matrix)[0]), dtype=np.float32) # shape (n_edges, n_nodes)
        send_rel = np.array(self.encode_onehot(np.where(self.fc_adj_matrix)[1]), dtype=np.float32)

        # make batch_size number of relation matrices
        n_edges = rec_rel.shape[0]
        rec_rel = np.tile(rec_rel, (self.batch_size, 1, 1)).reshape(self.batch_size, n_edges, self.n_nodes)
        send_rel = np.tile(send_rel, (self.batch_size, 1, 1)).reshape(self.batch_size, n_edges, self.n_nodes)

        rec_rel = torch.FloatTensor(rec_rel).to(self.device)
        send_rel = torch.FloatTensor(send_rel).to(self.device)

        return rec_rel, send_rel


class SparisifiedGraph:
    """
    It will contains methods to make sparsified graph
    The methods would include:
    - expert knowledge infusion
    - undirected graph learning (using the MakeUndirectedGraph class)
    """
    def __init__(self, n_nodes):
        pass

    def get_relation_matrix(self):
        """
        Generate relation matrices for the sparsified graph.

        Returns
        -------
        rec_rel : torch.Tensor, shape (n_edges, n_nodes)
            Receiver relation matrix that indicate which edges are on reciever end of nodes.
        send_rel : torch.Tensor, shape (n_edges, n_nodes)   
            Sender relation matrix that indicate which edges are senders of nodes.
        """
        pass

class UndirectedGraph:
    """
    - similarity matrix generation (the formuals will be in utils folder)
    - edge matrix construction
    """
    pass