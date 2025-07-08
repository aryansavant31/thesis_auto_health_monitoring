import torch.nn as nn
import torch

class MLP(nn.Module):
    def __init__(self, input_size, mlp_config, do_prob=0.0, is_batch_norm=False):
        super(MLP, self).__init__()

        current_dim = input_size
        self.model_type = 'MLP' 
        self.layers = nn.ModuleList()
        self.is_batch_norm = is_batch_norm

        activation_map = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU(),
            'tanh': nn.Tanh(),
            'elu': nn.ELU(),
            'softmax': nn.Softmax(dim=1),
            'sigmoid': nn.Sigmoid()
        }

        # make MLP layers
        for layer_num, layer_config in enumerate(mlp_config):
            layer_output_size, activation_type = layer_config
            self.layers.append(nn.Linear(current_dim, layer_output_size))

            # add batch normalization
            if self.is_batch_norm:
                self.layers.append(nn.BatchNorm1d(layer_output_size))

            # add activation functions
            if activation_type in activation_map:
                self.layers.append(activation_map[activation_type])

            # add dropout
            if layer_num < len(mlp_config) - 1: # no dropout on the last layer
                self.layers.append(nn.Dropout(do_prob))

            current_dim = layer_output_size
        
        # output layer
        # self.layers.append(nn.Linear(current_dim, output_size))

    def batch_norm(self, inputs, bn_layer):
        """
        Changes input shape if MLP used for GNNs
        """
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = bn_layer(x)
        return x.view(inputs.size(0), inputs.size(1), -1)

    def check_invalid_layer_indices(self, get_fex):
        """
        Check if the indices in get_fex exceed the MLP layers.
        """
        invalid_indices = []
        for i in get_fex:
            if i > len(self.layers):
                invalid_indices.append(i)
        if invalid_indices:
             print(f"Warning: The layer(s) {invalid_indices} exceeds the number of layers in the MLP ({len(self.layers)}).")

    def forward(self, x, get_fex=None):
        """
        Parameters
        ----------
        x : torch.Tensor, shape (batch_size, n_nodes, n_features).
            Input tensor 

        get_fex : list
            List of layer indices to return features from. If None, no features are returned.
        
        Returns
        -------
        x : torch.Tensor, shape (batch_size, n_nodes, n_output_features).
        fex : list
            List of feature tensors from specified layers, if get_fex is not None.

        """
        if get_fex is not None:
            self.check_invalid_layer_indices(get_fex)

        fex = []
        for layer_num, layer in enumerate(self.layers):
            # check if layer is batchnorm
            if isinstance(layer, nn.BatchNorm1d):
                x = self.batch_norm(x, layer)
            else:    
                x = layer(x)
            if get_fex:
                if layer_num in get_fex:
                    fex.append(x)

        return x

class LSTM(nn.Module):
    def __init__(self, n_dim, n_layers, hidden_size):
        """
        Parameters
        ----------
        n_dim : int
            Dimension of the input features (e.g., number of parameters per sample).
        n_layers: int
            number of LSTMs to stack
        n_hid: int
            output_size per roll from the LSTM

        """
        super(LSTM, self).__init__()
        self.model_type = 'LSTM' 
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(n_dim, hidden_size, n_layers, batch_first=True)
        # self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, h):
        """
        Parameters
        ----------
        x: torch.tensor, shape: (batch_size, n_timesteps, n_nodes, n_dim)
            Input tensor.
        h: torch.tensor, shape: (batch_size, n_nodes, hidden_size)

        Returns
        -------
        output: torch.tensor, shape: (batch_size, n_timesteps, n_nodes, hidden_size)
            Output tensor containing the hidden states (short term memory) from the LSTM unit for each timestep.
        """
        c0 = torch.zeros(self.n_layers, x.size(0), x.size(2), self.hidden_size) # shape = (n_layers, batch_size, output_size (per step))

        # reshape h to (n_layers, batch_size, n_nodes, hidden_size)
        h = h.view(self.n_layers, h.size(0), h.size(1), h.size(2)) 

        output, (hidden, _) = self.lstm(x, (h,c0)) # shape(out) = (batch_size, num_features, 1 <output_size (per step)>)
        
        # output contains the short term memory or hidden states from all the rolls of LSTM unit. 
        # So if I have 10 timesteps as input, output contains 10 hidden state, each pertaining to the individual timestep
        # features extracted
    
        return output
    
    
class RNN(nn.Module):
    def __init__(self, n_dim, n_layers, hidden_size):
        """
        Parameters
        ----------
        n_dim : int
            Dimension of the input features (e.g., number of parameters per sample).
        n_layers: int
            number of RNNs to stack
        hidden_size: int
            output_size per roll from the LSTM

        """
        super(RNN, self).__init__()
        self.model_type = 'RNN' 
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(n_dim, hidden_size, n_layers, batch_first=True)
        # self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, h):
        """
        Parameters
        ----------
        x: torch.tensor, shape: (batch_size, n_timesteps, n_nodes, n_dim)
            Input tensor. 
        h: torch.tensor, shape: (batch_size, n_nodes, hidden_size)

        Returns
        -------
        output: torch.tensor, shape: (batch_size, n_timesteps, n_nodes, hidden_size)
            Output tensor containing the hidden states from the RNN unit for each timestep.
        """
        # reshape h to (n_layers, batch_size, n_nodes, hidden_size)
        h = h.view(self.n_layers, h.size(0), h.size(1), h.size(2))  
   
        output, hidden, = self.rnn(x, h)
    
        return output
    
    
class GRU(nn.Module):
    def __init__(self, n_dim, n_layers, hidden_size):
        """
        Parameters
        ----------
        n_dim : int
            Dimension of the input features (e.g., number of parameters per sample).
        n_layers: int
            number of GRUs to stack
        hidden_size: int
            output_size per roll from the LSTM

        """
        super(GRU, self).__init__()
        self.model_type = 'GRU' 
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.gru = nn.GRU(n_dim, hidden_size, n_layers, batch_first=True)

    def forward(self, x, h):
        """
        Parameters
        ----------
        x: torch.tensor, shape: (batch_size, n_timesteps, n_nodes, n_dim)
            Input tensor.
        h: torch.tensor, shape: (batch_size, n_nodes, hidden_size) 

        Returns
        -------
        output: torch.tensor, shape: (batch_size, n_timesteps, n_nodes, hidden_size)
            Output tensor containing the hidden states from the GRU unit for each timestep.
        """
        # reshape h to (n_layers, batch_size, n_nodes, hidden_size)
        h = h.view(self.n_layers, h.size(0), h.size(1), h.size(2)) 
   
        output, hidden = self.gru(x, h)
    
        return output