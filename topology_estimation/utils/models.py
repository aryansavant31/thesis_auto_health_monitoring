import torch.nn as nn
import torch
from pytorch_lightning import LightningModule

class MLP(LightningModule):
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

# class LSTM(nn.Module):
#     def __init__(self, n_dim, n_layers, hidden_size):
#         """
#         Parameters
#         ----------
#         n_dim : int
#             Dimension of the input features (e.g., number of parameters per sample).
#         n_layers: int
#             number of LSTMs to stack
#         n_hid: int
#             output_size per roll from the LSTM

#         """
#         super(LSTM, self).__init__()
#         self.model_type = 'LSTM' 
#         self.n_layers = n_layers
#         self.hidden_size = hidden_size
#         self.lstm = nn.LSTM(n_dim, hidden_size, n_layers, batch_first=True)
#         # self.linear = nn.Linear(hidden_size, output_size)

#     def forward(self, input, h):
#         """
#         Parameters
#         ----------
#         x: torch.tensor, shape: (batch_size, n_timesteps, n_nodes, n_dim)
#             Input tensor.
#         h: torch.tensor, shape: (batch_size, n_nodes, hidden_size)

#         Returns
#         -------
#         output: torch.tensor, shape: (batch_size, n_timesteps, n_nodes, hidden_size)
#             Output tensor containing the hidden states (short term memory) from the LSTM unit for each timestep.
#         """
#         c0 = torch.zeros(self.n_layers, x.size(0)*x.size(2), self.hidden_size) # shape = (n_layers, batch_size, output_size (per step))

#         # reshape h to (n_layers, batch_size * n_nodes, hidden_size)
#         h = h.view(self.n_layers, h.size(0)*h.size(1), h.size(2)) 

#         # reshape x to (batch_size*n_nodes, n_timesteps, n_dim)
#         x = input.permute(0, 2, 1, 3)  # (batch_size, n_nodes, n_timesteps, n_dim)
#         x = x.contiguous().view(x.size(0) * x.size(1), x.size(2), x.size(3))  # (batch_size*n_nodes, n_timesteps, n_dim)

#         output, (hidden, _) = self.lstm(x, (h,c0)) 
        
#         # output contains the short term memory or hidden states from all the rolls of LSTM unit. 
#         # So if I have 10 timesteps as input, output contains 10 hidden state, each pertaining to the individual timestep

#         output = output.view(input.size(0), input.size(1), input.size(2), input.size(3))  # (batch_size, n_nodes, n_timesteps, hidden_size)
#         output = output.permute(0, 2, 1, 3)  # (batch_size, n_timesteps, n_nodes, hidden_size)
#         return output
    
    
# class RNN(nn.Module):
#     def __init__(self, n_dim, n_layers, hidden_size):
#         """
#         Parameters
#         ----------
#         n_dim : int
#             Dimension of the input features (e.g., number of parameters per sample).
#         n_layers: int
#             number of RNNs to stack
#         hidden_size: int
#             output_size per roll from the LSTM

#         """
#         super(RNN, self).__init__()
#         self.model_type = 'RNN' 
#         self.n_layers = n_layers
#         self.hidden_size = hidden_size
#         self.rnn = nn.RNN(n_dim, hidden_size, n_layers, batch_first=True)
#         # self.linear = nn.Linear(hidden_size, output_size)

#     def forward(self, x, h):
#         """
#         Parameters
#         ----------
#         x: torch.tensor, shape: (batch_size, n_timesteps, n_nodes, n_dim)
#             Input tensor. 
#         h: torch.tensor, shape: (batch_size, n_nodes, hidden_size)

#         Returns
#         -------
#         output: torch.tensor, shape: (batch_size, n_timesteps, n_nodes, hidden_size)
#             Output tensor containing the hidden states from the RNN unit for each timestep.
#         """
#         # reshape h to (n_layers, batch_size, n_nodes, hidden_size)
#         h = h.view(self.n_layers, h.size(0)*h.size(1), h.size(2))  
   
#         output, hidden, = self.rnn(x, h)
    
#         return output
    
    
class GRU(LightningModule):
    def __init__(self, n_dim, hidden_size):
        """
        Parameters
        ----------
        n_dim : int
            Dimension of the input features (e.g., number of parameters per sample).
        hidden_size: int
            output_size per roll from the GRU

        """
        super(GRU, self).__init__()
        self.model_type = 'GRU' 
        self.hidden_size = hidden_size
        
        # update gate parameters
        self.input_u = nn.Linear(n_dim, hidden_size)
        self.hidden_u = nn.Linear(hidden_size, hidden_size)

        # reset gate parameters
        self.input_r = nn.Linear(n_dim, hidden_size)
        self.hidden_r = nn.Linear(hidden_size, hidden_size)
        
        # candidate hidden state parameters
        self.input_h = nn.Linear(n_dim, hidden_size)
        self.hidden_h = nn.Linear(hidden_size, hidden_size)

        
    def forward(self, input, agg_msgs, hidden_prev):
        """
        Parameters
        ----------
        input: torch.tensor, shape: (batch_size, n_nodes, n_dims)
            Input tensor.
        agg_msgs: torch.tensor, shape: (batch_size, n_nodes, hidden_size)
            Aggregated messages from the edges
        hidden_prev: torch.tensor, shape: (batch_size, n_nodes, hidden_size)
            Previous hidden state
        Returns
        -------
        hidden: torch.tensor, shape: (batch_size, n_nodes, hidden_size)
            Updated hidden state after processing the input and aggregated messages through update and reset gates
        """
        
        # update gate
        u = torch.sigmoid(self.input_u(input) + self.hidden_u(agg_msgs))

        # reset gate
        r = torch.sigmoid(self.input_r(input) + self.hidden_r(agg_msgs))

        # candidate hidden state
        h_tilde = torch.tanh(self.input_h(input) + self.hidden_h(r * agg_msgs))

        # new hidd(en state
        hidden = ((1 - u) * hidden_prev) + (u * h_tilde)

        return hidden
        