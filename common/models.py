import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, mlp_config, do_prob=0.0, is_batch_norm=False, is_gnn=False):
        super(MLP, self).__init__()

        current_dim = input_size
        self.model_type = 'MLP' 
        self.layers = nn.ModuleList()
        self.is_batch_norm = is_batch_norm
        self.is_gnn = is_gnn

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
        x : torch.Tensor, shape (batch_size, (n_nodes), n_features).
            Input tensor 

        get_fex : list
            List of layer indices to return features from. If None, no features are returned.
        
        Returns
        -------
        x : torch.Tensor, shape (batch_size, (n_nodes), n_output_features).
        fex : list
            List of feature tensors from specified layers, if get_fex is not None.

        """
        if get_fex is not None:
            self.check_invalid_layer_indices(get_fex)

        fex = []
        for layer_num, layer in enumerate(self.layers):
            # check if layer is batchnorm
            if isinstance(layer, nn.BatchNorm1d):
                x = self.batch_norm(x, layer) if self.is_gnn else layer(x)
            else:    
                x = layer(x)
            if layer_num in get_fex:
                fex.append(x)

        return x, fex if get_fex is not None else x
        
# Put a nn.linear layer as output in the CNN    
class CNN(nn.Module):
    def __init__(self, n_inp_chn, input_size, conv_config):
        """
        arg:
            n_inp_chn: number of input channels (shape = (n_inp_chn,))
            input_size: number of features per sample (shape = (n_features,))
            conv_config: list of convolutional layer specifications (n_out_chn, kernal_size, pool_kernal_size)
        """
        super(CNN, self).__init__()

        self.n_conv_layer = len(conv_config)
        dim = input_size
        self.model_type = 'CNN' 
        n_curr_chn= n_inp_chn
        self.conv_layers = nn.ModuleList()

        # add convolution layer

        for conv_spec in conv_config:
            n_out_chn, conv_kernal_size, pool_kernal_size = conv_spec
            self.conv_layers.append(nn.Conv1d(n_curr_chn, n_out_chn, conv_kernal_size))
            self.conv_layers.append(nn.ReLU())
            self.conv_layers.append(nn.MaxPool1d(pool_kernal_size, 2))

            # update spatial dimensions
            dim = (dim - conv_kernal_size + 1) // pool_kernal_size
            
            n_curr_chn = n_out_chn

        # add FCN layers
        self.flatten_input_size = n_curr_chn * dim
        #self.curr_input_size = self.flatten_input_size

        # if self.hidden_config != None:
        #     self.fcn_layers = nn.ModuleList()
        #     for config in self.hidden_config:
        #         layer_output_size, activation_type = config
        #         self.fcn_layers.append(nn.Linear(curr_input_size, layer_output_size))
        #         if activation_type == 'relu':
        #             self.fcn_layers.append(nn.ReLU())
        #         elif activation_type == 'sigmoid':
        #             self.fcn_layers.append(nn.Sigmoid())
        #         curr_input_size = layer_output_size

        # self.output_layer = nn.Linear(curr_input_size, output_size)


    def forward(self, x):
        """
        arg:
            x shape = (batch_size, n_inp_chn, n_inp_feat)

        return:
            output shape = (batch_size, n_out_chn, n_out_feat)
        """
        output = x
        for conv_layer in self.conv_layers:
            output = conv_layer(output)
        
        # # flattening
        # output = output.view(-1, self.flatten_input_size)
        # if self.hidden_config != None:
        #     for fcn_layer in self.fcn_layers:
        #         output = fcn_layer(output)
        # output = self.output_layer(output)

        return output, self.flatten_input_size