"""
This module contains 
- pipeline class `DataTransformer` to apply domain transformations and normalization to the data.
- domain transofrm functions and 
- normalization functions.
"""

class DataTransformer:
    """
    Domain transform (time, freq)
    |
    v
    Normalization (std, min-max)
    """
    def __init__(self, domain='time', norm_type=None, data_stats=None):
        """
        Parameters
        ----------
        domain : str
            The domain transformation to be applied (e.g., 'time', 'freq').
        norm_type : str
            The type of normalization to be applied (e.g., 'std', 'min_max').
        data_stats : dict
            Statistics for normalization (e.g., mean, std, min, max).
        """
        self.domain = domain
        self.norm_type = norm_type
        self.data_stats = data_stats

    def __call__(self, data):
        """
        Apply the domain transform and normalization to the data.
        
        Parameters
        ----------
        data : torch.Tensor
            Input data tensor of shape (batch_size, n_nodes, n_datapoints, n_dims).
        
        """
        # apply domain transformation

        # [TODO] Implement domain transformation logic here

        # apply normalization
        if self.norm_type == 'min_max':
            data = min_max_normalize(data, self.data_stats['min'], self.data_stats['max'])

        elif self.norm_type == 'std':
            data = std_normalize(data, self.data_stats['mean'], self.data_stats['std'])

        elif self.norm_type is None:
            pass

        return data

def min_max_normalize(data, min, max):
    """
    Normalize the data based on min and max values along the components axis (axis=2)

    Parameters
    ----------
    data : torch.Tensor, shape (batch_size, n_nodes, n_components, n_dims)
        Input data tensor
    max : torch.Tensor, shape (n_nodes, 1, n_dims)
    min : torch.Tensor, shape (n_nodes, 1, n_dims)
        Min and max values for normalization

    """
    norm_data = (data - min) / (max - min + 1e-8)
    return norm_data

def std_normalize(data, mean, std):
    """
    Normalize the data based on mean and standard deviation along the components axis (axis=2)

    Parameters
    ----------
    data : torch.Tensor, shape (batch_size, n_nodes, n_components, n_dims)
        Input data tensor
    mean : torch.Tensor, shape (n_nodes, 1, n_dims)  
    std : torch.Tensor, shape (n_nodes, 1, n_dims)
        Mean and standard deviation for normalization 
    """
    norm_data = (data - mean) / (std + 1e-8)
    return norm_data