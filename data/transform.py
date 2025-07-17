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
    def __init__(self, domain, norm_type, data_stats:dict):
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

def min_max_normalize(data, min, max):
    """
    Normalize the data based on min and max values.
    """
    pass

def std_normalize(data, mean, std):
    """
    Normalize the data based on mean and standard deviation.
    """
    pass