"""
This module contains:
- Pipeline class
- Feature extraction Functions
"""
def get_fex_config(fex_type, **kwargs):
        """
        All the feature extraction configurations are defined here.

        Parameters
        ----------
        fex_type : str
            The type of feature extraction to be used (e.g., 'first_n_modes', 'PCA').

        **kwargs : dict
            For all options of `fex_type`:
            - 'first_n_modes': `n_modes`
            - 'PCA': `n_components`
            - 'AMD': `None`
            - 'lucas': `weight`, `height`, `age`
        """
        config = {}
        config['type'] = fex_type

        if fex_type == 'first_n_modes':
            config['n_modes'] = kwargs.get('n_modes', 5)  # default to 5 modes if not specified
        
        elif fex_type == 'PCA':
            config['n_components'] = kwargs.get('n_components', 5)  # default to 5 components if not specified

        elif fex_type == 'lucas':
            config['weight'] = kwargs.get('weight', 0.5)  # default weight
            config['height'] = kwargs.get('height', 6)
            config['age'] = kwargs.get('age', 20) 

        return config

class FeatureExtractor:
    def __init__(self, fex_configs):
        """
        Initialize the feature extractor model with the specified type.
        
        Parameters
        ----------
        fex_config : list
            Type of feature extraction to be used (e.g., 'first_n_modes', 'lucas').
        """
        self.fex_configs = fex_configs

    def __call__(self, data):
        for fex_config in self.fex_configs:

            if fex_config['type'] == 'first_n_modes':
                feature = first_n_modes(data, fex_config['n_modes'])
            
            #[TODO] add rest of the feature extraction methods here

        # [TODO] concatenate the features obtained from all the fex types into a single tensor

def first_n_modes(data, n_modes):
    # [TODO] Implement function to extract the first n modes from the data
    pass

# [TODO] Add rest of the LUCAS's feature extraction functions below