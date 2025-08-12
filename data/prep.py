"""
This moduel contains:
- pipeline class `DataLoader` which load the data using the address and put the loaded data in the dataloaders (trainloader, valloader, testlaoder, custum_loader)
- data loading functions
- trainset dataloader and custom dataloader functions.
"""

import sys, os

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, DATA_DIR) if DATA_DIR not in sys.path else None

import numpy as np
import torch
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader, random_split
import h5py
from torch.utils.data import Subset

# local imports
from config import DataConfig
from augment import add_gaussian_noise
from collections import defaultdict

class DataPreprocessor:
    """
    Step 1. Load the data usign address
    Step 2. Process the data (augmentation and segmentation of data)
    Step 3. Then see if they are part of fault_detection or topology_estimation
                if fault_detection, label is node_label
                if topology_estimation, label is edge_label
    Step 4. call either get_train_val_test_dataloaders or get_custom_dataloader
    """
    def __init__(self, package):
        """
        Parameters
        ----------
        package : str
            The package to load data for.
            ('fault_detection', 'topology_estimation')
        """
        self.package = package
      
    def _load_dataset(self):
        """
        Load the dataset based on the run type.
        """
        
        # load node and edge data paths
        node_path_map, edge_path_map = self.data_config.get_dataset_paths()

        # prepare data and labels from paths
        x_node, y_node, y_edge = {}, {}, {}

        for ds_type in node_path_map.keys():
            node_type_map = node_path_map[ds_type]
            ds_subtype_map = edge_path_map[ds_type]
            if node_type_map is not None and ds_subtype_map is not None:
                x_node[ds_type], y_node[ds_type], y_edge[ds_type] = self._prepare_data_from_path(node_type_map, ds_subtype_map, ds_type)
            else:
                x_node[ds_type], y_node[ds_type], y_edge[ds_type] = None, None, None

        # create datasets
        if self.package == 'fault_detection':
            dataset = self._make_fdet_dataset(x_node, y_node)
        elif self.package == 'topology_estimation':
            dataset = self._make_tp_dataset(x_node, y_edge, y_node)    

        return dataset
    
    def _get_dataset_stats(self, subset:Subset):
        """
        Compute statistic metrics for the subset.

        Note
        ----
        - Statistic metrics like `mean`, `std`, `min` and `max` are computed across all the samples and timesteps. 

            - For example, if data is of shape (samples=5, n_nodes=2, n_timesteps=10, n_dims=3), the statistical metrics is computed for (samples x n_timesteps) = 50 components.
            So mean of 50 components, std of 50 components etc.

        - Each node and dimension will have is own `mean`, `std`, `min` and `max`. 
            - This is becasue the data distribution can be different for each node and dimension and by doing per-node and per-dimension normalization, the data distribution after normalziation can be preserved. 
            - This will help in better training of the model.
        
        Parameters
        ----------
        subset : torch.utils.data.Subset
            The dataset to compute statistics for.

        Returns
        -------
        data_stats : dict
            Dictionary containing statistics of the dataset
        """
        if isinstance(subset, Subset):
            original_dataset = subset.dataset
            indices = subset.indices
            data = torch.stack([original_dataset[i][0] for i in indices]) # shape (total_samples, n_nodes, n_timesteps, n_dims)
        else:
            data = subset.tensors[0]  # For TensorDataset

        min_val = data.min(dim=2, keepdim=True).values  
        max_val = data.max(dim=2, keepdim=True).values  # Shape: (n_samples, n_nodes, 1, n_dims)
        
        data_stats = {
            'mean': torch.mean(data, dim=(0, 2), keepdim=True),
            'std': torch.std(data, dim=(0, 2), keepdim=True),
            'min': torch.min(min_val, dim=0, keepdim=True).values,
            'max': torch.max(max_val, dim=0, keepdim=True).values
        }
        for k in data_stats:
            data_stats[k] = data_stats[k].squeeze(0)

        return data_stats
    
    def _get_label_counts(self, dataset):
        """
        Get the counts of each label in the dataset.

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            The dataset to count labels from.

        Returns
        -------
        label_counts : dict
            Dictionary containing counts of each label.
        """
        label_counts = {0: 0, 1: 0, -1: 0}  # Assuming labels are 0 for healthy, 1 for unhealthy, and -1 for unknown

        for data in dataset:
            label_value = int(data[-1].item())
            label_counts[label_value] += 1
        return label_counts

    def get_custom_data_package(self, data_config:DataConfig, batch_size=10, num_workers=10):
        """
        Create a custom dataloader and the data stats for the specified run type.

        Parameters
        ----------
        data_config : DataConfig
            The data configuration object.
        batch_size : int

        Returns
        -------
        (custom_loader, data_stats) : tuple

        **_Note_**
            **data_stats** is dictionary containing statistics of the dataset.
            - `mean`: Mean of the dataset.
            - `std`: Standard deviation of the dataset.
            - `min`: Minimum value in the dataset.
            - `max`: Maximum value in the dataset.
        """
        # set the data config based on run type
        self.data_config = data_config
        
        # load the dataset
        dataset = self._load_dataset()

        # retain only the desired number of samples
        total_samples = len(dataset)

        desired_samples = int(total_samples * self.data_config.amt)

        remainder_samples = total_samples - desired_samples

        if desired_samples < total_samples:
            dataset, remain_dataset = random_split(dataset, [desired_samples, remainder_samples])
        
        # get number of OK and NOK samples
        des_label_counts = self._get_label_counts(dataset)
        rem_label_counts = self._get_label_counts(remain_dataset) if remainder_samples > 0 else {0: 0, 1: 0, -1: 0}

        print(f"\nTotal samples: {total_samples}, \nDesired samples: {desired_samples} [OK={des_label_counts[0]}, NOK={des_label_counts[1]}, UK={des_label_counts[-1]}], \nRemainder samples: {remainder_samples} [OK={rem_label_counts[0]}, NOK={rem_label_counts[1]}, UK={rem_label_counts[-1]}]")

        # get dataset statistics
        data_stats = self._get_dataset_stats(dataset)

        # create custom dataloader
        custom_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)

        return (custom_loader, data_stats)
    
    def get_training_data_package(self, data_config:DataConfig, train_rt=0.8, test_rt=0.2, val_rt=0, batch_size=50, num_workers=10):
        """
        Create train, validation, and test dataloaders and compute their statistical metrics.

        Parameters
        ----------
        data_config : DataConfig
            The data configuration object.
        batch_size : int
            The batch size for the dataloaders.

        Returns
        -------
        (train_loader, train_data_stats) : tuple       
        (test_loader, test_data_stats) : tuple
        (val_loader, val_data_stats) : tuple
        
        **_Note_**
            **data_stats** of train, test and val is dictionary containing following statistics:
                - `mean`: Mean of the dataset.
                - `std`: Standard deviation of the dataset.
                - `min`: Minimum value in the dataset.
                - `max`: Maximum value in the dataset.
        """
        # set the data config for training
        self.data_config = data_config

        # load the dataset
        dataset = self._load_dataset()

        # split the dataset into train, validation, and test sets
        total_samples = len(dataset)

        # error checks
        if train_rt + test_rt + val_rt > 1:
            raise ValueError("The sum of train, test, and validation ratios must not exceed 1.")
        
        if self.package == 'topology_estimation' and val_rt == 0:
            raise ValueError("Validation set is required for topology estimation. Please provide a non-zero validation ratio.")
        
        n_train = int(train_rt * total_samples)
        n_test = int(test_rt * total_samples)
        n_val = int(val_rt * total_samples)
        remainder_samples = total_samples - n_train - n_test - n_val

        if self.package == 'topology_estimation':
            if n_train + n_test + n_val < total_samples:
                train_set, test_set, val_set, remain_dataset = random_split(dataset, [n_train, n_test, n_val, remainder_samples])
            else:
                train_set, test_set, val_set = random_split(dataset, [n_train, n_test, n_val])

        elif self.package == 'fault_detection':
            if n_train + n_test < total_samples:
                train_set, test_set, remain_dataset = random_split(dataset, [n_train, n_test, remainder_samples])
            else:
                train_set, test_set = random_split(dataset, [n_train, n_test])

            val_set = None  # No validation set for fault detection
        
        # get number of OK and NOK samples in each set
        train_label_counts = self._get_label_counts(train_set)
        test_label_counts = self._get_label_counts(test_set)
        val_label_counts = self._get_label_counts(val_set) if val_set is not None else {0: 0, 1: 0, -1: 0}
        rem_label_counts = self._get_label_counts(remain_dataset) if remainder_samples > 0 else {0: 0, 1: 0, -1: 0}

        print(f"\nTotal samples: {total_samples}, \nTrain: {n_train} [OK={train_label_counts[0]}, NOK={train_label_counts[1]}, UK={train_label_counts[-1]}], Test: {n_test} [OK={test_label_counts[0]}, NOK={test_label_counts[1]}, UK={test_label_counts[-1]}], Val: {n_val} [OK={val_label_counts[0]}, NOK={val_label_counts[1]}, UK={val_label_counts[-1]}], \nRemainder: {remainder_samples} [OK={rem_label_counts[0]}, NOK={rem_label_counts[1]}, UK={rem_label_counts[-1]}]")

        # get dataset statistics
        train_data_stats = self._get_dataset_stats(train_set)
        test_data_stats = self._get_dataset_stats(test_set)
        val_data_stats = self._get_dataset_stats(val_set) if val_set is not None else None

        # create dataloaders
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers) if val_set is not None else None
        
        return (train_loader, train_data_stats), (test_loader, test_data_stats), (val_loader, val_data_stats)
        

    def _make_tp_dataset(self, x, y, z):
        """
        Create a TensorDataset for topology estimation (tp) from the provided data and labels.

        Parameters
        ----------
        x : dict
            Dictionary containing tensor node data for each dataset type.
        y : dict
            Dictionary containing tensor edge labels for each dataset type.
        z : dict
            Dictionary containing tensor node labels for each dataset type (to count number of OK and NOK samples).

        Returns
        -------
        TensorDataset
            A dataset containing the (concatenated) data and labels.
        """
        x_list, y_list, z_list = [], [], []

        for ds_type in x.keys():
            if x[ds_type] is not None and y[ds_type] is not None and z[ds_type] is not None:
                x_list.append(x[ds_type])
                y_list.append(y[ds_type])
                z_list.append(z[ds_type])

        if not x_list or not y_list or not z_list:
            raise ValueError("No data available to create datasets.")
        
        x_all = torch.cat(x_list, dim=0)
        y_all = torch.cat(y_list, dim=0)
        z_all = torch.cat(z_list, dim=0)

        return TensorDataset(x_all, y_all, z_all)
    
    def _make_fdet_dataset(self, x, y):
        """
        Create a TensorDataset for topology estimation (tp) from the provided data and labels.

        Parameters
        ----------
        x : dict
            Dictionary containing tensor node data for each dataset type.
        y : dict
            Dictionary containing tensor node labels for each dataset type.

        Returns
        -------
        TensorDataset
            A dataset containing the (concatenated) data and labels.
        """
        x_list, y_list = [], []

        for ds_type in x.keys():
            if x[ds_type] is not None and y[ds_type] is not None:
                x_list.append(x[ds_type])
                y_list.append(y[ds_type])

        if not x_list or not y_list:
            raise ValueError("No data available to create datasets.")
        
        x_all = torch.cat(x_list, dim=0)
        y_all = torch.cat(y_list, dim=0)

        return TensorDataset(x_all, y_all)

    def _prepare_data_from_path(self, node_type_map, ds_subtype_map, ds_type):
        """
        Parameters
        ----------
        node_type_map : dict
            A dictionary containing paths to the datasets.
        ds_subtype_map : dict
            A dictionary containing paths to the edge datasets.
        ds_type : str
            The type of dataset
            ('OK' for healthy, 'NOK' for unhealthy).

        Returns
        -------
        final_node_data : torch.Tensor
            The processed node data.
        final_node_labels : torch.Tensor
            The labels for the node data.
        final_edge_data : torch.Tensor
            The processed edge label data.
        """
        # step 1: process node data
        node_dim_collect = self._process_node_data(node_type_map, ds_type)

        # step 2: flatten edge matrices per ds_subtype
        ds_subtype_edge_map = self._process_edge_data(ds_subtype_map)

        # step 3: prepare node and edge data
        all_ds_subtype_node_blocks = []  # each: (n_samples, n_nodes, t, d)
        all_ds_subtype_edges = []        # each: (n_samples, n_edges)

        for ds_subtype in ds_subtype_map.keys():
            per_node_tensors = []

            for node_type in sorted(node_type_map.keys()):
                dim_segment_arrays = []

                for dim_idx in sorted(node_dim_collect[node_type][ds_subtype].keys()):
                    all_segments = np.concatenate(node_dim_collect[node_type][ds_subtype][dim_idx], axis=0)
                    dim_segment_arrays.append(all_segments)

                # truncate to min_segments across dims
                min_segments = min(arr.shape[0] for arr in dim_segment_arrays)
                trimmed_segments = [arr[:min_segments] for arr in dim_segment_arrays]

                # concatenate dims → (min_segments, t, n_dims)
                node_tensor = np.concatenate(trimmed_segments, axis=-1)
                per_node_tensors.append(node_tensor)

            # Now we have list of (min_segments, t, n_dims) for each node
            # stack nodes → (min_segments, n_nodes, t, n_dims)
            node_block = np.stack(per_node_tensors, axis=1)
            all_ds_subtype_node_blocks.append(node_block)

            # copy flat edge per segment
            edge_vec = ds_subtype_edge_map[ds_subtype]
            edge_block = np.tile(edge_vec, (min_segments, 1))  # (min_segments, n_edges)
            all_ds_subtype_edges.append(edge_block)

        # step 4: Final concatenation across ds_subtypes
        final_node_data_np = np.concatenate(all_ds_subtype_node_blocks, axis=0)  # (n_samples, n_nodes, t, d)
        final_edge_data_np = np.concatenate(all_ds_subtype_edges, axis=0)        # (n_samples, n_edges)
        
        if ds_type == 'OK':
            final_node_labels_np = np.zeros((final_node_data_np.shape[0], 1), dtype=np.float32)
        elif ds_type == 'NOK':
            final_node_labels_np = np.ones((final_node_data_np.shape[0], 1), dtype=np.float32)
        elif ds_type == 'UK':
            final_node_labels_np = (-1) * np.ones((final_node_data_np.shape[0], 1), dtype=np.float32)

        # convert to torch tensors
        final_node_data = torch.tensor(final_node_data_np, dtype=torch.float32)
        final_node_labels = torch.tensor(final_node_labels_np, dtype=torch.float32)
        final_edge_data = torch.tensor(final_edge_data_np, dtype=torch.float32)

        return final_node_data, final_node_labels, final_edge_data

    def _process_node_data(self, node_type_map, ds_type):
        # node_type -> ds_subtype -> dim -> segments
        node_dim_collect = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        # process node data
        for node_type, ds_subtype_map in node_type_map.items():
            for ds_subtype, signal_type_paths in ds_subtype_map.items():
                for dim_idx, hdf5_path in enumerate(signal_type_paths):

                    # load node data
                    with h5py.File(hdf5_path, 'r') as f:
                        data = f['data'][:]

                    # Apply augmentations
                    if ds_type == 'OK':
                       data = self.add_augmentations(data, self.data_config.healthy_configs[ds_subtype], ds_subtype)   

                    elif ds_type == 'NOK':
                       data = self.add_augmentations(data, self.data_config.unhealthy_configs[ds_subtype], ds_subtype) 
                                                     
                    data_segments = segment_data(data, self.data_config.window_length, self.data_config.stride)
                    data_segments = np.expand_dims(data_segments, axis=-1)  # (n_segments, n_timesteps, 1)

                    # store segments
                    node_dim_collect[node_type][ds_subtype][dim_idx].append(data_segments)

        return node_dim_collect
    
    def add_augmentations(self, data, augment_configs, ds_subtype="_"):
        augmented_data_list = []

        if augment_configs == []:
            raise ValueError(f"No original or augmentation configs for the dataset subtype {ds_subtype} provided.")

        for augment_config in augment_configs:
            # original data
            if augment_config['type'] == 'OG':
                augmented_data = data
            # gaussian noise
            if augment_config['type'] == 'gau':
                augmented_data = add_gaussian_noise(data, augment_config['mean'], augment_config['std'])

            # apply other augmentations if needed
            # ...

            augmented_data_list.append(augmented_data)

        return np.concatenate(augmented_data_list, axis=0)
    
    def _process_edge_data(self, ds_subtype_map):
        ds_subtype_edge_map = {}

        for ds_subtype, edge_hdf5_path in ds_subtype_map.items():
            with h5py.File(edge_hdf5_path, 'r') as f:
                adj_mat = f['adj_matrix'][:]  # shape: (n_nodes, n_nodes)

            flat_edge = []
            for r in range(adj_mat.shape[0]):
                for c in range(adj_mat.shape[1]):
                    if r != c:
                        flat_edge.append(adj_mat[r, c])
            ds_subtype_edge_map[ds_subtype] = np.array(flat_edge)
        
        return ds_subtype_edge_map
    


# =======================================================================
# Helper functions for data processing
# =======================================================================    

def segment_data(data, window_length, stride):
    """
    Segment the data into windows of window_length with given stride.
    Parameters
    ----------
    data : np.ndarray, shape (n_samples, n_timesteps)
        Input data to be segmented.
    window_length : int
        Length of each segment.
    stride : int
        Step size for moving the window.
    
    Returns
    -------
    segments : np.ndarray, shape (n_segments, window_length)
    """
    n_samples, n_timesteps = data.shape
    segments = []
    
    for i in range(0, n_timesteps - window_length + 1, stride):
        segment = data[:, i:i + window_length]
        segments.append(segment)
    
    return np.concatenate(segments, axis=0)


def load_spring_particle_data(batch_size=10):
    
    # for node_path in node_ds_path['H']:
    #     if os.path.basename(node_path) == 'pos':
    #         loc_path = node_path
    #     elif os.path.basename(node_path) == 'vel':
    #         vel_path = node_path

    # edge_path = edge_ds_path['H'][0]  # Assuming there's only one edge dataset for 'H'

    # loc_path = "C:\\AFD\\data\\datasets\spring_particles\P005\scenario_1\healthy\H1\processed\\nodes\T50\OG\\all_nodes\pos"
    # vel_path = "C:\AFD\data\datasets\spring_particles\P005\scenario_1\healthy\H1\processed\\nodes\T50\OG\\all_nodes\\vel"
    # edge_path = "C:\AFD\data\datasets\spring_particles\P005\scenario_1\healthy\H1\processed\edges"

    loc_train = np.load(f'{loc_path}\\loc_train_springs' + '5' + '.npy')
    vel_train = np.load(f'{vel_path}\\vel_train_springs' + '5' + '.npy')
    edges_train = np.load(f'{edge_path}\\edges_train_springs' + '5' + '.npy')

    loc_valid = np.load(f'{loc_path}\\loc_valid_springs' + '5' + '.npy')
    vel_valid = np.load(f'{vel_path}\\vel_valid_springs' + '5' + '.npy')
    edges_valid = np.load(f'{edge_path}\\edges_valid_springs' + '5' + '.npy')

    loc_test = np.load(f'{loc_path}\\loc_test_springs' + '5' + '.npy')
    vel_test = np.load(f'{vel_path}\\vel_test_springs' + '5' + '.npy')
    edges_test = np.load(f'{edge_path}\\edges_test_springs' + '5' + '.npy')

    # [num_samples, num_timesteps, num_dims, num_atoms]
    num_atoms = loc_train.shape[3]

    loc_max = loc_train.max()
    loc_min = loc_train.min()
    vel_max = vel_train.max()
    vel_min = vel_train.min()

    # Normalize to [-1, 1]
    loc_train = (loc_train - loc_min) * 2 / (loc_max - loc_min) - 1
    vel_train = (vel_train - vel_min) * 2 / (vel_max - vel_min) - 1

    loc_valid = (loc_valid - loc_min) * 2 / (loc_max - loc_min) - 1
    vel_valid = (vel_valid - vel_min) * 2 / (vel_max - vel_min) - 1

    loc_test = (loc_test - loc_min) * 2 / (loc_max - loc_min) - 1
    vel_test = (vel_test - vel_min) * 2 / (vel_max - vel_min) - 1

    # Reshape to: [num_sims, num_atoms, num_timesteps, num_dims]
    loc_train = np.transpose(loc_train, [0, 3, 1, 2])
    vel_train = np.transpose(vel_train, [0, 3, 1, 2])
    feat_train = np.concatenate([loc_train, vel_train], axis=3)
    edges_train = np.reshape(edges_train, [-1, num_atoms ** 2])
    edges_train = np.array((edges_train + 1) / 2, dtype=np.int64)

    loc_valid = np.transpose(loc_valid, [0, 3, 1, 2])
    vel_valid = np.transpose(vel_valid, [0, 3, 1, 2])
    feat_valid = np.concatenate([loc_valid, vel_valid], axis=3)
    edges_valid = np.reshape(edges_valid, [-1, num_atoms ** 2])
    edges_valid = np.array((edges_valid + 1) / 2, dtype=np.int64)

    loc_test = np.transpose(loc_test, [0, 3, 1, 2])
    vel_test = np.transpose(vel_test, [0, 3, 1, 2])
    feat_test = np.concatenate([loc_test, vel_test], axis=3)
    edges_test = np.reshape(edges_test, [-1, num_atoms ** 2])
    edges_test = np.array((edges_test + 1) / 2, dtype=np.int64)

    feat_train = torch.FloatTensor(feat_train)
    edges_train = torch.LongTensor(edges_train)
    feat_valid = torch.FloatTensor(feat_valid)
    edges_valid = torch.LongTensor(edges_valid)
    feat_test = torch.FloatTensor(feat_test)
    edges_test = torch.LongTensor(edges_test)

    # Exclude self edges
    off_diag_idx = np.ravel_multi_index(
        np.where(np.ones((num_atoms, num_atoms)) - np.eye(num_atoms)),
        [num_atoms, num_atoms])
    edges_train = edges_train[:, off_diag_idx]
    edges_valid = edges_valid[:, off_diag_idx]
    edges_test = edges_test[:, off_diag_idx]

    train_data = TensorDataset(feat_train, edges_train)
    valid_data = TensorDataset(feat_valid, edges_valid)
    test_data = TensorDataset(feat_test, edges_test)

    train_data_loader = DataLoader(train_data, batch_size=batch_size)
    valid_data_loader = DataLoader(valid_data, batch_size=batch_size)
    test_data_loader = DataLoader(test_data, batch_size=batch_size)

    data_stats = {
        'min': [], # minimum of all dimensions
        'max': [], # maximum of all dimensions
        'mean': [], # mean of all dimensions
        'std': []  # standard deviation of all dimensions
    }
    return train_data_loader, valid_data_loader, test_data_loader, data_stats