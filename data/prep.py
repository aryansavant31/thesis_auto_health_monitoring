"""
This moduel contains:
- pipeline class `DataLoader` which load the data using the address and put the loaded data in the dataloaders (trainloader, valloader, testlaoder, custum_loader)
- data loading functions
- trainset dataloader and custom dataloader functions.
"""

import sys, os

# other imports
import numpy as np
import torch
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader, random_split
import h5py
from torch.utils.data import Subset
from collections import defaultdict
from scipy.interpolate import interp1d

# local imports
from .config import DataConfig
from .augment import *


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
        x_node, y_node, y_edge, y_rep = {}, {}, {}, {}

        for ds_type in node_path_map.keys():
            node_type_map = node_path_map[ds_type]
            ds_subtype_map = edge_path_map[ds_type]
            if node_type_map is not None and ds_subtype_map is not None:
                x_node[ds_type], y_node[ds_type], y_edge[ds_type], y_rep[ds_type] = self._prepare_data_from_path(node_type_map, ds_subtype_map, ds_type)
            else:
                x_node[ds_type], y_node[ds_type], y_edge[ds_type], y_rep[ds_type] = None, None, None, None

        # create datasets
        if self.package == 'fault_detection':
            dataset = self._make_fdet_dataset(x_node, y_node, y_rep)
        elif self.package == 'topology_estimation':
            dataset = self._make_tp_dataset(x_node, y_edge, y_node, y_rep)

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
    
    def _get_label_counts(self, data_loader):
        """
        Get the counts of each label in the dataset.

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
            The data loader containing the dataset.

        Returns
        -------
        label_counts : dict
            Dictionary containing counts of each label.
        """
        label_counts = {0: 0, 1: 0, -1: 0}  # Assuming labels are 0 for healthy, 1 for unhealthy, and -1 for unknown

        for data in data_loader:
            for label_value in data[-2].view(-1).tolist(): # data has (..., y_node (we want this), y_rep)
                label_counts[label_value] += 1

        total_samples = sum(label_counts.values())

        return label_counts, total_samples
    
    def _get_augment_str(self, augments):
        """
        Get the string representation of the augmentations.
        """
        augment_strings = []
        for augment in augments:
            additional_keys = ', '.join([f"{key}={value}" for key, value in augment.items() if key != 'type'])
            if additional_keys and augment['type'] != 'OG':
                augment_strings.append(f"{augment['type']}({additional_keys})")
            else:
                augment_strings.append(f"{augment['type']}")

        return ', '.join(augment_strings)
    
    def get_data_selection_text(self):
        """
        - the data selections for healthy, unhealthy, and unknown configurations.
        - the node and signal types.
        """
        text = "Dataset selections:\n"
        text += 45*'-' + "\n"
        text += "*_(<ds_subtype_num>) <ds_subtype> : [<augments>]_*\n\n"

        text += "- **Healthy configs**\n"
        for idx, (ds_subtype, augments) in enumerate(self.data_config.healthy_configs.items()):
            text += f"  ({idx+1}) {ds_subtype}    : [{self._get_augment_str(augments)}]\n"

        text += "\n- **Unhealthy configs**\n"
        for idx, (ds_subtype, augments) in enumerate(self.data_config.unhealthy_configs.items()):
            text += f"  ({idx+1}) {ds_subtype}    : [{self._get_augment_str(augments)}]\n"

        text += "\n- **Unknown configs**\n"
        for idx, (ds_subtype, augments) in enumerate(self.data_config.unknown_configs.items()):
            text += f"  ({idx+1}) {ds_subtype}    : [{self._get_augment_str(augments)}]\n"

        text += "\n\nNode and signal types:\n"
        text += 45*'-' + "\n"
        text += "*_(<node_num>) <node_type> : [<signal_types>]_*\n\n"
        
        for node_num, (node, signals) in enumerate(self.data_config.signal_types['group'].items()):
            text += f"  ({node_num+1}) {node}   : [{', '.join(signals)}]\n"
        
        text += f"\nNode group name: {self.data_config.signal_types['node_group_name']}\n"
        text += f"Signal group name: {self.data_config.signal_types['signal_group_name']}\n"

        return text

        # # print the data selections
        # print(f"\n\nds_subtype selections:\n")
        # print("(<ds_subtype_num>) <ds_subtype> : [<augments>]")
        # print(45*'-')
        # print(">> Healthy configs")
        # for idx, (ds_subtype, augments) in enumerate(self.data_config.healthy_configs.items()):
        #     print(f"({idx+1}) {ds_subtype}    : [{self._get_augment_str(augments)}]")

        # print("\n>> Unhealthy configs")
        # for idx, (ds_subtype, augments) in enumerate(self.data_config.unhealthy_configs.items()):
        #     print(f"({idx+1}) {ds_subtype}    : [{self._get_augment_str(augments)}]")

        # print("\n>> Unknown configs")
        # for idx, (ds_subtype, augments) in enumerate(self.data_config.unknown_configs.items()):
        #     print(f"({idx+1}) {ds_subtype}    : [{self._get_augment_str(augments)}]")

        # # print node and signal types
        # print("\n\nNode and signal types are set as follows:\n")
        # print("(<node_num>) <node_type> : [<signal_types>]")
        # print(45*'-')
        # for node_num, (node, signals) in enumerate(self.data_config.signal_types['group'].items()):
        #     print(f"({node_num+1}) {node}   : [{', '.join(signals)}]")
        
        # print(f'\nNode group name: {self.data_config.signal_types['node_group_name']}')
        # print(f'Signal group name: {self.data_config.signal_types['signal_group_name']}')

    def get_custom_data_package(self, data_config:DataConfig, batch_size=10, num_workers=1):
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
        print(f"\n'{self.data_config.run_type.capitalize()}' type dataset selected:")

        data_select_text = self.get_data_selection_text()        
        print("\n\n" + data_select_text)

        # load the dataset
        dataset = self._load_dataset()

        # retain only the desired number of samples
        total_samples = len(dataset)

        desired_samples = int(total_samples * self.data_config.amt)

        remainder_samples = total_samples - desired_samples

        if desired_samples < total_samples:
            dataset, remain_dataset = random_split(dataset, [desired_samples, remainder_samples])

        # get dataset statistics
        data_stats = self._get_dataset_stats(dataset)

        # create custom dataloader
        custom_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=num_workers)
        remain_loader = DataLoader(remain_dataset, batch_size=1, shuffle=False, drop_last=False) if remainder_samples > 0 else None

        # get number of OK and NOK samples
        des_label_counts, n_des = self._get_label_counts(custom_loader)

        if remainder_samples > 0:
            rem_label_counts, _ = self._get_label_counts(remain_loader)
        else:
            rem_label_counts = {0: 0, 1: 0, -1: 0}

        print("\n\n[1 sample = (n_nodes, n_timesteps (window_length), n_dims)]")
        print(45*'-')
        print(f"Total samples: {total_samples}", f"\nDesired samples: {n_des}/{desired_samples} [OK={des_label_counts[0]}, NOK={des_label_counts[1]}, UK={des_label_counts[-1]}],\nRemainder samples: {remainder_samples} [OK={rem_label_counts[0]}, NOK={rem_label_counts[1]}, UK={rem_label_counts[-1]}]")
        
        # print loader statistics
        self.print_loader_stats(custom_loader, "custom")

        print('\n' + 75*'-')

        return (custom_loader, data_stats)
    
    def get_training_data_package(self, data_config:DataConfig, train_rt=0.8, test_rt=0.2, val_rt=0, batch_size=10, num_workers=1):
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

        if self.data_config.run_type == 'train':
            print(f"\n'{self.data_config.run_type.capitalize()}' type dataset selected:")
        else:
            raise ValueError(f"'{self.data_config.run_type}' is selected for training. Please set run_type to 'train' in the data config.")
        
        data_select_text = self.get_data_selection_text()        
        print("\n\n" + data_select_text)

        # load the dataset
        dataset = self._load_dataset()

        # split the dataset into train, validation, and test sets
        total_samples = len(dataset)

        # error checks
        if train_rt + test_rt + val_rt > 1:
            raise ValueError("The sum of train, test, and validation ratios must not exceed 1.")
        
        if self.package == 'topology_estimation' and val_rt == 0:
            raise ValueError("Validation set is required for topology estimation. Please provide a non-zero validation ratio.")
        
        train_total = int(train_rt * total_samples)
        test_total = int(test_rt * total_samples)
        val_total = int(val_rt * total_samples)
        remainder_samples = total_samples - train_total - test_total - val_total

        if self.package == 'topology_estimation':
            shuffle_data_loader = True

            if train_total + test_total + val_total < total_samples:
                train_set, test_set, val_set, remain_dataset = random_split(dataset, [train_total, test_total, val_total, remainder_samples])
            else:
                train_set, test_set, val_set = random_split(dataset, [train_total, test_total, val_total])

        elif self.package == 'fault_detection':
            shuffle_data_loader = False

            if train_total + test_total < total_samples:
                train_set, test_set, remain_dataset = random_split(dataset, [train_total, test_total, remainder_samples])
            else:
                train_set, test_set = random_split(dataset, [train_total, test_total])

            val_set = None  # No validation set for fault detection

        # get dataset statistics
        train_data_stats = self._get_dataset_stats(train_set)
        test_data_stats = self._get_dataset_stats(test_set)
        val_data_stats = self._get_dataset_stats(val_set) if val_set is not None else None

        # create dataloaders
        
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle_data_loader, drop_last=True, num_workers=num_workers, persistent_workers=True)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=num_workers, persistent_workers=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=num_workers, persistent_workers=True) if val_set is not None else None
        remain_loader = DataLoader(remain_dataset, batch_size=1, shuffle=False, drop_last=False) if remainder_samples > 0 else None

        # get number of OK and NOK samples in each set
        train_label_counts, n_train = self._get_label_counts(train_loader)
        test_label_counts, n_test = self._get_label_counts(test_loader)
        if val_loader is not None:
            val_label_counts, n_val = self._get_label_counts(val_loader) 
        else: 
            val_label_counts, n_val = {0: 0, 1: 0, -1: 0}, 0

        if remainder_samples > 0:
            rem_label_counts, _ = self._get_label_counts(remain_loader)
        else:
            rem_label_counts = {0: 0, 1: 0, -1: 0}

        print("\n\n[1 sample = (n_nodes, n_timesteps (window_length), n_dims)]")
        print(60*'-')
        print(f"Total samples: {total_samples}", f"\nTrain: {n_train}/{train_total} [OK={train_label_counts[0]}, NOK={train_label_counts[1]}, UK={train_label_counts[-1]}], Test: {n_test}/{test_total} [OK={test_label_counts[0]}, NOK={test_label_counts[1]}, UK={test_label_counts[-1]}], Val: {n_val}/{val_total} [OK={val_label_counts[0]}, NOK={val_label_counts[1]}, UK={val_label_counts[-1]}],\nRemainder: {remainder_samples} [OK={rem_label_counts[0]}, NOK={rem_label_counts[1]}, UK={rem_label_counts[-1]}]")

        # print loader statistics
        self.print_loader_stats(train_loader, "train")
        self.print_loader_stats(test_loader, "test")
        if val_loader is not None:
            self.print_loader_stats(val_loader, "val")
        
        print('\n' + 75*'-')

        return (train_loader, train_data_stats), (test_loader, test_data_stats), (val_loader, val_data_stats)
        
    def print_loader_stats(self, loader, type):
        dataiter = iter(loader)
        data = next(dataiter)
        shape_str = " => (batch_size, n_nodes, n_timesteps, n_dims)" if type in ['train', 'custom'] else ""

        print(f"\n{type}_data_loader statistics:")
        print(f"Number of batches: {len(loader)}")
        print(data[0].shape, f"{shape_str}")

    def _make_tp_dataset(self, x_node, y_edge, y_node, y_rep):
        """
        Create a TensorDataset for topology estimation (tp) from the provided data and labels.

        Parameters
        ----------
        x_node : dict
            Dictionary containing tensor node data for each dataset type.
        y_edge : dict
            Dictionary containing tensor edge labels for each dataset type.
        y_node : dict
            Dictionary containing tensor node labels for each dataset type (to count OK and NOK samples).
        y_rep : dict
            Dictionary containing rep labels for each dataset type.
        
        Returns
        -------
        TensorDataset
            A dataset containing the (concatenated) data and labels.
        """
        x_node_list, y_edge_list, y_node_list, y_rep_list = [], [], [], []

        for ds_type in x_node.keys():
            if x_node[ds_type] is not None and y_edge[ds_type] is not None and y_node[ds_type] is not None and y_rep[ds_type] is not None:
                x_node_list.append(x_node[ds_type])
                y_edge_list.append(y_edge[ds_type])
                y_node_list.append(y_node[ds_type])
                y_rep_list.append(y_rep[ds_type])

        if not x_node_list or not y_edge_list or not y_node_list or not y_rep_list:
            raise ValueError("No data available to create datasets.")

        x_node_all = torch.cat(x_node_list, dim=0)
        y_edge_all = torch.cat(y_edge_list, dim=0)
        y_node_all = torch.cat(y_node_list, dim=0)
        y_rep_all = torch.cat(y_rep_list, dim=0)

        return TensorDataset(x_node_all, y_edge_all, y_node_all, y_rep_all)

    def _make_fdet_dataset(self, x_node, y_node, y_rep):
        """
        Create a TensorDataset for topology estimation (tp) from the provided data and labels.

        Parameters
        ----------
        x_node : dict
            Dictionary containing tensor node data for each dataset type.
        y_node : dict
            Dictionary containing tensor node labels for each dataset type.
        y_rep : dict
            Dictionary containing rep labels for each dataset type

        Returns
        -------
        TensorDataset
            A dataset containing the (concatenated) data and labels.
        """
        x_node_list, y_node_list, y_rep_list = [], [], []

        for ds_type in x_node.keys():
            if x_node[ds_type] is not None and y_node[ds_type] is not None and y_rep[ds_type] is not None:
                x_node_list.append(x_node[ds_type])
                y_node_list.append(y_node[ds_type])
                y_rep_list.append(y_rep[ds_type])

        if not x_node_list or not y_node_list or not y_rep_list:
            raise ValueError("No data available to create datasets.")
        
        x_node_all = torch.cat(x_node_list, dim=0)
        y_node_all = torch.cat(y_node_list, dim=0)
        y_rep_all = torch.cat(y_rep_list, dim=0)


        return TensorDataset(x_node_all, y_node_all, y_rep_all)

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
        node_dim_collect, rep_num_collect = self._process_node_data(node_type_map, ds_type)

        # step 2: flatten edge matrices per ds_subtype
        ds_subtype_edge_map = self._process_edge_data(ds_subtype_map)

        # step 3: prepare node and edge data
        all_ds_subtype_node_blocks = []  # each: (n_samples, n_nodes, t, d)
        all_ds_subtype_edges = []        # each: (n_samples, n_edges)
        all_ds_subtype_rep_labels = []   # each: (n_samples,)

        for ds_subtype in ds_subtype_map.keys():
            node_tensors = []
            rep_tensors = []
            min_samples_per_node = np.inf

            for node_idx, node_type in enumerate(node_type_map.keys()):
                dim_segment_arrays = []
                dim_rep_arrays = []

                for dim_idx in node_dim_collect[node_type][ds_subtype].keys():
                    all_segments = np.concatenate(node_dim_collect[node_type][ds_subtype][dim_idx], axis=0)
                    all_reps = np.concatenate(rep_num_collect[node_type][ds_subtype][dim_idx], axis=0)
                    dim_segment_arrays.append(all_segments)
                    dim_rep_arrays.append(all_reps)

                # truncate to min_segments across dims
                try:
                    min_segments = min(arr.shape[0] for arr in dim_segment_arrays)
                except ValueError:
                    print(f"ds_subtype '{ds_subtype}' is missing data")
                    
                trimmed_segments = [arr[:min_segments] for arr in dim_segment_arrays]
                trimmed_segments_reps = [arr[:min_segments] for arr in dim_rep_arrays]

                # concatenate dims → (min_segments, t, n_dims)
                node_tensor = np.concatenate(trimmed_segments, axis=-1)
                rep_tensor = trimmed_segments_reps[0] # taking 1st dim, assuming all dims have same rep numbers

                # update min_samples_per_node and track the corresponding node_type
                if node_tensor.shape[0] < min_samples_per_node:
                    min_samples_per_node = node_tensor.shape[0]
                    min_samples_node_idx = node_idx

                trimmed_node_tensor = node_tensor[:min_samples_per_node] # so that each sample has access to all nodes
                trimmed_rep_tensor = rep_tensor[:min_samples_per_node]

                node_tensors.append(trimmed_node_tensor)
                rep_tensors.append(trimmed_rep_tensor)

            # Now we have list of (min_segments, t, n_dims) for each node
            # stack nodes → (min_segments, n_nodes, t, n_dims)
            node_block = np.stack(node_tensors, axis=1)
            all_ds_subtype_node_blocks.append(node_block)

            # pick rep labels from the node that determined min_samples_per_node
            rep_block = rep_tensors[min_samples_node_idx]
            all_ds_subtype_rep_labels.append(rep_block)
            
            # copy flat edge per segment
            edge_vec = ds_subtype_edge_map[ds_subtype]
            edge_block = np.tile(edge_vec, (min_segments, 1))  # (min_segments, n_edges)
            all_ds_subtype_edges.append(edge_block)

            # collect rep labels from the node that determined min_samples_per_node
            # rep_labels = np.array(rep_num_collect[min_samples_node_type][ds_subtype][:min_samples_per_node])
            # all_ds_subtype_rep_labels.append(rep_labels)

        # step 4: Final concatenation across ds_subtypes
        final_node_data_np = np.concatenate(all_ds_subtype_node_blocks, axis=0)  # (n_samples, n_nodes, t, d)
        final_edge_data_np = np.concatenate(all_ds_subtype_edges, axis=0)        # (n_samples, n_edges)
        final_rep_labels_np = np.concatenate(all_ds_subtype_rep_labels, axis=0)  # (n_samples,)
        
        if ds_type == 'OK':
            final_node_labels_np = np.zeros((final_node_data_np.shape[0], 1), dtype=np.float32)
        elif ds_type == 'NOK':
            final_node_labels_np = np.ones((final_node_data_np.shape[0], 1), dtype=np.float32)
        elif ds_type == 'UK':
            final_node_labels_np = (-1) * np.ones((final_node_data_np.shape[0], 1), dtype=np.float32)

        # convert to torch tensors
        final_node_data = torch.from_numpy(final_node_data_np).to(torch.float32)
        final_node_labels = torch.from_numpy(final_node_labels_np).to(torch.float32)
        final_edge_data = torch.from_numpy(final_edge_data_np).to(torch.float32)
        final_rep_labels = torch.from_numpy(final_rep_labels_np)

        return final_node_data, final_node_labels, final_edge_data, final_rep_labels


    def _process_node_data(self, node_type_map, ds_type):
        # node_type -> ds_subtype -> dim -> segments
        node_dim_collect = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        rep_num_collect = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # to store rep numbers for each node and subtype
        fs_matrix = [] # to store fs values for each node

        # get reference values for timesteps
        if not self.data_config.use_custom_max_timesteps:
            self.data_config.max_timesteps = self._get_max_timesteps(node_type_map)

        # process node data
        for node_type, ds_subtype_map in node_type_map.items():

            for ds_subtype_idx, (ds_subtype, signal_type_paths) in enumerate(ds_subtype_map.items()):
                node_fs_list = []  # to store fs values for all dimensions of the current node

                for dim_idx, hdf5_path in enumerate(signal_type_paths):

                    # load node data
                    with h5py.File(hdf5_path, 'r') as f:
                        data_list = [f[key][:] for key in f.keys() if key.startswith('data')]
                        time_list = [f[key][:] for key in f.keys() if key.startswith('time')]
                        
                        try:
                            is_default_rep = False
                            rep_num_list = [int(key.split('_')[-1].removeprefix('rep')) for key in f.keys() if key.startswith('data')]

                        except ValueError:
                            is_default_rep = True
                            rep_num_list = [i + 1 for i in range(len(data_list))]

                        data = np.concatenate(data_list, axis=0)  # shape: (n_reps, n_timesteps)
                        time = np.concatenate(time_list, axis=0) if len(time_list) > 0 else np.array([]) # shape: (n_reps, n_timesteps)

                    if time.size != 0:
                        # interpolate data with lesser timesteps to match global max_timesteps (if time is available)
                        if data.shape[1] < self.data_config.max_timesteps or data.shape[1] > self.data_config.max_timesteps:
                            data, time = self._interpolate_data(data, time, self.data_config.max_timesteps)
                            is_interpolated = True

                        # calcualte fs
                        fs = 1 / np.mean(np.diff(time, axis=1), axis=1)  # shape: (n_reps,)
                        node_fs_list.append(fs[0])  # assumes fs is consistent across reps for a dimension
                    else:
                        fs = [self.data_config.fs[0, 0]]
      
                    # Apply augmentations
                    if ds_type == 'OK':
                       data = self.add_augmentations(data, self.data_config.healthy_configs[ds_subtype], fs[0], ds_subtype)   

                    elif ds_type == 'NOK':
                       data = self.add_augmentations(data, self.data_config.unhealthy_configs[ds_subtype], fs[0], ds_subtype)

                    elif ds_type == 'UK':
                       data = self.add_augmentations(data, self.data_config.unknown_configs[ds_subtype], fs[0], ds_subtype)

                    # segment the data                              
                    data_segments = segment_data(data, self.data_config.window_length, self.data_config.stride)
                    data_segments = np.expand_dims(data_segments, axis=-1)  # (n_segments, n_timesteps, 1)

                    # generate segmented rep numbers
                    segmented_rep_nums = []
                    for rep_num in rep_num_list:
                        for seg_num in range(len(data_segments) // len(rep_num_list)): # iterate over number of segments per rep
                            segmented_rep_nums.append(float(f"{rep_num:02d}{ds_subtype_idx+1:03d}.{seg_num+1:03d}"))
                            
                    # store segments and rep numbers
                    node_dim_collect[node_type][ds_subtype][dim_idx].append(data_segments)
                    rep_num_collect[node_type][ds_subtype][dim_idx].append(segmented_rep_nums)

            # append the fs values for the current node to the fs_matrix
            fs_matrix.append(node_fs_list)

        # convert fs_matrix to numpy array of shape (n_nodes, n_dims)
        fs_matrix = np.array(fs_matrix, dtype=np.float32)

        # verbose output
        if ds_type == 'OK' or ds_type == 'UK':
            print(f"\nFor ds_type '{ds_type}' and others....")
            print(45*'-')
            print(f"Maximum timesteps across all node types: {self.data_config.max_timesteps:,}")

            if self.__dict__.get('is_interpolated', False):
                print(f"\nData interpolation applied to match max_timesteps for node types with lesser timesteps.")
            else:
                print(f"\nNo data interpolation applied.")
            
            # save fs values to data_config for only OK type data
            if fs_matrix.size != 0:
                self.data_config.fs = fs_matrix
                print(f"\n'fs' is updated in data_config as given in loaded healthy (or unknown) data.\nNew fs:")
                print(np.array2string(fs_matrix, separator=', '))
            else:
                print("\nNo 'fs_matrix' recieved from the data. Hence, using the currently set 'fs' in data_config. Current fs:")
                print(np.array2string(self.data_config.fs, separator=', '))

            # print if rep number is default or not
            if is_default_rep:
                print(f"\nNo exclusive rep numbers found in keys of hfd5 file. Hence, using default rep numbers.")
            else:
                print(f"\nExclusive rep numbers found in keys of hdf5 file. Hence, using them as rep numbers.")

        return node_dim_collect, rep_num_collect
    
    def _interpolate_data(self, data, time, target_timesteps):
        """
        Interpolate data to match the target number of timesteps.

        Parameters
        ----------
        data : np.ndarray, shape (n_reps, n_timesteps)
        time : np.ndarray, shape (n_reps, n_timesteps)
        target_timesteps : int
            The desired number of timesteps.

        Returns
        -------
        interp_data : np.ndarray, shape (n_reps, target_timesteps)
        new_time_data : np.ndarray, shape (n_reps, target_timesteps)
        """
        n_reps, _ = data.shape
        interp_data = []
        new_time_list = []

        for rep in range(n_reps):
            interp_func = interp1d(time[rep], data[rep], kind='linear', fill_value="extrapolate")
            new_time = np.linspace(time[rep, 0], time[rep, -1], target_timesteps)
            interp_data.append(interp_func(new_time))
            new_time_list.append(new_time)
        
        return np.array(interp_data), np.array(new_time_list)
    
    def _get_max_timesteps(self, node_type_map):
        """
        Get the maximum timesteps for each node and ds_subtype.

        Parameters
        ----------
        node_type_map : dict
            A dictionary containing paths to the datasets.
        """
        # min_end_time = float('inf')
        max_timesteps = 0

        for node_type, ds_subtype_map in node_type_map.items():
            for ds_subtype, signal_type_paths in ds_subtype_map.items():
                for dim_idx, hdf5_path in enumerate(signal_type_paths):
                    with h5py.File(hdf5_path, 'r') as f:
                        data_keys = [key for key in f.keys() if key.startswith('data')]
                        # end_times = [f[key][-1] for key in time_keys]  # Get the last time value for each rep
                        # min_end_time = min(min_end_time, *end_times)

                        # Calculate max_timesteps globally
                        for key in data_keys:
                            max_timesteps = max(max_timesteps, len(f[key][:].flatten()))

        # print(f"\nMinimum end time across all ds_subtypes: {min_end_time}")

        return max_timesteps

    
    def add_augmentations(self, data, augment_configs, fs, ds_subtype="_"):
        """
        Apply augmentations to the data based on the provided configurations.

        Parameters
        ----------
        data : np.ndarray, shape (n_samples, n_timesteps)
            The input data to be augmented.
        augment_configs : list of dict
            List of augmentation configurations.
            Each dict should have a 'type' key indicating the type of augmentation and other keys for parameters.
        fs : float
            Sampling frequency of the data.
        ds_subtype : str
            The dataset subtype (for error messages).
        """
        augmented_data_list = []
        og_data = data.copy()

        if augment_configs == []:
            raise ValueError(f"No original or augmentation configs for the dataset subtype {ds_subtype} provided.")

        for idx, augment_config in enumerate(augment_configs):
            # original data
            if augment_config['type'] == 'OG':
                augmented_data = data
            # gaussian noise
            if augment_config['type'] == 'gau':
                augmented_data = add_gaussian_noise(data, augment_config['mean'], augment_config['std'])
            # sine wave (freq modulation)
            elif augment_config['type'] == 'sine':
                augmented_data = add_sine_waves(data, augment_config['freqs'], augment_config['amps'], fs)
            # gitches
            elif augment_config['type'] == 'glitch':
                augmented_data = add_glitches(data, augment_config['prob'], augment_config['amp'])

            # chain augmentations
            if augment_config['add_next'] and idx < len(augment_configs) - 1:
                data = augmented_data 
            else:
                augmented_data_list.append(augmented_data)
                data = og_data  # reset to original data for next augmentation

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

    # loc_train = np.load(f'{loc_path}\\loc_train_springs' + '5' + '.npy')
    # vel_train = np.load(f'{vel_path}\\vel_train_springs' + '5' + '.npy')
    # edges_train = np.load(f'{edge_path}\\edges_train_springs' + '5' + '.npy')

    # loc_valid = np.load(f'{loc_path}\\loc_valid_springs' + '5' + '.npy')
    # vel_valid = np.load(f'{vel_path}\\vel_valid_springs' + '5' + '.npy')
    # edges_valid = np.load(f'{edge_path}\\edges_valid_springs' + '5' + '.npy')

    # loc_test = np.load(f'{loc_path}\\loc_test_springs' + '5' + '.npy')
    # vel_test = np.load(f'{vel_path}\\vel_test_springs' + '5' + '.npy')
    # edges_test = np.load(f'{edge_path}\\edges_test_springs' + '5' + '.npy')

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