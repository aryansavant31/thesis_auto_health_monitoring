import sys, os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR) if ROOT_DIR not in sys.path else None

from typing import List, Dict
from pathlib import Path
from rich.tree import Tree
from rich.console import Console
import glob
import numpy as np

from data.datasets.asml.groups import NXEGroupMaker
from data.datasets.mass_sp_dm.groups import MSDGroupMaker

class DataConfig:
    def __init__(self, run_type='train'):
        """
        Data structure:

        `application`
        |── `machine_type`
        |   |── `scenario`

        |   |   |── (healthy/unhealthy)
        |   |   |   |── `healthy/unhealthy_type`

        |   |   |   |   |── (processed_data)
        |   |   |   |       |── (edges)

        |   |   |   |       |── (nodes)
        |   |   |   |          |── `timestep_id`
        |   |   |   |               |── `augment` (N1.25, N2.5)
        |   |   |   |                   |── `node_type`
        |   |   |   |                       |── `signal_type`

        Attributes
        ----------
        application : str
            Dataset applications available:
            - BER: Bearing dataset
            - MSD: Mass-Spring-Damper dataset (fs=1000)
            - SPP: Spring-Particles dataset
            - ASM: ASML dataset

        To view rest of the attribute options, run this file directly.
        """
        self.run_type  = run_type  # options: train, custom_test, predict
    
        self.application_map = {'BER':'bearing',
                                'MSD':'mass_sp_dm',
                                'SPP':'spring_particles',
                                'ASM':'asml',
                                'ASMT':'asml_trial'}
        
        self.application = 'ASM'
        self.machine_type = 'NXE'
        self.scenario = 'full_wafer'

        self.signal_types = NXEGroupMaker().ammf_acc
        
        self.fs = None # np.array([[48000]])    # sampling frequency matrix, set in the data.prep.py
        self.format = 'hdf5'  # options: hdf5

        # segement data
        self.window_length      = 2000
        self.stride             = 2000

        self.view = DatasetViewer(self)

        if self.run_type == 'train':
            self.set_train_dataset()
        elif self.run_type == 'custom_test':
            self.set_custom_test_dataset()
        elif self.run_type == 'predict':
            self.set_predict_dataset()
        
    def set_train_dataset(self):
        self.healthy_configs   = {
            key: [get_augment_config('OG')] for key in self.view.healthy_types if key.startswith('E1')
        }
        
        self.unhealthy_configs = {
            # '0_B-021': [get_augment_config('OG')],
            
        }

        self.unknown_configs = {
        }
    
    def set_custom_test_dataset(self):
        self.amt = 1
        self.healthy_configs   = {
            key: [get_augment_config('OG')] for key in self.view.healthy_types[:50] if key.startswith('E1')
        }
        
        self.unhealthy_configs = {
            '(sim)_E1_set01_M=mAI26': [get_augment_config('glitch', prob=0.1, amps=1)]
        }

        self.unknown_configs = {
            # '1_N': [get_augment_config('OG')],
        }
        
    def set_predict_dataset(self):
        self.amt = 0.8
        self.healthy_configs   = {
            '0_N': [get_augment_config('OG')],
            '1_N': [get_augment_config('OG')],
        }
        
        self.unhealthy_configs = {
            '0_B-021': [get_augment_config('OG')],
            # '0_B-007': [get_augment_config('OG')],
            # '0_IR-007': [get_augment_config('OG')],
            # '0_IR-021': [get_augment_config('OG')],
        }

        self.unknown_configs = {
        }
    
    def _process_ds_addresses(self, config:dict, ds_type):
        """
        Process the dataset address list

        Parameters
        ----------
        config : dict
            Configuration dictionary containing dataset types and their augmentations.
        ds_type : str
            Type of dataset to be processed,
            Options: 'healthy' or 'unhealthy'

        Returns
        -------
        node_ds_paths : list
        edge_ds_paths : list
        """
        node_ds_paths = {}
        edge_ds_paths = {}

        if config != {}:
            # Build edge path
            for ds_subtype, _ in config.items():     
                edge_path = os.path.join(
                    self.main_ds_path, ds_type, ds_subtype, 'processed', 'edges'
                )
                # find .hdf5 file in the signal_type folder
                edge_hdf5_files = glob.glob(os.path.join(edge_path, f"*.{self.format}"))[0]
                edge_ds_paths[ds_subtype] = edge_hdf5_files
                
            # Build node paths
            for node_type, signal_types in self.signal_types['group'].items():
                node_ds_paths[node_type] = {}

                for ds_subtype, _ in config.items():
                    node_ds_paths[node_type][ds_subtype] = []

                    for signal_type in signal_types:
                        node_path = os.path.join(
                            self.main_ds_path, ds_type, ds_subtype, 'processed', 
                            'nodes', node_type, signal_type
                        )
                        # find .hdf5 file in the signal_type folder
                        node_hdf5_files = glob.glob(os.path.join(node_path, f"*.{self.format}"))
                        # append each file path to node_ds_paths
                        node_ds_paths[node_type][ds_subtype].extend(node_hdf5_files)
        else:
            node_ds_paths = None
            edge_ds_paths = None

        return node_ds_paths, edge_ds_paths

    def get_dataset_paths(self):
        """
        Set the dataset path for the processed data based on the requirements

        Returns
        -------
        node_ds_path : dict
        edge_ds_path : dict
            Dictionary containing paths of PROCCESSED healthy and unhealthy node datasets
            Each key has a list of all paths that are selected by user to load.
        """
        # set main dataset path
        base_dir = os.path.dirname(os.path.abspath(__file__))  # directory of config.py
        self.application_full = self.application_map[self.application]

        self.main_ds_path = os.path.join(
            base_dir, 'datasets', self.application_full, self.machine_type, self.scenario)
        
        if os.path.exists(self.main_ds_path) is False:
            raise FileNotFoundError(f"Dataset path does not exist: {self.main_ds_path}. Please check the dataset configuration.")
        
        # initialize dictionaries
        node_ds_path_main = {}
        edge_ds_path_main = {}
        
        # get actual node types to iterate over
        
        # self.node_options = self.view.node_types  if self.node_type == ['ALL'] else self.node_type

        
        # Process healthy and unhealthy dataset addresses
        node_ds_path_main['OK'], edge_ds_path_main['OK'] = self._process_ds_addresses(self.healthy_configs, 'healthy')
        node_ds_path_main['NOK'], edge_ds_path_main['NOK'] = self._process_ds_addresses(self.unhealthy_configs, 'unhealthy')                                                                 
        node_ds_path_main['UK'], edge_ds_path_main['UK'] = self._process_ds_addresses(self.unknown_configs, 'unknown')
        
        return node_ds_path_main, edge_ds_path_main
    
def get_augment_config(augment_type, **kwargs):
        """
        Get the configuration for a specific augmentation type.

        Parameters
        ----------
        augment_type : str
            Type of augmentation to get configuration for.

        **kwargs : dict
            For all `augment_type`, the following parameters are available:
            - `OG` (Original data): No additional parameters
            - `gau` (Gaussian noise): **mean**, **std**
            - `sine` (Sine wave): **freqs** (_list_), **amps** (_list_)
            - `glitch` (Random glitches): **prob**, **amp**

        """
        config = {}
        config['type'] = augment_type

        if augment_type == 'gau':
            config['mean'] = kwargs.get('mean', 0.0)
            config['std'] = kwargs.get('std', 0.1)
        elif augment_type == 'sine':
            config['freqs'] = kwargs.get('freqs', 10.0)
            config['amps'] = kwargs.get('amps', 5.0)
        elif augment_type == 'glitch':
            config['prob'] = kwargs.get('prob', 0.01)
            config['amp'] = kwargs.get('amp', 5.0)

        config['add_next'] = kwargs.get('add_next', False)  # whether to add the next augmentation to the current one
        
        return config

def get_domain_config(domain_type, **kwargs):
    """
    Get the domain configuration based on the specified domain.

    Parameters
    ----------
    domain_type : str
        The domain of the data (e.g., 'time', 'freq').

    **kwargs : dict
        Additional parameters for the domain configuration.
        - `time`: **cutoff_freq** (for high pass filter)
        - `freq`: **cutoff_freq** (for high pass filter)
    """
    config = {}
    config['type'] = domain_type

    if domain_type == 'time':
        config['cutoff_freq'] = kwargs.get('cutoff_freq', 0)  # default cutoff frequency for time domain
        
    elif domain_type == 'freq':
        config['cutoff_freq'] = kwargs.get('cutoff_freq', 100)  # default cutoff frequency
       
    return config 

# =====================================================
# Helper class to view dataset structure
# =====================================================

class DatasetViewer:
    """
    A class to view hierarchical dataset structure and store folder names.
    """
    def __init__(self, data_config:DataConfig):
        self.base_dir = Path(os.path.dirname(os.path.abspath(__file__))) 
        self.base_path = "datasets"
        self.application_full = data_config.application_map[data_config.application]
        
        # Initialize lists for folder names
        self.machine_types: List[str] = []
        self.scenarios: List[str] = []
        self.healthy_types: List[str] = []
        self.unhealthy_types: List[str] = []
        self.unknown_types: List[str] = []
        self.node_types: List[str] = []
        self.signal_types: List[str] = []

        self.view_dataset()
    
    def view_dataset(self) -> Dict:
        """
        View dataset structure for the configured application and populate folder name lists.
        
        Returns:
            Dict: Nested dictionary of the dataset structure for the configured application
        """
        dataset_path = self.base_dir / self.base_path
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
        
        # Look for the specific application folder
        app_path = dataset_path / self.application_full
        if not app_path.exists():
            raise FileNotFoundError(f"Application path does not exist: {app_path}")
        
        self.structure = {}
        
        # Explore machine types within the configured application
        for machine_folder in sorted(app_path.iterdir(), key=lambda x: x.name):
            if machine_folder.is_dir():
                self.machine_types.append(machine_folder.name)
                self.structure[machine_folder.name] = {}
                
                # Explore scenarios within each machine type
                for scenario_folder in sorted(machine_folder.iterdir(), key=lambda x: x.name):
                    if scenario_folder.is_dir():
                        self.scenarios.append(scenario_folder.name)
                        self.structure[machine_folder.name][scenario_folder.name] = {}
                        
                        # Explore healthy and unhealthy folders within each scenario
                        scenario_structure = self._explore_scenario(scenario_folder)
                        self.structure[machine_folder.name][scenario_folder.name] = scenario_structure
        
        # Remove duplicates
        self.machine_types = list(set(self.machine_types))
        self.scenarios = list(set(self.scenarios))
        self.healthy_types = list(set(self.healthy_types))
        self.unhealthy_types = list(set(self.unhealthy_types))
        self.unknown_types = list(set(self.unknown_types))
        self.node_types = list(set(self.node_types))
        self.signal_types = list(set(self.signal_types))

        # sort node types by their numeric prefix
        # self.node_types = sorted(set(self.node_types), key=lambda x: int(x.split('_')[0]) if x.split('_')[0].isdigit() else (_ for _ in ()).throw(ValueError(f"Invalid node type format: {x}. Node types should start with a number.")))
        # sort ds subtypes by their numeric prefix
        self.healthy_types = sorted(set(self.healthy_types))
        self.unhealthy_types = sorted(set(self.unhealthy_types))
        self.unknown_types = sorted(set(self.unknown_types))


    def _explore_scenario(self, scenario_path: Path) -> Dict:
        """
        Explore scenario folder structure (healthy/unhealthy).
        
        Args:
            scenario_path (Path): Path to scenario folder
            
        Returns:
            Dict: Structure of scenario folder
        """
        structure = {}
        
        # Explore healthy folder
        healthy_path = scenario_path / "healthy"
        if healthy_path.exists():
            structure["healthy"] = {}
            for health_type in sorted(healthy_path.iterdir(), key=lambda x: x.name):
                if health_type.is_dir():
                    self.healthy_types.append(health_type.name)
                    structure["healthy"][health_type.name] = self._explore_processed_data(health_type)
        
        # Explore unhealthy folder
        unhealthy_path = scenario_path / "unhealthy"
        if unhealthy_path.exists():
            structure["unhealthy"] = {}
            for health_type in sorted(unhealthy_path.iterdir(), key=lambda x: x.name):
                if health_type.is_dir():
                    self.unhealthy_types.append(health_type.name)
                    structure["unhealthy"][health_type.name] = self._explore_processed_data(health_type)

        # Explore unknown folder
        unknown_path = scenario_path / "unknown"
        if unknown_path.exists():
            structure["unknown"] = {}
            for health_type in sorted(unknown_path.iterdir(), key=lambda x: x.name):
                if health_type.is_dir():
                    self.unhealthy_types.append(health_type.name)
                    structure["unknown"][health_type.name] = self._explore_processed_data(health_type)
        
        return structure

    def _explore_processed_data(self, health_type_path: Path) -> Dict:
        """
        Explore processed_data folder structure.
        
        Args:
            health_type_path (Path): Path to healthy/unhealthy type folder
            
        Returns:
            Dict: Structure of processed_data folder
        """
        processed_data_path = health_type_path / "processed"
        if not processed_data_path.exists():
            return {}
        
        structure = {}
        
        # Check edges
        edges_path = processed_data_path / "edges"
        if edges_path.exists():
            structure["edges"] = True
        
        # Check nodes
        nodes_path = processed_data_path / "nodes"
        if nodes_path.exists():
            structure["nodes"] = {}
            
            # # Get timesteps
            # for timestep in nodes_path.iterdir():
            #     if timestep.is_dir():
            #         self.ds_timesteps.append(timestep.name)
            #         structure["nodes"][timestep.name] = {}
                    
            #         # Get augmentations
            #         for augment in timestep.iterdir():
            #             if augment.is_dir():
            #                 self.augment_types.append(augment.name)
            #                 structure["nodes"][timestep.name][augment.name] = {}
                            
            # Get node types
            for node_type in sorted(nodes_path.iterdir(), key=lambda x: int(x.name.split('_')[0]) if x.name.split('_')[0].isdigit() else x.name):
                if node_type.is_dir():
                    self.node_types.append(node_type.name)
                    structure["nodes"][node_type.name] = []
                    
                    # Get signal types
                    for signal_type in sorted(node_type.iterdir(), key=lambda x: x.name):
                        if signal_type.is_dir():
                            self.signal_types.append(signal_type.name)
                            structure["nodes"][node_type.name].append(signal_type.name)
                            
                            # # get all files in the signal type folder
                            # for file_type in signal_type.iterdir():
                            #     self.file_types.append(file_type.name)
                            #     structure["nodes"][node_type.name].append
        
        return structure
    
    def get_node_path(self, application: str, machine_type: str, scenario: str, 
                     ds_type: str, ds_subtype: str,
                     node_type: str, signal_type: str) -> Path:
        """
        Get full path to a specific node dataset.
        """
        return (self.base_dir / self.base_path / application / machine_type / scenario /
                ds_type / ds_subtype / "processed" / "nodes" /
                node_type / signal_type)
    
    def get_edges_path(self, application: str, machine_type: str, scenario: str,
                      ds_type: str, ds_subtype: str) -> Path:
        """
        Get full path to edges dataset.
        """
        return (self.base_dir / self.base_path / application / machine_type / scenario /
                ds_type / ds_subtype / "processed" / "edges")
    
    def print_rich_tree(self) -> None:
        """
        Print dataset structure for the configured application using rich library.
        """
        console = Console()  
        tree = Tree(f"[green]{self.application_full}[/green]")
        self._build_rich_tree(tree, self.structure)
        console.print(tree)

    def _build_rich_tree(self, parent_node, structure, level=0):
        """Helper method to build rich tree structure."""
        # Define folder names that should be white
        white_folders = {'healthy', 'unhealthy', 'unknown', 'nodes', 'edges', 'processed'}
        
        # Track which type labels have been added at this level
        added_labels = set()
        
        for key, value in structure.items():
            if isinstance(value, dict):
                # Add type label before the folder name (only once per type)
                if key in self.machine_types and 'machine_type' not in added_labels:
                    parent_node.add(f"[blue]<machine_type>[/blue]")
                    added_labels.add('machine_type')
                if key in self.scenarios and 'scenario' not in added_labels:
                    parent_node.add(f"[blue]<scenario>[/blue]")
                    added_labels.add('scenario')
                if key in self.healthy_types and 'healthy_type' not in added_labels:
                    parent_node.add(f"[blue]<healthy_type>[/blue]")
                    added_labels.add('healthy_type')
                if key in self.unhealthy_types and 'unhealthy_type' not in added_labels:
                    parent_node.add(f"[blue]<unhealthy_type>[/blue]")
                    added_labels.add('unhealthy_type')
                if key in self.unknown_types and 'unknown_type' not in added_labels:
                    parent_node.add(f"[blue]<unknown_type>[/blue]")
                    added_labels.add('unknown_type')
                if key in self.node_types and 'node_type' not in added_labels:
                    parent_node.add(f"[blue]<node_type>[/blue]")
                    added_labels.add('node_type')
                
                # Add the actual folder
                if key in white_folders:
                    branch = parent_node.add(f"[white]{key}[/white]")
                else:
                    branch = parent_node.add(f"[bright_yellow]{key}[/bright_yellow]")
                
                self._build_rich_tree(branch, value, level + 1)
            elif isinstance(value, list):
                # Add type label for signal types
                if key in self.node_types and 'node_type' not in added_labels:
                    parent_node.add(f"[blue]<node_type>[/blue]")
                    added_labels.add('node_type')
                
                # Add the node type folder
                if key in white_folders:
                    branch = parent_node.add(f"[white]{key}[/white]")
                else:
                    branch = parent_node.add(f"[bright_yellow]{key}[/bright_yellow]")
                
                # Add signal type label and items
                signal_label_added = False
                for item in value:
                    if item in self.signal_types and not signal_label_added:
                        branch.add(f"[blue]<signal_type>[/blue]")
                        signal_label_added = True
                    if item in self.signal_types:
                        branch.add(f"[bright_yellow]{item}[/bright_yellow]")
                    else:
                        branch.add(f"[bright_yellow]{item}[/bright_yellow]")
            else:
                # Single items
                if key in white_folders:
                    parent_node.add(f"[white]{key}[/white]")
                else:
                    parent_node.add(f"[bright_yellow]{key}[/bright_yellow]")

    def view_dataset_tree(self) -> None:
        """
        View and print dataset structure for the configured application in tree format.
        """
        print(f"Dataset Structure for Application: {self.application_full}")
        print("=" * 60)
        self.print_rich_tree()
        
    
if __name__ == "__main__":
    # data_config.set_train_valid_dataset()
    data_config = DataConfig()
    data_viewer = DatasetViewer(data_config)

    data_viewer.view_dataset_tree()

    # a, b =data_config.get_dataset_path()
    # print(a ,b)

    # data_config.set_test_dataset()
    # a, b = data_config.get_dataset_path()
    # print(a, b)


