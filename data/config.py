import os
from typing import List, Dict
from pathlib import Path
from rich.tree import Tree
from rich.console import Console

class DataConfig:
    def __init__(self, ):
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
            - MSD: Mass-Spring-Damper dataset
            - SPP: Spring-Particles dataset
            - ASM: ASML dataset

        To view rest of the attribute options, run this file directly.
        """
        self.application_map = {'BER':'bearing_cwru',
                                'MSD':'mass_spring_damper',
                                'SPP':'spring_particles',
                                'ASM':'ASML'}
        
        self.application    = 'SPP'

        self.machine_type   = 'P005'
        self.scenario       = 'scenario_1'
        self.timestep_id    = 'T50'  
        self.node_type      = 'all_nodes'           # options ALL or the specific node type
        self.signal_types   = ['vel', 'pos'] # in hdf5 format
        self.custom_test_ds = False
        
    def set_train_dataset(self):
        self.healthy_config   = [['H1', ['OG']], # N = Noise
        ]
        
        self.unhealthy_config = [       
        ]
    
    def set_custom_test_dataset(self):
        self.custom_test_ratio = 0.6
        self.healthy_config   = [
        ]
        
        self.unhealthy_config = [
        ]
        
    def set_predict_dataset(self):
        self.healthy_config   = [
        ]
        
        self.unhealthy_config = [
        ]

    def _process_ds_addresses(self, config, ds_type):
        """
        Process the dataset address list

        Parameters
        ----------
        config : 2D list
            Configuration list containing dataset types and their augmentations.
        ds_type : str
            Type of dataset to be processed,
            Options: 'healthy' or 'unhealthy'

        Returns
        -------
        node_ds_paths : list
        edge_ds_paths : list
        """
        node_ds_paths = []
        edge_ds_paths = []

        if config != []:
            for data_level in config:
                ds_subtype = data_level[0]  # e.g., 'H1'
                augments = data_level[1]     # e.g., ['N1.25', 'N2.5']

                # Build edge path
                edge_path = os.path.join(
                    self.main_ds_path, ds_type, ds_subtype, 'processed_data', 'edges'
                )
                edge_ds_paths.append(edge_path)
                
                # Build node paths
                for augment in augments:
                    for node_type in self.node_options:
                        for signal_type in self.signal_types:
                            node_path = os.path.join(
                                self.main_ds_path, ds_type, ds_subtype, 'processed_data', 
                                'nodes', self.timestep_id, augment, node_type, signal_type
                            )
                            node_ds_paths.append(node_path)
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
        node_ds_path_main = {'H': [], 'UH': []}
        edge_ds_path_main = {'H': [], 'UH': []}
        
        # get actual node types to iterate over
        self.view = DatasetViewer()
        self.node_options = self.view.node_types  if self.node_type == 'ALL' else [self.node_type]
        
        # Process healthy and unhealthy dataset addresses
        node_ds_path_main['H'], edge_ds_path_main['H'] = self._process_ds_addresses(self.healthy_config, 'healthy')
        node_ds_path_main['UH'], edge_ds_path_main['UH'] = self._process_ds_addresses(self.unhealthy_config, 'unhealthy')                                                                 
        
        
        return node_ds_path_main, edge_ds_path_main

class DatasetViewer(DataConfig):
    """
    A class to view hierarchical dataset structure and store folder names.
    """
    def __init__(self):
        super().__init__()
        self.base_dir = Path(os.path.dirname(os.path.abspath(__file__))) 
        self.base_path = "datasets"
        self.application_full = self.application_map[self.application]
        
        # Initialize lists for folder names
        self.machine_types: List[str] = []
        self.scenarios: List[str] = []
        self.healthy_types: List[str] = []
        self.unhealthy_types: List[str] = []
        self.ds_timesteps: List[str] = []
        self.augment_types: List[str] = []
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
        for machine_folder in app_path.iterdir():
            if machine_folder.is_dir():
                self.machine_types.append(machine_folder.name)
                self.structure[machine_folder.name] = {}
                
                # Explore scenarios within each machine type
                for scenario_folder in machine_folder.iterdir():
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
        self.ds_timesteps = list(set(self.ds_timesteps))
        self.augment_types = list(set(self.augment_types))
        self.node_types = list(set(self.node_types))
        self.signal_types = list(set(self.signal_types))
    
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
            for health_type in healthy_path.iterdir():
                if health_type.is_dir():
                    self.healthy_types.append(health_type.name)
                    structure["healthy"][health_type.name] = self._explore_processed_data(health_type)
        
        # Explore unhealthy folder
        unhealthy_path = scenario_path / "unhealthy"
        if unhealthy_path.exists():
            structure["unhealthy"] = {}
            for health_type in unhealthy_path.iterdir():
                if health_type.is_dir():
                    self.unhealthy_types.append(health_type.name)
                    structure["unhealthy"][health_type.name] = self._explore_processed_data(health_type)
        
        return structure

    def _explore_processed_data(self, health_type_path: Path) -> Dict:
        """
        Explore processed_data folder structure.
        
        Args:
            health_type_path (Path): Path to healthy/unhealthy type folder
            
        Returns:
            Dict: Structure of processed_data folder
        """
        processed_data_path = health_type_path / "processed_data"
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
            
            # Get timesteps
            for timestep in nodes_path.iterdir():
                if timestep.is_dir():
                    self.ds_timesteps.append(timestep.name)
                    structure["nodes"][timestep.name] = {}
                    
                    # Get augmentations
                    for augment in timestep.iterdir():
                        if augment.is_dir():
                            self.augment_types.append(augment.name)
                            structure["nodes"][timestep.name][augment.name] = {}
                            
                            # Get node types
                            for node_type in augment.iterdir():
                                if node_type.is_dir():
                                    self.node_types.append(node_type.name)
                                    structure["nodes"][timestep.name][augment.name][node_type.name] = []
                                    
                                    # Get signal types
                                    for signal_type in node_type.iterdir():
                                        if signal_type.is_dir():
                                            self.signal_types.append(signal_type.name)
                                            structure["nodes"][timestep.name][augment.name][node_type.name].append(signal_type.name)
        
        return structure
    
    def get_node_path(self, application: str, machine_type: str, scenario: str, 
                     ds_type: str, ds_subtype: str, timestep: str, 
                     augment: str, node_type: str, signal_type: str) -> Path:
        """
        Get full path to a specific node dataset.
        """
        return (self.base_dir / self.base_path / application / machine_type / scenario /
                ds_type / ds_subtype / "processed_data" / "nodes" /
                timestep / augment / node_type / signal_type)
    
    def get_edges_path(self, application: str, machine_type: str, scenario: str,
                      ds_type: str, ds_subtype: str) -> Path:
        """
        Get full path to edges dataset.
        """
        return (self.base_dir / self.base_path / application / machine_type / scenario /
                ds_type / ds_subtype / "processed_data" / "edges")
    
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
        white_folders = {'healthy', 'unhealthy', 'nodes', 'edges', 'processed_data'}
        
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
                if key in self.ds_timesteps and 'timestep' not in added_labels:
                    parent_node.add(f"[blue]<timestep>[/blue]")
                    added_labels.add('timestep')
                if key in self.augment_types and 'augment_type' not in added_labels:
                    parent_node.add(f"[blue]<augment_type>[/blue]")
                    added_labels.add('augment_type')
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
    data_viewer = DatasetViewer()

    data_viewer.view_dataset_tree()

    # a, b =data_config.get_dataset_path()
    # print(a ,b)

    # data_config.set_test_dataset()
    # a, b = data_config.get_dataset_path()
    # print(a, b)


