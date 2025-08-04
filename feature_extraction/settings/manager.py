import os
import sys

ROOT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT_DIR) if ROOT_DIR not in sys.path else None

SETTINGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SETTINGS_DIR) if SETTINGS_DIR not in sys.path else None

FEX_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(FEX_DIR, "logs")

# other imports
import shutil
from pathlib import Path
from rich.console import Console
from rich.tree import Tree
import numpy as np

# global imports
from data.config import DataConfig

# local imports
from feature_extraction.settings.rank_config import FeatureRankingConfig


class FeatureRankingManager(FeatureRankingConfig):
    def __init__(self, data_config:DataConfig):
        super().__init__()
        self.helper = HelperClass()
        self.data_config = data_config
        self.rank_version = f"[a={self.alpha}]" # [a=0.8,r=0.5,t=7]

    def get_perf_log_path(self, check_version=True):
        """
        Returns the path for saving the feature performance logs.

        Parameters
        ----------
        data_config : DataConfig
            The data configuration object. Since the data_config attribute values can change outside, 
            it is being passed as a parameter here to provide new log paths.
        """
        self.node_type = f"({'+'.join(self.data_config.node_type)})"
        
        base_path = os.path.join(LOGS_DIR, 
                                f'{self.data_config.application_map[self.data_config.application]}', 
                                f'{self.data_config.machine_type}',
                                f'{self.data_config.scenario}')
        
        # add node type
        self.perf_path = os.path.join(base_path, f'{self.node_type}')
        
        # add perf number
        self.perf_log_path = os.path.join(self.perf_path, f"{self.node_type}_perf_{self.perf_version}")

        if check_version:
            self.check_if_perf_version_exists()

        return self.perf_log_path
    
    def get_ranking_log_path(self, is_avail=False):
        perf_log_path = self.get_perf_log_path(check_version=False)

        if not os.path.exists(perf_log_path):
            raise FileNotFoundError(f"Performance log path {perf_log_path} does not exist. Please run the feature performance ranking first or type the correct version.")

        self.ranking_log_path = os.path.join(perf_log_path, f'ranks_{self.rank_version}')

        # make ranking id
        # add data type and subtype to the path
        ranking_path = self.helper.set_ds_types_in_path(self.data_config, self.perf_path)

        # add timestep id and signal type to path
        ranking_path = os.path.join(ranking_path, f"T{self.data_config.window_length} [{', '.join(self.data_config.signal_types)}]")

        self.ranking_id = os.path.join(ranking_path, f"{self.node_type}_perf_{self.perf_version}")

        if is_avail:
            if not os.path.exists(self.ranking_log_path):
                raise FileNotFoundError(f"Ranking log path {self.ranking_log_path} does not exist. Please run the feature ranking first or type the correct version.")
        else:    
            self.check_if_ranking_version_exists()

        return self.ranking_log_path

    def save_ranking_id(self):
        if not os.path.exists(self.ranking_log_path):
            os.makedirs(self.ranking_log_path)

        rank_path = os.path.join(self.ranking_log_path, f'ranks_{self.rank_version}.txt')
        with open(rank_path, 'w') as f:
            f.write(self.ranking_id)

        print(f"Ranking id saved to {self.ranking_log_path}.")


    def _remove_perf_version(self):
        """
        Removes the performance version from the log path.
        """
        if os.path.exists(self.perf_log_path):
            user_input = input(f"Are you sure you want to remove '{self.node_type}_perf_{self.perf_version}' from the log path {self.perf_log_path}? (y/n): ")
            if user_input.lower() == 'y':
                shutil.rmtree(self.perf_log_path)
                print(f"Overwrote exsiting '{self.node_type}_perf_{self.perf_version}' from the log path {self.perf_log_path}.")

            else:
                print(f"Operation cancelled. {self.node_type}_perf_{self.perf_version} still remains.")
    
    def _get_next_perf_version(self):
        parent_dir = os.path.dirname(self.perf_log_path)

        # List all folders in parent_dir that match 'v<number>'
        perf_folders = [f for f in os.listdir(parent_dir) if f.startswith(f'{self.node_type}_perf_')]

        if perf_folders:
            # Extract numbers and find the max
            max_perf_version = max(int(f.split('_')[-1]) for f in perf_folders)
            self.version = max_perf_version + 1
            new_feature_perf = f"{self.node_type}_perf_{self.perf_version}"
            print(f"Next feature performance folder will be: {new_feature_perf}")
        else:
            new_feature_perf = f"{self.node_type}_perf_1"

        return os.path.join(parent_dir, new_feature_perf)
    
    def check_if_ranking_version_exists(self):
        if os.path.isdir(self.ranking_log_path):
            print(f"'ranks_{self.rank_version}' already exists in the log path '{self.ranking_log_path}'.")
            user_input = input("\nOverwrite exsiting version? (y/n):  ")
            if user_input.lower() == 'y':
                shutil.rmtree(self.ranking_log_path)
                print(f"Overwrote exsiting 'ranks_{self.rank_version}' from the log path {self.ranking_log_path}.")

            else:
                print(f"Operation cancelled. ranks_{self.rank_version} still remains.")
    
    def check_if_perf_version_exists(self):
        if os.path.isdir(self.perf_log_path):
            print(f"'{self.node_type}_perf_{self.perf_version}' already exists in the log path '{self.perf_log_path}'.")
            user_input = input("(a) Overwrite exsiting version, (b) create new version, (c) stop operation (Choose 'a', 'b' or 'c'):  ")

            if user_input.lower() == 'a':
                self._remove_perf_version()

            elif user_input.lower() == 'b':
                self.perf_log_path = self._get_next_perf_version()

            elif user_input.lower() == 'c':
                print("Stopped training.")
                sys.exit()  # Exit the program gracefully    


class HelperClass:
    def get_augmment_config_str_list(self, augment_configs):
        """
        Returns a list of strings representing the augment configurations.
        """
        augment_str_list = []
        
        for augment_config in augment_configs:
            idx = 0
            augment_str = f"{augment_config['type']}"
            for key, value in augment_config.items():
                if key != 'type':
                    if idx == 0:
                        augment_str += "_" 
                        idx += 1
                    augment_str += f"{key[0]}={value}"

            augment_str_list.append(augment_str)

        return augment_str_list
    
    def set_ds_types_in_path(self, data_config:DataConfig, log_path):
        """
        Takes into account both empty healthy and unhealthy config and sets the path accordingly.
        """
        if data_config.unhealthy_configs == {}:
            log_path = os.path.join(log_path, 'healthy')

        elif data_config.unhealthy_configs != {}:
            log_path = os.path.join(log_path, 'healthy_unhealthy')

        # add ds_subtype to path
        config_str = ''
        
        if data_config.healthy_configs != {}:    
            healthy_config_str_list = []

            for healthy_type, augment_configs  in data_config.healthy_configs.items():
                augment_str_list = self.get_augmment_config_str_list(augment_configs)                    
                augment_str_main = ', '.join(augment_str_list) 

                healthy_config_str_list.append(f'{healthy_type}[{augment_str_main}]')

            config_str = ' + '.join(healthy_config_str_list)

        # add unhealthy config to path if exists
        if data_config.unhealthy_configs != {}:
            unhealthy_config_str_list = []

            for unhealthy_type, augment_configs in data_config.unhealthy_configs.items():
                augment_str_list = self.get_augmment_config_str_list(augment_configs)
                augment_str_main = ', '.join(augment_str_list) 

                unhealthy_config_str_list.append(f'{unhealthy_type}[{augment_str_main}]')

            if config_str:
                config_str += f" + {' + '.join(unhealthy_config_str_list)}"
            else:
                config_str += ' + '.join(unhealthy_config_str_list)

        log_path = os.path.join(log_path, config_str)
        return log_path
    
    
class ViewRankings():
    def __init__(self, application=None, machine=None, scenario=None, logs_dir=LOGS_DIR):
        data_config = DataConfig()

        self.logs_dir = Path(logs_dir)
        if application is None or machine is None or scenario is None:
            self.application = data_config.application_map[data_config.application]
            self.machine = data_config.machine_type
            self.scenario = data_config.scenario
        else:
            self.application = application
            self.machine = machine
            self.scenario = scenario

        self.file_name = 'ranks'
   

        self.structure = {}
        self.version_paths = []
        self.version_txt_files = []
        self._build_structure_from_txt()

    def _build_structure_from_txt(self):
        """
        Build the tree structure from all model_x.txt files under the framework directory.
        """
        base = self.logs_dir / self.application / self.machine / self.scenario
        os.makedirs(base, exist_ok=True)  # Ensure the base directory exists

        txt_files = list(base.rglob(f"*{self.file_name}_*.txt"))
        self.version_txt_files = txt_files

        path_map = {}
        for txt_file in txt_files:
            try:
                with open(txt_file, "r") as f:
                    model_path = f.read().strip()
            except Exception as e:
                print(f"Could not read {txt_file}: {e}")
                continue
            # Remove base logs dir and split by os.sep
            rel_path = os.path.relpath(model_path, str(self.logs_dir))
            path_parts = rel_path.split(os.sep)
            # Only keep the parts after framework (skip first 4: app, machine, scenario)
            path_parts = path_parts[3:]
            model_name = txt_file.stem  # e.g., model_3
            key = tuple(path_parts)
            if key not in path_map:
                path_map[key] = []
            path_map[key].append((str(txt_file), model_name))

        structure = {}
        for path_parts, versions in path_map.items():
            node = structure
            # Add all parts as nodes, including the last one
            for part in path_parts:
                node = node.setdefault(part, {})
            if "_versions" not in node:
                node["_versions"] = []
    
            for idx, (txt_file, model_name) in enumerate(versions, 1):
                node["_versions"].append({
                    "model_name": model_name,
                    "txt_file": txt_file,
                    "vnum": idx,
                    "full_path": path_parts,
                })
                self.version_paths.append(txt_file)
        self.structure = structure

    def print_tree(self):
        console = Console()
        # Green up to and including framework
        tree = Tree(f"[green]{self.application}[/green]")
        machine_node = tree.add(f"[green]{self.machine}[/green]")

        scenario_node = machine_node.add(f"[green]{self.scenario}[/green]")
  
        self._build_rich_tree(scenario_node, self.structure, 0, [])
        console.print(tree)
        print("\nAvailable version paths:")
        for idx, txt_file in enumerate(self.version_paths):
            print(f"{idx}: {os.path.dirname(txt_file)}")

    def _build_rich_tree(self, parent_node, structure, level, parent_keys):
       
        label_map = {
            0: "<node_name>",
            1: "<ds_type>",
            2: "<ds_subtype>",
            3: "<ds_stats>",
            4: "<perf_version>",
            5: "<rank_version>"
        }
                    
        added_labels = set()
        # Add all keys except _versions first
        for key, value in structure.items():
            if key == "_versions":
                continue
            safe_key = key.replace('[', '\\[')
            # Add blue label if at the correct level and not already added
            if level in label_map and label_map[level] not in added_labels:
                parent_node.add(f"[blue]{label_map[level]}[/blue]")
                added_labels.add(label_map[level])
            
            if level == 3:
                branch = parent_node.add(f"[bright_yellow]{safe_key}[/bright_yellow]")
                self._build_rich_tree(branch, value, level + 1, parent_keys + [key])
                continue
            
            # All other folders: white
            branch = parent_node.add(f"[white]{safe_key}[/white]")
            self._build_rich_tree(branch, value, level + 1, parent_keys + [key])

        # Now add <versions> label and version nodes if present
        if "_versions" in structure:
            if "<versions>" not in added_labels:
                parent_node.add(f"[blue]<rank_versions>[/blue]")
                added_labels.add("<rank_versions>")
            
            for v in structure["_versions"]:
                model_disp = f"{v['model_name']} (v{v['vnum']})"
                safe_model_disp = model_disp.replace('[', '\\[')
                idx = self.version_paths.index(v["txt_file"])
      
                parent_node.add(f"[bright_yellow]{safe_model_disp}[/bright_yellow] [bright_cyan][{idx}][/bright_cyan]")

    def view_ranking_tree(self):
        self.print_tree()
        if not self.version_paths:
            print("No version paths found.")
            return None
        
def get_feature_list(n, perf_v, rank_v, domain, data_config:DataConfig):
    """
    Get the list of features based on the ranks from the specified version.

    Parameters
    ----------
    n : int
        First 'n' features to extract.
    perf_v : int
        Performance version to get the feature ranks from.
    rank_v : int
        Ranking version to get the feature ranks from.
    domain : str
        Domain of the features (`time` or `freq`)
    data_config : DataConfig
        Data configuration object containing the data settings.
    """
    rank_config = FeatureRankingManager(data_config)

    rank_config.perf_version = perf_v
    rank_config.rank_version = rank_v

    rank_log_path = rank_config.get_ranking_log_path(is_avail=True)
    rank_dict = np.load(os.path.join(rank_log_path, f"{domain}_feature_ranking.npy"), allow_pickle=True).item()

    # sort the features based on the ranks in decending order
    sorted_features = sorted(rank_dict.items(), key=lambda x: x[1], reverse=True)

    # get the top 'n' features
    top_features = [feature_name.lower() for feature_name, _ in sorted_features[:n]]
    return top_features