import sys, os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR) if ROOT_DIR not in sys.path else None

from typing import List, Dict
from pathlib import Path
from rich.tree import Tree
from rich.console import Console
import glob
import numpy as np
import itertools

from data.datasets.asml.groups import NXEGroupMaker
from data.datasets.mass_sp_dm.groups import MSDGroupMaker
from data.datasets.bearing.groups import BERGroupMaker
import data.faults as f
from data.faults import get_augment_config

class DataConfig:
    def __init__(self, run_type='train', view_dataset=False):
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
        
        self.application = 'MSD'
        self.machine_type = 'M005'
        self.scenario = 'scene_1'

        self.signal_types = MSDGroupMaker().m005_m1_acc
        
        self.fs = None #np.array([[48000]])    # sampling frequency matrix, set in the data.prep.py
        self.format = 'hdf5'  # options: hdf5

        # segement data
        self.window_length      = 1100
        self.stride             = 1100
        self.start_from_timestep = 2000  #1000 number of initial samples to chop off from the start of the signal

        self.use_custom_max_timesteps = False
        self.max_timesteps     = 10000


        self.view = DatasetViewer(self)

        if self.run_type == 'train':
            self.set_train_dataset()
        elif self.run_type == 'custom_test':
            self.set_custom_test_dataset()
        elif self.run_type == 'predict':
            self.set_predict_dataset()

        if not view_dataset:
            self.set_id = self.get_set_id()

    
        
    def set_train_dataset(self):
        # key: [get_augment_config('OG')] for key in self.view.healthy_types if key.startswith(self.set_id)
        e1_keys = [key for key in self.view.healthy_types if key.startswith('E1')][:100]
        ds_keys_ok = [f"ok_ds_{i}" for i in range(1, 101)] # if int(key.split("_")[1]) in ds_nums]

        self.healthy_configs   = {
            # '0_N': [get_augment_config('OG')],
            # '1_N': [get_augment_config('OG')],
            # 'series_tp': [get_augment_config('OG')]  
            key: [get_augment_config('OG')] for key in ds_keys_ok
            # 'ds_1': [get_augment_config('OG'), get_augment_config('gau', mean=0, snr_db=45), get_augment_config('gau', mean=0, snr_db=42), get_augment_config('gau', mean=0, snr_db=36), get_augment_config('gau', mean=0, snr_db=38), get_augment_config('gau', mean=0, snr_db=37)],
            # 'ds_2': [get_augment_config('OG'), get_augment_config('gau', mean=0, snr_db=45), get_augment_config('gau', mean=0, snr_db=42), get_augment_config('gau', mean=0, snr_db=36), get_augment_config('gau', mean=0, snr_db=38), get_augment_config('gau', mean=0, snr_db=37)],
            # 'ds_3': [get_augment_config('OG'), get_augment_config('gau', mean=0, snr_db=45), get_augment_config('gau', mean=0, snr_db=42), get_augment_config('gau', mean=0, snr_db=36), get_augment_config('gau', mean=0, snr_db=38), get_augment_config('gau', mean=0, snr_db=37)],
            # 'ds_4': [get_augment_config('OG'), get_augment_config('gau', mean=0, snr_db=45), get_augment_config('gau', mean=0, snr_db=42), get_augment_config('gau', mean=0, snr_db=36), get_augment_config('gau', mean=0, snr_db=38), get_augment_config('gau', mean=0, snr_db=37)],
            # 'ds_5': [get_augment_config('OG')],

            # key: [get_augment_config('OG')] for key in e1_keys
            #key: [get_augment_config('gau', mean=0, snr_db=6)] for key in e1_keys

        }

        #self.faults_ctrl()
        # all_ammf_l2_faults = [self.fault_ammf_l2_1, self.fault_ammf_l2_2, self.fault_ammf_l2_3, self.fault_ammf_l2_4, self.fault_ammf_l2_5, self.fault_ammf_l2_6, self.fault_ammf_l2_1, self.fault_ammf_l2_2, self.fault_ammf_l2_3, self.fault_ammf_l2_4]
        #all_ctrl_l2_faults = [self.fault_ctrl_l2_1, self.fault_ctrl_l2_2, self.fault_ctrl_l2_3, self.fault_ctrl_l2_4, self.fault_ctrl_l2_5, self.fault_ctrl_l2_6, self.fault_ctrl_l2_1, self.fault_ctrl_l2_2, self.fault_ctrl_l2_3, self.fault_ctrl_l2_4]
        #e1_keys_nok = [key for key in self.view.unhealthy_types if key.startswith('(sim)_E1')][:10]
        #all_ctrl_l2_faults = [self.fault_ctrl_l2_1, self.fault_ctrl_l2_2, self.fault_ctrl_l2_3, self.fault_ctrl_l2_4, self.fault_ctrl_l2_5, self.fault_ctrl_l2_6, self.fault_ctrl_l2_1, self.fault_ctrl_l2_2, self.fault_ctrl_l2_3, self.fault_ctrl_l2_4]
        e1_keys_nok = [key for key in self.view.unhealthy_types if key.startswith('(sim)_E1')][:10]


        top_faults = ['5', '6', '7', '3', '8', '9', '10']
        ds_keys_top_nok = [f"top_add_{i}_ds_{j}" for i in top_faults for j in range(1, 3)]
        ds_keys_mod_nok = [f"mod_5_f{i}_ds_1" for i in range(1, 5)]
        self.unhealthy_configs = {

            #key : value for key, value in zip(e1_keys_nok, all_ctrl_l2_faults)
            
            key: [get_augment_config('OG')] for key in ds_keys_mod_nok
            # 'ds_1_top_mod_fault_1': [get_augment_config('OG')],
            #'ds_1_mod_fault_1': [get_augment_config('OG'), get_augment_config('gau', mean=0, snr_db=45), get_augment_config('gau', mean=0, snr_db=42), get_augment_config('gau', mean=0, snr_db=36), get_augment_config('gau', mean=0, snr_db=38), get_augment_config('gau', mean=0, snr_db=37)],
            #'0_B-007': [get_augment_config('OG')],
            # obvious 
            # '(sim)_E1_set01_M=mAQ87': [
            #             # get_augment_config('glitch', prob=0.01, std_fac=1, add_next=True),
            #             # get_augment_config('sine', freqs=[10, 15], amps=[1, 1.2], add_next=False),
            #             get_augment_config('glitch', prob=0.1, std_fac=1.6, add_next=True),
            #             get_augment_config('sine', freqs=[1, 10, 15], amps=[1, 1.5, 1.7], add_next=False)
            #         ] 
            # '(sim)_E1_set01_M=mAQ87': [
            #             get_augment_config('glitch', prob=0.1, std_fac=2, add_next=True),
            #             get_augment_config('sine', freqs=[3, 2], std_facs=[1.6, 2.2])
            #             ], 
            #         '(sim)_E1_set01_M=mAS23': [
            #             get_augment_config('glitch', prob=0.05, std_fac=2, add_next=True),
            #             get_augment_config('sine', freqs=[1, 2, 3], std_facs=[1.8, 2.4, 2.1])
            #             ],  

            # # medium

            # 
            # POB LOS
            # # medium
            # '(sim)_E1_set01_M=mAQ87': [
            #             # get_augment_config('gau', mean=0, snr_db=35, add_next=True),
            #             get_augment_config('glitch', prob=0.07, std_fac=2.4, add_next=True),
            #             get_augment_config('sine', freqs=[200, 60], std_facs=[2, 2.3]),

            #             # get_augment_config('gau', mean=0, snr_db=35, add_next=True),
            #             # get_augment_config('glitch', prob=0.07, std_fac=2.3, add_next=True),
            #             get_augment_config('sine', freqs=[75, 420], std_facs=[2.3, 2]),

            #             # get_augment_config('gau', mean=0, snr_db=35, add_next=True),
            #             get_augment_config('glitch', prob=0.07, std_fac=2.2, add_next=True),
            #             get_augment_config('sine', freqs=[150, 40, 320], std_facs=[2.3, 2, 2]),
            #             ], 
            #         '(sim)_E1_set01_M=mAS23': [
            #             # get_augment_config('gau', mean=0, snr_db=35, add_next=True),
            #             # get_augment_config('glitch', prob=0.02, std_fac=2.3, add_next=True),
            #             get_augment_config('sine', freqs=[25, 290, 315], std_facs=[2, 1.9, 2.5]),

            #             # get_augment_config('gau', mean=0, snr_db=35, add_next=True),
            #             get_augment_config('glitch', prob=0.02, std_fac=2.1, add_next=True),
            #             get_augment_config('sine', freqs=[100, 425, 45], std_facs=[2.2, 2.4, 2.5]),

            #             # get_augment_config('gau', mean=0, snr_db=35, add_next=True),
            #             # get_augment_config('glitch', prob=0.02, std_fac=2.1, add_next=True),
            #             get_augment_config('sine', freqs=[55, 125, 270, 465], std_facs=[1.6, 2.1, 1.8, 2.3]),
            #             ],   
                    
            
        }

        self.unknown_configs = {
        }

    
    def set_custom_test_dataset(self):
        # key: [get_augment_config('OG')] for key in self.view.healthy_types[:50] if key.startswith('E1')
        self.amt = 1
       # e1_keys = [key for key in self.view.healthy_types if key.startswith('E1')][100:]
        ds_keys_ok = [f"ok_ds_{i}" for i in range(101, 141)]
        self.healthy_configs   = {
            #"ds_12": [get_augment_config('OG')],
            key: [get_augment_config('OG')] for key in ds_keys_ok
            #'ds_5': [get_augment_config('OG'), get_augment_config('gau', mean=0, snr_db=45), get_augment_config('gau', mean=0, snr_db=42), get_augment_config('gau', mean=0, snr_db=36), get_augment_config('gau', mean=0, snr_db=38), get_augment_config('gau', mean=0, snr_db=37)],
            #'0_N': [get_augment_config('OG')], 
            #'series_tp': [get_augment_config('OG')]  
            #key: [get_augment_config('OG')] for key in e1_keys
        }
        
        #self.faults_ctrl()
        # all_ammf_l2_faults = [self.fault_ammf_l2_1, self.fault_ammf_l2_2, self.fault_ammf_l2_3, self.fault_ammf_l2_4, self.fault_ammf_l2_5, self.fault_ammf_l2_6, self.fault_ammf_l2_1, self.fault_ammf_l2_2]
        # all_ammf_l3_faults = [self.fault_ammf_l3_1, self.fault_ammf_l3_2, self.fault_ammf_l3_3, self.fault_ammf_l3_4, self.fault_ammf_l3_5, self.fault_ammf_l3_6, self.fault_ammf_l3_1, self.fault_ammf_l3_2]
        #all_ammf_l4_faults = [self.fault_ammf_l4_1, self.fault_ammf_l4_2, self.fault_ammf_l4_3, self.fault_ammf_l4_4, self.fault_ammf_l4_5, self.fault_ammf_l4_6, self.fault_ammf_l4_1, self.fault_ammf_l4_2]

        #all_ctrl_l2_faults = [self.fault_ctrl_l2_1, self.fault_ctrl_l2_2, self.fault_ctrl_l2_3, self.fault_ctrl_l2_4, self.fault_ctrl_l2_5, self.fault_ctrl_l2_6, self.fault_ctrl_l2_1, self.fault_ctrl_l2_2]
        #all_ctrl_l4_faults = [self.fault_ctrl_l4_1, self.fault_ctrl_l4_2, self.fault_ctrl_l4_3, self.fault_ctrl_l4_4, self.fault_ctrl_l4_5, self.fault_ctrl_l4_6, self.fault_ctrl_l4_1, self.fault_ctrl_l4_2]

        #e1_keys_nok = [key for key in self.view.unhealthy_types if key.startswith('(sim)_E1')][10:]
        ds_keys_mod_nok = [f"mod_5_f{i}_ds_{j}" for i in range(1, 5) for j in range(2, 5)]

        self.unhealthy_configs = {
            key: [get_augment_config('OG')] for key in ds_keys_mod_nok
            #key : value for key, value in zip(e1_keys_nok, all_ctrl_l4_faults)
            # 'top_add_10_ds_1': [get_augment_config('OG')],
            #'series_intcon_fault_l1': [get_augment_config('OG')]  
            #'0_B-021': [get_augment_config('OG')],
            # '(sim)_E1_set01_M=mAQ87': [
            #     #get_augment_config('glitch', prob=0.01, std_fac=4, add_next=True),
            #     # get_augment_config('gau', mean=0, std=0.01, add_next=True),
            #     get_augment_config('glitch', prob=0.1, std_fac=1.6, add_next=True),
            #     get_augment_config('sine', freqs=[1, 10, 15], amps=[1, 1.5, 1.7], add_next=False),
            #     get_augment_config('OG')
            #     ], 
            # '(sim)_E1_set01_M=mAQ10': [
            #     get_augment_config('glitch', prob=0.01, std_fac=4, add_next=True),
            #     get_augment_config('gau', mean=0, std=0.01, add_next=True),
            #     get_augment_config('sine', freqs=[1, 5, 10], std_facs=[5, 4, 4])
            #     ], 
            
            # # medium
            # '(sim)_E1_set01_M=mAI26': [
            #             get_augment_config('glitch', prob=0.07, std_fac=1.5, add_next=True),
            #             get_augment_config('gau', mean=0, std=0.0001, add_next=True),
            #             get_augment_config('sine', freqs=[2, 4], std_facs=[1.3, 1.5])
            #             ],

            # # subtle
            # '(sim)_E1_set01_M=mAI26': [
            #             get_augment_config('glitch', prob=0.0015, std_fac=0.8, add_next=True),
            #             get_augment_config('sine', freqs=[2, 4], std_facs=[0.6, 0.8])
            #             ], 

            # # healthy
            # '(sim)_E1_set01_M=mAI26': [
            #             get_augment_config('OG'),
                        
            #             ], 

            # healthy (level 2 noise)
            # '(sim)_E1_set01_M=mAI26': [
            #     get_augment_config('gau', mean=0.0, snr_db=6.6),
                        
            #             ], 

            # '(sim)_E1_set01_M=mAI26': [
            #             # get_augment_config('gau', mean=0, snr_db=35, add_next=True),
            #             get_augment_config('glitch', prob=0.07, std_fac=1.5, add_next=True),
            #             get_augment_config('sine', freqs=[2, 4], std_facs=[1.3, 1.5])
            #             ], 
            #         '(sim)_E1_set01_M=mAQ10': [
            #             # get_augment_config('gau', mean=0, snr_db=35, add_next=True),
            #             get_augment_config('glitch', prob=0.02, std_fac=1.3, add_next=True),
            #             get_augment_config('sine', freqs=[1, 2, 5], std_facs=[1.3, 1, 1.5])
            #             ], 
        }

        self.unknown_configs = {
            # '1_N': [get_augment_config('OG')],
            
            
        }
        
    def set_predict_dataset(self):
        self.amt = 1
        self.healthy_configs   = {
            #'series_tp': [get_augment_config('OG')]
        }
        
        self.unhealthy_configs = {
            'mod_1_4_f4_ds_1': [get_augment_config('OG')]
        #     '0_B-021': [get_augment_config('OG')],
            # '0_B-007': [get_augment_config('OG')],
            # '0_IR-007': [get_augment_config('OG')],
            # '0_IR-021': [get_augment_config('OG')],
        }

        self.unknown_configs = {
        }
    
    def get_set_id(self):
        try:
            if self.healthy_configs != {}:   
                set_id = list(self.healthy_configs.keys())[0].split('_')[0]
            elif self.unhealthy_configs != {}:
                set_id = list(self.unhealthy_configs.keys())[0].split('_')[0]
            elif self.unknown_configs != {}:
                set_id = list(self.unknown_configs.keys())[0].split('_')[0]

            if set_id not in ['E1', 'E2']:
                set_id = 'G'

        except ValueError as e:
            set_id = 'G'

        return set_id
    
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
    
class DataSweep:
    def __init__(self, run_type):
        self.run_type = run_type
        self.view = DatasetViewer(DataConfig())


        self.signal_types = [MSDGroupMaker().m005_m1_acc, MSDGroupMaker().m005_m1_pos, MSDGroupMaker().m005_m1_vel,
                             MSDGroupMaker().m005_m2_acc, MSDGroupMaker().m005_m2_pos, MSDGroupMaker().m005_m2_vel,
                             MSDGroupMaker().m005_m3_acc, MSDGroupMaker().m005_m3_pos, MSDGroupMaker().m005_m3_vel,
                             MSDGroupMaker().m005_m4_acc, MSDGroupMaker().m005_m4_pos, MSDGroupMaker().m005_m4_vel,]
                             #MSDGroupMaker().m005_m5_acc, MSDGroupMaker().m005_m5_pos, MSDGroupMaker().m005_m5_vel]
        self.window_length = [1100]
        self.stride = [1100]


        if self.run_type == 'train':
            e1_keys = [key for key in self.view.healthy_types if key.startswith('E1')][:100]
            ds_keys_ok = [f"ok_ds_{i}" for i in range(1, 101)]
            # e2_keys = [key for key in self.view.healthy_types if key.startswith('E2')][:50]

            self.healthy_configs = [
                #{key: [get_augment_config('OG')] for key in e1_keys},
                {key: [get_augment_config('OG')] for key in ds_keys_ok}
            ]

            # medium
            ds_keys_mod_nok = [f"mod_5_f{i}_ds_1" for i in range(1, 5)]
            self.unhealthy_configs = [
                {key: [get_augment_config('OG')] for key in ds_keys_mod_nok}
                # {'(sim)_E1_set01_M=mAQ87': [
                #         get_augment_config('glitch', prob=0.1, std_fac=2, add_next=True),
                #         get_augment_config('sine', freqs=[3, 2], std_facs=[2, 2.2]),

                #         get_augment_config('glitch', prob=0.1, std_fac=2, add_next=True),
                #         get_augment_config('sine', freqs=[1, 2], std_facs=[2.1, 2.2])
                #         ], 
                #     '(sim)_E1_set01_M=mAS23': [
                #         get_augment_config('glitch', prob=0.05, std_fac=2.2, add_next=True),
                #         get_augment_config('sine', freqs=[1, 5, 3], std_facs=[2.1, 2.4, 2.1]), 

                #         get_augment_config('glitch', prob=0.1, std_fac=2.1, add_next=True),
                #         get_augment_config('sine', freqs=[3, 4], std_facs=[2, 2.2])
                #         ], 
                # }
            ]


        elif self.run_type == 'custom_test':
            e1_keys = [key for key in self.view.healthy_types if key.startswith('E1')][100:]
            ds_keys_ok = [f"ok_ds_{i}" for i in range(101, 141)]
            self.healthy_configs = [
                {key: [get_augment_config('OG')] for key in ds_keys_ok}
            #     # no noise
            #     # {key: [get_augment_config('OG')] for key in e1_keys}
            #     #{'ds_5': [get_augment_config('OG'), get_augment_config('gau', mean=0, snr_db=45), get_augment_config('gau', mean=0, snr_db=42), get_augment_config('gau', mean=0, snr_db=36), get_augment_config('gau', mean=0, snr_db=38), get_augment_config('gau', mean=0, snr_db=37),
            #               #get_augment_config('gau', mean=0, snr_db=45), get_augment_config('gau', mean=0, snr_db=41), get_augment_config('gau', mean=0, snr_db=38), get_augment_config('gau', mean=0, snr_db=38), get_augment_config('gau', mean=0, snr_db=40)],}

            #     # # level 1 noise
            #     # {key: [get_augment_config('gau', mean=0.0, snr_db=2)] for key in e1_keys},

            #     # # level 3 noise
            #     # {key: [get_augment_config('gau', mean=0.0, std=0.0001)] for key in e1_keys},
            #     # {key: [get_augment_config('gau', mean=0.0, std=0.01)] for key in e1_keys},
            #     # {key: [get_augment_config('gau', mean=0.0, std=0.1)] for key in e1_keys},
            #     # {key: [get_augment_config('gau', mean=0.0, std=0.3)] for key in e1_keys},
            #     # {key: [get_augment_config('gau', mean=0.0, std=0.8)] for key in e1_keys},
            ]

            ds_keys_mod_nok = [f"mod_5_f{i}_ds_{j}" for i in range(1, 5) for j in range(2, 5)]

            self.unhealthy_configs = [
                # {'top_add_2_ds_1': [get_augment_config('OG')]},
                # {'top_add_1_ds_3': [get_augment_config('OG')]},
                # {'top_add_2_ds_3': [get_augment_config('OG')]},
                # {'top_add_3_ds_3': [get_augment_config('OG')]},
                # {'top_add_4_ds_3': [get_augment_config('OG')]},
                # {'top_add_5_ds_3': [get_augment_config('OG')]},
                # {'top_add_6_ds_3': [get_augment_config('OG')]},
                # {'top_add_7_ds_3': [get_augment_config('OG')]},
                # {'top_add_8_ds_3': [get_augment_config('OG')]},
                # {'top_add_9_ds_3': [get_augment_config('OG')]},
                # {'top_add_10_ds_3': [get_augment_config('OG')]},

                {key: [get_augment_config('OG')] for key in ds_keys_mod_nok}
                #{'ds_1_mod_fault_1': [get_augment_config('OG'), get_augment_config('gau', mean=0, snr_db=45), get_augment_config('gau', mean=0, snr_db=42), get_augment_config('gau', mean=0, snr_db=36), get_augment_config('gau', mean=0, snr_db=38), get_augment_config('gau', mean=0, snr_db=37)],}
                # # obvious fault
                # {
                #      '(sim)_E1_set01_M=mAI26': [
                #         # get_augment_config('gau', mean=0, snr_db=35, add_next=True),
                #         get_augment_config('glitch', prob=0.1, std_fac=3, add_next=True),
                #         get_augment_config('sine', freqs=[290, 478], std_facs=[3, 3.2])
                #         ], 
                #     '(sim)_E1_set01_M=mAQ10': [
                #         # get_augment_config('gau', mean=0, snr_db=35, add_next=True),
                #         get_augment_config('glitch', prob=0.05, std_fac=2.8, add_next=True),
                #         get_augment_config('sine', freqs=[106, 50, 535], std_facs=[3, 3.4, 3.1])
                #         ],
                # },

                # # medium fault
                # {
                #     '(sim)_E1_set01_M=mAI26': [
                #         # get_augment_config('gau', mean=0, snr_db=35, add_next=True),
                #         get_augment_config('glitch', prob=0.1, std_fac=2, add_next=True),
                #         get_augment_config('sine', freqs=[267, 434], std_facs=[2, 2])
                #         ], 
                #     '(sim)_E1_set01_M=mAQ10': [
                #         # get_augment_config('gau', mean=0, snr_db=35, add_next=True),
                #         get_augment_config('glitch', prob=0.05, std_fac=2, add_next=True),
                #         get_augment_config('sine', freqs=[100, 223, 500], std_facs=[2, 2.4, 2])
                #         ],
                # },

                # # subtle fault
                # {
                #     '(sim)_E1_set01_M=mAI26': [
                #         # get_augment_config('gau', mean=0, snr_db=35, add_next=True),
                #         get_augment_config('glitch', prob=0.07, std_fac=1.5, add_next=True),
                #         get_augment_config('sine', freqs=[200, 400], std_facs=[1.3, 1.5])
                #         ], 
                #     '(sim)_E1_set01_M=mAQ10': [
                #         # get_augment_config('gau', mean=0, snr_db=35, add_next=True),
                #         get_augment_config('glitch', prob=0.02, std_fac=1.3, add_next=True),
                #         get_augment_config('sine', freqs=[25, 290, 500], std_facs=[1.3, 1, 1.5])
                #         ], 
                # },

                # # subtler fault
                # {
                #     '(sim)_E1_set01_M=mAI26': [
                #         # get_augment_config('gau', mean=0, snr_db=35, add_next=True),
                #         get_augment_config('glitch', prob=0.07, std_fac=0.7, add_next=True),
                #         get_augment_config('sine', freqs=[256, 325], std_facs=[0.9, 0.8])
                #         ], 
                #     '(sim)_E1_set01_M=mAQ10': [
                #         # get_augment_config('gau', mean=0, snr_db=35, add_next=True),
                #         get_augment_config('glitch', prob=0.02, std_fac=0.7, add_next=True),
                #         get_augment_config('sine', freqs=[100, 200, 55], std_facs=[1, 0.9, 0.8])
                #         ], 
                # },

                # # more subtler fault
                # {
                #     '(sim)_E1_set01_M=mAI26': [
                #         # get_augment_config('gau', mean=0, snr_db=35, add_next=True),
                #         get_augment_config('glitch', prob=0.07, std_fac=0.7, add_next=True),
                #         get_augment_config('sine', freqs=[256, 300], std_facs=[0.5, 0.6])
                #         ], 
                #     '(sim)_E1_set01_M=mAQ10': [
                #         # get_augment_config('gau', mean=0, snr_db=35, add_next=True),
                #         get_augment_config('glitch', prob=0.02, std_fac=0.7, add_next=True),
                #         get_augment_config('sine', freqs=[150, 200, 55], std_facs=[0.5, 0.4, 0.5])
                #         ], 
                # }

        
            ]


        elif self.run_type == 'predict':
            #self.set_id = ['G']
            self.healthy_configs = [
                {}
            ]
            self.unhealthy_configs = [
                {'mod_1_f2_ds_6': [get_augment_config('OG')]}
            ]
            
    def get_sweep_configs(self):
        """
        Generate all possible combinations of parameters for sweeping.
            
        Returns
        -------
        list
            List of DataConfig objects with different parameter combinations
        """
        # Convert sweep config to dictionary
        sweep_dict = self._to_dict()
        
        # Get all parameter names and their values
        param_names = list(sweep_dict.keys())
        param_values = list(sweep_dict.values())
        
        # Generate all combinations using cartesian product
        combinations = list(itertools.product(*param_values))
        
        # Create train configs for each combination
        data_configs = []
        for idx, combo in enumerate(combinations):
            # Create base train config
            data_config = DataConfig(self.run_type)
            
            # Update parameters based on current combination
            for param_name, param_value in zip(param_names, combo):
                setattr(data_config, param_name, param_value)
            
            # update set id
            data_config.set_id = data_config.get_set_id()
            data_configs.append(data_config)
        
        return data_configs
    
    def _to_dict(self):
        """
        Convert sweep config attributes to dictionary, excluding private methods and non-list attributes.
        """
        sweep_dict = {}
        for attr_name in dir(self):
            if not attr_name.startswith('_') and not callable(getattr(self, attr_name)):
                attr_value = getattr(self, attr_name)
                if isinstance(attr_value, list):
                    sweep_dict[attr_name] = attr_value
        return sweep_dict
    
    def get_total_combinations(self):
        """
        Get the total number of parameter combinations.
        """
        sweep_dict = self._to_dict()
        total = 1
        for values in sweep_dict.values():
            total *= len(values)
        return total
    
    def print_sweep_summary(self):
        """
        Print a summary of the parameter sweep.
        """
        sweep_dict = self._to_dict()
        print(f"\nData Config Sweep Summary:")
        print(f"Run type: {self.run_type}")
        print(30*'-')
        print(f"Total combinations: {self.get_total_combinations()}")
        print("\nParameters and their values:")
        for param_name, param_values in sweep_dict.items():
            print(f"  {param_name}: {len(param_values)} values -> {param_values}")
    

    


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
        - `time+freq`: **cutoff_freq** (for high pass filter)
    """
    config = {}
    config['type'] = domain_type

    if domain_type == 'time':
        config['cutoff_freq'] = kwargs.get('cutoff_freq', 0)  # default cutoff frequency for time domain
        
    elif domain_type == 'freq':
        config['cutoff_freq'] = kwargs.get('cutoff_freq', 0)  # default cutoff frequency

    elif domain_type == 'time+freq':
        config['cutoff_freq'] = kwargs.get('cutoff_freq', 0)  # default cutoff frequency
       
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
    data_config = DataConfig(view_dataset=True)
    
    data_viewer = DatasetViewer(data_config)

    data_viewer.view_dataset_tree()

    # a, b =data_config.get_dataset_path()
    # print(a ,b)

    # data_config.set_test_dataset()
    # a, b = data_config.get_dataset_path()
    # print(a, b)


