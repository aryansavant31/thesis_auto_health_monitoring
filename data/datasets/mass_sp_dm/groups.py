class MSDGroupMaker:
    """
    This class is used to create groups for the MSD dataset.
    """
    def __init__(self):
        """
        Group names
        ----------
        M004
            - m004_all: contains all masses and all signal types
        """
        self.m004_all = {
            'node_group_name'  : 'm004',  
            'signal_group_name': 'apv',
            'group' : {'mass_1': ['acc', 'pos', 'vel'],
                       'mass_2': ['acc', 'pos', 'vel'],
                       'mass_3': ['acc', 'pos', 'vel'],
                       'mass_4': ['acc', 'pos', 'vel'],},
            'subsystems':  [
                            [0, 1],      # Subsystem 1
                            [2, 3],      # Subsystem 2
                        ]        
            }
        
        
        # ============== M005 =================

        # tOPOLOGY ESTIAMTION
        
        self.m005_all = {
            'node_group_name'  : 'm005',  
            'signal_group_name': 'apv',
            'group' : {'mass_1': ['acc', 'pos', 'vel'],
                       'mass_2': ['acc', 'pos', 'vel'],
                       'mass_3': ['acc', 'pos', 'vel'],
                       'mass_4': ['acc', 'pos', 'vel'],
                       'mass_5': ['acc', 'pos', 'vel'],},
            'subsystems':  [
                            [0, 1, 2], # Subsystem 1
                            [3, 4],    # Subsystem 2 
                        ]         
            }
        
        self.m005_acc = {
            'node_group_name'  : 'm005',  
            'signal_group_name': 'acc',
            'group' : {'mass_1': ['acc'],
                       'mass_2': ['acc'],
                       'mass_3': ['acc'],
                       'mass_4': ['acc'],
                       'mass_5': ['acc'],},
            'subsystems':  [
                            [0, 1, 2], # Subsystem 1
                            [3, 4],    # Subsystem 2  
                        ]         
            }
        
        
        # FAULT DETECTION

        # Mass 1
        self.m005_m1_acc = {
            'node_group_name'  : 'mass_1',  
            'signal_group_name': 'acc',
            'group' : {'mass_1': ['acc'],}
        }

        self.m005_m1_pos = {
            'node_group_name'  : 'mass_1',  
            'signal_group_name': 'pos',
            'group' : {'mass_1': ['pos']}
        }

        self.m005_m1_vel = {
            'node_group_name'  : 'mass_1',  
            'signal_group_name': 'vel',
            'group' : {'mass_1': ['vel']}
        }

        # Mass 2
        self.m005_m2_acc = {
            'node_group_name'  : 'mass_2',  
            'signal_group_name': 'acc',
            'group' : {'mass_2': ['acc'],}
        }
        
        self.m005_m2_pos = {
            'node_group_name'  : 'mass_2',  
            'signal_group_name': 'pos',
            'group' : {'mass_2': ['pos']}
        }

        self.m005_m2_vel = {
            'node_group_name'  : 'mass_2',  
            'signal_group_name': 'vel',
            'group' : {'mass_2': ['vel']}
        }

        # Mass 3
        self.m005_m3_acc = {
            'node_group_name'  : 'mass_3',  
            'signal_group_name': 'acc',
            'group' : {'mass_3': ['acc'],}
        }

        self.m005_m3_pos = {
            'node_group_name'  : 'mass_3',  
            'signal_group_name': 'pos',
            'group' : {'mass_3': ['pos']}
        }

        self.m005_m3_vel = {
            'node_group_name'  : 'mass_3',
            'signal_group_name': 'vel',
            'group' : {'mass_3': ['vel']}
        }

        # Mass 4
        self.m005_m4_acc = {
            'node_group_name'  : 'mass_4',  
            'signal_group_name': 'acc',
            'group' : {'mass_4': ['acc'],}
        }

        self.m005_m4_pos = {
            'node_group_name'  : 'mass_4',
            'signal_group_name': 'pos',
            'group' : {'mass_4': ['pos']}
        }

        self.m005_m4_vel = {
            'node_group_name'  : 'mass_4',
            'signal_group_name': 'vel',
            'group' : {'mass_4': ['vel']}
        }

        # Mass 5

        self.m005_m5_acc = {
            'node_group_name'  : 'mass_5',  
            'signal_group_name': 'acc',
            'group' : {'mass_5': ['acc'],}
        }

        self.m005_m5_pos = {
            'node_group_name'  : 'mass_5',
            'signal_group_name': 'pos',
            'group' : {'mass_5': ['pos']}
        }

        self.m005_m5_vel = {
            'node_group_name'  : 'mass_5',
            'signal_group_name': 'vel',
            'group' : {'mass_5': ['vel']}
        }

        # ============== M012 =================
        
        self.m012_all = {
            'node_group_name'  : 'm012',
            'signal_group_name': 'apv',
            'group' : {'mass_1': ['acc', 'pos', 'vel'],
                       'mass_2': ['acc', 'pos', 'vel'],
                       'mass_3': ['acc', 'pos', 'vel'],
                       'mass_4': ['acc', 'pos', 'vel'],
                       'mass_5': ['acc', 'pos', 'vel'],
                       'mass_6': ['acc', 'pos', 'vel'],
                       'mass_7': ['acc', 'pos', 'vel'],
                       'mass_8': ['acc', 'pos', 'vel'],
                       'mass_9': ['acc', 'pos', 'vel'],
                       'mass_10': ['acc', 'pos', 'vel'],
                       'mass_11': ['acc', 'pos', 'vel'],
                       'mass_12': ['acc', 'pos', 'vel'],},
            'subsystems':  [
                            [0, 1, 2, 3, 4],      # Subsystem 1
                            [5, 6, 7],      # Subsystem 2
                            [8, 9],      # Subsystem 3
                            [10, 11],      # Subsystem 4
                            ]
        }

        self.m012_mass_1_all = {
            'node_group_name'  : 'mass_1',
            'signal_group_name': 'apv',
            'group' : {'mass_1': ['acc', 'pos', 'vel']}

        }