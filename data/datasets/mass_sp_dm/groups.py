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