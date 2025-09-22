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
                       'mass_4': ['acc', 'pos', 'vel'],}         
            }
        
        self.m005_all = {
            'node_group_name'  : 'm005',  
            'signal_group_name': 'apv',
            'group' : {'mass_1': ['acc', 'pos', 'vel'],
                       'mass_2': ['acc', 'pos', 'vel'],
                       'mass_3': ['acc', 'pos', 'vel'],
                       'mass_4': ['acc', 'pos', 'vel'],
                       'mass_5': ['acc', 'pos', 'vel'],}         
            }