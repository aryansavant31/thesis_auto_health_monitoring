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
            'mass_1': ['acc', 'pos', 'vel'],
            'mass_2': ['acc', 'pos', 'vel'],
            'mass_3': ['acc', 'pos', 'vel'],
            'mass_4': ['acc', 'pos', 'vel'],
            }