class BERGroupMaker:
    """
    This class is used to create groups for the Bearing dataset.
    """
    def __init__(self):
        self.gb_acc1 = {
            'node_group_name'  : 'gearbox_1',
            'signal_group_name': 'acc',
            'group' : {'gearbox': ['acc'] }
            }
        self.gb_acc2 = {
            'node_group_name'  : 'gearbox_2',
            'signal_group_name': 'acc',
            'group' : {'gearbox': ['acc'] }
            }