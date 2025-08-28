class BERGroupMaker:
    """
    This class is used to create groups for the Bearing dataset.
    """
    def __init__(self):
        self.gb_acc = {
            'node_group_name'  : 'gearbox',
            'signal_group_name': 'acc',
            'group' : {'gearbox': ['acc'] }
            }