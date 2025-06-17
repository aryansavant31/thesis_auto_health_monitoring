class TopologyConfig:
    def __init__(self):
        self.sim_num = 1.1
        self.is_save = False

    def set_tp_dataset_params(self):
        self.ds_type_map = {'S':'mass_spring_damper',
                            'P':'spring_particles',
                            'A':'ASML'}

        self.ds_type_key = 'S'

        self.ds_subtype = 'M100' # The subtype must be applicable for the selected type
        self.ds_id = '1.N' # N = Noise
        self.data_format = 'hdf5'

    def set_result_path(self):
        pass