class DataPrepConfig:
    def __init__(self):
        self.ds_type_map = {'B':'bearing_cwru', # IDs: BR1.1
                            'S':'mass_spring_damper',
                            'P':'spring_particles',
                            'A':'ASML'}

        self.ds_type_key = 'S'

        self.ds_subtype = 'M100' # The subtype must be applicable for the selected type
        self.ds_id = '1.N' # N = Noise
        self.data_format = 'hdf5'