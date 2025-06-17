class DataPrepConfig:
    def __init__(self):
        self.ds_app_map = {'BER':'bearing_cwru', # IDs: BR1.1
                            'MSD':'mass_spring_damper',
                            'SPP':'spring_particles',
                            'ASM':'ASML'}

        self.ds_app_key = 'MSD'


        """
        Dataset types available for the applications:
        MSD: M10, M100
        ASM: NXE
        """
        self.ds_type = 'M10'

        
        """
        Scenarios available for the dataset types:
        MSD
            M10: S1, S2
            M100: S1, S2, S3
        """
        self.ds_scenario = 'S1'


        """
        Unhealthy types available for the scenarios:
        MSD
            M10
                S1: U0, U1, U2, U3
                S2: U0, U1, U2
        ASM
            NXE
                S1: U0
        """
        self.unhealthy_type = ['U1', 'U2', 'U3'] # 1, 2, 3, ... or all


        """
        Data ids available for the scenarios
        MSD
            M10
                S1: 'T60.N1.25', 'T25.G'

        """
        self.ds_id = 'T60.N1.25' # N = Noise, T = Timesteps


        """
        Signal type available for the dataset application
        MSD
            vel, pos, acc
        """
        self.signal_type = ['vel', 'pos', 'acc']


        """
        Data formats available
        MSD
            hdf5
        """
        self.data_format = 'hdf5'

    def set_dataset_path():
        pass
