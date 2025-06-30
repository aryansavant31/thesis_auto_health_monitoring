class DataPrepConfig:
    def __init__(self):
        """
        Attributes
        ----------
        ds_type 
            Dataset types available for the applications:
            - MSD: M10, M100
            - ASM: NXE

        ds_scenarios
            Scenarios available for the dataset types:
            - MSD
                - M10: S1, S2
                - M100: S1, S2, S3

        unhealthy_type
            Unhealthy types available for the scenarios:
            - MSD
                - M10
                    - S1: U0, U1, U2, U3
                    - S2: U0, U1, U2
            -ASM
                - NXE
                    - S1: U0

        ds_id
            Dataset IDs available for the scenarios:
            - MSD
                - M10
                    - S1: 'T60.N1.25', 'T25.G'

        signal_type
            Signal type available for the dataset application
            - MSD
                - vel, pos, acc
        
        data_format
            Data formats available:
            - MSD
                - hdf5

        """
        self.ds_app_map = {'BER':'bearing_cwru', # IDs: BR1.1
                            'MSD':'mass_spring_damper',
                            'SPP':'spring_particles',
                            'ASM':'ASML'}

        self.ds_app_key     = 'MSD'
        self.ds_type        = 'M10'
        self.ds_scenario    = 'S1'
        self.unhealthy_type = ['U1', 'U2', 'U3'] # 1, 2, 3, ... or all
        self.ds_id          = 'T60.N1.25' # N = Noise, T = Timesteps
        self.signsaal_type  = ['vel', 'pos', 'acc']
        self.data_format    = 'hdf5'

    def set_dataset_path():
        pass
