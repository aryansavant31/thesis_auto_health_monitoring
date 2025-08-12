import os
import scipy.io
import h5py

class MatToHDF5Processor:
    """
    A class to process .mat files and convert them into .hdf5 files
    while organizing them into a structured folder hierarchy.
    """

    def __init__(self, machine, scenario, ds_type):
        """
        Initialize the processor with input parameters.
        :param base_path: Base directory containing the .mat files.
        :param machine: Machine type (e.g., 'machine').
        :param scenario: Scenario type (e.g., 'scene_1').
        :param ds_type: Dataset type (e.g., 'healthy', 'unhealthy', 'unknown').
        """
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        self.machine = machine
        self.scenario = scenario
        self.ds_type = ds_type

        self.dummy_dict = {  
            'a1':   {'m_1': ['measure_1', 'ARYAN'], 
                    'm_2': ['measure_2', 'ARYAN']},
            'a2':   {'n_1': ['noise_1', 'RAHUL'],
                    'n_2': ['noise_2', 'RAHUL']},
            'b1':   {'t_1': ['temp_1', 'HIMANI'],
                     't_2': ['temp_2', 'HIMANI'],
                     't_3': ['temp_3', 'SHREYA'],
                     't_4': ['temp_4', 'SHREYA']},
            'b2':   {'r_1': ['resist_1', 'NIKITA'],
                    'r_2': ['resist_2', 'NIKITA'],
                     'r_3': ['resist_3', 'ANUJA'],
                     'r_4': ['resist_4', 'SHYAM']}
        }

    def find_mat_files(self):
        """
        Step 1: Find all .mat files in the specified directory structure.
        :return: List of paths to .mat files.
        """
        search_path = os.path.join(
            self.base_path, self.machine, self.scenario, self.ds_type
        )
        mat_files = []
        for root, _, files in os.walk(search_path):
            for file in files:
                if file.endswith('.mat'):
                    mat_files.append(os.path.join(root, file))
        return mat_files

    def create_ds_subtype_folders(self, mat_files):
        """
        Step 2: Create ds_subtype folders based on .mat file names.
        :param mat_files: List of paths to .mat files.
        :return: Dictionary mapping ds_subtype folder paths to their .mat files.
        """
        ds_subtype_folders = {}
        for mat_file in mat_files:
            file_name = os.path.basename(mat_file)
            machine_id = mat_file.split(os.sep)[-2]  # Extract machine_id from path
            e_set = "_".join(file_name.split('_')[:2])  # Extract E<e>_set<set_num>
            ds_subtype_folder = f"M={machine_id}_{e_set}"
            folder_path = os.path.join(
                self.base_path, self.machine, self.scenario, self.ds_type, ds_subtype_folder
            )
            os.makedirs(folder_path, exist_ok=True)
            ds_subtype_folders.setdefault(folder_path, []).append(mat_file)
        return ds_subtype_folders

    def process_mat_files(self, ds_subtype_folders):
        """
        Step 3: Process .mat files and organize data into the folder structure.
        :param ds_subtype_folders: Dictionary mapping ds_subtype folders to .mat files.
        """
        for ds_subtype_folder, mat_files in ds_subtype_folders.items():
            processed_path = os.path.join(ds_subtype_folder, 'processed', 'nodes')
            os.makedirs(processed_path, exist_ok=True)

            for mat_file in mat_files:
                file_name = os.path.basename(mat_file)
                m_name = file_name.split('_')[-1].split('.')[0]  # Extract m_name
                mat_data = scipy.io.loadmat(mat_file)  # Load .mat file
                structure_data = mat_data.get('structure_data', {})  # Replace with actual key

                for field in structure_data:
                    trace_id = field['name'][0]  # Extract trace_id
                    time_data = field['time']  # Extract time column
                    if m_name in self.dummy_dict and trace_id in self.dummy_dict[m_name]:
                        short_form, module_name = self.dummy_dict[m_name][trace_id]

                        # Create module_name folder
                        module_path = os.path.join(processed_path, module_name)
                        os.makedirs(module_path, exist_ok=True)

                        # Create short_form folder
                        signal_path = os.path.join(module_path, short_form)
                        os.makedirs(signal_path, exist_ok=True)

                        # Save time data to .hdf5 file
                        hdf5_file = os.path.join(signal_path, f"{trace_id}.hdf5")
                        with h5py.File(hdf5_file, 'w') as hdf:
                            hdf.create_dataset('time', data=time_data)

    def run(self):
        """
        Execute the entire processing pipeline.
        """
        print("Finding .mat files...")
        mat_files = self.find_mat_files()
        print(f"Found {len(mat_files)} .mat files.")

        print("Creating ds_subtype folders...")
        ds_subtype_folders = self.create_ds_subtype_folders(mat_files)

        print("Processing .mat files...")
        self.process_mat_files(ds_subtype_folders)
        print("Processing complete.")


# Example usage
if __name__ == "__main__":
    processor = MatToHDF5Processor(machine="machine", 
                                   scenario="scene_1", 
                                   ds_type="healthy")
    processor.run()