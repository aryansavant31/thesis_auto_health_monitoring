import os, sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT_DIR) if ROOT_DIR not in sys.path else None

# other imports
import scipy.io
import h5py
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# global imports
from console_logger import ConsoleLogger

class MatToHDF5Processor:
    """
    A class to process .mat files and convert them into .hdf5 files
    while organizing them into a structured folder hierarchy.
    """

    def __init__(self, machine, scenario, ds_type):
        """
        Initialize the processor with input parameters.

        Parameters
        ----------
        machine: str
            Machine type (e.g., 'machine').
        scenario: str
            Scenario type (e.g., 'scene_1').
        ds_type: str
            Dataset type (e.g., 'healthy', 'unhealthy', 'unknown').
        """
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        self.machine = machine
        self.scenario = scenario
        self.ds_type = ds_type

        self.signal_dict = {  
            'a1':   {'m_1': ['measure_1', '1_ARYAN'], 
                    'm_2': ['measure_2', '1_ARYAN']},
            'a2':   {'n_11': ['noise_1', '2_RAHUL'],
                    'n_2': ['noise_2', '2_RAHUL']},
            'b1':   {'t_1': ['temp_1', '3_HIMANI'],
                     't_2': ['temp_2', '3_HIMANI'],
                     't_3': ['temp_3', '4_SHREYA'],
                     't_4': ['temp_4', '4_SHREYA']},
            'b2':   {'r_1': ['resist_1', '5_NIKITA'],
                    'r_2': ['resist_2', '5_NIKITA'],
                     'r_3': ['resist_3', '6_ANUJA'],
                     'r_4': ['resist_4', '7_SHYAM']}
        }

    def find_mat_files(self):
        """
        Step 1: Find all .mat files in the specified directory structure.

        Return 
        -------
        List[str]
            A list of paths to .mat files.
        """
        search_path = os.path.join(
            self.base_path, self.machine, self.scenario, self.ds_type
        )
        mat_files = []
        for root, _, files in os.walk(search_path):
            for file in files:
                if file.endswith('.mat'):
                    mat_files.append(os.path.join(root, file))

        # sort the list of .mat files alphabetic order
        mat_files.sort() 

        print("All the .mat files found:")
        for idx, mat_file in enumerate(mat_files):
            print(f"{idx+1}: {os.path.basename(mat_file)}")

        return mat_files

    def create_ds_subtype_folders(self, mat_files):
        """
        Step 2: Create ds_subtype folders based on .mat file names.

        Parameters
        ----------
        mat_files: List[str]
            A list of paths to .mat files.

        Returns
        -------
        dict
            A dictionary mapping ds_subtype folder paths to lists of .mat files.
        """
        ds_subtype_folders = {}
        for mat_file in mat_files:
            file_name = os.path.basename(mat_file)
            machine_id = mat_file.split(os.sep)[-3]  # Extract machine_id from path
            e_set = "_".join(file_name.split('_')[:2])  # Extract E<e>_set<set_num>
            ds_subtype_folder = f"M={machine_id}_{e_set}"
            folder_path = os.path.join(
                self.base_path, self.machine, self.scenario, self.ds_type, ds_subtype_folder
            )
            os.makedirs(folder_path, exist_ok=True)
            ds_subtype_folders.setdefault(folder_path, []).append(mat_file)
        return ds_subtype_folders
    
    def process_single_mat_file(self, args):
        """
        Process a single .mat file and save its data to the appropriate folder structure.
        
        Parameters
        ----------
        args: tuple
            Contains (mat_file, ds_subtype_folder, processed_path).
        """
        mat_file, ds_subtype_folder, processed_path = args
        file_name = os.path.basename(mat_file)
        mat_mdl_name = file_name.split('.')[0].split('_')[-1]  # extract module name from file name
        rep_num = file_name.split('_')[2]

        mat_data = scipy.io.loadmat(mat_file)  # Load .mat file
        structure_data = mat_data.get('my_struct', {})  # Replace with actual key

        for fields in structure_data:
            for trace_id, time_data, time in zip(fields['name'], fields['data'], fields['time']):
                trace_id = trace_id[0].item()

                # For asml DATA only
                fs = 1 / (time[1] - time[0])  # Calculate sampling frequency

                # Check if mat_mdl_name exists in signal_dict
                if mat_mdl_name not in self.signal_dict:
                    print(f"Warning: mat_module_name '{mat_mdl_name}' not found in signal_dict for '{rep_num}' in ds_subtype {os.path.basename(ds_subtype_folder)}. Skipping...")
                    continue

                # Check if trace_id exists in signal_dict[mat_module_name]
                if trace_id not in self.signal_dict[mat_mdl_name]:
                    print(f"Warning: trace_id '{trace_id}' not found in signal_dict['{mat_mdl_name}'] for '{rep_num}' in ds_subtype {os.path.basename(ds_subtype_folder)}. Skipping...")
                    continue

                # If both checks pass, proceed with saving hdf5 file
                short_form, module_name = self.signal_dict[mat_mdl_name][trace_id]

                # Create module_name folder
                module_path = os.path.join(processed_path, module_name)
                os.makedirs(module_path, exist_ok=True)

                # Create short_form folder
                signal_path = os.path.join(module_path, short_form)
                os.makedirs(signal_path, exist_ok=True)

                # Save time data to .hdf5 file
                hdf5_file = os.path.join(signal_path, f"{trace_id}.hdf5")
                with h5py.File(hdf5_file, 'a') as hdf:
                    dataset_name = f"time_data_{rep_num}"  # Use rep_num to create unique dataset names
                    fs_name = f"fs_{rep_num}"

                    if dataset_name in hdf:
                        print(f"Warning: Dataset '{dataset_name}' already exists in '{hdf5_file}'. Overwriting...")
                        del hdf[dataset_name]

                    if fs_name in hdf:
                        print(f"Warning: fs '{fs_name}' already exists in '{hdf5_file}'. Overwriting...")
                        del hdf[fs_name]

                    hdf.create_dataset(dataset_name, data=time_data)
                    hdf.create_dataset(fs_name, data=fs)

    def process_mat_files(self, ds_subtype_folders):
        """
        Step 3: Process .mat files and organize data into the folder structure.
        
        Parameters
        ----------
        ds_subtype_folders: dict
            A dictionary mapping ds_subtype folder paths to lists of .mat files.
        """
        tasks = []
        for ds_subtype_folder, mat_files in ds_subtype_folders.items():
            processed_path = os.path.join(ds_subtype_folder, 'processed', 'nodes')
            os.makedirs(processed_path, exist_ok=True)

            for mat_file in mat_files:
                tasks.append((mat_file, ds_subtype_folder, processed_path))

        # Use ProcessPoolExecutor for parallel processing
        num_cpus = multiprocessing.cpu_count()
        with ProcessPoolExecutor(max_workers=num_cpus-2) as executor:
            executor.map(self.process_single_mat_file, tasks)

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


if __name__ == "__main__":
    console_logger = ConsoleLogger()

    with console_logger.capture_output():
        print("Starting processing of .mat files...")

        processor = MatToHDF5Processor(machine="machine", 
                                    scenario="scene_1", 
                                    ds_type="healthy")
        processor.run()
        print("All .mat files processed and organized into .hdf5 format.")

    base_path = os.path.dirname(os.path.abspath(__file__))
    log_file_path = os.path.join(base_path, 'mat_processing_log.txt')
    console_logger.save_to_file(log_file_path, "process_raw_node_mat_files.py")