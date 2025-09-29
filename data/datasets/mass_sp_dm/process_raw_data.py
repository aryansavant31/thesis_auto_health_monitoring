import os
import h5py
import scipy.io

class ProcessRawMSDNodeData:
    """
    Class to process raw data for MSD fault detection.
    Converts .mat files to .hdf5 format and stores them in processed folders.
    """ 

    def __init__(self, machine, scenario, base_dir=None):
        self.machine = machine
        self.scenario = scenario
        self.base_dir = base_dir or os.path.dirname(os.path.abspath(__file__))

    def get_raw_path(self):
        return os.path.join(self.base_dir, self.machine, self.scenario)

    def find_mat_files(self):
        mat_files = []
        raw_root = self.get_raw_path()
        if not os.path.exists(raw_root):
            raise FileNotFoundError(f"Raw data path {raw_root} does not exist.")
        for ds_type in ['healthy', 'unhealthy']:
            ds_type_path = os.path.join(raw_root, ds_type)
            for root, _, files in os.walk(ds_type_path):
                for file in files:
                    if file.endswith('node.mat'):
                        mat_files.append((os.path.join(root, file), ds_type))
        return mat_files

    def get_label(self, ds_type):
        return 0 if ds_type == 'healthy' else 1

    def construct_processed_path(self, mat_file_path, ds_type):
        # Extract subfolders after ds_type
        rel_path = os.path.relpath(mat_file_path, os.path.join(self.get_raw_path(), ds_type))
        parts = rel_path.split(os.sep)
        # Insert 'processed', 'timestep', 'augment' after ds_subtype
        ds_subtype = parts[0]
        node_type = parts[3]
        signal_type = parts[4]

        processed_path = os.path.join(
            self.get_raw_path(), ds_type, ds_subtype, 'processed',
            'nodes', node_type, signal_type
        )
        os.makedirs(processed_path, exist_ok=True)
        return processed_path

    def convert_mat_to_hdf5(self, mat_file_path, processed_path, label):
        mat_data = scipy.io.loadmat(mat_file_path)
        structure_data = mat_data.get('S', {})

        for fields in structure_data:
            for name, data, time in zip(fields['name'], fields['data'], fields['time']):
                name = name[0].item()
                data = data.reshape(1, -1)
                time = time.reshape(1, -1)

                hdf5_filename = name + '.hdf5'
                hdf5_path = os.path.join(processed_path, hdf5_filename)

                with h5py.File(hdf5_path, 'w') as f:
                    f.create_dataset('data', data=data)
                    f.create_dataset('time', data=time)

    def process_all(self):
        mat_files = self.find_mat_files()
        for mat_file_path, ds_type in mat_files:
            label = self.get_label(ds_type)
            processed_path = self.construct_processed_path(mat_file_path, ds_type)
            self.convert_mat_to_hdf5(mat_file_path, processed_path, label)

if __name__ == "__main__":
    processor = ProcessRawMSDNodeData(machine='M012', 
                                      scenario='scene_1',
    )
    processor.process_all()
    print("Processing complete. All .mat files converted to .hdf5 format.")