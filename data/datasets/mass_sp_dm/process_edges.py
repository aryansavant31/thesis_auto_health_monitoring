import os
import h5py
import scipy.io
import numpy as np

class ProcessRawMSDEdgeData:
    """
    Class to process raw edge data for MSD fault detection.
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
            for ds_subtype in os.listdir(ds_type_path):
                edges_path = os.path.join(ds_type_path, ds_subtype, 'raw', 'edges')
                if os.path.exists(edges_path):  # Check specifically for the 'edges' folder
                    for root, _, files in os.walk(edges_path):
                        for file in files:
                            if file.endswith('_adj.mat'):  # Adjusted for edge files
                                mat_files.append((os.path.join(root, file), ds_type, ds_subtype))
        return mat_files

    def get_label(self, ds_type):
        return 0 if ds_type == 'healthy' else 1

    def construct_processed_path(self, ds_type, ds_subtype):
        processed_path = os.path.join(
            self.get_raw_path(), ds_type, ds_subtype, 'processed', 'edges'
        )
        os.makedirs(processed_path, exist_ok=True)
        return processed_path

    def convert_mat_to_hdf5(self, mat_file_path, processed_path, label):
        if mat_file_path.endswith('.mat') and os.path.exists(mat_file_path):
            mat_data = scipy.io.loadmat(mat_file_path)
            key_name = next((signal for signal in mat_data.keys() if '_adj' in signal), None)

            if key_name is None:
                raise ValueError(f"No adjacency matrix containing '_adj' found in {mat_file_path}")
            adj_matrix = mat_data[key_name]

            # add an extra dimension to make it (N, N, 1)
            adj_matrix = np.expand_dims(adj_matrix, axis=-1)

            hdf5_filename = os.path.splitext(os.path.basename(mat_file_path))[0] + '.hdf5'
        
        else:
            # Create a default 2x2 adjacency matrix with all elements 0
            adj_matrix = -1 * np.ones((2, 2, 1))
            hdf5_filename = 'null_adj.hdf5'

        hdf5_path = os.path.join(processed_path, hdf5_filename)
        with h5py.File(hdf5_path, 'w') as f:
            f.create_dataset('adj_matrix', data=adj_matrix)
            f.create_dataset('label', data=label)

    def process_all(self):
        raw_root = self.get_raw_path()

        for ds_type in ['healthy', 'unhealthy']:
            ds_type_path = os.path.join(raw_root, ds_type)
            for ds_subtype in os.listdir(ds_type_path):
                edges_path = os.path.join(ds_type_path, ds_subtype, 'raw', 'edges')
                processed_path = self.construct_processed_path(ds_type, ds_subtype)
                if not os.path.exists(edges_path):  # If 'edges' folder does not exist
                    print(f"No raw edges folder found for {ds_type}/{ds_subtype}. Creating default adjacency matrix.")
                    self.convert_mat_to_hdf5("default", processed_path, self.get_label(ds_type))
                else:
                    mat_files = self.find_mat_files()
                    for mat_file_path, ds_type, ds_subtype in mat_files:
                        label = self.get_label(ds_type)
                        self.convert_mat_to_hdf5(mat_file_path, processed_path, label)

if __name__ == "__main__":
    processor = ProcessRawMSDEdgeData(machine="M004", scenario="scene_1")
    processor.process_all()
    print("Edge adjacency matrices processed and saved.")