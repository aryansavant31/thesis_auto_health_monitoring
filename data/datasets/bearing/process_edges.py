import os
import h5py
import numpy as np

class AdjacencyMatrixGenerator:
    def __init__(self, base_dir=None):
        self.base_dir = base_dir or os.path.dirname(os.path.abspath(__file__))

    def generate_matrix(self, seed=None):
        np.random.seed(seed)
        mat = np.random.randint(0, 2, size=(5, 5))
        np.fill_diagonal(mat, 0)  # No self-loops
        return mat

    def process(self):
        scene_path = os.path.join(self.base_dir, "cwru", "scene_1")
        for ds_type in os.listdir(scene_path):
            ds_type_path = os.path.join(scene_path, ds_type)

            # processed_path = os.path.join(ds_type_path, "processed", "edges")
            # os.makedirs(processed_path, exist_ok=True)
            ds_subtypes = [
                d for d in os.listdir(ds_type_path)
            ]
            if ds_type == "healthy":
                matrix = self.generate_matrix(seed=42)
                for ds_subtype in ds_subtypes:

                    processed_path = os.path.join(ds_type_path, ds_subtype, "processed", "edges")
                    os.makedirs(processed_path, exist_ok=True)

                    self.save_matrix(processed_path, ds_subtype, matrix)
            else:  # unhealthy
                for ds_subtype in ds_subtypes:

                    matrix = self.generate_matrix()
                    processed_path = os.path.join(ds_type_path, ds_subtype, "processed", "edges")
                    os.makedirs(processed_path, exist_ok=True)

                    self.save_matrix(processed_path, ds_subtype, matrix)

    def save_matrix(self, folder, ds_subtype, matrix):
        file_path = os.path.join(folder, f"{ds_subtype}.hdf5")
        with h5py.File(file_path, "w") as f:
            f.create_dataset("adj_mat", data=matrix)

if __name__ == "__main__":
    generator = AdjacencyMatrixGenerator()
    generator.process()
    print("Adjacency matrices generated and saved.")