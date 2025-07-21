from scipy.io import loadmat
from pathlib import Path

def load_mat(filepath):
    """
    Read a .mat file and return an object with its contents
    """
    print(f"Loading .mat file {filepath}")
    mat_data = loadmat(filepath)
    return mat_data


def create_folder_structure(base_path):
    """
    Create the folder structure for storing results
    """
    base_path = Path(base_path)
    base_path.mkdir(exist_ok=True)
    return base_path
