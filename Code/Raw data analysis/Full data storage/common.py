import sys
import numpy as np
from scipy.io import loadmat
from pathlib import Path


def get_cli_arguments(default_input, default_output):
    """
    Get the paths to the data directories from the command line
    """
    if len(sys.argv) == 3:
        input_directory = sys.argv[1]
        output_directory = sys.argv[2]
    else:
        input_directory = default_input
        output_directory = default_output
    return input_directory, output_directory


def load_mat(filepath):
    """
    Read a .mat file and return an object with its contents
    """
    print(f"Loading .mat file {filepath}")
    mat_data = loadmat(filepath)
    return mat_data


def validate_mat_data(mat_data):
    """
    Validate .mat data and return whether it's valid for processing
    """
    if mat_data is None or mat_data.size == 0:
        return False

    # Check if data contains only zeros or if it's too short
    if np.all(mat_data == 0) or mat_data.shape[0] < 256:
        return False

    # Check if data contains any infinite values
    if np.any(np.isinf(mat_data)):
        return False

    return True


def create_folder_structure(base_path):
    """
    Create the folder structure for storing results
    """
    base_path = Path(base_path)
    base_path.mkdir(exist_ok=True)
    return base_path


def save_data_to_csv(trial_data, output_base_dir, subject_ids, data_type):
    """
    Store the results of processing each trial in a separate CSV file.
    Output files are named after the subject, and placed in the folder
    corresponding to the trial.
    """
    # Create the required directory
    base_output_dir = create_folder_structure(output_base_dir)
    # Create the folders and files
    for trial_num, data_list in trial_data.items():
        if data_list:
            # Create trial folder
            trial_folder = base_output_dir / f"trial{trial_num:02d}"
            trial_folder.mkdir(exist_ok=True)

            # Save individual subject files within the trial folder
            for i, subject_df in enumerate(data_list):
                if not subject_df.empty:
                    # Get subject ID from the separate array
                    subject_id = subject_ids[trial_num][i]
                    # Name the file according to the type of data evaluated and the subject
                    output_file = trial_folder / f'{data_type}_{subject_id}.csv'
                    subject_df.to_csv(output_file, index=False)
        else:
            print(f"No data available for trial {trial_num}")
