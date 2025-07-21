import numpy as np
import pandas as pd
import heartpy as hp
import traceback
from pathlib import Path

from common import (get_cli_arguments,
                    load_mat,
                    save_data_to_csv,
                    validate_mat_data)


# Default values for the input and output directories
INPUT_DIRECTORY = "../../Colemak_Data"
OUTPUT_DIRECTORY = "../../30sec_ECG_data"


#from joblib import Parallel, delayed
    #results = Parallel(n_jobs=num_threads)(delayed(sum_primes_segment)(start, end) for start, end in zip(starts, ends))

# GEF: Should create another file with the common functions used in EEG ECG.
# This will avoid code duplication and make the codebase simpler
# - create_folder_structure
# - validate data
# - load mat file
# - select desired sample
# - configure sampling windows

# GEF: Speeding up
# Can launch a thread per subject. This way each thread only deals with one input file
# We need to verify if memory will be enough to load all files simultaneously
# If not, we can process all users sequentially, but processing the trials in parallel


def interpolate_missing_values(df, features):
    """
    Interpolate missing values in a DataFrame:
    1. If there are 3 or fewer consecutive NaN values in the middle, interpolate linearly
    2. If NaN values are at the start or end, fill with the nearest valid value

    Parameters:
    df (pandas.DataFrame): DataFrame containing the data
    features (list): List of column names to interpolate

    Returns:
    pandas.DataFrame: DataFrame with interpolated values
    """
    # Make a copy to avoid modifying the original DataFrame
    result_df = df.copy()

    for feature in features:
        # Skip if the feature is not in the DataFrame
        if feature not in result_df.columns:
            continue

        # Get the series
        series = result_df[feature]

        # Find runs of NaNs
        # Create a boolean mask for NaN values
        is_nan = series.isna()

        # Skip if all values are NaN
        if is_nan.all():
            continue

        # Create groups of consecutive NaN values
        # This trick uses the cumulative sum which increases only when is_nan changes
        nan_groups = (is_nan != is_nan.shift()).cumsum()

        # Handle the first values if they are NaN (forward fill from first valid value)
        if is_nan.iloc[0]:
            # Find the first valid value
            first_valid_idx = is_nan.idxmin()  # Index of first False in is_nan
            first_valid_value = series.loc[first_valid_idx]

            # Get indices of the first group of NaNs
            first_group_id = nan_groups.iloc[0]
            first_group_indices = nan_groups[nan_groups == first_group_id].index

            # Fill with the first valid value
            for idx in first_group_indices:
                result_df.loc[idx, feature] = first_valid_value

        # Handle the last values if they are NaN (backward fill from last valid value)
        if is_nan.iloc[-1]:
            # Find the last valid value
            last_valid_idx = is_nan[::-1].idxmin()  # Index of first False in reversed is_nan
            last_valid_value = series.loc[last_valid_idx]

            # Get indices of the last group of NaNs
            last_group_id = nan_groups.iloc[-1]
            last_group_indices = nan_groups[nan_groups == last_group_id].index

            # Fill with the last valid value
            for idx in last_group_indices:
                result_df.loc[idx, feature] = last_valid_value

        # For each group of NaNs in the middle, check the length
        for group_id in nan_groups[is_nan].unique():
            # Get the indices of this group
            group_indices = nan_groups[nan_groups == group_id].index

            # Skip if this is the first or last group (already handled)
            if group_indices.min() == 0 or group_indices.max() == len(series) - 1:
                continue

            # If the group has 3 or fewer NaNs, interpolate
            if len(group_indices) <= 3:
                # Get the start and end indices of the group
                start_idx = group_indices.min()
                end_idx = group_indices.max()

                # Get the values before and after the gap
                before_val = series.iloc[start_idx - 1]
                after_val = series.iloc[end_idx + 1]

                # Only interpolate if both before and after values are not NaN
                if not pd.isna(before_val) and not pd.isna(after_val):
                    # Calculate the step size for linear interpolation
                    step = (after_val - before_val) / (len(group_indices) + 1)

                    # Fill in the values
                    for i, idx in enumerate(group_indices):
                        result_df.loc[idx, feature] = before_val + step * (i + 1)

    return result_df


def analyze_ecg(mat_data, sampling_rate, cell_index, subject_id):
    """
    Analyze ECG data using HeartPy library
    """
    try:
        print(f"Processing {subject_id}, trial {cell_index+1}...")

        ecg_data = mat_data['Cn'][cell_index][0].flatten()

        # Display signal information
        total_samples = len(ecg_data)
        total_duration_seconds = total_samples / sampling_rate
        total_duration_minutes = total_duration_seconds / 60

        print(f"\nSignal Duration Information:")
        print(f"Total samples: {total_samples}")
        print(f"Duration in seconds: {total_duration_seconds:.2f} seconds")
        print(f"Duration in minutes: {total_duration_minutes:.2f} minutes")
        print(f"Data range: [{np.min(ecg_data):.2f}, {np.max(ecg_data):.2f}]")

        # Process the signal using HeartPy's segmentwise processing
        print("\nProcessing signal with HeartPy...")
        segment_width = 15 # seconds
        segment_overlap = 0.5

        # GEF: I am getting an message: UserWarning.  This seems to come from heartpy
        # Does this affect the results?

        # GEF: Should be more specific when catching exceptions to better determine what is going wrong
        #      avoid:  except Exception

        # Validate ECG data before processing
        if not validate_mat_data(ecg_data):
            print(f"Invalid ECG data for trial {cell_index+1}")
            # Create an empty DataFrame with the expected columns
            columns = ['segment', 'start_time', 'rmssd', 'bpm', 'sdnn', 'pnn50', 'sd1', 'sd2', 'breathingrate', 'ln_rmssd']
            results_df = pd.DataFrame(columns=columns)
        else:
            try:
                working_data, measures = hp.process_segmentwise(ecg_data,
                                                            sample_rate=sampling_rate,
                                                            segment_width=segment_width,
                                                            segment_overlap=segment_overlap,
                                                            calc_freq=False)

                print(f"\nSuccessfully processed signal")

                # Compute segment start times to ensure full coverage of the signal
                segment_times = np.arange(0, total_duration_seconds - segment_width + 1,
                                        segment_width * (1 - segment_overlap))

                # Get the number of segments from this method
                n_segments = len(segment_times)
                print(f"Number of segments: {n_segments}")

                # Create list to store segment data
                segment_data = []

                features = ['rmssd', 'bpm', 'sdnn', 'pnn50', 'sd1', 'sd2', 'breathingrate']

                # Combine measures for each segment
                for i in range(min(n_segments, len(measures['rmssd']))):
                    segment_dict = {
                        'segment': i+1,
                        'start_time': segment_times[i]
                    }

                    # Assign values safely, replacing missing ones with NaN
                    for feature in features:
                        segment_dict[feature] = measures[feature][i] if i < len(measures[feature]) else np.nan

                    # Handle logarithm safely (only for RMSSD)
                    if not np.isnan(segment_dict['rmssd']) and segment_dict['rmssd'] > 0:
                        segment_dict['ln_rmssd'] = np.log(segment_dict['rmssd'])
                    else:
                        segment_dict['ln_rmssd'] = np.nan

                    # Append the completed dictionary to the list
                    segment_data.append(segment_dict)

                # Convert to DataFrame
                results_df = pd.DataFrame(segment_data)

                # Apply interpolation for missing values (up to 3 consecutive NaNs)
                # GEF: Is this the same list as 'features'? Why not use that other one?
                features_to_interpolate = ['rmssd', 'bpm', 'sdnn', 'pnn50', 'sd1', 'sd2', 'breathingrate']
                results_df = interpolate_missing_values(results_df, features_to_interpolate)

                # Recalculate ln_rmssd after interpolation
                mask = (~pd.isna(results_df['rmssd'])) & (results_df['rmssd'] > 0)
                results_df.loc[mask, 'ln_rmssd'] = np.log(results_df.loc[mask, 'rmssd'])

                # Print summary statistics
                print("\nSummary Statistics (after interpolation):")
                stats = results_df.describe()
                print(stats.round(2))

            except Exception as e:
                print(f"Error in signal processing: {str(e)}")
                print("\nDetailed error information:")
                traceback.print_exc()
                # Create an empty DataFrame with the expected columns
                columns = ['segment', 'start_time', 'rmssd', 'bpm', 'sdnn', 'pnn50', 'sd1', 'sd2', 'breathingrate', 'ln_rmssd']
                results_df = pd.DataFrame(columns=columns)

        return results_df, True

    except Exception as e:
        print(f"Critical error processing trial {cell_index + 1} for {subject_id}: {str(e)}")
        traceback.print_exc()
        return None, False


def process_all_subjects(base_dir, num_subjects, trials_per_subject):
    """
    Process all subjects and their trials, organizing by trial instead of by subject
    """
    # List with the results of all trials and subjects
    # It should contain dictionaries with the trial, subject and data
    results = []

    for subject in range(1, num_subjects + 1):
        subject_id = f"subject{subject:02d}"
        print(f"\nProcessing {subject_id}")

        # Process file
        file_path = Path(base_dir) / f"{subject_id}.mat"
        if not file_path.exists():
            print(f"File not found: {file_path}")
            continue

        mat_data = load_mat(file_path)

        # Process each trial for the subject
        for trial in range(trials_per_subject):
            result = process_ecg_trial_data(mat_data, subject_id, trial)
            results.append(result)

    return results


def process_ecg_trial_data(mat_data, subject_id, trial):
    """
    Function to do the processing of a single trial for a single subject
    This function could be called in parallel for each of the trials
    Returns a dictionary with the id of the trial, the user, and the data obtained
    """
    trial_num = trial + 1
    results_df, success = analyze_ecg(
        mat_data=mat_data,
        sampling_rate=256,
        cell_index=trial,
        subject_id=subject_id
    )

    if success:
        return {'trial': trial,
                'subject_id': subject_id,
                'data': results_df
                }
    else:
        print(f"Failed to process trial {trial_num} for {subject_id}")
        return {'trial': trial,
                'subject_id': subject_id,
                'data': None
                }


def main():
    """
    Entry function for the program
    """
    input_directory, output_directory, num_subjects = get_cli_arguments(INPUT_DIRECTORY, OUTPUT_DIRECTORY)

    # Process all subjects
    results = process_all_subjects(
        base_dir=input_directory,
        num_subjects=num_subjects,
        trials_per_subject=15
    )

    # After processing all subjects, save data by trial
    print(f"Saving output files into: {output_directory}")
    save_data_to_csv(results, output_directory, 'ecg')


if __name__ == "__main__":
    main()
