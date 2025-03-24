import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy import signal
import matplotlib.pyplot as plt
import os
from pathlib import Path
import traceback

# Define EEG channel names
EEG_CHANNELS = ['F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'P3', 'POz', 'P4']

def create_folder_structure(base_path):
    """
    Create the folder structure for storing results
    """
    base_path = Path(base_path)
    base_path.mkdir(exist_ok=True)
    return base_path

def load_eeg_data(matlab_path, cell_index):
    """
    Load EEG data from MATLAB file for a specific trial (cell index).
    """
    # Load EEG data from MATLAB file
    mat_data = loadmat(matlab_path)
    
    # Extract EEG channels (columns 1-9, which are indices 1-9 in Cn[cell_index])
    eeg_data = []
    for i in range(1, 10):  # Channels 1-9 (indices 1-9)
        channel_data = mat_data['Cn'][cell_index][i].flatten()
        eeg_data.append(channel_data)
    
    # Convert to numpy array and transpose (samples x channels)
    eeg_data = np.array(eeg_data).T
    
    print(f"Loaded EEG data: shape={eeg_data.shape}")
    
    return eeg_data

def validate_eeg_data(eeg_data):
    """
    Validate EEG data and return whether it's valid for processing
    """
    if eeg_data is None or eeg_data.size == 0:
        return False
    
    # Check if data contains only zeros or if it's too short
    if np.all(eeg_data == 0) or eeg_data.shape[0] < 256:
        return False
    
    # Check if data contains any infinite values
    if np.any(np.isinf(eeg_data)):
        return False
    
    return True

def extract_individual_frequencies(eeg_data, fs, window_duration, overlap, max_freq):
    """
    Extract power at individual frequencies from EEG data for multiple time segments.
    Groups the frequency values into 1 Hz bins (e.g., 1 Hz, 2 Hz, 3 Hz).
    
    Parameters:
    -----------
    eeg_data: numpy.ndarray 
        EEG data with shape (samples x channels)
    fs: int
        Sampling frequency
    window_duration: float
        Duration of each segment window in seconds
    overlap: float
        Overlap between consecutive windows (0-1)
    max_freq: float
        Maximum frequency to include in the output
        
    Returns:
    --------
    pandas.DataFrame with frequency data for each segment
    """
    # Calculate window parameters in samples
    segment_samples = int(fs * window_duration)  # Number of samples per window
    step_samples = int(segment_samples * (1 - overlap))  # Step size based on overlap
    
    total_samples = len(eeg_data)
    num_segments = max(0, (total_samples - segment_samples) // step_samples + 1)
    
    print(f"Analysis parameters:")
    print(f"  - Signal duration: {total_samples/fs:.2f} seconds")
    print(f"  - Window size: {window_duration} seconds ({segment_samples} samples)")
    print(f"  - Window overlap: {overlap*100}% ({segment_samples - step_samples} samples)")
    print(f"  - Number of segments: {num_segments}")
    
    # List to store segment-wise frequency data
    all_segments_data = []
    
    # Set parameters for Welch's method
    nperseg = max(256, segment_samples // 4)  # Use shorter windows within the segment
    
    # Process each time segment
    for seg_idx in range(num_segments):
        start_idx = seg_idx * step_samples
        end_idx = start_idx + segment_samples
        
        if end_idx > total_samples:
            # If we reach the end of the signal, pad with zeros
            segment_data = np.pad(eeg_data[start_idx:total_samples, :], 
                                  ((0, end_idx - total_samples), (0, 0)), 
                                  mode='constant')
        else:
            segment_data = eeg_data[start_idx:end_idx, :]
        
        segment_time = (start_idx + segment_samples/2) / fs  # Center time of segment
        print(f"Processing segment {seg_idx+1}/{num_segments} (time: {segment_time:.2f}s)")
        
        # Dictionary to store this segment's data
        segment_dict = {'segment': seg_idx + 1, 'time_sec': segment_time}
        
        # Process each channel and calculate average PSD across channels
        channel_psds = {}
        freqs = None
        
        for ch_idx in range(segment_data.shape[1]):
            channel_data = segment_data[:, ch_idx]
            channel_name = EEG_CHANNELS[ch_idx]
            
            # Apply Welch's method to get power spectral density
            # Window options:
            # - ('tukey', 0.25): Preserves signal amplitude (75% flat middle). Better for amplitude relationships.
            # - 'hann': Better frequency separation. Preferred for CCA and frequency analysis.
            f, psd = signal.welch(
                channel_data,
                fs=fs,
                nperseg=nperseg,
                detrend='constant',
                window = 'hann',
                #window=('tukey', 0.25)
            )
            
            # Store frequency array on first iteration
            if freqs is None:
                freqs = f

            # filter PSD with 10log 10
            psd = 10 * np.log10(psd)
            # Safetty check for invalid values
            # psd = 10 * np.log10(np.maximum(psd, 1e-10))

            # Store this channel's PSD
            channel_psds[channel_name] = psd
        
        # For each frequency, calculate the average power across all channels
        # and store it in the segment dictionary
        grouped_freqs = {}

        for freq_idx, freq in enumerate(freqs):
            # Only include frequencies up to max_freq
            if freq <= max_freq:
                # Calculate the bin key (rounded down to nearest integer)
                bin_key = int(np.floor(freq) + 1)
                # Calculate average power at this frequency across all channels
                avg_power = np.mean([channel_psds[ch][freq_idx] for ch in EEG_CHANNELS])

                if bin_key not in grouped_freqs:
                    grouped_freqs[bin_key] = []
                grouped_freqs[bin_key].append(avg_power)

        # Calculate the average power for each 1 Hz frequency bin
        for bin_key, values in grouped_freqs.items():
            segment_dict[f'freq_{bin_key}'] = np.mean(values)  

        # Add to the list of segment data
        all_segments_data.append(segment_dict)

    # Convert to DataFrame
    freq_df = pd.DataFrame(all_segments_data)
    
    print(f"Extracted {len(freq_df)} segments with individual frequency data")
    print(f"Frequency columns: {len([col for col in freq_df.columns if 'freq_' in col])}")
    
    return freq_df

def process_subject_trial(file_path, subject_id, trial, fs):
    """
    Process a single subject/trial and return its frequency data
    """
    try:
        print(f"Processing {subject_id}, trial {trial+1}...")
        
        # Load EEG data for this trial
        eeg_data = load_eeg_data(str(file_path), trial)
        
        # Skip if data is invalid
        if not validate_eeg_data(eeg_data):
            print(f"Invalid EEG data for {subject_id}, trial {trial+1}, skipping")
            return None, False, subject_id
        
        # Extract individual frequency data
        freq_df = extract_individual_frequencies(
            eeg_data=eeg_data,
            fs=fs,
            window_duration=30,  # 15 seconds window
            overlap=0.5,         # 50% overlap
            max_freq=50          # Maximum frequency to include
        )
        
        if not freq_df.empty:
            print(f"Successfully processed {subject_id}, trial {trial+1}")
            return freq_df, True, subject_id
        else:
            print(f"No valid frequency data for {subject_id}, trial {trial+1}, skipping")
            return None, False, subject_id
            
    except Exception as e:
        print(f"Error processing {subject_id}, trial {trial+1}: {str(e)}")
        traceback.print_exc()
        return None, False, subject_id

def process_all_subjects(base_dir, output_base_dir, num_subjects, trials_per_subject, fs):
    """
    Process all subjects and their trials for EEG data, organizing by trial
    """
    # Create base output directory
    base_output_dir = create_folder_structure(output_base_dir)
    
    # Dictionary to collect results by trial
    trial_data = {trial: [] for trial in range(1, trials_per_subject + 1)}
    subject_ids = {trial: [] for trial in range(1, trials_per_subject + 1)}  # Store subject IDs separately
    
    # Process each subject
    for subject in range(1, num_subjects + 1):
        subject_id = f"subject{subject:02d}"
        print(f"\nProcessing {subject_id}")
        
        # Process file
        file_path = Path(base_dir) / f"{subject_id}.mat"
        if not file_path.exists():
            print(f"File not found: {file_path}")
            continue
            
        # Process each trial for the subject
        for trial in range(trials_per_subject):
            trial_num = trial + 1
            freq_df, success, subj_id = process_subject_trial(
                file_path=file_path,
                subject_id=subject_id,
                trial=trial,
                fs=fs
            )
            
            if success and freq_df is not None and not freq_df.empty:
                # Append this subject's data to the appropriate trial list
                trial_data[trial_num].append(freq_df)
                subject_ids[trial_num].append(subj_id)
            else:
                print(f"Failed to process trial {trial_num} for {subject_id}")
    
    # After processing all subjects, save data by trial
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
                    output_file = trial_folder / f'eeg_{subject_id}.csv'
                    subject_df.to_csv(output_file, index=False)
        else:
            print(f"No data available for trial {trial_num}")

def main():
    # Set your paths here
    base_directory = "/Users/dokyuhan/Documents/ECG_EEG/Colemak_Data"
    output_directory = "/Users/dokyuhan/Documents/ECG_EEG/30sec_EEG_data_hann"
    
    # Process all subjects
    process_all_subjects(
        base_dir=base_directory,
        output_base_dir=output_directory,
        num_subjects=10,
        trials_per_subject=15,
        fs=256
    )

if __name__ == "__main__":
    main()