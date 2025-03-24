import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy import signal
import matplotlib.pyplot as plt
import os

# Define EEG channel names
EEG_CHANNELS = ['F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'P3', 'POz', 'P4']

# Define frequency bands (Hz)
FREQ_BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 12),
    'beta': (12, 25),
    'gamma': (25, 50)
}

def load_eeg_data(matlab_path):
    """
    Load EEG data from MATLAB file.
    """
    # Load EEG data from MATLAB file
    mat_data = loadmat(matlab_path)
    
    # Extract EEG channels (columns 2-10)
    eeg_data = []
    for i in range(1, 10):  # Channels 2-10
        channel_data = mat_data['Cn'][0][i].flatten()
        eeg_data.append(channel_data)
    
    # Convert to numpy array and transpose (samples x channels)
    eeg_data = np.array(eeg_data).T
    
    print(f"Loaded EEG data: shape={eeg_data.shape}")
    
    return eeg_data

def extract_frequency_bands_welch(eeg_data, fs, window_duration, overlap):
    """
    Extract different frequency bands from EEG data using Welch's method for multiple time segments.
    This version creates multiple time segments to match RMSSD calculations.
    """
    # Calculate window parameters in samples
    segment_samples = int(fs * window_duration)  # Number of samples per window
    step_samples = int(segment_samples * (1 - overlap))  # Step size overlap
    
    total_samples = len(eeg_data)
    num_segments = max(0, (total_samples - segment_samples) // step_samples + 1)
    
    print(f"Analysis parameters:")
    print(f"  - Signal duration: {total_samples/fs:.2f} seconds")
    print(f"  - Window size: {window_duration} seconds ({segment_samples} samples)")
    print(f"  - Window overlap: {overlap*100}% ({segment_samples - step_samples} samples)")
    print(f"  - Expected number of segments: {num_segments}")
    
    # Dictionary to store segment-wise frequency band data
    all_segments_data = []
    segment_times = []
    
    # Process each time segment
    for seg_idx in range(num_segments):
        start_idx = seg_idx * step_samples
        end_idx = start_idx + segment_samples
        
        if end_idx > total_samples:
            segment_data = np.pad(eeg_data[start_idx:total_samples, :], ((0, end_idx - total_samples), (0, 0)), mode='constant')
        else:
            segment_data = eeg_data[start_idx:end_idx, :]

            
        segment_time = (start_idx + segment_samples/2) / fs  # Center time of segment
        segment_times.append(segment_time)
        
        print(f"Processing segment {seg_idx+1}/{num_segments} (time: {segment_time:.2f}s)")
        
        # Dictionary to store band results for this segment
        segment_bands = {'segment': seg_idx + 1, 'time_sec': segment_time}
        
        # Process each frequency band
        for band_name, (low_freq, high_freq) in FREQ_BANDS.items():
            segment_data = eeg_data[start_idx:end_idx, :]
            
            # Calculate average band power across channels
            band_powers = []
            for ch_idx in range(segment_data.shape[1]):
                channel_data = segment_data[:, ch_idx]
                
                # Apply FFT-based power spectral density estimation
                f, psd = signal.welch(
                    channel_data,
                    fs=fs,
                    nperseg = max(256, segment_samples // 4),  # Use shorter windows within the segment
                    detrend='constant',
                    # Different window types 
                    # window='hann'
                    window=('tukey', 0.25)
                )
                
                # Find indices for the frequency band
                idx_band = np.logical_and(f >= low_freq, f <= high_freq)
                
                # Calculate power in the band
                # if np.any(idx_band):
                #     power = np.trapezoid(psd[idx_band], f[idx_band])
                #     band_powers.append(power)
                # else:
                #     band_powers.append(0)
                power = np.trapezoid(psd[idx_band], f[idx_band]) if np.any(idx_band) else np.nan
                band_powers.append(power)
            
            # Average across all channels
            segment_bands[f'{band_name}_avg'] = np.mean(band_powers)
        
        # Add to the list of segment data
        all_segments_data.append(segment_bands)
    
    # Convert to DataFrame
    bands_df = pd.DataFrame(all_segments_data)
    
    print(f"Extracted {len(bands_df)} segments with frequency band data")
    print(bands_df.head())
    
    return bands_df

def visualize_current_results(band_data):
    """
    Create plots to visualize the results immediately without saving.
    """
    # Plot average power for each band
    plt.figure(figsize=(12, 8))
    for band_name in FREQ_BANDS.keys():
        df = band_data[band_name]
        plt.plot(df['time_sec'], df['average'], label=f'{band_name.capitalize()} ({FREQ_BANDS[band_name][0]}-{FREQ_BANDS[band_name][1]} Hz)')
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Average Band Power')
    plt.title('Average EEG Band Power Over Time')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def main(matlab_path, output_dir, fs):
    """
    Main function to extract frequency bands and save to CSV.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Load EEG data
    print("\nLoading EEG data...")
    eeg_data = load_eeg_data(matlab_path)
    
    # Extract frequency bands for multiple segments
    print("\nExtracting frequency bands using Welch's method for multiple segments...")
    bands_df = extract_frequency_bands_welch(
        eeg_data=eeg_data,
        fs=fs,
        window_duration=15,  # 15 seconds window
        overlap=0.5          # 50% overlap
    )
    
    # Save to CSV
    summary_path = os.path.join(output_dir, "frequency_bands_summary.csv")
    bands_df.to_csv(summary_path, index=False)
    print(f"Saved segment-wise frequency band data to {summary_path}")
    
    # Visualize results
    print("\nVisualizing results...")
    plt.figure(figsize=(12, 8))
    for band in FREQ_BANDS.keys():
        plt.plot(bands_df['time_sec'], bands_df[f'{band}_avg'], label=f'{band.capitalize()}')
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Band Power')
    plt.title('EEG Frequency Band Power Over Time')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    print("\nExtraction complete!")
    return bands_df

if __name__ == "__main__":
    # Parameters
    matlab_path = "Colemak_Data/subject01.mat"
    output_dir = "EEG_frequency_bands"
    fs = 256  # Sampling frequency (Hz)
    
    # Run the extraction
    band_data = main(matlab_path, output_dir, fs)