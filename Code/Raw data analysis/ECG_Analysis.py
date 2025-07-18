import scipy.io
import heartpy as hp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

def create_folder_structure(base_path):
    """
    Create the folder structure for storing results
    """
    base_path = Path(base_path)
    base_path.mkdir(exist_ok=True)
    return base_path

def analyze_ecg(file_path, sampling_rate):
    """
    Analyze entire ECG data using HeartPy library with segmentwise processing
    Returns a list of RMSSD values for each segment
    """
    print("Starting ECG analysis...")
    
    try:
        # Load the .mat file
        print("Loading .mat file...")
        mat_data = scipy.io.loadmat(file_path)
        ecg_data = mat_data['Cn'][0][0].flatten()
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
        segment_width = 15
        segment_overlap = 0.5
        
        working_data, measures = hp.process_segmentwise(ecg_data, 
                                                      sample_rate=sampling_rate,
                                                      segment_width=segment_width,
                                                      segment_overlap=segment_overlap,
                                                      calc_freq=False)
        
        print(f"\nSuccessfully processed signal")
        
        # Compute segment start times to ensure full coverage of the signal
        segment_times = np.arange(0, total_duration_seconds - segment_width + 1, segment_width * (1 - segment_overlap))

        # Get the number of segments from this corrected method
        n_segments = len(segment_times)
        print(f"Number of segments (corrected): {n_segments}")

        # Create list to store segment data
        segment_data = []
        
        features = ['rmssd', 'bpm', 'sdnn', 'pnn50', 'sd1', 'sd2', 'breathingrate']

        # Combine measures for each segment
        for i in range(n_segments):
            segment_dict = {'segment': i+1, 'start_time': segment_times[i]}

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
        
        if not results_df.empty:
            print("\nSuccessfully created DataFrame with measures")
            print("\nFirst few rows of data:")
            print(results_df.head())
            
            # Create visualization
            plt.figure(figsize=(15, 15))
            
            # Plot 1: RMSSD
            plt.subplot(3,1,1)
            plt.plot(results_df['start_time'], results_df['rmssd'], 'bo-', label='RMSSD')
            plt.title('RMSSD Values Across Signal Segments')
            plt.xlabel('Start Time (seconds)')
            plt.ylabel('RMSSD (ms)')
            plt.grid(True)
            plt.legend()
            
            # Plot 2: Heart Rate
            plt.subplot(3,1,2)
            plt.plot(results_df['start_time'], results_df['bpm'], 'ro-', label='Heart Rate')
            plt.title('Heart Rate Across Signal Segments')
            plt.xlabel('Start Time (seconds)')
            plt.ylabel('Heart Rate (BPM)')
            plt.grid(True)
            plt.legend()
            
            # Plot 3: SDNN
            # plt.subplot(3,1,3)
            # plt.plot(results_df['start_time'], results_df['sdnn'], 'go-', label='SDNN')
            # plt.title('SDNN Across Signal Segments')
            # plt.xlabel('Start Time (seconds)')
            # plt.ylabel('SDNN (ms)')
            # plt.grid(True)
            # plt.legend()
            
            plt.tight_layout()
            plt.show()
            
            # Print summary statistics
            print("\nSummary Statistics:")
            stats = results_df.describe()
            print(stats.round(2))
            
            # Save to CSV
            output_file = 'ecg_analysis_results.csv'
            results_df.to_csv(output_file, index=False)
            print(f"\nResults saved to {output_file}")
            
            # Print mean values for each measure
            print("\nMean values across all segments:")
            means = results_df.mean()
            for measure in ['rmssd', 'bpm', 'sdnn', 'pnn50']:
                print(f"Mean {measure.upper()}: {means[measure]:.2f}")
            
        else:
            if total_samples < sampling_rate * 15:
                print("Error: ECG signal too short for a 15s segment analysis.")
                return pd.DataFrame(), {}, {}
                
            if 'peaklist' not in working_data or len(working_data['peaklist']) == 0:
                print("Warning: No heartbeats detected. Check ECG signal quality.")


            
            # Debug: Plot raw signal for visual inspection
            plt.figure(figsize=(15,5))
            time = np.arange(len(ecg_data)) / sampling_rate
            plt.plot(time, ecg_data)
            plt.title('Raw ECG Signal')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Amplitude')
            plt.grid(True)
            plt.show()
        
        return results_df, working_data, measures
        
    except Exception as e:
        print(f"\nError in signal processing: {str(e)}")
        print("\nDetailed error information:")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(), {}, {}

if __name__ == "__main__":
    # Individual file for the process ex."subject_1.mat"
    file_path = "{Path to your ECG .mat file}"
    
    print(f"Starting analysis with file: {file_path}")
    print(f"Sampling rate: 256 Hz")
    
    results_df, working_data, measures = analyze_ecg(file_path, sampling_rate=256)