import numpy as np
import pandas as pd
from scipy import stats
import os
import glob
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

def load_subject_data(subject_id, eeg_path_pattern, ecg_path_pattern, ecg_value):
    """
    Load EEG and ECG data for a specific subject.
    
    Parameters:
    -----------
    subject_id : int
        Subject ID number
    eeg_path_pattern : str
        File path pattern for EEG data files (e.g., "subject{:02d}_*.csv")
    ecg_path_pattern : str
        File path pattern for ECG data files (e.g., "subject{:02d}_*.csv")
    
    Returns:
    --------
    Load EEG and ECG data for a specific subject into a dictionary.
    """
    # Format file paths with subject ID
    eeg_pattern = eeg_path_pattern.format(subject_id)
    ecg_pattern = ecg_path_pattern.format(subject_id)
    
    # Find matching files
    eeg_files = glob.glob(eeg_pattern)
    ecg_files = glob.glob(ecg_pattern)
    
    if not eeg_files:
        raise FileNotFoundError(f"No EEG files found for subject {subject_id} with pattern {eeg_pattern}")
    if not ecg_files:
        raise FileNotFoundError(f"No ECG files found for subject {subject_id} with pattern {ecg_pattern}")
    
    # Use the first matching file for each type
    eeg_file = eeg_files[0]
    ecg_file = ecg_files[0]
    
    # Load data
    eeg_df = pd.read_csv(eeg_file)
    ecg_df = pd.read_csv(ecg_file)
    
    freq_columns = []
    # For 50 band hz data
    # Find frequency columns
    for col in eeg_df.columns:
        if col.startswith('freq_'):
            freq_columns.append(col)
    
    freq_columns.sort(key=lambda x: int(x.split('_')[1]))

    # Frequency bands for 5 band data
    # for col in eeg_df.columns:
    #     if col in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
    #         freq_columns.append(col)

    # # Sort bands in order of frequency
    # band_order = {'delta': 1, 'theta': 2, 'alpha': 3, 'beta': 4, 'gamma': 5}
    # freq_columns.sort(key=lambda x: band_order[x])

    # if len(eeg_df) > len(ecg_df):
    #     print(f"Subject {subject_id}: Truncating EEG data from {len(eeg_df)} to {len(ecg_df)} rows to match ECG data length")
    #     eeg_df = eeg_df.iloc[:len(ecg_df)]

    # Extract EEG frequency data and ECG data
    eeg_data = eeg_df[freq_columns].values
    
    # Identify and filter out rows with empty ECG values
    # Check for NaN, empty string, or other missing value indicators
    valid_rows = ~(ecg_df[ecg_value].isna() | (ecg_df[ecg_value] == '') | (ecg_df[ecg_value].astype(str) == 'nan'))

    # Apply the filter to both EEG and ECG data to keep them aligned
    filtered_eeg_data = eeg_data[valid_rows]
    filtered_ecg_data = ecg_df.loc[valid_rows, [ecg_value]].values

    # Report how many rows were skipped
    skipped_rows = len(eeg_data) - len(filtered_eeg_data)
    if skipped_rows > 0:
        print(f"Subject {subject_id}: Skipped {skipped_rows} rows with empty ECG values")
    
    subject_data = {
        'subject_id': subject_id,
        'eeg_data': filtered_eeg_data,
        'ecg_data': filtered_ecg_data,
        'n_samples': len(filtered_eeg_data),
        'freq_bands': freq_columns
    }

    print(f"Subject {subject_id}: Loaded {subject_data['eeg_data'].shape[0]} valid samples from {eeg_file} and {ecg_file}")

    return subject_data

def run_cca_analysis(eeg_data, ecg_data, n_components=1):
    """
    Run Canonical Correlation Analysis on the data.
    
    Parameters:
    -----------
    eeg_data : numpy.ndarray
        EEG data matrix
    ecg_data : numpy.ndarray
        ECG data matrix
    n_components : int
        Number of canonical components to extract
    
    Returns:
    --------
    results : dict
        Dictionary with CCA results
    """
    # Step 1: Standardize the data
    print("Standardizing data...")
    scaler_eeg = StandardScaler()
    scaler_ecg = StandardScaler()
    
    eeg_scaled = scaler_eeg.fit_transform(eeg_data)
    ecg_scaled = scaler_ecg.fit_transform(ecg_data)
    
    # Step 2: Apply CCA
    print("Applying CCA...")
    cca = CCA(n_components=n_components)
    eeg_c, ecg_c = cca.fit_transform(eeg_scaled, ecg_scaled)
    
    # Step 3: Calculate correlation between canonical variates
    corr = np.corrcoef(eeg_c.flatten(), ecg_c.flatten())[0, 1]
    print(f"Canonical correlation: {corr:.4f}")
    
    # Get the weights for each frequency band
    eeg_weights = cca.x_weights_
    ecg_weights = cca.y_weights_
    
    # Prepare results dictionary
    results = {
        'correlation': corr,
        'eeg_canonical': eeg_c,
        'ecg_canonical': ecg_c,
        'eeg_weights': eeg_weights,
        'ecg_weights': ecg_weights,
        'cca_model': cca
    }
    
    return results

def visualize_cca_results(results, subject_id):
    """
    Visualize CCA results.
    
    Parameters:
    -----------
    results : dict
        Dictionary with CCA results
    subject_id : int
        Subject ID for title
    """
    correlation = results['correlation']
    eeg_c = results['eeg_canonical']
    ecg_c = results['ecg_canonical']
    
    # Plot 1: Canonical correlation scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(eeg_c, ecg_c, alpha=0.5, c='blue', edgecolors='none')
    plt.title(f'Subject {subject_id} - EEG-ECG Canonical Correlation: {correlation:.4f}', fontsize=14)
    plt.xlabel('EEG Canonical Variate', fontsize=12)
    plt.ylabel('ECG Canonical Variate', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def visualize_improved_cca_results(results, subject_id):
    """
    Visualize CCA results with an improved visualization style.
    
    Parameters:
    -----------
    results : dict
        Dictionary with CCA results
    subject_id : int
        Subject ID for title
    """
    correlation = results['correlation']
    eeg_c = results['eeg_canonical']
    ecg_c = results['ecg_canonical']
    
    # Get the slope and intercept for the best-fit line
    slope, intercept, r_value, p_value, std_err = stats.linregress(eeg_c.flatten(), ecg_c.flatten())
    line_x = np.linspace(min(eeg_c), max(eeg_c), 100)
    line_y = slope * line_x + intercept
    
    # Set figure size and style
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    ax.set_facecolor('white')
    ax.grid(True, linestyle='-', linewidth=0.5, alpha=0.7, color='#E6E6E6')
    
    # Create main scatter plot with semi-transparent dots
    plt.scatter(eeg_c, ecg_c, alpha=0.6, c='blue', edgecolors='none', s=50)
    
    # Add the regression line
    plt.plot(line_x, line_y, color='#ff1c45', linewidth=2.5)
    
    # Add canonical correlation annotation
    plt.annotate(f'Canonical Correlation: {correlation:.4f}', 
                xy=(0.05, 0.95), 
                xycoords='axes fraction',
                fontsize=14,
                color='#ff1c45')
    
    # Set up clean axes
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    
    # Add title
    plt.title(f'Subject {subject_id} - EEG-ECG Canonical Correlation', fontsize=16, y=1.02)
    
    # X and Y labels
    plt.xlabel('EEG Canonical Variate', fontsize=13, labelpad=10)
    plt.ylabel('ECG Canonical Variate', fontsize=13, labelpad=10)
    
    # Set equal aspect ratio to make the correlation visualization more intuitive
    plt.axis('equal')
    
    # Adjust layout
    plt.tight_layout()
    plt.show()

def visualize_frequency_weights(results, freq_bands, subject_id):
    """
    Visualize the weights of each frequency band in the CCA.
    
    Parameters:
    -----------
    results : dict
        Dictionary with CCA results
    freq_bands : list
        List of frequency band names
    subject_id : int
        Subject ID for title
    """
    eeg_weights = results['eeg_weights'][:, 0]
    
    # Create DataFrame for easier manipulation
    weights_df = pd.DataFrame({
        'frequency_band': freq_bands,
        'weight': eeg_weights,
        'abs_weight': np.abs(eeg_weights)
    })
    
    # Sort by absolute weight
    sorted_weights = weights_df.sort_values('abs_weight', ascending=False).reset_index(drop=True)
    
    # Plot top 10 most important frequencies
    plt.figure(figsize=(12, 7))
    top_n = min(10, len(sorted_weights))
    top_freqs = sorted_weights.head(top_n)
    
    # Create bar chart
    bars = plt.bar(
        top_freqs['frequency_band'], 
        top_freqs['weight'],
        color=[('red' if w < 0 else 'blue') for w in top_freqs['weight']]
    )
    
    # Add absolute value labels
    for bar, abs_val in zip(bars, top_freqs['abs_weight']):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            0.003 if height < 0 else height + 0.003,
            f'{abs_val:.4f}',
            ha='center', va='bottom', rotation=0, fontsize=9
        )
    
    # Add horizontal line at y=0
    plt.axhline(y=0, color='grey', linestyle='-', alpha=0.3)
    
    # Add labels and title
    plt.title(f'Subject {subject_id} - Top Frequency Bands Contributing to EEG-ECG Correlation', fontsize=16)
    plt.xlabel('Frequency Band', fontsize=14)
    plt.ylabel('Weight Value', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add note about interpretation
    plt.figtext(
        0.5, 0.01, 
        "Note: Larger absolute values indicate stronger contribution to the correlation.\n"
        "The sign (positive or negative) indicates the direction of the relationship.",
        ha='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.5)
    )
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()

def main():
    """
    Main function to run the single subject CCA analysis.
    """
    print("=== Single Subject CCA Analysis: EEG-ECG Data ===")
    
    # Configuration - can be modified by user
    subject_id = 1  # Set the subject ID you want to analyze
    
    # File path patterns - update these to match your file naming convention
    # example: eeg_path_pattern = "path/to/eeg_subject_01.csv"
    eeg_path_pattern = "{Path to the individual EEG .csv files}"
    ecg_path_pattern = "{Path to the individual ECG .csv files}"
    
    # Directory for saving results
    results_dir = f'subject_{subject_id}_results_csv'
    os.makedirs(results_dir, exist_ok=True)
    
    # Load data for the specific subject
    print(f"\nLoading data for Subject {subject_id}...")
    try:
        ecg_value = 'rmssd'
        subject_data = load_subject_data(subject_id, eeg_path_pattern, ecg_path_pattern, ecg_value)
        print(f"Successfully loaded data for Subject {subject_id} with {subject_data['n_samples']} samples")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Run CCA analysis
    print(f"\nRunning CCA analysis for Subject {subject_id}...")
    results = run_cca_analysis(subject_data['eeg_data'], subject_data['ecg_data'])
    
    # Visualize results
    print("\nVisualizing results...")
    visualize_cca_results(results, subject_id)
    visualize_improved_cca_results(results, subject_id)
    visualize_frequency_weights(results, subject_data['freq_bands'], subject_id)
    
    # Save CCA results to CSV files
    print(f"\nSaving CCA results to {results_dir} directory...")
    try:
        # Save canonical correlation value
        pd.DataFrame({'canonical_correlation': [results['correlation']]}).to_csv(
            f'{results_dir}/correlation.csv', index=False)
        
        # Save EEG weights (importance of each frequency band)
        eeg_weights_df = pd.DataFrame({
            'frequency_band': subject_data['freq_bands'],
            'weight': results['eeg_weights'][:, 0],
            'abs_weight': np.abs(results['eeg_weights'][:, 0])
        })
        eeg_weights_df.to_csv(f'{results_dir}/eeg_weights.csv', index=False)
        
        # Save sorted weights
        sorted_weights_df = eeg_weights_df.sort_values('abs_weight', ascending=False).reset_index(drop=True)
        sorted_weights_df.to_csv('Path to save csv', index=False)
        
        # Save ECG weights
        ecg_weights_df = pd.DataFrame({
            'measure': ['rmssd'],
            'weight': results['ecg_weights'][:, 0]
        })
        ecg_weights_df.to_csv(f'{results_dir}/ecg_weights.csv', index=False)
        
        # Save canonical variates (transformed data)
        canonical_df = pd.DataFrame({
            'eeg_canonical': results['eeg_canonical'].flatten(),
            'ecg_canonical': results['ecg_canonical'].flatten()
        })
        canonical_df.to_csv(f'{results_dir}/canonical_variates.csv', index=False)
        
        print(f"CCA results saved to '{results_dir}/' directory")
    except Exception as e:
        print(f"Error saving CCA results: {e}")

    print(f"\nSingle subject CCA analysis for Subject {subject_id} complete!")

if __name__ == "__main__":
    main()