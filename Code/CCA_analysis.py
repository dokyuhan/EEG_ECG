import numpy as np
import pandas as pd
from scipy import stats
import os
import glob
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.cross_decomposition import CCA

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

    # Extract EEG frequency data and ECG data
    eeg_data = eeg_df[freq_columns].values

    # Identify and filter out rows with empty ECG values
    # Check for NaN, empty string, or other missing value indicators
    valid_rows = ~(ecg_df[ecg_value].isna() | (ecg_df[ecg_value] == '') | (ecg_df[ecg_value].astype(str) == 'nan'))

    # Apply the filter to both EEG and ECG data to keep them aligned
    #filtered_eeg_data = eeg_data[valid_rows]
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


def pool_subject_data(num_subjects, eeg_path_pattern, ecg_path_pattern, ecg_value):
    """
    Pool data from multiple subjects using dictionaries.

    Parameters:
    -----------
    num_subjects : int
        Number of subjects to include
    eeg_path_pattern : str
        File path pattern for EEG data files
    ecg_path_pattern : str
        File path pattern for ECG data files
    expected_samples_per_subject : int
        Expected number of samples per subject

    Returns:
    --------
    all_subjects : dict
        Dictionary containing data for each subject
    pooled_eeg : numpy.ndarray
        Pooled EEG data matrix
    pooled_ecg : numpy.ndarray
        Pooled ECG data matrix
    """
    all_subjects = {}
    all_eeg = []
    all_ecg = []

    # For verification
    row_indices_by_subject = {}

    for subject_id in range(1, num_subjects + 1):
        try:
            # Load data for this subject
            subject_data = load_subject_data(subject_id, eeg_path_pattern, ecg_path_pattern, ecg_value)
            eeg_data = subject_data['eeg_data']
            ecg_data = subject_data['ecg_data']

            # Validate dimensions
            if eeg_data.shape[0] != ecg_data.shape[0]:
                print(f"Warning: Subject {subject_id} has mismatched sample counts. EEG: {eeg_data.shape[0]}, ECG: {ecg_data.shape[0]}")
                min_samples = min(eeg_data.shape[0], ecg_data.shape[0])
                eeg_data = eeg_data[:min_samples]
                ecg_data = ecg_data[:min_samples]

            # Skip subject entirely if they have no valid data after filtering
            if eeg_data.shape[0] == 0:
                print(f"Warning: Subject {subject_id} has no valid samples after filtering out empty ECG values. Skipping subject.")
                continue

            # Store this subject's data in the dictionary
            all_subjects[subject_id] = {
                'subject_id': subject_id,
                'eeg_data': eeg_data,
                'ecg_data': ecg_data,
                'n_samples': eeg_data.shape[0]
            }

            # Add to the pooled data
            all_eeg.append(eeg_data)
            all_ecg.append(ecg_data)

            # Track the starting and ending row indices for this subject
            start_idx = sum(len(x) for x in all_eeg[:-1])  # Sum of lengths of previous subjects
            end_idx = start_idx + len(eeg_data)
            row_indices_by_subject[subject_id] = (start_idx, end_idx - 1)

        except Exception as e:
            print(f"Error loading data for subject {subject_id}: {e}")

    # Stack all data
    if not all_eeg or not all_ecg:
        raise ValueError("No valid data loaded for any subject")

    pooled_eeg = np.vstack(all_eeg)
    pooled_ecg = np.vstack(all_ecg)

    print(f"Pooled data dimensions: EEG {pooled_eeg.shape}, ECG {pooled_ecg.shape}")

    return all_subjects, pooled_eeg, pooled_ecg


def run_cca_analysis(eeg_data, ecg_data, trial, n_components=1):
    """
    Run Canonical Correlation Analysis on the pooled data.

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
    # Standardize the data
    print("Standardizing data...")
    scaler_eeg = StandardScaler()
    scaler_ecg = StandardScaler()

    eeg_scaled = scaler_eeg.fit_transform(eeg_data)
    ecg_scaled = scaler_ecg.fit_transform(ecg_data)

    # Apply CCA
    print("Applying CCA...")
    cca = CCA(n_components=n_components)
    eeg_c, ecg_c = cca.fit_transform(eeg_scaled, ecg_scaled)

    # Calculate correlation between canonical variates
    corr = np.corrcoef(eeg_c.flatten(), ecg_c.flatten())[0, 1]
    print(f"Canonical correlation: {corr:.4f}")

    # Get the weights for each frequency band
    eeg_weights = cca.x_weights_
    ecg_weights = cca.y_weights_

    # Calculate per-row contributions (weighted values)
    print("Calculating per-row feature contributions using CCA weights...")
    eeg_contributions = np.abs(eeg_scaled * eeg_weights.T)  # (n_samples, n_features)
    ecg_contributions = np.abs(ecg_scaled * ecg_weights.T)  # (n_samples, 1)

    # Save contributions to CSV
    print("Saving weighted feature contributions...")
    eeg_columns = [f'freq_{i+1}_weighted' for i in range(eeg_contributions.shape[1])]
    df_contributions = pd.DataFrame(eeg_contributions, columns=eeg_columns)
    df_contributions['ECG_weighted'] = ecg_contributions.flatten()
    df_contributions.insert(0, 'Row', range(1, len(df_contributions) + 1))

    df_contributions.to_csv('subject_results_csv/row_feature_contributions.csv', index=False)
    print("Saved per-row weighted feature contributions to CSV.")

    # Haufe transformation (interpretable pattern)
    print("Applying Haufe transformation...")
    cov_X = np.cov(eeg_scaled, rowvar=False)  # EEG covariance
    haufe_activation = cov_X @ eeg_weights  # (features x components)

    haufe_df = pd.DataFrame({
        'frequency_band': [f'freq_{i+1}' for i in range(haufe_activation.shape[0])],
        'activation_value': haufe_activation[:, 0]
    })

    # Sort by descending activation value
    haufe_df_sorted = haufe_df.sort_values(by='activation_value', ascending=False)
    # Path to save Haufe results
    haufe_df_sorted.to_csv(f'subject_results/haufe_results/trial_{trial}.csv', index=False)
    print("Saved sorted Haufe activation pattern to CSV.")

    results = {
        'correlation': corr,
        'eeg_canonical': eeg_c,
        'ecg_canonical': ecg_c,
        'eeg_weights': eeg_weights,
        'ecg_weights': ecg_weights,
        'haufe_activation': haufe_activation,
        'cca_model': cca
    }

    return results


def visualize_cca_results(results):
    """
    Visualize CCA results.

    Parameters:
    -----------
    results : dict
        Dictionary with CCA results
    """
    correlation = results['correlation']
    eeg_c = results['eeg_canonical']
    ecg_c = results['ecg_canonical']
    eeg_weights = results['eeg_weights']

    # Plot 1: Canonical correlation scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(eeg_c, ecg_c, alpha=0.5, c='blue', edgecolors='none')
    plt.title(f'EEG-ECG Canonical Correlation: {correlation:.4f}', fontsize=14)
    plt.xlabel('EEG Canonical Variate', fontsize=12)
    plt.ylabel('ECG Canonical Variate', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    #plt.savefig('pooled_cca_correlation.png', dpi=300)
    plt.show()


def visualize_improved_cca_results(results):
    """
    Visualize CCA results with an improved visualization style similar to the reference image.

    Parameters:
    -----------
    results : dict
        Dictionary with CCA results
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
    #plt.style.use('seaborn-whitegrid')
    ax = plt.gca()
    ax.set_facecolor('white')
    ax.grid(True, linestyle='-', linewidth=0.5, alpha=0.7, color='#E6E6E6')

    # Create main scatter plot with semi-transparent gray dots
    plt.scatter(eeg_c, ecg_c, alpha=0.6, c='blue', edgecolors='none', s=50)

    # Add the regression line
    plt.plot(line_x, line_y, color='#ff1c45', linewidth=2.5)

    # Add canonical correlation annotation
    plt.annotate(f'Canonical Correlation',
                xy=(0.75, 0.85),
                xycoords='axes fraction',
                fontsize=14,
                color='#ff1c45')

    # Add subject annotation (example for one point)
    # Pick a point near the middle of the distribution for labeling
    # mid_idx = len(eeg_c) // 2
    # plt.annotate('Subj n',
    #             xy=(eeg_c[mid_idx], ecg_c[mid_idx]),
    #             xytext=(eeg_c[mid_idx] + 0.05, ecg_c[mid_idx] - 0.3),
    #             arrowprops=dict(arrowstyle='->', color='black', lw=1),
    #             fontsize=12)

    # Set up clean axes
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_major_locator(MultipleLocator(2))
    ax.xaxis.set_major_locator(MultipleLocator(0.5))

    # Add title
    plt.title(f'EEG-ECG Canonical Correlation: {correlation:.4f}', fontsize=16, y=1.02)

    # X and Y labels (minimal)
    plt.xlabel('EEG Canonical Variate', fontsize=13, labelpad=10)
    plt.ylabel('ECG Canonical Variate', fontsize=13, labelpad=10)

    # Set equal aspect ratio to make the correlation visualization more intuitive
    plt.axis('equal')

    # Adjust layout and save
    plt.tight_layout()
    #plt.savefig('pooled_cca_improved_correlation.png', dpi=300)
    plt.show()


def main():
    """
    Main function to run the pooled CCA analysis.
    """
    print("=== Pooled Subject CCA Analysis: EEG-ECG Data ===")

    # Configuration
    num_subjects = 9
    # ECG value to analyse
    ecg_v = 'rmssd'
    # Trial number
    trial = '1'
    # File path patterns - update these to match your file naming convention

    # The data should be matching the same size as both EEG and ECG data
    #eeg_path_pattern = "30sec_EEG_data_hann/trial15/eeg_subject{:d}.csv"
    #ecg_path_pattern = "30sec_ECG_data/trial15/ecg_subject{:d}.csv"
    eeg_path_pattern = "{Path to the EEG .csv files}"
    ecg_path_pattern = "{Path to the ECG .csv files}"

    # Create a directory to store results
    os.makedirs('subject_results_csv', exist_ok=True)# Create a directory to store results
    os.makedirs('subject_results_csv/haufe_results', exist_ok=True)

    # Pool data from all subjects
    print("\nPooling data from all subjects...")
    try:
        all_subjects, pooled_eeg, pooled_ecg = pool_subject_data(
            num_subjects=num_subjects,
            eeg_path_pattern=eeg_path_pattern,
            ecg_path_pattern=ecg_path_pattern,
            ecg_value = ecg_v
        )

        # Print summary of loaded data
        print(f"Successfully loaded data for {len(all_subjects)} subjects")
        for subject_id, subject_data in all_subjects.items():
            print(f"  Subject {subject_id}: {subject_data['n_samples']} samples")
    except Exception as e:
        print(f"Error pooling data: {e}")
        return

    # Run general CCA analysis
    print("\nRunning CCA analysis on pooled data...")
    results = run_cca_analysis(pooled_eeg, pooled_ecg, trial)

    # Visualize results
    print("\nVisualizing results...")
    # visualize_cca_results(results)
    # visualize_improved_cca_results(results)

    # Save individual subject data for potential future analysis
    # print("\nStep 4: Saving subject data dictionary...")
    # import pickle
    # try:
    #     with open('subject_data.pkl', 'wb') as f:
    #         pickle.dump(all_subjects, f)
    #     print("Subject data saved to 'subject_data.pkl'")
    # except Exception as e:
    #     print(f"Error saving subject data: {e}")

    # print("\nPooled CCA analysis complete!")

    # Save CCA results to CSV files
    print("\nSaving CCA results to CSV files...")

    try:
        # Save canonical variates (transformed data)
        # canonical_df = pd.DataFrame({
        #     'eeg_canonical': results['eeg_canonical'].flatten(),
        #     'ecg_canonical': results['ecg_canonical'].flatten()
        # })
        # canonical_df.to_csv('', index=False)

        # save canonical correlation value
        corr_df = pd.DataFrame({
            'trial': [trial],  # Adjust this if running for other trials
            'canonical_correlation': [results['correlation']]
        })
        corr_df.to_csv('subject_results_csv/canonical_correlation.csv', index=False)
        print("Canonical correlation value saved.")

        # Save EEG weights (importance of each frequency band)
        eeg_weights_df = pd.DataFrame({
            'frequency_band': [f'freq_{i+1}' for i in range(len(results['eeg_weights']))],
            'weight': results['eeg_weights'][:, 0]
        })
        eeg_weights_df.to_csv('subject_results_csv/eeg_weights.csv', index=False)

        # Save ECG weights
        ecg_weights_df = pd.DataFrame({
            'measure': ['rmssd'],
            'weight': results['ecg_weights'][:, 0]
        })
        ecg_weights_df.to_csv('subject_results_csv/ecg_weights.csv', index=False)

        print(f"CCA results saved to 'subject_results_csv/' directory")
    except Exception as e:
        print(f"Error saving CCA results: {e}")

    print("\nPooled CCA analysis complete!")


if __name__ == "__main__":
    main()
