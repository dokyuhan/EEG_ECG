import pandas as pd

def analyze_stability_from_csv(csv_path, top_n=10, save_summary_path=None):
    """
    Analyze per-row EEG bin stability from a CSV generated after CCA.

    Parameters:
    -----------
    csv_path : str
        Path to the CSV file containing per-row weighted EEG contributions.
    top_n : int
        Number of top stable bins to return.
    save_summary_path : str or None
        If given, saves the summary of bin stability to this path.

    Returns:
    --------
    top_bins_df : pd.DataFrame
        DataFrame with top_n EEG bins ranked by stability score.
    """
    df = pd.read_csv(csv_path)

    # Extract EEG bin columns (exclude ECG_weighted and Row)
    eeg_columns = [col for col in df.columns if col.startswith("freq_")]
    eeg_df = df[eeg_columns]

    # Compute mean, std, and stability score
    means = eeg_df.mean()
    stds = eeg_df.std()
    stability = means / stds

    summary_df = pd.DataFrame({
        'Bin': eeg_columns,
        'Mean': means.values,
        'Std': stds.values,
        'Stability_Score': stability.values
    }).sort_values(by='Stability_Score', ascending=False)

    # Save if needed
    if save_summary_path:
        summary_df.to_csv(save_summary_path, index=False)
        print(f"Saved bin stability summary to: {save_summary_path}")

    return summary_df.head(top_n)

# Example usage
if __name__ == "__main__":
    top_bins = analyze_stability_from_csv(
        csv_path="subject_results_csv/row_feature_contributions.csv",
        top_n=10,
        save_summary_path="General_CCA_results/Stability bands/trial_15.csv"
    )
    
    print("\nTop Stable EEG Bins:")
    print(top_bins)
