import pandas as pd
import os

def compute_band_weights(csv_path):
    """
    Reads a CSV file with CCA EEG frequency bin weights and computes total absolute weights per EEG band.
    EEG bands are defined as:
        - Delta: 0.5-4 Hz
        - Theta: 4-8 Hz
        - Alpha: 8-12 Hz
        - Beta: 12-25 Hz
        - Gamma: 25-50 Hz

    Parameters:
        csv_path (str): Path to the CSV file with columns ['frequency_band', 'weight']

    Returns:
        pd.DataFrame: Band importance table (total abs weights per band)
    """
    # Load the data
    df = pd.read_csv(csv_path)
    
    # Add absolute weight column if not present
    if 'abs_weight' not in df.columns:
        df['abs_weight'] = abs(df['weight'])

    # Extract frequency bin number
    df['frequency_hz'] = df['frequency_band'].str.extract(r'freq_(\d+)').astype(int)

    # Define EEG bands
    bands = {
        'Delta (0.5-4 Hz)': (0.5, 4),
        'Theta (4-8 Hz)': (4, 8),
        'Alpha (8-12 Hz)': (8, 12),
        'Beta (12-25 Hz)': (12, 25),
        'Gamma (25-50 Hz)': (25, 50)
    }

    # Compute total abs weights and normalized weights per band
    band_results = []
    for band_name, (low, high) in bands.items():
        band_data = df[(df['frequency_hz'] >= low) & (df['frequency_hz'] <= high)]
        bin_count = len(band_data)
        total_abs_weight = band_data['abs_weight'].sum()
        
        # Calculate normalized weight (average per bin)
        normalized_weight = total_abs_weight / bin_count if bin_count > 0 else 0
        
        band_results.append({
            'EEG Band': band_name,
            'Total Abs Weight': total_abs_weight,
            'Bin Count': bin_count,
            'Normalized Weight': normalized_weight
        })

    # Create output DataFrame
    band_df = pd.DataFrame(band_results)
    band_df = band_df.sort_values(by='Normalized Weight', ascending=False)
    
    return band_df

def visualize_in_terminal(band_df, max_width=50, weight_type='normalized'):
    """
    Creates a text-based visualization of band weights for terminal display.
    
    Parameters:
        band_df (pd.DataFrame): DataFrame with EEG band weights
        max_width (int): Maximum width of the ASCII bars
        weight_type (str): Type of weight to visualize ('total' or 'normalized')
    """
    weight_col = 'Total Abs Weight' if weight_type == 'total' else 'Normalized Weight'
    title = "EEG BAND IMPORTANCE" if weight_type == 'total' else "EEG BAND IMPORTANCE (NORMALIZED BY BIN COUNT)"
    
    max_weight = band_df[weight_col].max()
    
    # Calculate terminal width
    try:
        terminal_width = os.get_terminal_size().columns
    except (AttributeError, OSError):
        terminal_width = 80  # Default if can't determine
    
    # Print header
    print("\n" + "=" * terminal_width)
    print(title.center(terminal_width))
    print("=" * terminal_width + "\n")
    
    # Print bars
    max_band_name_len = max(len(band) for band in band_df['EEG Band'])
    format_str = f"{{:<{max_band_name_len}}} | {{:<8.4f}} | {{:<4d}} bins | {{}}"
    
    for _, row in band_df.iterrows():
        band = row['EEG Band']
        weight = row[weight_col]
        bin_count = row['Bin Count']
        bar_length = int((weight / max_weight) * max_width) if max_weight > 0 else 0
        bar = "█" * bar_length
        print(format_str.format(band, weight, bin_count, bar))
    
    print("\n" + "-" * terminal_width)

def main():
    #csv_path = 'General_CCA_results/Frequency_importance/LN_RMSSD/trial_5.csv'
    csv_path = 'General_CCA_results/Frequency_importance/30sec_turkey_lnrMSSD/trial_15.csv'
    
    try:
        band_df = compute_band_weights(csv_path)    
        
        # Display full numeric results
        print("\nNumeric Results:")
        print(band_df.to_string(index=False))
        
        # Display ASCII visualization for normalized weights (default)
        visualize_in_terminal(band_df, weight_type='normalized')
        
        # Also show total weights visualization
        visualize_in_terminal(band_df, weight_type='total')
        
    except FileNotFoundError:
        print(f"Error: File not found at path {csv_path}")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()