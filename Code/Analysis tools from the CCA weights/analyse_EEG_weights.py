import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def sort_cca_weights(weights_file, plot=True):
    """
    Load CCA weights from a CSV file, sort them by absolute value in descending order,
    and optionally create a visualization.
    
    Parameters:
    -----------
    weights_file : str
        Path to the CSV file containing the EEG weights
    plot : bool
        Whether to create a bar plot visualization
    save_plot : bool
        Whether to save the plot to a file
        
    Returns:
    --------
    sorted_weights_df : pandas.DataFrame
        DataFrame with weights sorted by absolute importance
    """
    # Check if the file exists
    if not os.path.exists(weights_file):
        print(f"Error: File '{weights_file}' not found.")
        print("Make sure you've run the CCA analysis first to generate the weights file.")
        return None
    
    try:
        # Load the EEG weights from CSV
        weights_df = pd.read_csv(weights_file)
        
        # Add absolute value column for sorting
        weights_df['abs_weight'] = weights_df['weight'].abs()
        
        # Sort by absolute weight in descending order
        sorted_weights_df = weights_df.sort_values('abs_weight', ascending=False).reset_index(drop=True)
        
        # Print the sorted weights
        print("Frequencies sorted by contribution to CCA (highest to lowest):")
        print("=" * 60)
        print(f"{'Rank':<6}{'Frequency Band':<20}{'Weight':<15}{'Abs Weight':<15}")
        print("-" * 60)
        
        for i, row in sorted_weights_df.iterrows():
            print(f"{i+1:<6}{row['frequency_band']:<20}{row['weight']:<15.6f}{row['abs_weight']:<15.6f}")
        
        # Save the sorted weights to a new CSV
        output_file = '{Output file name}'
        sorted_weights_df.to_csv(output_file, index=False)
        print(f"\nSorted weights saved to '{output_file}'")
        
        # Create visualization if requested
        if plot:
            create_weight_importance_plot(sorted_weights_df)
        
        return sorted_weights_df
        
    except Exception as e:
        print(f"Error processing weights file: {e}")
        return None

def create_weight_importance_plot(sorted_weights_df):
    """
    Create a bar plot showing frequency bands sorted by their contribution to the CCA.
    
    Parameters:
    -----------
    sorted_weights_df : pandas.DataFrame
        DataFrame with weights sorted by absolute importance
    save : bool
        Whether to save the plot to a file
    """
    # Get top 10 frequencies for better readability
    top_n = min(10, len(sorted_weights_df))
    plot_df = sorted_weights_df.head(top_n)
    
    # Create figure
    plt.figure(figsize=(12, 7))
    bars = plt.bar(
        plot_df['frequency_band'], 
        plot_df['weight'],
        color=[('red' if w < 0 else 'blue') for w in plot_df['weight']]
    )
    
    # Add absolute value as text on bars
    for bar, abs_val in zip(bars, plot_df['abs_weight']):
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
    plt.title('Top Frequency Bands Contributing to EEG-ECG Correlation', fontsize=16)
    plt.xlabel('Frequency Band', fontsize=14)
    plt.ylabel('Weight Value (Bar)\nAbsolute Importance (Text)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add a note about interpretation
    plt.figtext(
        0.5, 0.01, 
        "Note: Larger absolute values (shown in text) indicate stronger contribution to the correlation.\n"
        "The sign (positive or negative) indicates the direction of the relationship.",
        ha='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.5)
    )
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()

def main():
    """
    Main function to run the weight sorting and visualization.
    """
    print("=== CCA Frequency Importance Analysis ===\n")
    
    # Default path to weights file
    default_file = 'subject_results_csv/eeg_weights.csv'
    
    # Check if custom path is provided
    import sys
    if len(sys.argv) > 1:
        weights_file = sys.argv[1]
    else:
        weights_file = default_file
    
    # Run the analysis
    sorted_weights = sort_cca_weights(
        weights_file=weights_file,
        plot=True,
    )
    
    if sorted_weights is not None:
        # Additional analysis: what percentage of the total contribution comes from top frequencies?
        total_abs_contribution = sorted_weights['abs_weight'].sum()
        
        # Calculate cumulative contribution
        sorted_weights['cumulative_contribution'] = sorted_weights['abs_weight'].cumsum() / total_abs_contribution * 100
        
        # Find how many frequencies contribute to 80% of the total
        freq_for_80_percent = len(sorted_weights[sorted_weights['cumulative_contribution'] <= 80])
        
        print(f"\nTop frequency insights:")
        print(f"- The top frequency ({sorted_weights.iloc[0]['frequency_band']}) contributes {sorted_weights.iloc[0]['abs_weight']/total_abs_contribution*100:.1f}% of the total correlation")
        print(f"- The top 3 frequencies contribute {sorted_weights.iloc[:3]['abs_weight'].sum()/total_abs_contribution*100:.1f}% of the total correlation")
        print(f"- {freq_for_80_percent + 1} frequencies are needed to explain 80% of the correlation")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()