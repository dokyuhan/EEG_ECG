import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_contributions(contrib_path):
    df = pd.read_csv(contrib_path)
    df['Row'] = df['Row'].astype(int)
    return df

def compute_band_contributions(df):
    bands = {
        'Delta': list(range(1, 4)),
        'Theta': list(range(4, 8)),
        'Alpha': list(range(8, 12)),
        'Beta': list(range(12, 25)),
        'Gamma': list(range(25, 51))
    }

    for band, indices in bands.items():
        band_cols = [f"freq_{i}_weighted" for i in indices if f"freq_{i}_weighted" in df.columns]
        df[band] = df[band_cols].sum(axis=1)

    return df, list(bands.keys())

def run_regressions(df, bands):
    results = []
    for band in bands:
        X = sm.add_constant(df[[band]])
        y = df["ECG_weighted"]
        model = sm.OLS(y, X).fit()
        results.append({
            "Band": band,
            "Coefficient": model.params[band],
            "P-Value": model.pvalues[band],
            "R-Squared": model.rsquared
        })

    X_multi = sm.add_constant(df[bands])
    y_multi = df["ECG_weighted"]
    multi_model = sm.OLS(y_multi, X_multi).fit()

    return pd.DataFrame(results).sort_values(by="R-Squared", ascending=False), multi_model.rsquared

def plot_band_r2(df):
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x="R-Squared", y="Band", palette="viridis")
    plt.title("Univariate R-Squared by EEG Band (Weighted Contributions)")
    plt.xlabel("R-Squared")
    plt.ylabel("EEG Band")
    plt.tight_layout()
    plt.savefig('General_CCA_results/Regresion_EEG_ECG_weights/table view/trial_15.png', dpi=300)
    plt.show()

def main():
    contrib_file = "subject_results_csv/row_feature_contributions.csv"  # specify your file path here

    df = load_contributions(contrib_file)
    df, band_names = compute_band_contributions(df)

    univariate_results, multivariate_r2 = run_regressions(df, band_names)

    print("\nUnivariate Band Regression Results:")
    print(univariate_results.to_string(index=False))
    print(f"\nMultivariate Model R-Squared: {multivariate_r2:.4f}\n")

    plot_band_r2(univariate_results)

if __name__ == "__main__":
    main()
