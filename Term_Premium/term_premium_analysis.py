import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col
import matplotlib.pyplot as plt
from datetime import timedelta
from scipy.stats import mstats # For geometric mean

# --- Configuration ---
FILE_DGS10 = 'DGS10.csv'
FILE_DGS3MO = 'DGS3MO.csv'
FILE_THREEFYTP10 = 'THREEFYTP10.csv'
YEARS_FORWARD = 10 # For 10-year term premium
DATE_COLUMN_NAME = 'observation_date' # Changed from 'DATE'

# --- Helper Functions ---
def load_fred_series(filename, series_name):
    """Loads a FRED series from CSV, converts to numeric, and resamples to monthly."""
    try:
        df = pd.read_csv(filename, na_values='.')
        df[DATE_COLUMN_NAME] = pd.to_datetime(df[DATE_COLUMN_NAME])
        df = df.set_index(DATE_COLUMN_NAME)
        # Assuming the actual value is the first column after the date index
        # or the column that is NOT the date index.
        # If CSV has 'observation_date,VALUE_COL_NAME', then after set_index,
        # the remaining column is VALUE_COL_NAME. df.iloc[:,0] gets it.
        df[series_name] = pd.to_numeric(df.iloc[:, 0], errors='coerce')
        df = df[[series_name]] # Keep only the relevant column
        df = df.resample('M').last() # Resample to end of month, taking the last observation
        df = df / 100 # Convert percentage to decimal
        df = df.dropna()
        print(f"Successfully loaded and processed {filename} for {series_name}.")
        return df
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        print(f"Please ensure the CSV file is in the same directory and has '{DATE_COLUMN_NAME}' and a value column.")
        return pd.DataFrame()

def calculate_geometric_average_rolling(series, window_periods):
    """
    Calculates a rolling geometric average of future rates.
    """
    results = pd.Series(index=series.index, dtype=float)
    for i in range(len(series) - window_periods + 1):
        window = series.iloc[i : i + window_periods]
        if len(window) == window_periods and not window.isnull().any():
            if (window <= -1).any(): # Avoid issues with log(0) or log(negative) if 1+r is non-positive
                results.iloc[i] = np.nan
            else:
                results.iloc[i] = mstats.gmean(1 + window) - 1
        else:
            results.iloc[i] = np.nan
    return results


# --- Main Script ---
if __name__ == "__main__":
    print("Starting Term Premium Analysis...")

    # Load data
    dgs10 = load_fred_series(FILE_DGS10, 'DGS10')
    dgs3mo = load_fred_series(FILE_DGS3MO, 'DGS3MO')
    threefytp10 = load_fred_series(FILE_THREEFYTP10, 'ETP') # Ex-ante Term Premium

    if dgs10.empty or dgs3mo.empty or threefytp10.empty:
        print("One or more data series failed to load. Exiting.")
        exit()

    # Combine into a single DataFrame
    df_combined = pd.concat([dgs10, dgs3mo, threefytp10], axis=1)
    df_combined = df_combined.dropna() # Drop rows where any series has NaN at this stage

    print("\n--- Data Head (after initial load & merge) ---")
    print(df_combined.head())
    if df_combined.empty:
        print("Combined data is empty after initial load and merge. Check data alignment and availability.")
        exit()
    print(f"\nData ranges from {df_combined.index.min().strftime('%Y-%m-%d')} to {df_combined.index.max().strftime('%Y-%m-%d')}")

    # Calculate Realized Term Premium (RTP)
    window_months = YEARS_FORWARD * 12

    df_combined['Avg_Future_DGS3MO'] = calculate_geometric_average_rolling(df_combined['DGS3MO'], window_months)
    
    df_combined['RTP'] = df_combined['DGS10'] - df_combined['Avg_Future_DGS3MO']

    analysis_df = df_combined[['ETP', 'RTP']].dropna()

    if analysis_df.empty:
        print("\nNo overlapping data available to compare Ex-Ante and Realized Term Premiums.")
        print("This might happen if your data series are too short or don't overlap sufficiently after RTP calculation.")
        exit()
        
    print("\n--- Data Head (for regression analysis) ---")
    print(analysis_df.head())
    print(f"\nRegression analysis data from {analysis_df.index.min().strftime('%Y-%m-%d')} to {analysis_df.index.max().strftime('%Y-%m-%d')}")
    print(f"Number of observations for regression: {len(analysis_df)}")


    # Regression Analysis: RTP_t = alpha + beta * ETP_t + error_t
    Y = analysis_df['RTP']
    X = analysis_df['ETP']
    X = sm.add_constant(X) # Add intercept

    model = sm.OLS(Y, X)
    
    n_obs = len(Y)
    max_lags_nw = 12 * YEARS_FORWARD 
    
    # Check if n_obs is sufficient for max_lags_nw
    if max_lags_nw >= n_obs:
        print(f"Warning: maxlags ({max_lags_nw}) for Newey-West is >= number of observations ({n_obs}).")
        print("Reducing maxlags to n_obs - 1. Consider if your dataset is too short for reliable HAC estimates with this overlap.")
        max_lags_nw = n_obs - 1 if n_obs > 1 else 0


    if max_lags_nw > 0 :
        results_hac = model.fit(cov_type='HAC', cov_kwds={'maxlags': max_lags_nw, 'use_correction': True})
    else:
        print("Warning: Not enough observations to apply HAC with meaningful lags. Using non-robust standard errors.")
        results_hac = model.fit()


    print("\n--- Regression Results (HAC Standard Errors if applicable) ---")
    print(results_hac.summary())
    
    # --- Data Visualizations ---
    plt.style.use('seaborn-v0_8-darkgrid')

    # 1. Time Series of Ex-Ante (ETP) and Realized (RTP) Term Premiums
    plt.figure(figsize=(12, 6))
    plt.plot(analysis_df.index, analysis_df['ETP'], label='Ex-Ante Term Premium (ETP - Model)', color='blue')
    plt.plot(analysis_df.index, analysis_df['RTP'], label='Realized Term Premium (RTP)', color='red', linestyle='--')
    plt.title('Ex-Ante vs. Realized 10-Year Term Premiums')
    plt.xlabel('Date')
    plt.ylabel('Term Premium (Decimal)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('ETP_vs_RTP_timeseries.png')
    print("\nSaved plot: ETP_vs_RTP_timeseries.png")

    # 2. Scatter Plot of RTP vs ETP with Regression Line
    plt.figure(figsize=(8, 8))
    plt.scatter(analysis_df['ETP'], analysis_df['RTP'], alpha=0.5, label='Data Points')
    
    beta_hat = results_hac.params.get('ETP', np.nan) # Use .get for robustness if 'ETP' isn't in params
    alpha_hat = results_hac.params.get('const', np.nan)

    if not (np.isnan(beta_hat) or np.isnan(alpha_hat)):
        reg_line_x = np.array([analysis_df['ETP'].min(), analysis_df['ETP'].max()])
        reg_line_y = alpha_hat + beta_hat * reg_line_x
        plt.plot(reg_line_x, reg_line_y, color='red', label=f'Regression Line\n(RTP = {alpha_hat:.3f} + {beta_hat:.3f}*ETP)')
    else:
        print("Could not plot regression line as coefficients are missing.")
        
    min_val = min(analysis_df['ETP'].min(skipna=True), analysis_df['RTP'].min(skipna=True))
    max_val = max(analysis_df['ETP'].max(skipna=True), analysis_df['RTP'].max(skipna=True))
    if not (np.isnan(min_val) or np.isnan(max_val)):
        plt.plot([min_val, max_val], [min_val, max_val], color='green', linestyle=':', label='Perfect Alignment Line (y=x)')
    
    plt.title('Realized (RTP) vs. Ex-Ante (ETP) Term Premium')
    plt.xlabel('Ex-Ante Term Premium (ETP - Model)')
    plt.ylabel('Realized Term Premium (RTP)')
    plt.legend()
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('RTP_vs_ETP_scatter.png')
    print("Saved plot: RTP_vs_ETP_scatter.png")

    # 3. Plot of 10-Year Yield and Average Future 3-Month Yields
    plt.figure(figsize=(12, 6))
    plot_df_components = df_combined[['DGS10', 'Avg_Future_DGS3MO']].dropna()
    if not plot_df_components.empty:
        plt.plot(plot_df_components.index, plot_df_components['DGS10'], label='10-Year Treasury Yield (DGS10)', color='purple')
        plt.plot(plot_df_components.index, plot_df_components['Avg_Future_DGS3MO'], label='Avg. Realized Future 3m Yields (next 10yr)', color='orange', linestyle='--')
        plt.title('Components of Realized Term Premium')
        plt.xlabel('Date')
        plt.ylabel('Yield (Decimal)')
        plt.legend()
        plt.tight_layout()
        plt.savefig('RTP_components_timeseries.png')
        print("Saved plot: RTP_components_timeseries.png")
    else:
        print("Could not generate RTP components plot due to insufficient data.")

    print("\nAnalysis complete. Check for saved PNG plot files in the script's directory.")