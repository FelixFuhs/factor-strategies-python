# Analysis of Ex-Ante vs. Realized 10-Year Treasury Term Premium

This project investigates the relationship between model-based ex-ante predictions of the 10-year U.S. Treasury term premium and the subsequently realized term premium. The analysis was conducted by Felix (with conceptual and scripting assistance from Gemini).

## Research Question

The primary question addressed is: How well does an ex-ante model of the 10-year term premium (specifically, the Adrian, Crump, and Moench (ACM) model estimates available on FRED) predict the actual, realized term premium over a 10-year horizon?

## Data Used

The analysis utilizes the following time series data, downloaded as CSV files from FRED (Federal Reserve Economic Data):

* `DGS10.csv`: 10-Year Treasury Constant Maturity Rate
* `DGS3MO.csv`: 3-Month Treasury Constant Maturity Rate
* `THREEFYTP10.csv`: ACM 10-Year Treasury Term Premium (Ex-Ante Term Premium, ETP)

## Methodology

1.  **Data Preparation:** All series were resampled to monthly (end-of-period) frequency, and rates were converted to decimals.
2.  **Realized Term Premium (RTP) Calculation:** For each month `t`, the RTP was calculated as:
    `RTP_t = DGS10_t - GeometricAverage(DGS3MO from t to t+10 years)`
3.  **Regression Analysis:** The realized term premium was regressed on the ex-ante term premium:
    `RTP_t = α + β * ETP_t + ε_t`
    The regression was estimated using Ordinary Least Squares (OLS), with standard errors corrected for heteroskedasticity and autocorrelation (HAC) using the Newey-West method (120 lags for monthly data over a 10-year overlap).

## Key Results (Sample: 1990-01-31 to 2015-06-30 for regression)

* **Significant Relationship:** There is a statistically significant positive relationship between the ex-ante term premium (ETP) and the realized term premium (RTP).
    * The coefficient for ETP ($\beta$) was approximately **0.682** (p-value = 0.023).
* **Intercept ($\alpha$):** The intercept was approximately **0.0184** (1.84%) and highly significant (p-value = 0.000). This suggests that when the ETP model predicts a zero premium, the realized premium averages around 1.84%.
* **Explanatory Power:** The ETP model explained approximately **22.3%** (R-squared) of the variance in the RTP.
* **Overall Verdict:** While the ACM ex-ante term premium has some statistically significant predictive value for the direction of realized term premiums, it does not explain the majority of their variation. Realized premiums have often been higher than predicted by the model, particularly when the model predicted lower premiums. The relationship is noisy, and the ex-ante model is much smoother than the volatile realized outcomes.

## Files

* `term_premium_analysis.py`: The Python script used for the analysis.
* `DGS10.csv`, `DGS3MO.csv`, `THREEFYTP10.csv`: Raw data files (user-provided).
* `ETP_vs_RTP_timeseries.png`: Plot of Ex-Ante vs. Realized Term Premiums over time.
* `RTP_vs_ETP_scatter.png`: Scatter plot of Realized vs. Ex-Ante Term Premiums with regression line.
* `RTP_components_timeseries.png`: Plot of the 10-Year Treasury Yield and the average of realized future 3-month yields.

## How to Run

1.  Ensure Python 3 is installed along with the necessary libraries: `pandas`, `numpy`, `statsmodels`, `matplotlib`, `scipy`.
    ```bash
    pip install pandas numpy statsmodels matplotlib scipy
    ```
2.  Download the required CSV data files from FRED and place them in the same directory as the script, ensuring they are named as listed above.
3.  Run the script from the command line:
    ```bash
    python term_premium_analysis.py
    ```
    The script will print regression results to the console and save the plots as PNG files in the same directory.