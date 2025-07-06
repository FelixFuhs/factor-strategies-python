import pandas as pd
import numpy as np


def sample_cov(returns, min_periods=30):
    """
    Calculate sample covariance matrix.
    
    Input:
        returns: pd.DataFrame of returns (dates x tickers)
                 e.g., 252 rows (days) x 10 columns (stocks)
        min_periods: minimum observations required
    
    Output:
        cov_matrix: np.array (N x N) where N = number of stocks
                    e.g., 10 x 10 matrix for 10 stocks
    """
    # Check if we have enough observations
    if len(returns) < min_periods:
        raise ValueError(f"Need at least {min_periods} observations, got {len(returns)}")
    
    # Check if we have enough assets (at least 2)
    if returns.shape[1] < 2:
        raise ValueError(f"Need at least 2 assets, got {returns.shape[1]}")
    
    # Calculate annualized covariance
    cov_matrix = returns.cov(min_periods=min_periods) * 252
    # Convert to numpy array
    cov_matrix = cov_matrix.to_numpy()
    # Ensure the covariance matrix is positive definite
    if not np.all(np.linalg.eigvals(cov_matrix) > 0):
        raise ValueError("Covariance matrix is not positive definite.")
    return cov_matrix


