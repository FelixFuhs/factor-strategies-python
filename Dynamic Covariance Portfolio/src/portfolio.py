import numpy as np
import pandas as pd 

def min_var_weights(cov_matrix):
    """
    Calculate minimum variance portfolio weights.
    
    Input:
        cov_matrix: np.array (N x N) where N = number of assets
                    e.g., 10 x 10 matrix for 10 assets
    
    Output:
        weights: np.array (N,) where N = number of assets
                 e.g., 10 weights for 10 assets
    """
    # Ensure covariance matrix is a numpy array
    if not isinstance(cov_matrix, np.ndarray):
        raise ValueError("Covariance matrix must be a numpy array.")
    
    # Check if covariance matrix is square
    if cov_matrix.shape[0] != cov_matrix.shape[1]:
        raise ValueError("Covariance matrix must be square.")
    
    # Calculate inverse of covariance matrix
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    
    # Calculate weights
    ones = np.ones(cov_matrix.shape[0])
    weights = inv_cov_matrix @ ones / (ones @ inv_cov_matrix @ ones)
    
    return weights


def portfolio_variance(weights, cov_matrix):
    """
    Calculate portfolio variance: σ²_p = w'Σw
    
    Input:
        weights: np.array (N,) portfolio weights
        cov_matrix: np.array (N x N) covariance matrix
    
    Output:
        variance: float - portfolio variance
    """
    # Ensure weights and covariance matrix are numpy arrays
    if not isinstance(weights, np.ndarray):
        raise ValueError("Weights must be a numpy array.")
    if not isinstance(cov_matrix, np.ndarray):
        raise ValueError("Covariance matrix must be a numpy array.")
    
    # Check dimensions
    if weights.shape[0] != cov_matrix.shape[0]:
        raise ValueError("Weights and covariance matrix dimensions do not match.")
    
    # Calculate portfolio variance
    variance = weights.T @ cov_matrix @ weights
    
    return variance

    
def portfolio_returns(weights, returns):
    """
    Calculate realized portfolio returns over time
    
    Input:
        weights: np.array (N,) portfolio weights  
        returns: pd.DataFrame - asset returns (T x N)
    
    Output:
        port_returns: pd.Series - time series of portfolio returns
    """
    # Ensure weights are a numpy array
    if not isinstance(weights, np.ndarray):
        raise ValueError("Weights must be a numpy array.")
    
    # Check if returns is a DataFrame
    if not isinstance(returns, pd.DataFrame):
        raise ValueError("Returns must be a pandas DataFrame.")
    
    # Calculate portfolio returns
    port_returns = returns.dot(weights)
    
    return port_returns