import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Parameters 
tickers = ['SPY', 'EFA', 'EEM', 'IWM', 'QQQ', 'TLT', 'GLD', 'DBC', 'VNQ', 'AGG']
start, end  = '2010-01-01', '2024-12-31'
decay = 0.94                      # RiskMetrics λ
alpha = 1 - decay                 # = 0.06
trading_days = 252                # for annualising

# Download adjusted close prices 
data = yf.download(
    tickers,
    start=start,
    end=end,
    auto_adjust=True               # avoids the FutureWarning
)['Close']

# Daily log-returns 
returns = np.log(data / data.shift(1)).dropna()

# RiskMetrics EWMA volatility (vectorised) 
# EWMA variance: σ²_t = α · r²_{t-1} + λ · σ²_{t-1}
ewma_var = returns.pow(2).ewm(alpha=alpha, adjust=False).mean()
vol_ewma = np.sqrt(ewma_var)                 # daily σ
vol_ewma_annual = vol_ewma * np.sqrt(trading_days)

# Rank by predicted volatility
vol_rank = vol_ewma_annual.rank(axis=1, pct=True)   

# Select top 3 assets (lowest volatility)
top_3 = vol_rank <= 0.3

# Equal weight for selected assets
weights = top_3.div(top_3.sum(axis=1), axis=0)
# Calculate strategy returns
strategy_returns = (weights.shift(1) * returns).sum(axis=1)
# Calculate Sharpe ratio
def calculate_sharpe(returns, risk_free_rate=0.01):
    excess_returns = returns - risk_free_rate / trading_days  # Daily risk-free rate
    return excess_returns.mean() / excess_returns.std() * np.sqrt(trading_days)
sharpe_ratio = calculate_sharpe(strategy_returns)
print(f'Sharpe Ratio: {sharpe_ratio:.2f}')

# Calculate cumulative returns
cumulative_returns = (1 + strategy_returns).cumprod()
# Plot cumulative returns
plt.figure(figsize=(12, 6))
cumulative_returns.plot(title='Low Volatility Strategy Cumulative Returns', figsize=(12, 6))
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.grid()
plt.show()

# Select bottom 3 assets (highest volatility)
bottom_3 = vol_rank >= 0.7
# Equal weight for short positions
short_weights = bottom_3.div(bottom_3.sum(axis=1), axis=0)
# Calculate short strategy returns
short_returns = (short_weights.shift(1) * -returns).sum(axis=1)
# Calculate Sharpe ratio for short strategy
short_sharpe_ratio = calculate_sharpe(short_returns)
print(f'Sharpe Ratio Short: {short_sharpe_ratio:.2f}')
# Calculate cumulative returns for short strategy
cumulative_short_returns = (1 + short_returns).cumprod()
# Plot cumulative returns for short strategy
plt.figure(figsize=(12, 6))
cumulative_short_returns.plot(title='Short Low Volatility Strategy Cumulative Returns', figsize=(12, 6))
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.grid()
plt.show()

# Calculate long/short strategy returns
long_short_returns = strategy_returns + short_returns
# Calculate Sharpe ratio for long/short strategy
long_short_sharpe = calculate_sharpe(long_short_returns)
print(f'Sharpe Ratio Long/Short: {long_short_sharpe:.2f}')
# Plot cumulative returns for long/short strategy
plt.figure(figsize=(12, 6))
cumulative_long_short_returns = (1 + long_short_returns).cumprod()
cumulative_long_short_returns.plot(title='Long/Short Low Volatility Strategy Cumulative Returns', figsize=(12, 6))
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.grid()
plt.show()


#Calcute a volatiltiy neutral l/s portfolio under the assumption of independent assets
# This is a simple approach, assuming independence and equal risk contribution  
#long leg expected voolatility
vol_long = vol_ewma_annual[top_3].mean(axis=1)
#short leg expected volatility
vol_short = vol_ewma_annual[bottom_3].mean(axis=1)

# Calculate the volatility neutral weights by making the expected vola of the long leg equal to the short leg
weights_neutral = top_3.div(top_3.sum(axis=1), axis=0) * (vol_short / vol_long).values[:, None]
# Calculate strategy returns for volatility neutral portfolio
strategy_returns_neutral = (weights_neutral.shift(1) * returns).sum(axis=1)
# Calculate Sharpe ratio for volatility neutral strategy
sharpe_ratio_neutral = calculate_sharpe(strategy_returns_neutral)
print(f'Sharpe Ratio Volatility Neutral: {sharpe_ratio_neutral:.2f}')
# Plot cumulative returns for volatility neutral strategy
plt.figure(figsize=(12, 6))
cumulative_returns_neutral = (1 + strategy_returns_neutral).cumprod()
cumulative_returns_neutral.plot(title='Volatility Neutral Low Volatility Strategy Cumulative Returns', figsize=(12, 6))
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.grid()
plt.show()



# quick plot 
#vol_ewma_annual['SPY'].plot(figsize=(10, 4), title='SPY EWMA Volatility (annualised)')
#plt.ylabel('Annualised σ')
#plt.show()
