import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Download historical data for ETFs
tickers = ['SPY', 'EFA', 'EEM', 'IWM', 'QQQ', 'TLT', 'GLD', 'DBC', 'VNQ', 'AGG']
# US, Intl Developed, EM, Small Cap, Tech, Bonds, Gold, Commodities, Real Estate, Agg Bonds

# Download data using close since auto adj true
data = yf.download(tickers, start='2010-01-01', end='2024-12-31')['Close']

# Calculate 12-month returns
returns_12m = data.pct_change(252).shift(21)  # 12mo returns, skip last month

# Rank and select top 3
top_3 = returns_12m.rank(axis=1, pct=True) > 0.7

#Select bottom 3 for shorting
bottom_3 = returns_12m.rank(axis=1, pct=True) < .3

# Simple equal weight when selected
weights = top_3.div(top_3.sum(axis=1), axis=0)

# Calculate strategy returns
strategy_returns = (weights.shift(1) * data.pct_change()).sum(axis=1)

#calcutlate the returns of the short positions
short_weights = bottom_3.div(bottom_3.sum(axis=1), axis=0)

# Calculate short strategy returns
short_returns = (short_weights.shift(1) * -data.pct_change()).sum(axis=1)


# Done! Calculate Sharpe, plot cumulative returns
def calculate_sharpe(returns, risk_free_rate=0.01):
    excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
    return excess_returns.mean() / excess_returns.std() * (252 ** 0.5)

sharpe_ratio = calculate_sharpe(strategy_returns)
print(f'Sharpe Ratio: {sharpe_ratio:.2f}')
print(f'Sharpe Ratio Short: {calculate_sharpe(short_returns):.2f}')


#Calacute long/short strategy returns
long_short_returns = strategy_returns + short_returns
# Calculate Sharpe ratio for long/short strategy
long_short_sharpe = calculate_sharpe(long_short_returns)
print(f'Sharpe Ratio Long/Short: {long_short_sharpe:.2f}')

# Plot cummulative  long/short strategy returns
plt.figure(figsize=(12, 6))
cumulative_long_short_returns = (1 + long_short_returns).cumprod()
cumulative_long_short_returns.plot(title='Long/Short Momentum Strategy Cumulative Returns', figsize=(12, 6))
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.grid()
plt.show()



# Plot cumulative returns
plt.figure(figsize=(12, 6))
cumulative_returns = (1 + strategy_returns).cumprod()
cumulative_returns.plot(title='Momentum Strategy Cumulative Returns', figsize=(12, 6))
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.grid()
plt.show()


# Plot short strategy cumulative returns
plt.figure(figsize=(12, 6)) 
cumulative_short_returns = (1 + short_returns).cumprod()
cumulative_short_returns.plot(title='Short Momentum Strategy Cumulative Returns', figsize=(12, 6))
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.grid()
plt.show()


# Add max drawdown calculation
def calculate_max_drawdown(returns):
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()

# Added these metrics
print(f'Annual Return: {long_short_returns.mean() * 252:.1%}')
print(f'Annual Vol: {long_short_returns.std() * np.sqrt(252):.1%}')
print(f'Max Drawdown: {calculate_max_drawdown(long_short_returns):.1%}')

# Add a comparison plot with SPY
spy_returns = data['SPY'].pct_change()
comparison = pd.DataFrame({
    'Long/Short Momentum': (1 + long_short_returns).cumprod(),
    'SPY Buy & Hold': (1 + spy_returns).cumprod()
})
comparison.plot(figsize=(12, 6), title='Long/Short Momentum vs SPY Buy & Hold')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.grid()
plt.legend()
plt.show()
