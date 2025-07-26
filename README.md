# Monte Carlo Portfolio Risk Engine

Compute Value at Risk (VaR), Conditional Value at Risk (CVaR), backtest risk models, and create visualizations.

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)

## Features

- Multiple distribution models (Normal, Student-t, Bootstrap)
- VaR and CVaR calculation with Monte Carlo simulation
- Backtesting with statistical validation tests
- Visualization tools for risk analysis
- Easy-to-use API for portfolio risk management

## What is Monte Carlo VaR?

Value at Risk (VaR) measures potential portfolio losses over a specific time period at a given confidence level. For example, a 1-day 95% VaR of $10,000 means there's a 5% chance of losing more than $10,000 in one day.

Monte Carlo simulation generates thousands of possible future scenarios based on statistical models fitted to historical data.

## Installation

```bash
# From source
git clone https://github.com/yourusername/mc-risk-engine.git
cd mc-risk-engine
pip install -r requirements.txt
pip install -e .
```

## Requirements

Python 3.8+ with numpy, pandas, scipy, matplotlib, yfinance, and other dependencies listed in requirements.txt.

## Quick Start

```python
from mcrisk import RiskEngine

# Initialize the risk engine
engine = RiskEngine(
    tickers=['SPY', 'AAPL', 'QQQ'], 
    lookback=250
)

# Load historical data
engine.load_data(period="2y")

# Compute Monte Carlo VaR and CVaR
var95, cvar95 = engine.compute_mc_var(alpha=0.95, sims=10000)

# Run backtesting
backtest_results = engine.backtest(alpha=0.95)

print(f"95% VaR: {var95:.4f}")
print(f"95% CVaR: {cvar95:.4f}")
```

## Usage Examples

### Initialize Risk Engine
```python
engine = RiskEngine(
    tickers=['SPY', 'AAPL', 'QQQ'],
    portfolio_weights=[0.4, 0.3, 0.3],  # Optional, defaults to equal weights
    lookback=250,                        # Days for parameter estimation
    random_seed=42                       
)
```

### Load Data and Compute Risk Metrics
```python
# Load historical data
prices = engine.load_data(period="2y")

# Compute VaR and CVaR using Monte Carlo simulation
var95, cvar95 = engine.compute_mc_var(alpha=0.95, sims=10000)

# Get comprehensive risk analysis
risk_metrics = engine.compute_comprehensive_risk_metrics(
    confidence_levels=[0.95, 0.99],
    initial_value=1000000
)
```

### Backtesting
```python
# Run VaR backtesting
backtest_results = engine.backtest(alpha=0.95)
summary = engine.backtester.get_backtest_summary()
```

### Visualizations
```python
# Plot return distribution with VaR markers
engine.plot_return_distribution()

# Plot simulated price paths
engine.plot_price_paths(n_paths=100)

# Plot backtesting results
engine.plot_backtest_results()
```

## Example Output

```
Monte Carlo VaR 95%: 0.0187 (1.87%)
Monte Carlo CVaR 95%: 0.0241 (2.41%)

Comprehensive Risk Metrics ($1M Portfolio):
==================================================
Expected Daily Return   : 0.0008 (0.08%)
Daily Volatility        : 0.0121 (1.21%)
Skewness               : -0.1234
Kurtosis               : 2.8765
95% VaR (Dollar)       : $18,673.45
95% CVaR (Dollar)      : $24,112.89
99% VaR (Dollar)       : $28,901.23
99% CVaR (Dollar)      : $34,567.89

Backtest Results for VaR 95%:
- Expected breach rate: 5.0%
- Actual breach rate: 4.8%
- Total breaches: 12
- Kupiec test p-value: 0.8234
- Model accepted at 5% level
```

## Demo Notebook

Run the complete demo using the provided Jupyter notebook:

```bash
jupyter notebook demo_notebook.ipynb
```