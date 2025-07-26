"""
Monte Carlo Risk Engine

A comprehensive Python library for portfolio risk management using Monte Carlo simulation.
Supports VaR, CVaR calculation, backtesting, and various return distribution models.
"""

__version__ = "0.1.0"
__author__ = "Msde-7"

from .risk_engine import RiskEngine
from .data_loader import DataLoader
from .parameter_estimation import ParameterEstimator
from .simulation import MonteCarloSimulator
from .risk_metrics import RiskCalculator
from .backtest import Backtester
from .visualization import RiskVisualizer

__all__ = [
    "RiskEngine",
    "DataLoader", 
    "ParameterEstimator",
    "MonteCarloSimulator",
    "RiskCalculator",
    "Backtester",
    "RiskVisualizer",
] 