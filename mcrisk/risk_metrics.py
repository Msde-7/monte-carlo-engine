import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, Optional, List


class RiskCalculator:
    """Calculates various risk metrics from Monte Carlo simulation results."""
    
    def __init__(self):
        self.last_calculation = None
    
    def calculate_var(self, returns: np.ndarray, confidence_level: float = 0.95, 
                     method: str = 'historical') -> float:
        """Calculate Value at Risk (VaR)."""
        if method == 'historical':
            var = np.percentile(returns, (1 - confidence_level) * 100)
        elif method == 'gaussian':
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            z_score = stats.norm.ppf(1 - confidence_level)
            var = mean_return + z_score * std_return
        else:
            raise ValueError("Method must be 'historical' or 'gaussian'")
        
        return -var if var < 0 else var
    
    def calculate_cvar(self, returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """Calculate Conditional Value at Risk (CVaR)."""
        var_threshold = np.percentile(returns, (1 - confidence_level) * 100)
        tail_losses = returns[returns <= var_threshold]
        
        if len(tail_losses) == 0:
            return 0.0
        
        cvar = np.mean(tail_losses)
        return -cvar if cvar < 0 else cvar
    
    def calculate_portfolio_risk_metrics(self, portfolio_returns: np.ndarray,
                                       confidence_levels: List[float] = [0.95, 0.99],
                                       initial_value: float = 1.0) -> Dict[str, Any]:
        """Calculate comprehensive risk metrics for a portfolio."""
        if portfolio_returns.ndim == 1:
            pnl = portfolio_returns * initial_value
            final_values = initial_value + pnl
        else:
            portfolio_values = np.zeros((portfolio_returns.shape[0], portfolio_returns.shape[1] + 1))
            portfolio_values[:, 0] = initial_value
            
            for sim in range(portfolio_returns.shape[0]):
                for t in range(portfolio_returns.shape[1]):
                    portfolio_values[sim, t + 1] = portfolio_values[sim, t] * (1 + portfolio_returns[sim, t])
            
            final_values = portfolio_values[:, -1]
        
        returns_for_risk = (final_values - initial_value) / initial_value
        
        results = {
            'initial_value': initial_value,
            'mean_final_value': np.mean(final_values),
            'std_final_value': np.std(final_values),
            'mean_return': np.mean(returns_for_risk),
            'std_return': np.std(returns_for_risk),
            'skewness': stats.skew(returns_for_risk),
            'kurtosis': stats.kurtosis(returns_for_risk),
            'min_value': np.min(final_values),
            'max_value': np.max(final_values),
            'n_simulations': len(final_values)
        }
        
        for conf_level in confidence_levels:
            conf_str = f"{int(conf_level * 100)}%"
            
            var_historical = self.calculate_var(returns_for_risk, conf_level, 'historical')
            cvar = self.calculate_cvar(returns_for_risk, conf_level)
            
            results[f'VaR_{conf_str}_historical'] = var_historical
            results[f'CVaR_{conf_str}'] = cvar
            results[f'VaR_{conf_str}_dollar'] = var_historical * initial_value
            results[f'CVaR_{conf_str}_dollar'] = cvar * initial_value
        
        self.last_calculation = results
        return results 