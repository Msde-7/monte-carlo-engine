import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime, timedelta
import warnings
from scipy import stats
warnings.filterwarnings('ignore')

from .parameter_estimation import ParameterEstimator
from .simulation import MonteCarloSimulator
from .risk_metrics import RiskCalculator


class Backtester:
    """VaR model backtesting using rolling window analysis."""
    
    def __init__(self, returns_data: pd.DataFrame, window_size: int = 250):
        self.returns_data = returns_data
        self.window_size = window_size
        self.backtest_results = None
        self.risk_calculator = RiskCalculator()
        
        if len(returns_data) < window_size + 50:
            raise ValueError(f"Insufficient data. Need at least {window_size + 50} observations.")
    
    def run_backtest(self, portfolio_weights: Optional[np.ndarray] = None,
                    confidence_levels: List[float] = [0.95, 0.99],
                    n_simulations: int = 10000,
                    distribution: str = 'normal',
                    horizon: int = 1,
                    random_seed: Optional[int] = 42) -> pd.DataFrame:
        """Run VaR backtesting."""
        print(f"Starting backtest with {self.window_size}-day rolling window...")
        print(f"Distribution: {distribution}, Simulations: {n_simulations}, Horizon: {horizon} day(s)")
        
        # Set equal weights if none provided
        if portfolio_weights is None:
            n_assets = len(self.returns_data.columns)
            portfolio_weights = np.ones(n_assets) / n_assets
        
        # Validate weights
        if not np.isclose(np.sum(portfolio_weights), 1.0, rtol=1e-5):
            raise ValueError("Portfolio weights must sum to 1.0")
        
        # Calculate actual portfolio returns
        portfolio_returns = (self.returns_data * portfolio_weights).sum(axis=1)
        
        backtest_results = []
        start_idx = self.window_size
        end_idx = len(self.returns_data) - horizon + 1
        
        print(f"Running backtest from {self.returns_data.index[start_idx].date()} to {self.returns_data.index[end_idx-1].date()}")
        
        for i in range(start_idx, end_idx):
            current_date = self.returns_data.index[i]
            
            # Get training window
            train_data = self.returns_data.iloc[i-self.window_size:i]
            
            # Estimate parameters
            estimator = ParameterEstimator(train_data)
            
            try:
                if distribution == 'normal':
                    params = estimator.estimate_normal_parameters()
                elif distribution == 't':
                    params = estimator.estimate_t_parameters()
                elif distribution == 'bootstrap':
                    params = estimator.prepare_bootstrap_data()
                else:
                    raise ValueError(f"Unsupported distribution: {distribution}")
                
                # Run simulation
                simulator = MonteCarloSimulator(params, random_seed)
                portfolio_sim_returns, _ = simulator.simulate_portfolio_returns(
                    weights=portfolio_weights,
                    n_simulations=n_simulations,
                    horizon=horizon,
                    method=distribution
                )
                
                # Calculate VaR predictions
                if horizon == 1:
                    sim_returns_for_var = portfolio_sim_returns[:, 0]
                else:
                    # For multi-day horizon, use final period returns
                    sim_returns_for_var = portfolio_sim_returns[:, -1]
                
                var_predictions = {}
                cvar_predictions = {}
                
                for conf_level in confidence_levels:
                    conf_str = f"{int(conf_level * 100)}%"
                    var_predictions[conf_str] = self.risk_calculator.calculate_var(
                        sim_returns_for_var, conf_level, 'historical'
                    )
                    cvar_predictions[conf_str] = self.risk_calculator.calculate_cvar(
                        sim_returns_for_var, conf_level
                    )
                
                # Get actual returns for the forecast period
                if horizon == 1:
                    actual_return = portfolio_returns.iloc[i]
                else:
                    # Calculate cumulative return over horizon
                    actual_returns_horizon = portfolio_returns.iloc[i:i+horizon]
                    actual_return = (1 + actual_returns_horizon).prod() - 1
                
                # Record results
                result_row = {
                    'date': current_date,
                    'actual_return': actual_return,
                    'window_start': train_data.index[0],
                    'window_end': train_data.index[-1]
                }
                
                # Add VaR predictions and breaches
                for conf_level in confidence_levels:
                    conf_str = f"{int(conf_level * 100)}%"
                    var_pred = var_predictions[conf_str]
                    cvar_pred = cvar_predictions[conf_str]
                    
                    result_row[f'VaR_{conf_str}'] = var_pred
                    result_row[f'CVaR_{conf_str}'] = cvar_pred
                    result_row[f'breach_{conf_str}'] = 1 if actual_return < -var_pred else 0
                    result_row[f'excess_loss_{conf_str}'] = max(0, -actual_return - var_pred)
                
                backtest_results.append(result_row)
                
            except Exception as e:
                print(f"Error at date {current_date}: {str(e)}")
                continue
        
        # Convert to DataFrame
        results_df = pd.DataFrame(backtest_results)
        results_df.set_index('date', inplace=True)
        
        self.backtest_results = results_df
        print(f"Backtest completed. {len(results_df)} observations analyzed.")
        
        return results_df
    
    def calculate_backtest_statistics(self, confidence_levels: List[float] = [0.95, 0.99]) -> Dict[str, Any]:
        """
        Calculate comprehensive backtesting statistics.
        
        Args:
            confidence_levels: List of confidence levels to analyze
            
        Returns:
            Dictionary with backtesting statistics
        """
        if self.backtest_results is None:
            raise ValueError("No backtest results available. Run run_backtest() first.")
        
        results = self.backtest_results
        stats = {}
        
        for conf_level in confidence_levels:
            conf_str = f"{int(conf_level * 100)}%"
            expected_breach_rate = 1 - conf_level
            
            # Basic breach statistics
            breach_col = f'breach_{conf_str}'
            if breach_col in results.columns:
                actual_breaches = results[breach_col].sum()
                total_observations = len(results)
                actual_breach_rate = actual_breaches / total_observations
                
                # Unconditional coverage test (Kupiec test)
                kupiec_lr = self._kupiec_test(actual_breaches, total_observations, expected_breach_rate)
                
                # Conditional coverage test (Christoffersen test)
                christoffersen_lr = self._christoffersen_test(results[breach_col].values, expected_breach_rate)
                
                # Average excess loss (when breaches occur)
                excess_loss_col = f'excess_loss_{conf_str}'
                if excess_loss_col in results.columns:
                    breach_mask = results[breach_col] == 1
                    avg_excess_loss = results.loc[breach_mask, excess_loss_col].mean() if breach_mask.sum() > 0 else 0
                else:
                    avg_excess_loss = 0
                
                # VaR statistics
                var_col = f'VaR_{conf_str}'
                if var_col in results.columns:
                    avg_var = results[var_col].mean()
                    var_volatility = results[var_col].std()
                else:
                    avg_var = var_volatility = 0
                
                stats[conf_str] = {
                    'expected_breach_rate': expected_breach_rate,
                    'actual_breach_rate': actual_breach_rate,
                    'total_breaches': actual_breaches,
                    'total_observations': total_observations,
                    'kupiec_lr_stat': kupiec_lr['lr_stat'],
                    'kupiec_p_value': kupiec_lr['p_value'],
                    'kupiec_reject': kupiec_lr['reject_null'],
                    'christoffersen_lr_stat': christoffersen_lr['lr_stat'],
                    'christoffersen_p_value': christoffersen_lr['p_value'],
                    'christoffersen_reject': christoffersen_lr['reject_null'],
                    'average_var': avg_var,
                    'var_volatility': var_volatility,
                    'average_excess_loss': avg_excess_loss,
                    'breach_clustering': self._calculate_breach_clustering(results[breach_col].values)
                }
        
        return stats
    
    def _kupiec_test(self, violations: int, observations: int, alpha: float) -> Dict[str, Any]:
        """
        Perform Kupiec unconditional coverage test.
        
        Args:
            violations: Number of VaR violations
            observations: Total number of observations
            alpha: Expected violation rate
            
        Returns:
            Dictionary with test results
        """
        if violations == 0:
            lr_stat = 0
        else:
            p_hat = violations / observations
            if p_hat == 1:
                lr_stat = float('inf')
            else:
                lr_stat = -2 * (violations * np.log(alpha) + (observations - violations) * np.log(1 - alpha) -
                               violations * np.log(p_hat) - (observations - violations) * np.log(1 - p_hat))
        
        # Critical value at 5% significance level (chi-squared with 1 df)
        critical_value = 3.841
        p_value = 1 - stats.chi2.cdf(lr_stat, df=1) if lr_stat != float('inf') else 0
        
        return {
            'lr_stat': lr_stat,
            'p_value': p_value,
            'critical_value': critical_value,
            'reject_null': lr_stat > critical_value
        }
    
    def _christoffersen_test(self, violations: np.ndarray, alpha: float) -> Dict[str, Any]:
        """
        Perform Christoffersen conditional coverage test.
        
        Args:
            violations: Binary array of violations (1 = violation, 0 = no violation)
            alpha: Expected violation rate
            
        Returns:
            Dictionary with test results
        """
        # Count transitions
        n00 = n01 = n10 = n11 = 0
        
        for i in range(1, len(violations)):
            if violations[i-1] == 0 and violations[i] == 0:
                n00 += 1
            elif violations[i-1] == 0 and violations[i] == 1:
                n01 += 1
            elif violations[i-1] == 1 and violations[i] == 0:
                n10 += 1
            elif violations[i-1] == 1 and violations[i] == 1:
                n11 += 1
        
        # Calculate transition probabilities
        if n00 + n01 > 0:
            p01 = n01 / (n00 + n01)
        else:
            p01 = 0
            
        if n10 + n11 > 0:
            p11 = n11 / (n10 + n11)
        else:
            p11 = 0
        
        # Calculate likelihood ratio statistic
        total_violations = np.sum(violations)
        total_observations = len(violations)
        
        if total_violations == 0 or total_violations == total_observations:
            lr_stat = 0
        else:
            p = total_violations / total_observations
            
            # Likelihood under null hypothesis (independence)
            if p == 0 or p == 1:
                l_null = 0
            else:
                l_null = (total_observations - total_violations) * np.log(1 - p) + total_violations * np.log(p)
            
            # Likelihood under alternative hypothesis
            l_alt = 0
            if n00 > 0 and p01 < 1:
                l_alt += n00 * np.log(1 - p01)
            if n01 > 0 and p01 > 0:
                l_alt += n01 * np.log(p01)
            if n10 > 0 and p11 < 1:
                l_alt += n10 * np.log(1 - p11)
            if n11 > 0 and p11 > 0:
                l_alt += n11 * np.log(p11)
            
            lr_stat = -2 * (l_null - l_alt)
        
        # Critical value at 5% significance level (chi-squared with 1 df)
        critical_value = 3.841
        p_value = 1 - stats.chi2.cdf(lr_stat, df=1) if lr_stat != float('inf') else 0
        
        return {
            'lr_stat': lr_stat,
            'p_value': p_value,
            'critical_value': critical_value,
            'reject_null': lr_stat > critical_value,
            'p01': p01,
            'p11': p11
        }
    
    def _calculate_breach_clustering(self, violations: np.ndarray) -> float:
        """
        Calculate breach clustering measure.
        
        Args:
            violations: Binary array of violations
            
        Returns:
            Clustering measure (higher values indicate more clustering)
        """
        if np.sum(violations) <= 1:
            return 0.0
        
        # Count consecutive violation runs
        runs = []
        current_run = 0
        
        for v in violations:
            if v == 1:
                current_run += 1
            else:
                if current_run > 0:
                    runs.append(current_run)
                current_run = 0
        
        if current_run > 0:
            runs.append(current_run)
        
        if len(runs) == 0:
            return 0.0
        
        # Calculate clustering measure (average run length)
        return np.mean(runs)
    
    def get_backtest_summary(self) -> pd.DataFrame:
        """
        Create a summary table of backtesting results.
        
        Returns:
            DataFrame with formatted backtesting summary
        """
        if self.backtest_results is None:
            raise ValueError("No backtest results available. Run run_backtest() first.")
        
        stats = self.calculate_backtest_statistics()
        
        summary_data = []
        
        for conf_level, results in stats.items():
            summary_data.append([
                f"VaR {conf_level}",
                f"{results['expected_breach_rate']:.1%}",
                f"{results['actual_breach_rate']:.1%}",
                results['total_breaches'],
                f"{results['kupiec_p_value']:.4f}",
                "Reject" if results['kupiec_reject'] else "Accept",
                f"{results['christoffersen_p_value']:.4f}",
                "Reject" if results['christoffersen_reject'] else "Accept",
                f"{results['average_excess_loss']:.4f}"
            ])
        
        columns = [
            'Model', 'Expected BR', 'Actual BR', 'Breaches', 
            'Kupiec p-val', 'UC Test', 'CC p-val', 'CC Test', 'Avg Excess Loss'
        ]
        
        return pd.DataFrame(summary_data, columns=columns) 