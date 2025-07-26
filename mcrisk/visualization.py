import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class RiskVisualizer:
    """Handles visualization of risk analysis results."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_price_paths(self, prices: np.ndarray, dates: Optional[pd.DatetimeIndex] = None,
                        asset_names: Optional[List[str]] = None, n_paths_display: int = 100,
                        title: str = "Simulated Price Paths") -> plt.Figure:
        """Plot simulated price paths."""
        n_sims, horizon, n_assets = prices.shape
        
        if asset_names is None:
            asset_names = [f'Asset_{i}' for i in range(n_assets)]
        
        if dates is None:
            dates = pd.date_range(start='2024-01-01', periods=horizon, freq='D')
        
        display_paths = np.random.choice(n_sims, min(n_paths_display, n_sims), replace=False)
        
        fig, axes = plt.subplots(1, n_assets, figsize=(5*n_assets, 6))
        if n_assets == 1:
            axes = [axes]
        
        for i, asset in enumerate(asset_names):
            for path_idx in display_paths:
                axes[i].plot(dates, prices[path_idx, :, i], alpha=0.3, linewidth=0.5)
            
            mean_path = np.mean(prices[:, :, i], axis=0)
            axes[i].plot(dates, mean_path, color='red', linewidth=2, label='Mean')
            
            p5 = np.percentile(prices[:, :, i], 5, axis=0)
            p95 = np.percentile(prices[:, :, i], 95, axis=0)
            axes[i].fill_between(dates, p5, p95, alpha=0.2, color='gray', label='5th-95th %ile')
            
            axes[i].set_title(f'{asset} Price Paths')
            axes[i].set_xlabel('Date')
            axes[i].set_ylabel('Price')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        return fig
    
    def plot_return_distribution(self, returns: np.ndarray, confidence_levels: List[float] = [0.95, 0.99],
                               title: str = "Portfolio Return Distribution") -> plt.Figure:
        """Plot return distribution with VaR and CVaR markers."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        ax1.hist(returns, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Portfolio Returns')
        ax1.set_ylabel('Density')
        ax1.set_title('Return Distribution')
        ax1.grid(True, alpha=0.3)
        
        colors = ['red', 'darkred']
        for i, conf_level in enumerate(confidence_levels):
            if i < len(colors):
                var_value = np.percentile(returns, (1-conf_level)*100)
                cvar_value = np.mean(returns[returns <= var_value])
                
                ax1.axvline(var_value, color=colors[i], linestyle='--', linewidth=2, 
                           label=f'VaR {conf_level:.0%}: {var_value:.4f}')
                ax1.axvline(cvar_value, color=colors[i], linestyle=':', linewidth=2,
                           label=f'CVaR {conf_level:.0%}: {cvar_value:.4f}')
        
        ax1.legend()
        
        from scipy import stats
        stats.probplot(returns, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot (Normal)')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        return fig
    
    def plot_backtest_results(self, backtest_results: pd.DataFrame, 
                            confidence_level: float = 0.95,
                            title: str = "VaR Backtesting Results") -> plt.Figure:
        """Plot VaR backtesting results."""
        conf_str = f"{int(confidence_level * 100)}%"
        
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        dates = backtest_results.index
        actual_returns = backtest_results['actual_return']
        var_values = backtest_results[f'VaR_{conf_str}']
        breaches = backtest_results[f'breach_{conf_str}']
        
        axes[0].plot(dates, actual_returns, color='blue', alpha=0.7, label='Actual Returns')
        axes[0].plot(dates, -var_values, color='red', linestyle='--', label=f'VaR {conf_str}')
        
        breach_dates = dates[breaches == 1]
        breach_returns = actual_returns[breaches == 1]
        axes[0].scatter(breach_dates, breach_returns, color='red', s=50, alpha=0.8, 
                       label=f'Breaches ({len(breach_dates)})')
        
        axes[0].set_xlabel('Date')
        axes[0].set_ylabel('Returns')
        axes[0].set_title(f'Actual Returns vs VaR {conf_str}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        window = 60
        breach_rate = breaches.rolling(window).mean() * 100
        expected_rate = (1 - confidence_level) * 100
        
        axes[1].plot(dates, breach_rate, color='blue', label='Actual Breach Rate')
        axes[1].axhline(expected_rate, color='red', linestyle='--', 
                       label=f'Expected Rate ({expected_rate:.1f}%)')
        
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel('Breach Rate (%)')
        axes[1].set_title(f'{window}-Day Rolling Breach Rate')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        return fig
    
    def plot_correlation_heatmap(self, correlation_matrix: pd.DataFrame,
                               title: str = "Asset Correlation Matrix") -> plt.Figure:
        """Plot correlation heatmap."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(correlation_matrix, annot=True, cmap='RdYlBu_r', center=0,
                   square=True, ax=ax, cbar_kws={'shrink': 0.8})
        
        ax.set_title(title, fontsize=16)
        plt.tight_layout()
        return fig
    
    def create_risk_dashboard(self, risk_results: Dict[str, Any], 
                            backtest_results: Optional[pd.DataFrame] = None) -> None:
        """Create a simple risk dashboard (placeholder for plotly version)."""
        print("Interactive dashboard requires plotly - use individual plot methods instead") 