import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

from .data_loader import DataLoader
from .parameter_estimation import ParameterEstimator
from .simulation import MonteCarloSimulator
from .risk_metrics import RiskCalculator
from .backtest import Backtester
from .visualization import RiskVisualizer


class RiskEngine:
    """Main interface for Monte Carlo portfolio risk analysis."""
    
    def __init__(self, 
                 tickers: List[str],
                 lookback: int = 250,
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None,
                 portfolio_weights: Optional[np.ndarray] = None,
                 random_seed: Optional[int] = 42):
        self.tickers = tickers
        self.lookback = lookback
        self.start_date = start_date
        self.end_date = end_date
        self.random_seed = random_seed
        
        # Set portfolio weights
        if portfolio_weights is None:
            self.portfolio_weights = np.ones(len(tickers)) / len(tickers)
        else:
            if not np.isclose(np.sum(portfolio_weights), 1.0, rtol=1e-5):
                raise ValueError("Portfolio weights must sum to 1.0")
            self.portfolio_weights = portfolio_weights
        
        # Initialize components
        self.data_loader = DataLoader(tickers, start_date, end_date)
        self.parameter_estimator = None
        self.simulator = None
        self.risk_calculator = RiskCalculator()
        self.backtester = None
        self.visualizer = RiskVisualizer()
        
        # Data storage
        self.prices = None
        self.returns = None
        self.parameters = {}
        self.simulation_results = {}
        self.risk_metrics = {}
        
        print(f"Risk Engine initialized for {len(tickers)} assets")
    
    def load_data(self, period: str = "2y", source: str = "yfinance") -> pd.DataFrame:
        """Load historical price data."""
        if source == "yfinance":
            self.prices = self.data_loader.fetch_data(period)
        else:
            raise ValueError("Only 'yfinance' source is currently supported")
        
        self.returns = self.data_loader.compute_returns(method="simple")
        self.parameter_estimator = ParameterEstimator(self.returns)
        return self.prices
    
    def load_data_from_csv(self, filepath: str, date_column: str = 'Date') -> pd.DataFrame:
        """Load data from CSV file."""
        self.prices = self.data_loader.load_from_csv(filepath, date_column)
        self.returns = self.data_loader.compute_returns(method="simple")
        self.parameter_estimator = ParameterEstimator(self.returns)
        return self.prices
    
    def fit_distribution(self, distribution: str = 'normal') -> Dict[str, Any]:
        """Fit return distribution parameters."""
        if self.parameter_estimator is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        print(f"Fitting {distribution} distribution...")
        
        if distribution == 'normal':
            params = self.parameter_estimator.estimate_normal_parameters()
        elif distribution == 't':
            params = self.parameter_estimator.estimate_t_parameters()
        elif distribution == 'bootstrap':
            params = self.parameter_estimator.prepare_bootstrap_data()
        else:
            raise ValueError("Distribution must be 'normal', 't', or 'bootstrap'")
        
        self.parameters[distribution] = params
        print(f"{distribution.title()} distribution fitted")
        return params
    
    def compute_mc_var(self, 
                      alpha: float = 0.95,
                      sims: int = 10000,
                      horizon: int = 1,
                      distribution: str = 'normal',
                      method: str = 'historical') -> Tuple[float, float]:
        """Compute Monte Carlo VaR and CVaR."""
        # Ensure parameters are fitted
        if distribution not in self.parameters:
            self.fit_distribution(distribution)
        
        print(f"Running Monte Carlo simulation: {sims} paths, {horizon} day horizon")
        
        # Initialize simulator
        self.simulator = MonteCarloSimulator(self.parameters[distribution], self.random_seed)
        
        # Run portfolio simulation
        portfolio_returns, portfolio_values = self.simulator.simulate_portfolio_returns(
            weights=self.portfolio_weights,
            n_simulations=sims,
            horizon=horizon,
            method=distribution
        )
        
        # Store simulation results
        self.simulation_results = {
            'portfolio_returns': portfolio_returns,
            'portfolio_values': portfolio_values,
            'parameters': self.parameters[distribution],
            'horizon': horizon,
            'n_simulations': sims,
            'distribution': distribution
        }
        
        # Calculate risk metrics
        if horizon == 1:
            returns_for_risk = portfolio_returns[:, 0]
        else:
            returns_for_risk = portfolio_returns[:, -1]
        
        var = self.risk_calculator.calculate_var(returns_for_risk, alpha, method)
        cvar = self.risk_calculator.calculate_cvar(returns_for_risk, alpha)
        
        # Store results
        conf_str = f"{int(alpha * 100)}%"
        self.risk_metrics[conf_str] = {
            'VaR': var,
            'CVaR': cvar,
            'method': method,
            'distribution': distribution
        }
        
        print(f"VaR {conf_str}: {var:.4f} ({var*100:.2f}%)")
        print(f"CVaR {conf_str}: {cvar:.4f} ({cvar*100:.2f}%)")
        
        return var, cvar
    
    def compute_comprehensive_risk_metrics(self,
                                         confidence_levels: List[float] = [0.95, 0.99],
                                         sims: int = 10000,
                                         horizon: int = 1,
                                         distribution: str = 'normal',
                                         initial_value: float = 1000000) -> Dict[str, Any]:
        """Compute comprehensive portfolio risk metrics."""
        # Run simulation if not already done
        if not self.simulation_results or self.simulation_results.get('n_simulations', 0) < sims:
            self.compute_mc_var(confidence_levels[0], sims, horizon, distribution)
        
        portfolio_returns = self.simulation_results['portfolio_returns']
        
        # Calculate comprehensive metrics
        risk_results = self.risk_calculator.calculate_portfolio_risk_metrics(
            portfolio_returns, confidence_levels, initial_value
        )
        
        self.risk_metrics['comprehensive'] = risk_results
        
        print("Risk metrics calculated:")
        print(f"- Mean return: {risk_results['mean_return']:.4f}")
        print(f"- Volatility: {risk_results['std_return']:.4f}")
        for conf_level in confidence_levels:
            conf_str = f"{int(conf_level * 100)}%"
            print(f"- VaR {conf_str}: ${risk_results[f'VaR_{conf_str}_dollar']:,.0f}")
            print(f"- CVaR {conf_str}: ${risk_results[f'CVaR_{conf_str}_dollar']:,.0f}")
        
        return risk_results
    
    def backtest(self,
                alpha: float = 0.95,
                window_size: Optional[int] = None,
                n_simulations: int = 5000,
                distribution: str = 'normal',
                horizon: int = 1) -> pd.DataFrame:
        """Perform VaR backtesting."""
        if self.returns is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        if window_size is None:
            window_size = self.lookback
        
        print(f"Starting VaR backtesting with {window_size}-day rolling window...")
        
        # Initialize backtester
        self.backtester = Backtester(self.returns, window_size)
        
        # Run backtest
        backtest_results = self.backtester.run_backtest(
            portfolio_weights=self.portfolio_weights,
            confidence_levels=[alpha],
            n_simulations=n_simulations,
            distribution=distribution,
            horizon=horizon,
            random_seed=self.random_seed
        )
        
        stats = self.backtester.calculate_backtest_statistics([alpha])
        conf_str = f"{int(alpha * 100)}%"
        
        if conf_str in stats:
            result = stats[conf_str]
            print(f"\nBacktest Results for VaR {conf_str}:")
            print(f"- Expected breach rate: {result['expected_breach_rate']:.1%}")
            print(f"- Actual breach rate: {result['actual_breach_rate']:.1%}")
            print(f"- Total breaches: {result['total_breaches']}")
            print(f"- Kupiec test p-value: {result['kupiec_p_value']:.4f}")
            print(f"- Model {'rejected' if result['kupiec_reject'] else 'accepted'} at 5% level")
        
        return backtest_results
    
    def get_summary_statistics(self) -> pd.DataFrame:
        """Get summary statistics for the loaded data."""
        if self.data_loader is None or self.returns is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        return self.data_loader.get_summary_statistics()
    
    def get_correlation_matrix(self) -> pd.DataFrame:
        """Get correlation matrix of returns."""
        if self.data_loader is None or self.returns is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        return self.data_loader.get_correlation_matrix()
    
    def plot_price_paths(self, n_paths: int = 100) -> None:
        """Plot simulated price paths."""
        if not self.simulation_results:
            raise ValueError("No simulation results available. Run compute_mc_var() first.")
        
        # Get asset returns and convert to price paths
        simulator = MonteCarloSimulator(self.simulation_results['parameters'], self.random_seed)
        
        if self.simulation_results['distribution'] == 'normal':
            _, prices = simulator.simulate_normal_returns(
                n_paths, self.simulation_results['horizon']
            )
        elif self.simulation_results['distribution'] == 't':
            _, prices = simulator.simulate_t_returns(
                n_paths, self.simulation_results['horizon']
            )
        else:
            _, prices = simulator.simulate_bootstrap_returns(
                n_paths, self.simulation_results['horizon']
            )
        
        fig = self.visualizer.plot_price_paths(prices, asset_names=self.tickers, n_paths_display=n_paths)
        return fig
    
    def plot_return_distribution(self, confidence_levels: List[float] = [0.95, 0.99]) -> None:
        """Plot portfolio return distribution with VaR markers."""
        if not self.simulation_results:
            raise ValueError("No simulation results available. Run compute_mc_var() first.")
        
        portfolio_returns = self.simulation_results['portfolio_returns']
        if portfolio_returns.ndim > 1:
            returns = portfolio_returns[:, -1]  # Use final period returns
        else:
            returns = portfolio_returns
        
        fig = self.visualizer.plot_return_distribution(returns, confidence_levels)
        return fig
    
    def plot_backtest_results(self, confidence_level: float = 0.95) -> None:
        """Plot backtesting results."""
        if self.backtester is None or self.backtester.backtest_results is None:
            raise ValueError("No backtest results available. Run backtest() first.")
        
        fig = self.visualizer.plot_backtest_results(
            self.backtester.backtest_results, confidence_level
        )
        return fig
    
    def plot_correlation_heatmap(self) -> None:
        """Plot correlation heatmap of asset returns."""
        if self.returns is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        correlation_matrix = self.get_correlation_matrix()
        fig = self.visualizer.plot_correlation_heatmap(correlation_matrix)
        return fig
    
    def create_risk_dashboard(self) -> None:
        """Create interactive risk dashboard."""
        if not self.risk_metrics:
            raise ValueError("No risk metrics available. Run compute_comprehensive_risk_metrics() first.")
        
        backtest_data = None
        if self.backtester is not None:
            backtest_data = self.backtester.backtest_results
        
        fig = self.visualizer.create_risk_dashboard(
            self.risk_metrics.get('comprehensive', {}),
            backtest_data
        )
        return fig
    
    def export_results(self, filepath: str, format: str = 'excel') -> None:
        """
        Export results to file.
        
        Args:
            filepath: Output file path
            format: Export format ('excel', 'csv', or 'json')
        """
        results_dict = {
            'summary_statistics': self.get_summary_statistics() if self.returns is not None else None,
            'correlation_matrix': self.get_correlation_matrix() if self.returns is not None else None,
            'risk_metrics': self.risk_metrics,
            'simulation_parameters': self.parameters
        }
        
        if self.backtester is not None and self.backtester.backtest_results is not None:
            results_dict['backtest_results'] = self.backtester.backtest_results
            results_dict['backtest_statistics'] = self.backtester.calculate_backtest_statistics()
        
        if format == 'excel':
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                for sheet_name, data in results_dict.items():
                    if isinstance(data, pd.DataFrame):
                        data.to_excel(writer, sheet_name=sheet_name)
                    elif isinstance(data, dict):
                        pd.DataFrame([data]).to_excel(writer, sheet_name=sheet_name)
        
        elif format == 'json':
            import json
            # Convert DataFrames to dict for JSON serialization
            json_dict = {}
            for key, value in results_dict.items():
                if isinstance(value, pd.DataFrame):
                    json_dict[key] = value.to_dict()
                else:
                    json_dict[key] = value
            
            with open(filepath, 'w') as f:
                json.dump(json_dict, f, indent=2, default=str)
        
        else:
            raise ValueError("Format must be 'excel', 'csv', or 'json'")
        
        print(f"Results exported to: {filepath}")
    
    def __repr__(self) -> str:
        """String representation of the RiskEngine."""
        return (f"RiskEngine(tickers={self.tickers}, "
                f"lookback={self.lookback}, "
                f"weights={self.portfolio_weights.round(3).tolist()})") 