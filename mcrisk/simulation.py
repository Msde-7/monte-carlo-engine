import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, Optional, Tuple, List
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class MonteCarloSimulator:
    """Monte Carlo simulation for portfolio returns."""
    
    def __init__(self, parameters: Dict[str, Any], random_seed: Optional[int] = None):
        self.parameters = parameters
        self.random_seed = random_seed
        self.simulated_returns = None
        self.simulated_prices = None
        
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def simulate_normal_returns(self, n_simulations: int, horizon: int, 
                               initial_prices: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate returns using multivariate normal distribution."""
        if self.parameters['distribution'] != 'normal':
            raise ValueError("Parameters must be for normal distribution")
        
        mean = self.parameters['mean']
        cov = self.parameters['cov']
        n_assets = len(mean)
        
        if initial_prices is None:
            initial_prices = np.ones(n_assets)
        
        # Generate random returns
        returns = np.random.multivariate_normal(
            mean=mean,
            cov=cov,
            size=(n_simulations, horizon)
        )
        
        # Convert to price paths
        prices = self._returns_to_prices(returns, initial_prices)
        
        self.simulated_returns = returns
        self.simulated_prices = prices
        
        return returns, prices
    
    def simulate_t_returns(self, n_simulations: int, horizon: int,
                          initial_prices: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate returns using multivariate Student-t distribution.
        
        Args:
            n_simulations: Number of simulation paths
            horizon: Number of time periods to simulate
            initial_prices: Initial prices for each asset (defaults to 1.0)
            
        Returns:
            Tuple of (simulated_returns, simulated_prices)
        """
        if self.parameters['distribution'] != 't':
            raise ValueError("Parameters must be for t-distribution")
        
        mean = self.parameters['mean']
        cov = self.parameters['cov']
        df = self.parameters['df']
        n_assets = len(mean)
        
        if initial_prices is None:
            initial_prices = np.ones(n_assets)
        
        # Generate multivariate t random variables
        # Using the fact that t = Normal / sqrt(Chi2/df)
        returns = np.zeros((n_simulations, horizon, n_assets))
        
        for sim in range(n_simulations):
            for t in range(horizon):
                # Generate normal random vector
                normal_sample = np.random.multivariate_normal(
                    mean=np.zeros(n_assets), cov=cov
                )
                
                # Generate chi-squared random variable
                chi2_sample = np.random.chisquare(df)
                
                # Create t-distributed sample
                t_factor = np.sqrt(df / chi2_sample)
                returns[sim, t, :] = mean + normal_sample * t_factor
        
        # Convert to price paths
        prices = self._returns_to_prices(returns, initial_prices)
        
        self.simulated_returns = returns
        self.simulated_prices = prices
        
        return returns, prices
    
    def simulate_bootstrap_returns(self, n_simulations: int, horizon: int,
                                  initial_prices: Optional[np.ndarray] = None,
                                  block_size: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate returns using bootstrap resampling.
        
        Args:
            n_simulations: Number of simulation paths
            horizon: Number of time periods to simulate
            initial_prices: Initial prices for each asset (defaults to 1.0)
            block_size: Size of blocks for block bootstrap (None for standard bootstrap)
            
        Returns:
            Tuple of (simulated_returns, simulated_prices)
        """
        if self.parameters['distribution'] != 'bootstrap':
            raise ValueError("Parameters must be for bootstrap")
        
        historical_data = self.parameters['data']
        n_assets = historical_data.shape[1]
        n_observations = len(historical_data)
        
        if initial_prices is None:
            initial_prices = np.ones(n_assets)
        
        returns = np.zeros((n_simulations, horizon, n_assets))
        
        if block_size is None:
            # Standard bootstrap - sample individual observations
            for sim in range(n_simulations):
                for t in range(horizon):
                    random_idx = np.random.randint(0, n_observations)
                    returns[sim, t, :] = historical_data[random_idx, :]
        else:
            # Block bootstrap - sample blocks of observations
            if block_size > n_observations:
                raise ValueError("Block size cannot be larger than the number of observations")
            
            for sim in range(n_simulations):
                t = 0
                while t < horizon:
                    # Randomly select a starting point for the block
                    start_idx = np.random.randint(0, n_observations - block_size + 1)
                    block = historical_data[start_idx:start_idx + block_size]
                    
                    # Add the block to our simulation
                    remaining_periods = min(block_size, horizon - t)
                    returns[sim, t:t + remaining_periods, :] = block[:remaining_periods]
                    t += remaining_periods
        
        # Convert to price paths
        prices = self._returns_to_prices(returns, initial_prices)
        
        self.simulated_returns = returns
        self.simulated_prices = prices
        
        return returns, prices
    
    def simulate_portfolio_returns(self, weights: np.ndarray, n_simulations: int, 
                                  horizon: int, method: str = 'normal') -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate portfolio returns given asset weights.
        
        Args:
            weights: Portfolio weights (must sum to 1.0)
            n_simulations: Number of simulation paths
            horizon: Number of time periods to simulate
            method: Simulation method ('normal', 't', or 'bootstrap')
            
        Returns:
            Tuple of (portfolio_returns, portfolio_values)
        """
        if not np.isclose(np.sum(weights), 1.0, rtol=1e-5):
            raise ValueError("Portfolio weights must sum to 1.0")
        
        # Generate asset returns
        if method == 'normal':
            asset_returns, _ = self.simulate_normal_returns(n_simulations, horizon)
        elif method == 't':
            asset_returns, _ = self.simulate_t_returns(n_simulations, horizon)
        elif method == 'bootstrap':
            asset_returns, _ = self.simulate_bootstrap_returns(n_simulations, horizon)
        else:
            raise ValueError("Method must be 'normal', 't', or 'bootstrap'")
        
        # Calculate portfolio returns
        portfolio_returns = np.zeros((n_simulations, horizon))
        for sim in range(n_simulations):
            for t in range(horizon):
                portfolio_returns[sim, t] = np.dot(weights, asset_returns[sim, t, :])
        
        # Calculate portfolio values (starting at 1.0)
        portfolio_values = np.zeros((n_simulations, horizon + 1))
        portfolio_values[:, 0] = 1.0
        
        for sim in range(n_simulations):
            for t in range(horizon):
                portfolio_values[sim, t + 1] = portfolio_values[sim, t] * (1 + portfolio_returns[sim, t])
        
        return portfolio_returns, portfolio_values
    
    def _returns_to_prices(self, returns: np.ndarray, initial_prices: np.ndarray) -> np.ndarray:
        """
        Convert returns to cumulative price paths.
        
        Args:
            returns: Array of returns with shape (n_simulations, horizon, n_assets)
            initial_prices: Initial prices for each asset
            
        Returns:
            Array of prices with shape (n_simulations, horizon + 1, n_assets)
        """
        n_simulations, horizon, n_assets = returns.shape
        prices = np.zeros((n_simulations, horizon + 1, n_assets))
        
        # Set initial prices
        prices[:, 0, :] = initial_prices
        
        # Calculate cumulative prices
        for sim in range(n_simulations):
            for t in range(horizon):
                prices[sim, t + 1, :] = prices[sim, t, :] * (1 + returns[sim, t, :])
        
        return prices
    
    def get_simulation_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of the simulated returns.
        
        Returns:
            Dictionary with summary statistics
        """
        if self.simulated_returns is None:
            raise ValueError("No simulations have been run yet")
        
        # Flatten returns across simulations and time
        returns_flat = self.simulated_returns.reshape(-1, self.simulated_returns.shape[-1])
        
        summary = {
            'n_simulations': self.simulated_returns.shape[0],
            'horizon': self.simulated_returns.shape[1],
            'n_assets': self.simulated_returns.shape[2],
            'mean_returns': np.mean(returns_flat, axis=0),
            'std_returns': np.std(returns_flat, axis=0),
            'min_returns': np.min(returns_flat, axis=0),
            'max_returns': np.max(returns_flat, axis=0),
            'percentiles': {
                '5%': np.percentile(returns_flat, 5, axis=0),
                '25%': np.percentile(returns_flat, 25, axis=0),
                '50%': np.percentile(returns_flat, 50, axis=0),
                '75%': np.percentile(returns_flat, 75, axis=0),
                '95%': np.percentile(returns_flat, 95, axis=0)
            }
        }
        
        return summary
    
    def export_simulations(self, filepath: str, format: str = 'csv') -> None:
        """
        Export simulation results to file.
        
        Args:
            filepath: Path to save the file
            format: File format ('csv' or 'hdf5')
        """
        if self.simulated_returns is None:
            raise ValueError("No simulations have been run yet")
        
        if format == 'csv':
            # Reshape and save as CSV
            n_sims, horizon, n_assets = self.simulated_returns.shape
            asset_names = self.parameters.get('assets', [f'Asset_{i}' for i in range(n_assets)])
            
            # Create a DataFrame with multi-level columns
            columns = pd.MultiIndex.from_product([
                [f'Sim_{i}' for i in range(n_sims)],
                asset_names
            ])
            
            data = self.simulated_returns.reshape(horizon, -1)
            df = pd.DataFrame(data, columns=columns)
            df.to_csv(filepath)
            
        elif format == 'hdf5':
            import h5py
            with h5py.File(filepath, 'w') as f:
                f.create_dataset('returns', data=self.simulated_returns)
                f.create_dataset('prices', data=self.simulated_prices)
                
        else:
            raise ValueError("Format must be 'csv' or 'hdf5'")
        
        print(f"Simulations exported to {filepath}") 