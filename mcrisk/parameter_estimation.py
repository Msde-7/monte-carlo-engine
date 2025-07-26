import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, Optional, Tuple


class ParameterEstimator:
    """Handles parameter estimation for return distribution models."""
    
    def __init__(self, returns: pd.DataFrame):
        self.returns = returns
        self.parameters = {}
        
    def estimate_normal_parameters(self) -> Dict[str, Any]:
        """Estimate parameters for multivariate normal distribution."""
        mean_vector = self.returns.mean().values
        cov_matrix = self.returns.cov().values
        
        params = {
            'distribution': 'normal',
            'mean': mean_vector,
            'cov': cov_matrix,
            'assets': list(self.returns.columns)
        }
        
        self.parameters['normal'] = params
        return params
    
    def estimate_t_parameters(self) -> Dict[str, Any]:
        """Estimate parameters for multivariate Student-t distribution."""
        df_estimates = []
        
        for column in self.returns.columns:
            asset_returns = self.returns[column].values
            df_est, loc_est, scale_est = stats.t.fit(asset_returns)
            df_estimates.append(df_est)
        
        avg_df = np.mean(df_estimates)
        mean_vector = self.returns.mean().values
        cov_matrix = self.returns.cov().values
        
        params = {
            'distribution': 't',
            'mean': mean_vector,
            'cov': cov_matrix,
            'df': avg_df,
            'assets': list(self.returns.columns)
        }
        
        self.parameters['t'] = params
        return params
    
    def prepare_bootstrap_data(self, block_size: Optional[int] = None) -> Dict[str, Any]:
        """Prepare data for bootstrap resampling."""
        params = {
            'distribution': 'bootstrap',
            'data': self.returns.values,
            'block_size': block_size,
            'assets': list(self.returns.columns),
            'n_observations': len(self.returns)
        }
        
        self.parameters['bootstrap'] = params
        return params 