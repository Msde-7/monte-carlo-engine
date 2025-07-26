import numpy as np
import pandas as pd
import yfinance as yf
from typing import List, Optional, Union, Tuple
import warnings
warnings.filterwarnings('ignore')


class DataLoader:
    """Handles data loading and preprocessing for portfolio risk analysis."""
    
    def __init__(self, tickers: List[str], start_date: Optional[str] = None, 
                 end_date: Optional[str] = None):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.prices = None
        self.returns = None
        self.log_returns = None
        
    def fetch_data(self, period: str = "2y") -> pd.DataFrame:
        """Fetch historical price data using yfinance."""
        try:
            if self.start_date and self.end_date:
                data = yf.download(self.tickers, start=self.start_date, 
                                 end=self.end_date, progress=False)
            else:
                data = yf.download(self.tickers, period=period, progress=False)
            
            if len(self.tickers) == 1:
                # For single ticker, data is already a DataFrame
                if 'Adj Close' in data.columns:
                    prices = data[['Adj Close']].copy()
                    prices.columns = self.tickers
                elif 'Close' in data.columns:
                    prices = data[['Close']].copy()
                    prices.columns = self.tickers
                else:
                    raise ValueError("No price column found in data")
            else:
                # For multiple tickers, try Adj Close first, fallback to Close
                if 'Adj Close' in data.columns:
                    prices = data['Adj Close']
                elif 'Close' in data.columns:
                    prices = data['Close']
                else:
                    raise ValueError("No price column found in data")
            
            prices = prices.dropna()
            
            if prices.empty:
                raise ValueError("No price data available for the specified tickers and date range.")
            
            self.prices = prices
            print(f"Loaded data for {len(self.tickers)} assets: {len(prices)} observations")
            
            return prices
            
        except Exception as e:
            raise RuntimeError(f"Failed to fetch data: {str(e)}")
    
    def compute_returns(self, method: str = "simple") -> pd.DataFrame:
        """Compute returns from price data."""
        if self.prices is None:
            raise ValueError("No price data available. Call fetch_data() first.")
        
        if method == "simple":
            returns = self.prices.pct_change().dropna()
            self.returns = returns
        elif method == "log":
            returns = np.log(self.prices / self.prices.shift(1)).dropna()
            self.log_returns = returns
        else:
            raise ValueError("Method must be either 'simple' or 'log'")
        
        return returns
    
    def get_summary_statistics(self) -> pd.DataFrame:
        """Get summary statistics for returns data."""
        if self.returns is None:
            self.compute_returns()
        
        stats = pd.DataFrame({
            'Mean (Annualized)': self.returns.mean() * 252,
            'Std (Annualized)': self.returns.std() * np.sqrt(252),
            'Skewness': self.returns.skew(),
            'Kurtosis': self.returns.kurtosis(),
            'Min': self.returns.min(),
            'Max': self.returns.max(),
            'Observations': len(self.returns)
        })
        
        return stats
    
    def load_from_csv(self, filepath: str, date_column: str = 'Date') -> pd.DataFrame:
        try:
            data = pd.read_csv(filepath)
            data[date_column] = pd.to_datetime(data[date_column])
            data.set_index(date_column, inplace=True)
            
            # Filter for our tickers if they exist in the CSV
            available_tickers = [ticker for ticker in self.tickers if ticker in data.columns]
            if not available_tickers:
                raise ValueError(f"None of the specified tickers {self.tickers} found in CSV columns: {list(data.columns)}")
            
            self.prices = data[available_tickers].dropna()
            print(f"Loaded data from CSV for {len(available_tickers)} assets")
            
            return self.prices
            
        except Exception as e:
            raise RuntimeError(f"Failed to load CSV data: {str(e)}")
    
    def get_correlation_matrix(self) -> pd.DataFrame:
        """
        Compute correlation matrix of returns.
        
        Returns:
            Correlation matrix
        """
        if self.returns is None:
            self.compute_returns()
        
        return self.returns.corr()
    
    def get_covariance_matrix(self, annualized: bool = True) -> pd.DataFrame:
        """
        Compute covariance matrix of returns.
        
        Args:
            annualized: Whether to annualize the covariance matrix
            
        Returns:
            Covariance matrix
        """
        if self.returns is None:
            self.compute_returns()
        
        cov_matrix = self.returns.cov()
        
        if annualized:
            cov_matrix = cov_matrix * 252  # Assuming daily data
        
        return cov_matrix 