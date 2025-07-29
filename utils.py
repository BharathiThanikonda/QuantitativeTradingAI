#!/usr/bin/env python3
"""
QuantitativeTradingAI - Utility Functions
=========================================

Common utility functions used throughout the project.
"""

import os
import yaml
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime, timedelta
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class ConfigManager:
    """Configuration manager for loading and accessing project settings."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self._setup_logging()
    
    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except FileNotFoundError:
            print(f"Warning: Config file {self.config_path} not found. Using defaults.")
            return self._get_default_config()
        except yaml.YAMLError as e:
            print(f"Error parsing config file: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            'app': {'name': 'QuantitativeTradingAI', 'version': '1.0.0'},
            'trading': {'initial_capital': 10000, 'transaction_cost': 0.001},
            'models': {'window_size': 20, 'layer_size': 500}
        }
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_config = self.config.get('logging', {})
        log_level = getattr(logging, log_config.get('level', 'INFO'))
        log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_config.get('file', 'logs/app.log'))
            ]
        )
    
    def get(self, key: str, default=None):
        """Get configuration value by key."""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

class DataLoader:
    """Data loader for stock market data."""
    
    def __init__(self, data_dir: str = "Stock-Prediction-Models/dataset"):
        """Initialize data loader.
        
        Args:
            data_dir: Directory containing stock data files
        """
        self.data_dir = Path(data_dir)
        self.available_symbols = self._get_available_symbols()
    
    def _get_available_symbols(self) -> List[str]:
        """Get list of available stock symbols."""
        symbols = []
        if self.data_dir.exists():
            for file in self.data_dir.glob("*.csv"):
                symbols.append(file.stem)
        return symbols
    
    def load_stock_data(self, symbol: str, start_date: Optional[str] = None, 
                       end_date: Optional[str] = None) -> pd.DataFrame:
        """Load stock data for a given symbol.
        
        Args:
            symbol: Stock symbol (e.g., 'GOOG', 'TSLA')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            DataFrame with stock data
        """
        file_path = self.data_dir / f"{symbol}.csv"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file for {symbol} not found: {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            
            # Convert date column
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
            
            # Filter by date range
            if start_date:
                df = df[df.index >= start_date]
            if end_date:
                df = df[df.index <= end_date]
            
            return df
            
        except Exception as e:
            raise ValueError(f"Error loading data for {symbol}: {e}")
    
    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols."""
        return self.available_symbols.copy()

class PerformanceMetrics:
    """Calculate trading performance metrics."""
    
    @staticmethod
    def calculate_returns(prices: pd.Series) -> pd.Series:
        """Calculate daily returns."""
        return prices.pct_change().dropna()
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio.
        
        Args:
            returns: Series of returns
            risk_free_rate: Risk-free rate (annual)
            
        Returns:
            Sharpe ratio
        """
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()
    
    @staticmethod
    def calculate_max_drawdown(prices: pd.Series) -> float:
        """Calculate maximum drawdown.
        
        Args:
            prices: Series of prices
            
        Returns:
            Maximum drawdown as a percentage
        """
        cumulative = prices / prices.iloc[0]
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    @staticmethod
    def calculate_volatility(returns: pd.Series) -> float:
        """Calculate annualized volatility."""
        return returns.std() * np.sqrt(252)
    
    @staticmethod
    def calculate_total_return(prices: pd.Series) -> float:
        """Calculate total return."""
        return (prices.iloc[-1] / prices.iloc[0]) - 1

class TechnicalIndicators:
    """Calculate technical indicators."""
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index (RSI)."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, 
                      signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD (Moving Average Convergence Divergence)."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series, period: int = 20, 
                                 std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band
    
    @staticmethod
    def calculate_moving_averages(prices: pd.Series, short_period: int = 10, 
                                 long_period: int = 50) -> Tuple[pd.Series, pd.Series]:
        """Calculate short and long moving averages."""
        short_ma = prices.rolling(window=short_period).mean()
        long_ma = prices.rolling(window=long_period).mean()
        return short_ma, long_ma

class DataPreprocessor:
    """Data preprocessing utilities."""
    
    @staticmethod
    def normalize_data(data: np.ndarray, method: str = 'minmax') -> np.ndarray:
        """Normalize data using specified method.
        
        Args:
            data: Input data
            method: Normalization method ('minmax', 'zscore', 'robust')
            
        Returns:
            Normalized data
        """
        if method == 'minmax':
            return (data - data.min()) / (data.max() - data.min())
        elif method == 'zscore':
            return (data - data.mean()) / data.std()
        elif method == 'robust':
            median = np.median(data)
            mad = np.median(np.abs(data - median))
            return (data - median) / mad
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    @staticmethod
    def create_sequences(data: np.ndarray, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series prediction.
        
        Args:
            data: Input time series data
            window_size: Size of input window
            
        Returns:
            Tuple of (X, y) where X are input sequences and y are targets
        """
        X, y = [], []
        for i in range(len(data) - window_size):
            X.append(data[i:(i + window_size)])
            y.append(data[i + window_size])
        return np.array(X), np.array(y)
    
    @staticmethod
    def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to dataframe."""
        df = df.copy()
        
        # RSI
        df['RSI'] = TechnicalIndicators.calculate_rsi(df['Close'])
        
        # MACD
        macd, signal, hist = TechnicalIndicators.calculate_macd(df['Close'])
        df['MACD'] = macd
        df['MACD_Signal'] = signal
        df['MACD_Histogram'] = hist
        
        # Bollinger Bands
        upper, middle, lower = TechnicalIndicators.calculate_bollinger_bands(df['Close'])
        df['BB_Upper'] = upper
        df['BB_Middle'] = middle
        df['BB_Lower'] = lower
        
        # Moving Averages
        short_ma, long_ma = TechnicalIndicators.calculate_moving_averages(df['Close'])
        df['MA_Short'] = short_ma
        df['MA_Long'] = long_ma
        
        return df

def setup_project_structure():
    """Setup project directory structure."""
    directories = [
        'logs',
        'data',
        'output',
        'models',
        'checkpoints',
        'tests'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    print("✅ Project structure created successfully!")

def get_project_info() -> Dict:
    """Get project information."""
    return {
        'name': 'QuantitativeTradingAI',
        'version': '1.0.0',
        'description': 'Advanced quantitative trading and stock prediction platform',
        'models': {
            'deep_learning': 18,
            'trading_agents': 23,
            'ensemble_methods': 2
        },
        'features': [
            'Real-time trading API',
            'Portfolio optimization',
            'Monte Carlo simulations',
            'Technical analysis tools'
        ]
    }

# Global instances
config = ConfigManager()
data_loader = DataLoader()
performance_metrics = PerformanceMetrics()
technical_indicators = TechnicalIndicators()
data_preprocessor = DataPreprocessor()

if __name__ == "__main__":
    # Setup project structure
    setup_project_structure()
    
    # Print project info
    info = get_project_info()
    print(f"\n🎯 {info['name']} v{info['version']}")
    print(f"📊 {info['description']}")
    print(f"🤖 Models: {info['models']['deep_learning']} DL + {info['models']['trading_agents']} Agents")
    print(f"✨ Features: {', '.join(info['features'])}") 