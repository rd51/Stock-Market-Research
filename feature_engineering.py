"""
Feature Engineering Module
==========================

Creates advanced features for machine learning models using freshly scraped financial data.
Optimized for time series analysis and volatility modeling.

Author: GitHub Copilot
Date: February 8, 2026
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import List, Optional, Union, Tuple
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Advanced feature engineering for financial time series data.
    
    Creates lag features, rolling statistics, volatility measures,
    regime indicators, and sequences for machine learning models.
    """
    
    def __init__(self):
        """Initialize the feature engineer."""
        self.scalers = {}
        logger.info("Initialized FeatureEngineer")
    
    def create_lag_features(self, df: pd.DataFrame, columns: List[str], 
                           lags: List[int] = [1, 5, 10, 20]) -> pd.DataFrame:
        """
        Create lag features for specified columns.
        
        Args:
            df (pd.DataFrame): Input dataframe
            columns (List[str]): Columns to create lags for
            lags (List[int]): Lag periods to create
            
        Returns:
            pd.DataFrame: DataFrame with lag features added
        """
        logger.info(f"Creating lag features for {len(columns)} columns with lags: {lags}")
        
        df_copy = df.copy()
        
        for col in columns:
            if col not in df_copy.columns:
                logger.warning(f"Column {col} not found in dataframe")
                continue
                
            for lag in lags:
                lag_col_name = f"{col}_lag_{lag}"
                df_copy[lag_col_name] = df_copy[col].shift(lag)
                
                # Calculate lag returns for price columns
                if any(keyword in col.lower() for keyword in ['price', 'close', 'open', 'high', 'low']):
                    return_col_name = f"{col}_lag_return_{lag}"
                    df_copy[return_col_name] = (df_copy[col] - df_copy[col].shift(lag)) / df_copy[col].shift(lag)
        
        # Remove rows with NaN values created by lagging
        initial_rows = len(df_copy)
        df_copy = df_copy.dropna()
        final_rows = len(df_copy)
        
        logger.info(f"Lag features created: {initial_rows - final_rows} rows removed due to NaN values")
        return df_copy
    
    def create_rolling_statistics(self, df: pd.DataFrame, column: str, 
                                windows: List[int] = [7, 14, 30, 60, 90]) -> pd.DataFrame:
        """
        Create rolling statistics for a column.
        
        Args:
            df (pd.DataFrame): Input dataframe
            column (str): Column to create rolling stats for
            windows (List[int]): Rolling window sizes
            
        Returns:
            pd.DataFrame: DataFrame with rolling statistics added
        """
        logger.info(f"Creating rolling statistics for {column} with windows: {windows}")
        
        if column not in df.columns:
            logger.error(f"Column {column} not found in dataframe")
            return df
        
        df_copy = df.copy()
        
        for window in windows:
            # Rolling mean
            df_copy[f"{column}_rolling_mean_{window}"] = df_copy[column].rolling(window=window).mean()
            
            # Rolling standard deviation
            df_copy[f"{column}_rolling_std_{window}"] = df_copy[column].rolling(window=window).std()
            
            # Rolling min/max
            df_copy[f"{column}_rolling_min_{window}"] = df_copy[column].rolling(window=window).min()
            df_copy[f"{column}_rolling_max_{window}"] = df_copy[column].rolling(window=window).max()
            
            # Rolling skewness and kurtosis (for advanced analysis)
            df_copy[f"{column}_rolling_skew_{window}"] = df_copy[column].rolling(window=window).skew()
            df_copy[f"{column}_rolling_kurt_{window}"] = df_copy[column].rolling(window=window).kurt()
            
            # Rate of change (momentum)
            df_copy[f"{column}_roc_{window}"] = df_copy[column].pct_change(periods=window)
        
        # Remove rows with NaN values from rolling operations
        initial_rows = len(df_copy)
        df_copy = df_copy.dropna()
        final_rows = len(df_copy)
        
        logger.info(f"Rolling statistics created: {initial_rows - final_rows} rows removed due to NaN values")
        return df_copy
    
    def create_volatility_clustering_features(self, df: pd.DataFrame, column: str = 'VIX') -> pd.DataFrame:
        """
        Create features that capture volatility clustering effects.
        
        Args:
            df (pd.DataFrame): Input dataframe
            column (str): Volatility column (default: 'VIX')
            
        Returns:
            pd.DataFrame: DataFrame with volatility clustering features
        """
        logger.info(f"Creating volatility clustering features for {column}")
        
        if column not in df.columns:
            logger.error(f"Column {column} not found in dataframe")
            return df
        
        df_copy = df.copy()
        
        # Calculate returns if not present
        if 'Returns' not in df_copy.columns:
            # Try to find a price column to calculate returns
            price_cols = [col for col in df_copy.columns if 'close' in col.lower() or 'price' in col.lower()]
            if price_cols:
                df_copy['Returns'] = df_copy[price_cols[0]].pct_change()
            else:
                logger.warning("No price column found to calculate returns")
                return df
        
        # Absolute returns (volatility proxy)
        df_copy['abs_returns'] = df_copy['Returns'].abs()
        
        # Squared returns (volatility measure)
        df_copy['squared_returns'] = df_copy['Returns'] ** 2
        
        # ARCH effect: lagged squared returns
        for lag in [1, 5, 10]:
            df_copy[f'squared_returns_lag_{lag}'] = df_copy['squared_returns'].shift(lag)
        
        # Volatility clustering: correlation between current and past volatility
        df_copy['volatility_persistence'] = df_copy[column].rolling(20).corr(df_copy[column].shift(1))
        
        # High-low range (intraday volatility proxy)
        if 'High' in df_copy.columns and 'Low' in df_copy.columns:
            df_copy['daily_range'] = (df_copy['High'] - df_copy['Low']) / df_copy['Low']
            df_copy['range_volatility'] = df_copy['daily_range'].rolling(20).std()
        
        # Realized volatility (rolling standard deviation of returns)
        df_copy['realized_volatility_20'] = df_copy['Returns'].rolling(20).std() * np.sqrt(252)  # Annualized
        
        # Volatility of volatility
        df_copy['vol_of_vol'] = df_copy[column].rolling(20).std()
        
        # Extreme value indicators
        vol_threshold = df_copy[column].quantile(0.95)  # 95th percentile
        df_copy['high_volatility_regime'] = (df_copy[column] > vol_threshold).astype(int)
        
        # Remove NaN values
        df_copy = df_copy.dropna()
        
        logger.info(f"Volatility clustering features created: {len(df_copy)} rows")
        return df_copy
    
    def create_regime_indicators(self, df: pd.DataFrame, vix_col: str = 'VIX') -> pd.DataFrame:
        """
        Create market regime indicators based on volatility levels.
        
        Args:
            df (pd.DataFrame): Input dataframe
            vix_col (str): VIX column name
            
        Returns:
            pd.DataFrame: DataFrame with regime indicators
        """
        logger.info(f"Creating regime indicators based on {vix_col}")
        
        if vix_col not in df.columns:
            logger.error(f"Column {vix_col} not found in dataframe")
            return df
        
        df_copy = df.copy()
        
        # Define volatility regimes
        vix_values = df_copy[vix_col]
        
        # Low volatility regime (< 15)
        df_copy['low_vol_regime'] = (vix_values < 15).astype(int)
        
        # Normal volatility regime (15-25)
        df_copy['normal_vol_regime'] = ((vix_values >= 15) & (vix_values < 25)).astype(int)
        
        # High volatility regime (25-35)
        df_copy['high_vol_regime'] = ((vix_values >= 25) & (vix_values < 35)).astype(int)
        
        # Extreme volatility regime (>= 35)
        df_copy['extreme_vol_regime'] = (vix_values >= 35).astype(int)
        
        # Regime change indicators
        df_copy['regime_change'] = (df_copy['high_vol_regime'] != df_copy['high_vol_regime'].shift(1)).astype(int)
        
        # Bull/bear market indicators (if we have price data)
        if 'Returns' in df_copy.columns:
            # Cumulative returns over different periods
            df_copy['cumulative_return_20'] = (1 + df_copy['Returns']).rolling(20).apply(lambda x: x.prod() - 1)
            df_copy['cumulative_return_60'] = (1 + df_copy['Returns']).rolling(60).apply(lambda x: x.prod() - 1)
            
            # Bull market: positive cumulative returns
            df_copy['bull_market_20'] = (df_copy['cumulative_return_20'] > 0).astype(int)
            df_copy['bull_market_60'] = (df_copy['cumulative_return_60'] > 0).astype(int)
        
        # Risk-on/risk-off indicators
        df_copy['risk_on'] = ((df_copy['low_vol_regime'] == 1) & 
                             (df_copy.get('bull_market_20', 1) == 1)).astype(int)
        df_copy['risk_off'] = ((df_copy['high_vol_regime'] == 1) | 
                              (df_copy['extreme_vol_regime'] == 1)).astype(int)
        
        logger.info(f"Regime indicators created: {len(df_copy)} rows")
        return df_copy
    
    def create_shock_indicators(self, df: pd.DataFrame, columns: List[str] = ['VIX', 'Returns']) -> pd.DataFrame:
        """
        Create indicators for market shocks and extreme events.
        
        Args:
            df (pd.DataFrame): Input dataframe
            columns (List[str]): Columns to analyze for shocks
            
        Returns:
            pd.DataFrame: DataFrame with shock indicators
        """
        logger.info(f"Creating shock indicators for columns: {columns}")
        
        df_copy = df.copy()
        
        for col in columns:
            if col not in df_copy.columns:
                logger.warning(f"Column {col} not found, skipping shock indicators")
                continue
            
            # Calculate z-scores (standard deviations from mean)
            rolling_mean = df_copy[col].rolling(252).mean()  # 1 year lookback
            rolling_std = df_copy[col].rolling(252).std()
            df_copy[f'{col}_zscore'] = (df_copy[col] - rolling_mean) / rolling_std
            
            # Extreme positive shocks (3+ sigma)
            df_copy[f'{col}_positive_shock'] = (df_copy[f'{col}_zscore'] > 3).astype(int)
            
            # Extreme negative shocks (-3+ sigma)
            df_copy[f'{col}_negative_shock'] = (df_copy[f'{col}_zscore'] < -3).astype(int)
            
            # Jump detection (sudden large changes)
            pct_change = df_copy[col].pct_change()
            df_copy[f'{col}_jump_up'] = (pct_change > pct_change.quantile(0.99)).astype(int)
            df_copy[f'{col}_jump_down'] = (pct_change < pct_change.quantile(0.01)).astype(int)
            
            # Volatility spikes
            if col == 'VIX':
                df_copy['vix_spike'] = (df_copy[col] > 30).astype(int)  # VIX above 30
                df_copy['vix_crash'] = (df_copy[col] < 10).astype(int)  # VIX below 10
        
        # Market crash indicators (if we have price data)
        if 'Returns' in df_copy.columns:
            # 5-day cumulative return crash
            df_copy['crash_5d'] = ((1 + df_copy['Returns']).rolling(5).apply(lambda x: x.prod() - 1) < -0.10).astype(int)
            
            # 20-day cumulative return crash
            df_copy['crash_20d'] = ((1 + df_copy['Returns']).rolling(20).apply(lambda x: x.prod() - 1) < -0.20).astype(int)
        
        # Remove NaN values from calculations
        df_copy = df_copy.dropna()
        
        logger.info(f"Shock indicators created: {len(df_copy)} rows")
        return df_copy
    
    def create_seasonality_features(self, df: pd.DataFrame, date_col: str = 'Date') -> pd.DataFrame:
        """
        Create seasonality features from date information.
        
        Args:
            df (pd.DataFrame): Input dataframe
            date_col (str): Date column name
            
        Returns:
            pd.DataFrame: DataFrame with seasonality features
        """
        logger.info(f"Creating seasonality features from {date_col}")
        
        # Ensure date_col exists in the DataFrame
        if date_col not in df.columns:
            raise KeyError(f"The specified date column '{date_col}' does not exist in the DataFrame.")

        # Ensure date_col is a datetime type
        df_copy = df.copy()
        df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
        if df_copy[date_col].isnull().all():
            raise ValueError(f"All values in column '{date_col}' could not be converted to datetime.")

        # Basic date features
        df_copy['year'] = df_copy[date_col].dt.year
        df_copy['month'] = df_copy[date_col].dt.month
        df_copy['day'] = df_copy[date_col].dt.day
        df_copy['day_of_week'] = df_copy[date_col].dt.dayofweek  # 0=Monday, 6=Sunday
        df_copy['day_of_year'] = df_copy[date_col].dt.dayofyear
        df_copy['week_of_year'] = df_copy[date_col].dt.isocalendar().week
        df_copy['quarter'] = df_copy[date_col].dt.quarter
        
        # Cyclical encoding for periodic features
        df_copy['month_sin'] = np.sin(2 * np.pi * df_copy['month'] / 12)
        df_copy['month_cos'] = np.cos(2 * np.pi * df_copy['month'] / 12)
        
        df_copy['day_of_week_sin'] = np.sin(2 * np.pi * df_copy['day_of_week'] / 7)
        df_copy['day_of_week_cos'] = np.cos(2 * np.pi * df_copy['day_of_week'] / 7)
        
        # Weekend indicator
        df_copy['is_weekend'] = (df_copy['day_of_week'] >= 5).astype(int)
        
        # Business day indicator
        df_copy['is_business_day'] = (~df_copy[date_col].dt.dayofweek.isin([5, 6])).astype(int)
        
        # Month-end effect
        df_copy['is_month_end'] = df_copy[date_col].dt.is_month_end.astype(int)
        df_copy['is_month_start'] = df_copy[date_col].dt.is_month_start.astype(int)
        
        # Quarter-end effect
        df_copy['is_quarter_end'] = df_copy[date_col].dt.is_quarter_end.astype(int)
        df_copy['is_quarter_start'] = df_copy[date_col].dt.is_quarter_start.astype(int)
        
        # Holiday season (December)
        df_copy['is_december'] = (df_copy['month'] == 12).astype(int)
        
        # Tax season (March-April in India)
        df_copy['is_tax_season'] = df_copy['month'].isin([3, 4]).astype(int)
        
        logger.info(f"Seasonality features created: {len(df_copy)} rows")
        return df_copy
    
    def normalize_features(self, df: pd.DataFrame, columns: List[str], 
                          method: str = 'minmax') -> pd.DataFrame:
        """
        Normalize specified columns using various scaling methods.
        
        Args:
            df (pd.DataFrame): Input dataframe
            columns (List[str]): Columns to normalize
            method (str): Normalization method ('minmax', 'standard', 'robust')
            
        Returns:
            pd.DataFrame: DataFrame with normalized features
        """
        logger.info(f"Normalizing {len(columns)} columns using {method} scaling")
        
        if df.empty:
            logger.warning("DataFrame is empty, skipping normalization")
            return df
        
        df_copy = df.copy()
        
        for col in columns:
            if col not in df_copy.columns:
                logger.warning(f"Column {col} not found, skipping normalization")
                continue
            
            # Skip non-numeric columns
            if not pd.api.types.is_numeric_dtype(df_copy[col]):
                logger.warning(f"Column {col} is not numeric, skipping normalization")
                continue
            
            # Check if column has enough non-NaN values
            non_nan_count = df_copy[col].dropna().shape[0]
            if non_nan_count < 2:
                logger.warning(f"Column {col} has insufficient data for normalization ({non_nan_count} non-NaN values)")
                continue
            
            try:
                if method == 'minmax':
                    scaler = MinMaxScaler()
                    # Fit only on non-NaN values
                    valid_data = df_copy[col].dropna().to_numpy().reshape(-1, 1)
                    scaler.fit(valid_data)
                    scaled_values = scaler.transform(df_copy[[col]].fillna(df_copy[col].mean()))
                    df_copy[f"{col}_normalized"] = scaled_values.flatten()
                    self.scalers[f"{col}_minmax"] = scaler
                    
                elif method == 'standard':
                    scaler = StandardScaler()
                    valid_data = df_copy[col].dropna().to_numpy().reshape(-1, 1)
                    scaler.fit(valid_data)
                    scaled_values = scaler.transform(df_copy[[col]].fillna(df_copy[col].mean()))
                    df_copy[f"{col}_standardized"] = scaled_values.flatten()
                    self.scalers[f"{col}_standard"] = scaler
                    
                elif method == 'robust':
                    from sklearn.preprocessing import RobustScaler
                    scaler = RobustScaler()
                    valid_data = df_copy[col].dropna().to_numpy().reshape(-1, 1)
                    scaler.fit(valid_data)
                    scaled_values = scaler.transform(df_copy[[col]].fillna(df_copy[col].mean()))
                    df_copy[f"{col}_robust"] = scaled_values.flatten()
                    self.scalers[f"{col}_robust"] = scaler
                
                else:
                    logger.warning(f"Unknown normalization method: {method}")
                    
            except Exception as e:
                logger.error(f"Error normalizing column {col}: {str(e)}")
                continue
        
        logger.info(f"Feature normalization completed")
        return df_copy
    
    def create_sequences_for_lstm(self, df: pd.DataFrame, lookback: int = 30, 
                                target_col: str = 'Returns') -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM model training.
        
        Args:
            df (pd.DataFrame): Input dataframe
            lookback (int): Number of time steps to look back
            target_col (str): Target column for prediction
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (X_sequences, y_targets)
        """
        logger.info(f"Creating LSTM sequences with lookback={lookback}, target={target_col}")
        
        if target_col not in df.columns:
            logger.error(f"Target column {target_col} not found in dataframe")
            return np.array([]), np.array([])
        
        # Select numeric columns for features
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
        
        if not numeric_cols:
            logger.error("No numeric columns found for features")
            return np.array([]), np.array([])
        
        # Prepare data
        feature_data = df[numeric_cols].values
        target_data = df[target_col].values
        
        # Remove NaN values
        valid_indices = ~np.isnan(feature_data).any(axis=1) & ~np.isnan(target_data)
        feature_data = feature_data[valid_indices]
        target_data = target_data[valid_indices]
        
        if len(feature_data) < lookback + 1:
            logger.error(f"Insufficient data for sequences: {len(feature_data)} < {lookback + 1}")
            return np.array([]), np.array([])
        
        # Create sequences
        X_sequences = []
        y_targets = []
        
        for i in range(len(feature_data) - lookback):
            X_sequences.append(feature_data[i:i + lookback])
            y_targets.append(target_data[i + lookback])
        
        X_sequences = np.array(X_sequences)
        y_targets = np.array(y_targets)
        
        logger.info(f"LSTM sequences created: {len(X_sequences)} sequences, "
                   f"shape: {X_sequences.shape}")
        
        return X_sequences, y_targets
    
    def create_comprehensive_features(self, df: pd.DataFrame, 
                                    vix_col: str = 'VIX', 
                                    price_col: Optional[str] = None) -> pd.DataFrame:
        """
        Create a comprehensive set of features for machine learning.
        
        Args:
            df (pd.DataFrame): Input dataframe
            vix_col (str): VIX column name
            price_col (str): Price column name (optional)
            
        Returns:
            pd.DataFrame: DataFrame with comprehensive features
        """
        logger.info("Creating comprehensive feature set")
        
        df_featured = df.copy()
        
        # 1. Lag features
        lag_columns = [vix_col]
        if price_col is None:
            price_col = 'Close'
        if price_col and price_col in df_featured.columns:
            lag_columns.append(price_col)
        
        df_featured = self.create_lag_features(df_featured, lag_columns)
        
        # 2. Rolling statistics
        df_featured = self.create_rolling_statistics(df_featured, vix_col)
        
        # 3. Volatility clustering features
        df_featured = self.create_volatility_clustering_features(df_featured, vix_col)
        
        # 4. Regime indicators
        df_featured = self.create_regime_indicators(df_featured, vix_col)
        
        # 5. Shock indicators
        shock_cols = [vix_col]
        if 'Returns' in df_featured.columns:
            shock_cols.append('Returns')
        df_featured = self.create_shock_indicators(df_featured, shock_cols)
        
        # 6. Seasonality features
        df_featured = self.create_seasonality_features(df_featured)
        
        # 7. Normalize key features
        normalize_cols = [col for col in df_featured.columns 
                         if any(keyword in col.lower() for keyword in ['vix', 'return', 'volatility'])
                         and pd.api.types.is_numeric_dtype(df_featured[col])]
        
        if normalize_cols:
            df_featured = self.normalize_features(df_featured, normalize_cols[:5], method='standard')
        
        # Remove any remaining NaN values
        initial_rows = len(df_featured)
        df_featured = df_featured.dropna()
        final_rows = len(df_featured)
        
        logger.info(f"Comprehensive features created: {len(df_featured.columns)} features, "
                   f"{initial_rows - final_rows} rows removed due to NaN values")
        
        return df_featured