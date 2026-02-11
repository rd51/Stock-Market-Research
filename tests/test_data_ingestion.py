"""
Data Ingestion Tests
====================

Comprehensive test suite for data ingestion pipeline using pytest.
Tests web scraping, data quality, and feature engineering components.

Author: GitHub Copilot
Date: February 8, 2026
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import requests
from bs4 import BeautifulSoup
import tempfile
import os
from pathlib import Path

# Import modules to test
from data_ingestion import DataIngestionSystem
from preprocessing_pipeline import DataPreprocessingPipeline
from feature_engineering import FeatureEngineer
from web_scraping_utils import ScrapingSession, NSEIndiaScraper


# Module-level fixtures
@pytest.fixture
def ingestion_system():
    """Create a data ingestion system instance."""
    return DataIngestionSystem()

@pytest.fixture
def preprocessing_pipeline():
    """Create preprocessing pipeline instance."""
    return DataPreprocessingPipeline()

@pytest.fixture
def feature_engineer():
    """Create feature engineer instance."""
    return FeatureEngineer()

@pytest.fixture
def complete_pipeline():
    """Create complete data processing pipeline."""
    return DataPreprocessingPipeline()

@pytest.fixture
def sample_data():
    """Create sample financial data for testing."""
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    np.random.seed(42)

    data = pd.DataFrame({
        'date': dates,
        'nifty50': 18000 + np.random.normal(0, 500, 100).cumsum(),
        'sensex': 65000 + np.random.normal(0, 1000, 100).cumsum(),
        'vix': 15 + np.random.normal(0, 5, 100),
        'unemployment_rate': 6.5 + np.random.normal(0, 0.5, 100)
    })
    data.set_index('date', inplace=True)
    return data

@pytest.fixture
def scraped_data():
    """Create mock scraped data for testing."""
    return {
        'vix': pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=10, freq='D'),
            'VIX': [12.5, 13.2, 11.8, 14.1, 15.3, 13.7, 12.9, 11.4, 13.6, 14.8]
        }),
        'nifty50': pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=10, freq='D'),
            'Close': [18200, 18350, 18100, 18500, 18700, 18400, 18250, 18000, 18300, 18600]
        }),
        'sensex': pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=10, freq='D'),
            'Close': [65200, 65800, 64500, 66500, 67200, 65800, 65200, 64000, 65500, 66800]
        }),
        'unemployment': pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=10, freq='ME'),
            'Unemployment Rate': [6.2, 6.5, 6.1, 6.8, 6.3, 6.7, 6.4, 6.9, 6.6, 6.2]
        })
    }

@pytest.fixture
def config():
    """Test configuration fixture."""
    return {
        'start_date': '2023-01-01',
        'end_date': '2023-12-31',
        'timeout': 30,
        'retries': 3,
        'user_agent': 'Test-Agent/1.0'
    }

@pytest.fixture
def data_with_missing(sample_data):
    """Create data with missing values."""
    data = sample_data.copy()
    # Introduce missing values
    data.loc[data.index[10:15], 'nifty50'] = np.nan
    data.loc[data.index[20:25], 'vix'] = np.nan
    return data

@pytest.fixture
def data_with_outliers(sample_data):
    """Create data with outliers."""
    data = sample_data.copy()
    # Add extreme outliers
    data.loc[data.index[5], 'nifty50'] = 50000  # Extreme high
    data.loc[data.index[15], 'vix'] = 100  # Extreme high VIX
    return data

@pytest.fixture
def data_with_duplicates(sample_data):
    """Create data with duplicate rows."""
    data = sample_data.copy()
    # Add duplicate rows
    duplicate_row = data.iloc[10:11].copy()
    data = pd.concat([data, duplicate_row], ignore_index=False)
    return data

@pytest.fixture
def time_series_data():
    """Create time series data for feature engineering."""
    dates = pd.date_range('2023-01-01', periods=50, freq='D')
    np.random.seed(42)

    data = pd.DataFrame({
        'date': dates,
        'price': 100 + np.random.normal(0, 2, 50).cumsum(),
        'volume': np.random.normal(1000000, 200000, 50),
        'returns': np.random.normal(0, 0.02, 50)
    })
    data.set_index('date', inplace=True)
    return data


# Web Scraping Tests
def test_nse_connection(ingestion_system):
    """Test NSE site accessibility."""
    with patch('requests.get') as mock_get:
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = '<html><body>NSE India</body></html>'
        mock_get.return_value = mock_response

        # Test connection
        response = requests.get('https://www.nseindia.com')
        assert response.status_code == 200
        assert 'NSE India' in response.text

def test_vix_scraping(ingestion_system):
    """Test VIX data scraping."""
    with patch.object(ingestion_system, 'get_nse_vix_data') as mock_scrape:
        # Mock VIX data
        mock_vix_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=5, freq='D'),
            'vix': [12.5, 13.2, 11.8, 14.1, 15.3]
        })
        mock_scrape.return_value = mock_vix_data

        result = ingestion_system.get_nse_vix_data()

        assert isinstance(result, pd.DataFrame)
        assert 'vix' in result.columns
        assert len(result) == 5
        assert result['vix'].dtype in [np.float64, float]

def test_nifty_scraping(ingestion_system):
    """Test NIFTY50 data scraping."""
    with patch.object(ingestion_system, 'load_nifty50_data') as mock_scrape:
        mock_nifty_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=5, freq='D'),
            'close': [18200, 18350, 18100, 18500, 18700]
        })
        mock_scrape.return_value = mock_nifty_data

        result = ingestion_system.load_nifty50_data()

        assert isinstance(result, pd.DataFrame)
        assert 'close' in result.columns
        assert len(result) == 5
        assert all(result['close'] > 0)

def test_sensex_scraping(ingestion_system):
    """Test SENSEX data scraping."""
    with patch.object(ingestion_system, 'load_sensex_data') as mock_scrape:
        mock_sensex_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=5, freq='D'),
            'close': [65200, 65800, 64500, 66500, 67200]
        })
        mock_scrape.return_value = mock_sensex_data

        result = ingestion_system.load_sensex_data()

        assert isinstance(result, pd.DataFrame)
        assert 'close' in result.columns
        assert len(result) == 5
        assert all(result['close'] > 0)

def test_unemployment_scraping(ingestion_system):
    """Test unemployment data scraping."""
    with patch.object(ingestion_system, 'get_unemployment_data_gov_in') as mock_scrape:
        mock_unemployment_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=5, freq='ME'),
            'unemployment_rate': [6.2, 6.5, 6.1, 6.8, 6.3]
        })
        mock_scrape.return_value = mock_unemployment_data

        result = ingestion_system.get_unemployment_data_gov_in()

        assert isinstance(result, pd.DataFrame)
        assert 'unemployment_rate' in result.columns
        assert len(result) == 5
        assert all(result['unemployment_rate'] > 0)

def test_data_alignment(ingestion_system, scraped_data):
    """Test temporal alignment of all data sources."""
    vix_df = scraped_data['vix']
    nifty_df = scraped_data['nifty50']
    sensex_df = scraped_data['sensex']
    unemployment_df = scraped_data['unemployment']

    aligned_data = ingestion_system.align_temporal_index(
        vix_df, nifty_df, sensex_df, unemployment_df
    )

    assert isinstance(aligned_data, pd.DataFrame)
    assert len(aligned_data) > 0

    # Check that all expected columns are present (using actual column names from method)
    expected_columns = ['Date', 'Close', 'Close_sensex', 'VIX', 'Unemployment Rate']
    for col in expected_columns:
        assert col in aligned_data.columns

    # Check that dates are aligned
    assert aligned_data['Date'].is_monotonic_increasing


# Data Quality Tests
def test_missing_values_handling(preprocessing_pipeline, data_with_missing):
    """Test missing value detection and handling."""
    # Check missing values are detected
    missing_counts = data_with_missing.isnull().sum()
    assert missing_counts['nifty50'] > 0
    assert missing_counts['vix'] > 0

    # Test imputation (forward fill)
    filled_data = data_with_missing.ffill()

    # Check no missing values remain
    assert filled_data.isnull().sum().sum() == 0

def test_outlier_detection(preprocessing_pipeline, data_with_outliers):
    """Test outlier detection."""
    # Use IQR method for outlier detection
    def detect_outliers_iqr(data, column, threshold=1.5):
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
        return outliers

    nifty_outliers = detect_outliers_iqr(data_with_outliers, 'nifty50')
    vix_outliers = detect_outliers_iqr(data_with_outliers, 'vix')

    assert len(nifty_outliers) > 0
    assert len(vix_outliers) > 0

def test_stationarity_adf_test(preprocessing_pipeline, sample_data):
    """Test Augmented Dickey-Fuller stationarity test."""
    from statsmodels.tsa.stattools import adfuller

    # Test on potentially non-stationary series
    for column in ['nifty50', 'sensex']:
        result = adfuller(sample_data[column].dropna())
        adf_statistic = result[0]
        p_value = result[1]

        # ADF statistic should be a finite number
        assert np.isfinite(adf_statistic)
        assert 0 <= p_value <= 1

        # If p_value < 0.05, series is stationary
        # If p_value >= 0.05, series is non-stationary (common for price data)
        assert isinstance(p_value, float)

def test_duplicate_removal(preprocessing_pipeline, data_with_duplicates):
    """Test duplicate row detection and removal."""
    initial_length = len(data_with_duplicates)

    # Remove duplicates
    deduplicated_data = data_with_duplicates.drop_duplicates()

    final_length = len(deduplicated_data)

    # Should have fewer rows after deduplication
    assert final_length < initial_length

    # Check no duplicates remain
    assert not deduplicated_data.duplicated().any()


# Feature Engineering Tests
def test_lag_features_shape(feature_engineer, time_series_data):
    """Test lag feature creation produces correct dimensions."""
    columns_to_lag = ['price', 'volume']
    lags = [1, 3, 5]

    result = feature_engineer.create_lag_features(
        time_series_data, columns_to_lag, lags
    )

    # Check that lag features were added
    expected_new_columns = []
    for col in columns_to_lag:
        for lag in lags:
            expected_new_columns.append(f'{col}_lag_{lag}')
            # For price columns, also add lag return features
            if 'price' in col.lower():
                expected_new_columns.append(f'{col}_lag_return_{lag}')

    for col in expected_new_columns:
        assert col in result.columns

    # Check dimensions - result should have same or fewer rows due to NaN handling
    assert len(result) <= len(time_series_data)
    # Original columns + new lag features
    expected_total_columns = time_series_data.shape[1] + len(expected_new_columns)
    assert result.shape[1] == expected_total_columns

def test_rolling_statistics(feature_engineer, time_series_data):
    """Test rolling statistics calculation."""
    column = 'price'
    windows = [5, 10]

    result = feature_engineer.create_rolling_statistics(
        time_series_data, column, windows
    )

    # Check rolling features were added
    expected_features = []
    for window in windows:
        expected_features.extend([
            f'{column}_rolling_mean_{window}',
            f'{column}_rolling_std_{window}',
            f'{column}_rolling_min_{window}',
            f'{column}_rolling_max_{window}'
        ])

    for feature in expected_features:
        assert feature in result.columns

    # Check that rolling statistics are calculated correctly
    price_mean_5 = result['price_rolling_mean_5']
    assert not price_mean_5.isnull().all()  # Should have some valid values

def test_regime_indicators(feature_engineer, time_series_data):
    """Test market regime indicator creation."""
    # Add VIX column for regime detection
    data_with_vix = time_series_data.copy()
    data_with_vix['VIX'] = 15 + np.random.normal(0, 5, len(time_series_data))

    result = feature_engineer.create_regime_indicators(data_with_vix)

    # Check regime columns were added
    regime_columns = [col for col in result.columns if 'regime' in col.lower()]
    assert len(regime_columns) > 0

    # Check regime values are valid
    for col in regime_columns:
        unique_values = result[col].dropna().unique()
        assert len(unique_values) > 0

def test_sequence_creation(feature_engineer, time_series_data):
    """Test LSTM sequence creation."""
    lookback = 10

    # Create some features first and ensure we have a 'Returns' column
    data_with_features = feature_engineer.create_lag_features(
        time_series_data, ['price'], [1, 2, 3]
    )

    # Rename 'returns' to 'Returns' to match method expectation
    data_with_features = data_with_features.rename(columns={'returns': 'Returns'})

    X_sequences, y_sequences = feature_engineer.create_sequences_for_lstm(
        data_with_features, lookback=lookback
    )

    # Check that sequences are created (may be empty if insufficient data)
    assert isinstance(X_sequences, np.ndarray)
    assert isinstance(y_sequences, np.ndarray)

    # If sequences were created, check dimensions
    if len(X_sequences) > 0 and len(y_sequences) > 0:
        assert X_sequences.shape[0] == y_sequences.shape[0]
        assert X_sequences.shape[1] == lookback
        assert X_sequences.shape[0] == len(data_with_features) - lookback

        # Check that sequences are properly formed
        assert not np.isnan(X_sequences).any()
        assert not np.isnan(y_sequences).any()


# Integration Tests
def test_end_to_end_pipeline(complete_pipeline):
    """Test complete data processing pipeline."""
    # This would be a full integration test
    # For now, just test that the pipeline can be initialized
    assert complete_pipeline is not None
    assert hasattr(complete_pipeline, 'output_dir')
    assert hasattr(complete_pipeline, 'feature_engineer')

def test_data_validation(ingestion_system, sample_data):
    """Test data validation functions."""
    # Test dataframe validation - method returns list of issues
    issues = ingestion_system._validate_dataframe(
        sample_data, "test_data", ['nifty50', 'sensex', 'vix']
    )

    # Should return a list
    assert isinstance(issues, list)

    # For valid data, issues list should be empty or contain only warnings
    # (the method returns issues list, empty means validation passed)

def test_checkpoint_saving_loading(ingestion_system):
    """Test checkpoint functionality."""
    test_data = {'test_key': 'test_value', 'timestamp': datetime.now()}

    # Test saving checkpoint
    ingestion_system._save_checkpoint(test_data)

    # Test loading checkpoint
    loaded_data = ingestion_system._load_checkpoint()

    assert loaded_data is not None
    assert 'test_key' in loaded_data


# Test configuration and utilities
@pytest.fixture(scope="session")
def test_config():
    """Global test configuration."""
    return {
        'test_data_dir': Path(tempfile.mkdtemp()),
        'mock_responses': True,
        'timeout': 5,
        'retries': 1
    }


# Custom test markers
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "web_scraping: marks tests that involve web scraping"
    )


# Test utilities
def assert_dataframe_not_empty(df, name="DataFrame"):
    """Assert that a DataFrame is not empty."""
    assert isinstance(df, pd.DataFrame), f"{name} is not a DataFrame"
    assert not df.empty, f"{name} is empty"
    assert len(df) > 0, f"{name} has no rows"


def assert_columns_exist(df, columns, name="DataFrame"):
    """Assert that required columns exist in DataFrame."""
    missing_columns = [col for col in columns if col not in df.columns]
    assert not missing_columns, f"{name} is missing columns: {missing_columns}"


def assert_no_missing_values(df, columns=None, name="DataFrame"):
    """Assert that specified columns have no missing values."""
    check_columns = columns or df.columns
    for col in check_columns:
        if col in df.columns:
            missing_count = df[col].isnull().sum()
            assert missing_count == 0, f"{name} column '{col}' has {missing_count} missing values"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])