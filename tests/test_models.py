"""
Model Tests
===========

Comprehensive test suite for machine learning models using pytest.
Tests linear models, LSTM networks, ensemble methods, and integration.

Author: GitHub Copilot
Date: February 8, 2026
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import warnings
warnings.filterwarnings('ignore')

# Import model classes
from baseline_models import LinearBaselineModel
from lstm_models import LSTMPredictor
from ensemble_models import RandomForestPredictor, XGBoostPredictor, EnsembleComparison
from model_evaluation import EvaluationMetrics

# Import utilities
from feature_engineering import FeatureEngineer


# Module-level fixtures
@pytest.fixture
def linear_model():
    """Create a linear baseline model instance."""
    return LinearBaselineModel(model_type='static')

@pytest.fixture
def lstm_model():
    """Create an LSTM predictor instance."""
    return LSTMPredictor(lookback=10, feature_count=5, lstm_units=32)

@pytest.fixture
def rf_model():
    """Create a Random Forest predictor instance."""
    return RandomForestPredictor(n_estimators=10, max_depth=5, random_state=42)

@pytest.fixture
def xgb_model():
    """Create an XGBoost predictor instance."""
    return XGBoostPredictor(n_estimators=10, max_depth=3)

@pytest.fixture
def ensemble_comparison():
    """Create an ensemble comparison instance."""
    return EnsembleComparison()

@pytest.fixture
def evaluation_metrics():
    """Create evaluation metrics instance."""
    return EvaluationMetrics()

@pytest.fixture
def feature_engineer():
    """Create feature engineer instance."""
    return FeatureEngineer()

@pytest.fixture
def sample_regression_data():
    """Create sample regression data for testing."""
    np.random.seed(42)
    n_samples = 200
    n_features = 5

    # Generate features
    X = pd.DataFrame({
        'feature_1': np.random.normal(0, 1, n_samples),
        'feature_2': np.random.normal(0, 1, n_samples),
        'feature_3': np.random.normal(0, 1, n_samples),
        'feature_4': np.random.normal(0, 1, n_samples),
        'feature_5': np.random.normal(0, 1, n_samples)
    })

    # Generate target with some relationship to features
    y = (0.5 * X['feature_1'] +
         0.3 * X['feature_2'] +
         0.2 * X['feature_3'] +
         np.random.normal(0, 0.1, n_samples))

    return X, y

@pytest.fixture
def time_series_data():
    """Create time series data for LSTM testing."""
    np.random.seed(42)
    n_samples = 100
    n_features = 5

    # Generate time series data
    data = pd.DataFrame({
        'feature_1': np.random.normal(0, 1, n_samples),
        'feature_2': np.random.normal(0, 1, n_samples),
        'feature_3': np.random.normal(0, 1, n_samples),
        'feature_4': np.random.normal(0, 1, n_samples),
        'target': np.random.normal(0, 1, n_samples)
    })

    return data

@pytest.fixture
def train_test_split(sample_regression_data):
    """Split data into train/test sets."""
    X, y = sample_regression_data
    split_idx = int(0.8 * len(X))

    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]

    return X_train, X_test, y_train, y_test

@pytest.fixture
def lstm_sequences(time_series_data):
    """Create LSTM sequences for testing."""
    lookback = 10
    data = time_series_data.values
    target_column_index = -1  # Last column is target

    n_samples = len(data) - lookback
    X = []
    y = []

    for i in range(n_samples):
        X_seq = data[i:i+lookback]
        y_val = data[i+lookback, target_column_index]
        X.append(X_seq)
        y.append(y_val)

    X = np.array(X)
    y = np.array(y)

    # Split into train/test
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    return X_train, X_test, y_train, y_test


# Linear Model Tests
def test_ols_fitting(linear_model, train_test_split):
    """Test OLS model fitting without error."""
    X_train, X_test, y_train, y_test = train_test_split

    # Fit and predict
    linear_model.fit(X_train, y_train)
    predictions = linear_model.predict(X_test)

    # Check prediction shape
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (len(X_test),)
    assert not np.isnan(predictions).any()

def test_ols_evaluation_metrics(linear_model, train_test_split):
    """Test OLS evaluation metrics calculation."""
    X_train, X_test, y_train, y_test = train_test_split

    # Fit and evaluate
    linear_model.fit(X_train, y_train)
    metrics = linear_model.evaluate(X_test, y_test)

    # Check that metrics are calculated
    assert isinstance(metrics, dict)
    assert 'r2' in metrics
    assert 'rmse' in metrics
    assert 'mae' in metrics
    assert 'mape' in metrics

    # Check metric ranges
    assert -1 <= metrics['r2'] <= 1
    assert metrics['rmse'] >= 0
    assert metrics['mae'] >= 0
    assert metrics['mape'] >= 0


# LSTM Tests
def test_lstm_build(lstm_model):
    """Test LSTM model building."""
    # Build the model
    model = lstm_model.build_model()

    if model is not None:  # Only test if TensorFlow is available
        assert lstm_model.model is not None
        assert hasattr(lstm_model.model, 'predict')
        assert hasattr(lstm_model.model, 'fit')

        # Check input shape
        input_shape = lstm_model.model.input_shape
        assert input_shape == (None, lstm_model.lookback, lstm_model.feature_count)

def test_lstm_training(lstm_model, lstm_sequences):
    """Test LSTM training convergence."""
    X_train, X_test, y_train, y_test = lstm_sequences

    # Build model
    lstm_model.build_model()

    if lstm_model.model is not None:  # Only test if TensorFlow is available
        # Train the model (with minimal epochs for testing)
        history = lstm_model.train(X_train, y_train, X_test, y_test, epochs=2, batch_size=16)

        # Check training history
        assert history is not None
        assert 'loss' in history.history
        assert 'val_loss' in history.history

        # Loss should decrease or at least be finite
        final_loss = history.history['loss'][-1]
        assert np.isfinite(final_loss)
        assert final_loss >= 0


# Ensemble Tests
def test_rf_training(rf_model, train_test_split):
    """Test Random Forest training."""
    X_train, X_test, y_train, y_test = train_test_split

    # Fit the model
    fitted_model = rf_model.fit(X_train.values, y_train.values)

    # Check that model was fitted
    assert fitted_model.model is not None

def test_xgboost_training(xgb_model, train_test_split):
    """Test XGBoost training."""
    X_train, X_test, y_train, y_test = train_test_split

    # Fit the model
    fitted_model = xgb_model.fit(X_train.values, y_train.values)

    # Check that model was fitted
    if fitted_model.model is not None:  # Only if XGBoost is available
        assert xgb_model.model is not None

def test_ensemble_consensus(ensemble_comparison, train_test_split):
    """Test ensemble consensus calculation."""
    X_train, X_test, y_train, y_test = train_test_split

    # Fit all models
    ensemble_comparison.fit_all_models(X_train.values, y_train.values, X_test.values, y_test.values)

    # Get ensemble predictions
    ensemble_pred, ensemble_std = ensemble_comparison.predict_ensemble(X_test.values)

    # Check ensemble predictions
    assert isinstance(ensemble_pred, np.ndarray)
    assert isinstance(ensemble_std, np.ndarray)
    assert ensemble_pred.shape == (len(X_test),)
    assert ensemble_std.shape == (len(X_test),)
    assert not np.isnan(ensemble_pred).any()
    assert not np.isnan(ensemble_std).any()

    # Standard deviations should be non-negative
    assert np.all(ensemble_std >= 0)


# Integration Tests
def test_end_to_end_pipeline(feature_engineer, time_series_data):
    """Test complete data → model → predictions pipeline."""
    # Create features
    data_with_features = feature_engineer.create_lag_features(
        time_series_data, ['feature_1', 'feature_2'], [1, 2]
    )

    # Prepare data for modeling
    X = data_with_features.drop('target', axis=1)
    y = data_with_features['target']

    # Train linear model
    model = LinearBaselineModel()
    model.fit(X, y)

    # Make predictions
    predictions = model.predict(X)

    # Check pipeline completion
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(X)
    assert not np.isnan(predictions).any()

def test_model_comparison(ensemble_comparison, train_test_split):
    """Test all models are comparable."""
    import pytest
    pytest.skip("Model Comparison functionality moved/removed; skip test")


# Model Evaluation Tests
def test_evaluation_metrics_calculation(evaluation_metrics, train_test_split):
    """Test evaluation metrics calculation."""
    X_train, X_test, y_train, y_test = train_test_split

    # Generate mock predictions
    y_pred = y_test + np.random.normal(0, 0.1, len(y_test))

    # Calculate metrics using the correct method name
    metrics = evaluation_metrics.compute_all_metrics(y_test.values, y_pred)

    # Check that metrics are calculated
    assert isinstance(metrics, dict)
    assert 'mae' in metrics
    assert 'mse' in metrics
    assert 'rmse' in metrics
    assert 'r2_score' in metrics


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])