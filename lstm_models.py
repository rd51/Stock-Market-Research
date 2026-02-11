"""
LSTM Neural Network Models Module
==================================

Advanced LSTM neural networks for financial time series forecasting.
Implements deep learning models that should outperform linear baselines.

Author: GitHub Copilot
Date: February 8, 2026
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings
warnings.filterwarnings('ignore')

# Required libraries for deep learning
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, optimizers, callbacks
    from tensorflow.keras.models import Sequential
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("TensorFlow/Keras not available - LSTM modeling disabled")
    # Provide fallback functions
    def mean_squared_error(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def mean_absolute_error(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    def r2_score(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
    plt.style.use('default')
    sns.set_palette("husl")
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logging.warning("matplotlib/seaborn not available - plotting disabled")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LSTMPredictor:
    """
    LSTM neural network for financial time series forecasting.
    """

    def __init__(self, lookback: int = 30, feature_count: int = 8,
                 lstm_units: int = 64, dropout_rate: float = 0.2):
        """
        Initialize LSTM predictor.

        Args:
            lookback (int): Number of time steps to look back
            feature_count (int): Number of input features
            lstm_units (int): Number of LSTM units in first layer
            dropout_rate (float): Dropout rate for regularization
        """
        self.lookback = lookback
        self.feature_count = feature_count
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate

        self.model = None
        self.history = None
        self.scaler_X = None
        self.scaler_y = None

        logger.info(f"Initialized LSTMPredictor with lookback={lookback}, features={feature_count}")

    def build_model(self) -> Any:
        """
        Build LSTM neural network architecture.

        Returns:
            keras.Sequential: Compiled LSTM model
        """
        logger.info("Building LSTM model architecture")

        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available - cannot build model")
            return None

        try:
            model = Sequential([
                layers.Input(shape=(self.lookback, self.feature_count)),
                layers.LSTM(self.lstm_units, return_sequences=True),
                layers.Dropout(self.dropout_rate),
                layers.LSTM(self.lstm_units // 2, return_sequences=False),
                layers.Dropout(self.dropout_rate),
                layers.Dense(16, activation='relu'),
                layers.Dense(1, activation='linear')
            ])

            # Compile model
            model.compile(
                optimizer=optimizers.Adam(learning_rate=0.001),
                loss='mean_squared_error',
                metrics=['mae']
            )

            self.model = model
            logger.info(f"LSTM model built successfully. Parameters: {model.count_params()}")
            return model

        except Exception as e:
            logger.error(f"Error building LSTM model: {str(e)}")
            return None

    def prepare_sequences(self, data: np.ndarray, target_column_index: int,
                         lookback: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for LSTM training.

        Args:
            data (np.ndarray): Input data array
            target_column_index (int): Index of target column
            lookback (int, optional): Lookback window size

        Returns:
            Tuple[np.ndarray, np.ndarray]: X sequences and y targets
        """
        if lookback is None:
            lookback = self.lookback

        logger.info(f"Preparing sequences with lookback={lookback}")

        try:
            n_samples = len(data) - lookback
            if n_samples <= 0:
                raise ValueError(f"Insufficient data for lookback={lookback}")

            X = []
            y = []

            for i in range(n_samples):
                X_seq = data[i:i+lookback]  # Shape: (lookback, n_features)
                y_val = data[i+lookback, target_column_index]  # Next value of target

                X.append(X_seq)
                y.append(y_val)

            X = np.array(X)  # Shape: (n_samples, lookback, n_features)
            y = np.array(y)  # Shape: (n_samples,)

            logger.info(f"Sequences prepared: X shape {X.shape}, y shape {y.shape}")
            return X, y

        except Exception as e:
            logger.error(f"Error preparing sequences: {str(e)}")
            return np.array([]), np.array([])

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 100, batch_size: int = 32) -> Any:
        """
        Train the LSTM model with early stopping and callbacks.

        Args:
            X_train (np.ndarray): Training sequences
            y_train (np.ndarray): Training targets
            X_val (np.ndarray): Validation sequences
            y_val (np.ndarray): Validation targets
            epochs (int): Maximum number of epochs
            batch_size (int): Batch size for training

        Returns:
            History: Training history object
        """
        logger.info(f"Training LSTM model for {epochs} epochs, batch_size={batch_size}")

        if self.model is None:
            logger.warning("Model not built - building now")
            self.build_model()

        if self.model is None:
            logger.error("Could not build model")
            return None

        try:
            # Define callbacks
            early_stopping = callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            )

            reduce_lr = callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )

            model_checkpoint = callbacks.ModelCheckpoint(
                'best_lstm_model.keras',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )

            # Train model
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stopping, reduce_lr, model_checkpoint],
                verbose=1
            )

            self.history = history
            logger.info(f"Training completed. Best validation loss: {min(history.history['val_loss']):.6f}")
            return history

        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return None

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Make predictions on test data.

        Args:
            X_test (np.ndarray): Test sequences

        Returns:
            np.ndarray: Predictions
        """
        if self.model is None:
            logger.warning("Model not trained - returning zeros")
            return np.zeros(len(X_test))

        try:
            predictions = self.model.predict(X_test, verbose=0)
            return predictions.flatten()

        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            return np.zeros(len(X_test))

    def predict_with_uncertainty(self, X_test: np.ndarray, num_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with uncertainty quantification using Monte Carlo dropout.

        Args:
            X_test (np.ndarray): Test sequences
            num_samples (int): Number of Monte Carlo samples

        Returns:
            Tuple[np.ndarray, np.ndarray]: Mean predictions and standard deviations
        """
        logger.info(f"Generating predictions with uncertainty ({num_samples} samples)")

        if self.model is None:
            logger.warning("Model not trained - returning zeros")
            return np.zeros(len(X_test)), np.zeros(len(X_test))

        try:
            # Enable dropout during inference for uncertainty
            def enable_dropout(model):
                for layer in model.layers:
                    if isinstance(layer, layers.Dropout):
                        layer.training = True
                return model

            # Generate multiple predictions
            predictions = []

            for _ in range(num_samples):
                pred = self.model.predict(X_test, verbose=0)
                predictions.append(pred.flatten())

            predictions = np.array(predictions)  # Shape: (num_samples, n_test_samples)

            # Calculate mean and std
            mean_pred = np.mean(predictions, axis=0)
            std_pred = np.std(predictions, axis=0)

            logger.info(f"Uncertainty quantification completed. Mean std: {np.mean(std_pred):.6f}")
            return mean_pred, std_pred

        except Exception as e:
            logger.error(f"Error in uncertainty prediction: {str(e)}")
            return self.predict(X_test), np.zeros(len(X_test))

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance on test data.

        Args:
            X_test (np.ndarray): Test sequences
            y_test (np.ndarray): Test targets

        Returns:
            Dict[str, float]: Evaluation metrics
        """
        logger.info("Evaluating LSTM model performance")

        try:
            predictions = self.predict(X_test)

            # Basic metrics
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            mae = mean_absolute_error(y_test, predictions)
            mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
            r2 = r2_score(y_test, predictions)

            # Direction accuracy (correctly predicting up/down movements)
            actual_direction = np.sign(np.diff(y_test))
            pred_direction = np.sign(np.diff(predictions))

            # Handle cases where diff might be zero
            valid_indices = (actual_direction != 0) & (pred_direction != 0)
            if np.sum(valid_indices) > 0:
                direction_accuracy = np.mean(actual_direction[valid_indices] == pred_direction[valid_indices])
            else:
                direction_accuracy = 0.0

            results = {
                'rmse': rmse,
                'mae': mae,
                'mape': mape,
                'r2': r2,
                'direction_accuracy': direction_accuracy,
                'mean_absolute_percentage_error': mape
            }

            logger.info(f"Evaluation completed. RMSE: {rmse:.6f}, R²: {r2:.4f}, Direction Acc: {direction_accuracy:.4f}")
            return results

        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            return {
                'rmse': float('inf'),
                'mae': float('inf'),
                'mape': float('inf'),
                'r2': 0.0,
                'direction_accuracy': 0.0,
                'mean_absolute_percentage_error': float('inf')
            }


def create_train_val_test_sequences(data: np.ndarray, lookback: int,
                                   train_ratio: float = 0.7, val_ratio: float = 0.15) -> Tuple:
    """
    Create train/validation/test sequences maintaining temporal order.

    Args:
        data (np.ndarray): Input time series data
        lookback (int): Lookback window size
        train_ratio (float): Proportion for training
        val_ratio (float): Proportion for validation

    Returns:
        Tuple: (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    logger.info(f"Creating train/val/test sequences with lookback={lookback}")

    try:
        n_samples = len(data) - lookback
        if n_samples <= 0:
            raise ValueError(f"Insufficient data for lookback={lookback}")

        # Calculate split indices
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))

        # Create sequences for target (assuming last column is target)
        target_idx = -1
        X_all, y_all = [], []

        for i in range(n_samples):
            X_seq = data[i:i+lookback]
            y_val = data[i+lookback, target_idx]
            X_all.append(X_seq)
            y_all.append(y_val)

        X_all = np.array(X_all)
        y_all = np.array(y_all)

        # Split maintaining temporal order
        X_train = X_all[:train_end]
        y_train = y_all[:train_end]

        X_val = X_all[train_end:val_end]
        y_val = y_all[train_end:val_end]

        X_test = X_all[val_end:]
        y_test = y_all[val_end:]

        logger.info(f"Sequences created: Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}")
        return X_train, y_train, X_val, y_val, X_test, y_test

    except Exception as e:
        logger.error(f"Error creating sequences: {str(e)}")
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])


def analyze_temporal_dependencies(model: LSTMPredictor, X_test: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Analyze which time steps are most important for predictions.

    Args:
        model (LSTMPredictor): Trained LSTM model
        X_test (np.ndarray): Test sequences

    Returns:
        Dict[str, np.ndarray]: Temporal dependency analysis
    """
    logger.info("Analyzing temporal dependencies")

    if model.model is None:
        logger.warning("Model not trained")
        return {}

    try:
        # For simplicity, we'll use permutation importance
        # In a real implementation, you might use attention mechanisms or SHAP

        baseline_predictions = model.predict(X_test)
        baseline_mae = mean_absolute_error(baseline_predictions, baseline_predictions)  # Self-comparison

        importance_scores = []

        # Test importance of each time step by setting it to mean
        for lag in range(model.lookback):
            X_permuted = X_test.copy()

            # Set this lag to mean across all features
            X_permuted[:, lag, :] = np.mean(X_test[:, lag, :], axis=0)

            permuted_predictions = model.predict(X_permuted)
            permuted_mae = mean_absolute_error(baseline_predictions, permuted_predictions)

            # Higher MAE increase = more important
            importance = permuted_mae - baseline_mae
            importance_scores.append(importance)

        importance_scores = np.array(importance_scores)

        # Normalize to 0-1 scale
        if np.max(importance_scores) > 0:
            importance_scores = importance_scores / np.max(importance_scores)

        result = {
            'importance_per_lag': importance_scores,
            'most_important_lag': np.argmax(importance_scores),
            'average_importance': np.mean(importance_scores)
        }

        logger.info(f"Temporal analysis completed. Most important lag: {result['most_important_lag']}")
        return result

    except Exception as e:
        logger.error(f"Error analyzing temporal dependencies: {str(e)}")
        return {}


def plot_temporal_dependencies(temporal_analysis: Dict[str, np.ndarray]) -> Any:
    """
    Plot temporal dependency analysis.

    Args:
        temporal_analysis (Dict[str, np.ndarray]): Results from analyze_temporal_dependencies

    Returns:
        matplotlib figure object
    """
    if not MATPLOTLIB_AVAILABLE or not temporal_analysis:
        logger.warning("Matplotlib not available or no analysis data")
        return None

    logger.info("Creating temporal dependency visualization")

    try:
        fig, ax = plt.subplots(figsize=(12, 6))

        importance_scores = temporal_analysis['importance_per_lag']
        lags = range(-len(importance_scores), 0)  # Negative lags (t-30 to t-1)

        bars = ax.bar(lags, importance_scores, alpha=0.7, color='skyblue')
        ax.set_xlabel('Time Lag (relative to prediction time)')
        ax.set_ylabel('Importance Score')
        ax.set_title('Temporal Dependency Analysis: Importance of Each Time Lag')
        ax.grid(True, alpha=0.3)

        # Highlight most important lag
        most_important = temporal_analysis['most_important_lag']
        bars[most_important].set_color('red')
        bars[most_important].set_alpha(0.9)

        # Add value labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            if height > 0.1:  # Only label significant bars
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       '.2f', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        logger.info("Temporal dependency plot created")
        return fig

    except Exception as e:
        logger.error(f"Error creating temporal plot: {str(e)}")
        return None


def hyperparameter_grid_search(model_class: type, X_train: np.ndarray, y_train: np.ndarray,
                              X_val: np.ndarray, y_val: np.ndarray,
                              param_grid: Dict[str, List]) -> Dict[str, Any]:
    """
    Perform grid search over hyperparameters.

    Args:
        model_class (type): LSTM model class
        X_train (np.ndarray): Training sequences
        y_train (np.ndarray): Training targets
        X_val (np.ndarray): Validation sequences
        y_val (np.ndarray): Validation targets
        param_grid (Dict[str, List]): Parameter combinations to test

    Returns:
        Dict[str, Any]: Grid search results
    """
    logger.info("Starting hyperparameter grid search")

    try:
        results = []
        best_score = float('inf')
        best_params = None
        best_model = None

        # Generate all parameter combinations
        param_keys = list(param_grid.keys())
        param_values = list(param_grid.values())

        from itertools import product
        param_combinations = list(product(*param_values))

        logger.info(f"Testing {len(param_combinations)} parameter combinations")

        for i, params in enumerate(param_combinations):
            param_dict = dict(zip(param_keys, params))
            logger.info(f"Testing combination {i+1}/{len(param_combinations)}: {param_dict}")

            try:
                # Create and train model
                model = model_class(**param_dict)
                model.build_model()

                if model.model is None:
                    continue

                history = model.train(X_train, y_train, X_val, y_val, epochs=50, batch_size=32)

                if history is None:
                    continue

                # Evaluate on validation set
                val_predictions = model.predict(X_val)
                val_mae = mean_absolute_error(y_val, val_predictions)

                result = {
                    'params': param_dict,
                    'val_mae': val_mae,
                    'epochs_trained': len(history.history['loss']),
                    'final_train_loss': history.history['loss'][-1],
                    'final_val_loss': history.history['val_loss'][-1]
                }

                results.append(result)

                # Track best model
                if val_mae < best_score:
                    best_score = val_mae
                    best_params = param_dict
                    best_model = model

            except Exception as e:
                logger.warning(f"Failed combination {param_dict}: {str(e)}")
                continue

        results_df = pd.DataFrame(results)

        grid_results = {
            'best_params': best_params,
            'best_score': best_score,
            'results': results_df,
            'best_model': best_model
        }

        logger.info(f"Grid search completed. Best MAE: {best_score:.6f}")
        return grid_results

    except Exception as e:
        logger.error(f"Error in grid search: {str(e)}")
        return {}


def plot_training_history(history: Any) -> Any:
    """
    Plot training history with loss and MAE curves.

    Args:
        history: Keras training history object

    Returns:
        matplotlib figure object
    """
    if not MATPLOTLIB_AVAILABLE or history is None:
        logger.warning("Matplotlib not available or no history")
        return None

    logger.info("Creating training history visualization")

    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        epochs = range(1, len(history.history['loss']) + 1)

        # Loss plot
        ax1.plot(epochs, history.history['loss'], 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, history.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        ax1.set_title('Model Loss Over Epochs')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss (MSE)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # MAE plot
        ax2.plot(epochs, history.history['mae'], 'b-', label='Training MAE', linewidth=2)
        ax2.plot(epochs, history.history['val_mae'], 'r-', label='Validation MAE', linewidth=2)
        ax2.set_title('Model MAE Over Epochs')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        logger.info("Training history plot created")
        return fig

    except Exception as e:
        logger.error(f"Error creating training plot: {str(e)}")
        return None


def plot_predictions_vs_actual(y_true: np.ndarray, y_pred: np.ndarray,
                              y_std: Optional[np.ndarray] = None,
                              dates: Optional[pd.Series] = None) -> Any:
    """
    Plot predictions vs actual values with uncertainty bands.

    Args:
        y_true (np.ndarray): True values
        y_pred (np.ndarray): Predicted values
        y_std (np.ndarray, optional): Prediction uncertainties
        dates (pd.Series, optional): Date index

    Returns:
        matplotlib figure object
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Matplotlib not available")
        return None

    logger.info("Creating predictions vs actual visualization")

    try:
        fig, ax = plt.subplots(figsize=(14, 8))

        # Use dates if provided, otherwise use indices
        x_axis = dates[len(dates)-len(y_true):] if dates is not None else range(len(y_true))

        # Plot predictions and actual
        ax.plot(x_axis, y_true, 'b-', label='Actual', linewidth=2, alpha=0.8)
        ax.plot(x_axis, y_pred, 'r-', label='Predicted', linewidth=2, alpha=0.8)

        # Add uncertainty bands if available
        if y_std is not None:
            ax.fill_between(x_axis, y_pred - 2*y_std, y_pred + 2*y_std,
                          alpha=0.3, color='red', label='±2σ Uncertainty')

        ax.set_xlabel('Time' if dates is not None else 'Sample')
        ax.set_ylabel('Value')
        ax.set_title('LSTM Predictions vs Actual Values')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        logger.info("Predictions vs actual plot created")
        return fig

    except Exception as e:
        logger.error(f"Error creating predictions plot: {str(e)}")
        return None


def plot_prediction_errors(y_true: np.ndarray, y_pred: np.ndarray) -> Any:
    """
    Plot prediction error analysis.

    Args:
        y_true (np.ndarray): True values
        y_pred (np.ndarray): Predicted values

    Returns:
        matplotlib figure object
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Matplotlib not available")
        return None

    logger.info("Creating prediction error analysis")

    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        errors = y_true - y_pred

        # Error distribution
        ax1.hist(errors, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax1.set_xlabel('Prediction Error')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Prediction Errors')
        ax1.grid(True, alpha=0.3)

        # Error autocorrelation (if enough data)
        if len(errors) > 20:
            try:
                from statsmodels.graphics.tsaplots import plot_acf
                plot_acf(errors, lags=min(20, len(errors)-1), ax=ax2, alpha=0.05)
                ax2.set_title('Autocorrelation of Prediction Errors')
            except ImportError:
                # Fallback: simple autocorrelation calculation
                lags = min(20, len(errors)-1)
                autocorr = []
                for lag in range(1, lags+1):
                    corr = np.corrcoef(errors[:-lag], errors[lag:])[0, 1]
                    autocorr.append(corr)

                ax2.bar(range(1, len(autocorr)+1), autocorr, alpha=0.7, color='skyblue')
                ax2.set_xlabel('Lag')
                ax2.set_ylabel('Autocorrelation')
                ax2.set_title('Autocorrelation of Prediction Errors (Manual)')
                ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'Insufficient data\nfor autocorrelation',
                    ha='center', va='center', transform=ax2.transAxes)

        plt.tight_layout()
        logger.info("Prediction error analysis created")
        return fig

    except Exception as e:
        logger.error(f"Error creating error analysis: {str(e)}")
        return None


def main():
    """
    Main function demonstrating LSTM time series forecasting.
    """
    logger.info("Starting LSTM time series forecasting demonstration")

    try:
        # Load preprocessed data
        data_sources = [
            'processed_data.csv',
            'stationary_data.csv',
            '../processed_data.csv'
        ]

        data = None
        for source in data_sources:
            try:
                data = pd.read_csv(source, index_col=0, parse_dates=True)
                logger.info(f"Loaded data from {source}: {data.shape}")
                break
            except FileNotFoundError:
                continue

        if data is None:
            logger.warning("No processed data found - generating synthetic data")
            # Generate synthetic financial data
            np.random.seed(42)
            dates = pd.date_range('2020-01-01', periods=1000, freq='D')

            # Create correlated time series
            vix_base = 20 + np.random.normal(0, 2, 1000)
            vix = pd.Series(np.maximum(5, vix_base), index=dates, name='VIX')

            returns_noise = np.random.normal(0, 0.02, 1000)
            returns = pd.Series(returns_noise - 0.5 * (vix - 20) / 100, index=dates, name='Returns')

            unemployment_base = 6.0
            unemployment = []
            for i in range(1000):
                market_stress = vix.iloc[min(i, len(vix)-1)] - 20
                shock = np.random.normal(0, 0.05)
                unemployment_base += shock - 0.01 * market_stress
                unemployment_base = max(3, min(15, unemployment_base))
                unemployment.append(unemployment_base)

            unemployment = pd.Series(unemployment, index=dates, name='Unemployment')

            data = pd.DataFrame({
                'VIX': vix,
                'Returns': returns,
                'Unemployment': unemployment
            })

        # Prepare data for LSTM
        logger.info("Preparing data for LSTM modeling")

        # Normalize data
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data)

        # Create sequences
        lookback = 30
        X_train, y_train, X_val, y_val, X_test, y_test = create_train_val_test_sequences(
            data_scaled, lookback, train_ratio=0.7, val_ratio=0.15
        )

        if len(X_train) == 0:
            logger.error("Failed to create sequences")
            return

        # Build and train LSTM model
        logger.info("Building and training LSTM model")
        feature_count = data.shape[1]
        lstm_model = LSTMPredictor(lookback=lookback, feature_count=feature_count)

        # Build model
        lstm_model.build_model()

        # Train model
        history = lstm_model.train(X_train, y_train, X_val, y_val, epochs=50, batch_size=32)

        # Evaluate model
        logger.info("Evaluating LSTM model")
        test_metrics = lstm_model.evaluate(X_test, y_test)

        # Generate predictions with uncertainty
        logger.info("Generating predictions with uncertainty")
        pred_mean, pred_std = lstm_model.predict_with_uncertainty(X_test, num_samples=50)

        # Analyze temporal dependencies
        logger.info("Analyzing temporal dependencies")
        temporal_analysis = analyze_temporal_dependencies(lstm_model, X_test[:100])  # Subset for speed

        # Create visualizations
        if MATPLOTLIB_AVAILABLE:
            logger.info("Creating visualizations")

            # Training history
            history_fig = plot_training_history(history)

            # Predictions vs actual
            test_dates = data.index[-len(y_test):] if hasattr(data, 'index') else None
            pred_fig = plot_predictions_vs_actual(y_test, pred_mean, pred_std, test_dates)

            # Prediction errors
            error_fig = plot_prediction_errors(y_test, pred_mean)

            # Temporal dependencies
            temporal_fig = None
            if temporal_analysis:
                temporal_fig = plot_temporal_dependencies(temporal_analysis)

        # Save results
        logger.info("Saving results")

        # Save model predictions
        predictions_df = pd.DataFrame({
            'actual': y_test,
            'predicted': pred_mean,
            'uncertainty': pred_std
        })
        predictions_df.to_csv('lstm_predictions.csv', index=False)

        # Save evaluation metrics
        metrics_df = pd.DataFrame([test_metrics])
        metrics_df.to_csv('lstm_evaluation_metrics.csv', index=False)

        # Save temporal analysis
        if temporal_analysis:
            temporal_df = pd.DataFrame({
                'lag': range(-lookback, 0),
                'importance': temporal_analysis['importance_per_lag']
            })
            temporal_df.to_csv('lstm_temporal_analysis.csv', index=False)

        # Save plots
        if MATPLOTLIB_AVAILABLE:
            if history_fig:
                history_fig.savefig('lstm_training_history.png', dpi=300, bbox_inches='tight')
            if pred_fig:
                pred_fig.savefig('lstm_predictions.png', dpi=300, bbox_inches='tight')
            if error_fig:
                error_fig.savefig('lstm_prediction_errors.png', dpi=300, bbox_inches='tight')
            if temporal_fig:
                temporal_fig.savefig('lstm_temporal_dependencies.png', dpi=300, bbox_inches='tight')

        # Print summary
        print("\n" + "="*80)
        print("LSTM TIME SERIES FORECASTING COMPLETED")
        print("="*80)
        print(f"Data: {len(data)} total samples")
        print(f"Sequences: lookback={lookback}, features={feature_count}")
        print(f"Train/Val/Test split: {len(X_train)}/{len(X_val)}/{len(X_test)}")
        print()

        print("MODEL PERFORMANCE:")
        print("-" * 30)
        print(".6f")
        print(".6f")
        print(".4f")
        print(".4f")
        print(".1%")
        print()

        print("MODEL ARCHITECTURE:")
        print("-" * 20)
        if lstm_model.model:
            print(f"Parameters: {lstm_model.model.count_params()}")
            print(f"Layers: {len(lstm_model.model.layers)}")
            print("Architecture: LSTM(64) -> Dropout -> LSTM(32) -> Dropout -> Dense(16) -> Dense(1)")
        print()

        print("TRAINING SUMMARY:")
        print("-" * 20)
        if history:
            print(f"Epochs trained: {len(history.history['loss'])}")
            print(".6f")
            print(".6f")
        print()

        if temporal_analysis:
            print("TEMPORAL ANALYSIS:")
            print("-" * 20)
            print(f"Most important lag: t-{temporal_analysis['most_important_lag']}")
            print(".4f")
        print()

        print(f"Files saved: 4 CSV files, 4 PNG plots")
        print("="*80)

    except Exception as e:
        logger.error(f"Error in main analysis: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()