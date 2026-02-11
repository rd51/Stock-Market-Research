"""
Advanced LSTM Architectures Module
===================================

Sophisticated LSTM variants for financial time series analysis.
Includes bidirectional LSTMs, attention mechanisms, ensembles, and multi-task learning.

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
    from tensorflow.keras import layers, models, optimizers, callbacks, Model
    from tensorflow.keras.models import Sequential
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("TensorFlow/Keras not available - Advanced LSTM modeling disabled")
    # Provide fallback functions
    def mean_squared_error(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def mean_absolute_error(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    def r2_score(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    def accuracy_score(y_true, y_pred):
        return np.mean(y_true == y_pred)

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


class BidirectionalLSTM:
    """
    Bidirectional LSTM for pattern analysis in historical data.
    Sees both past and future context (not suitable for forecasting).
    """

    def __init__(self, lookback: int = 30, feature_count: int = 8):
        """
        Initialize Bidirectional LSTM.

        Args:
            lookback (int): Number of time steps to look back
            feature_count (int): Number of input features
        """
        self.lookback = lookback
        self.feature_count = feature_count
        self.model = None

        logger.info(f"Initialized BidirectionalLSTM with lookback={lookback}, features={feature_count}")

    def build_model(self) -> Any:
        """
        Build bidirectional LSTM architecture.

        Returns:
            keras.Sequential: Compiled bidirectional LSTM model
        """
        logger.info("Building bidirectional LSTM model")

        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available - cannot build model")
            return None

        try:
            model = Sequential([
                layers.Bidirectional(layers.LSTM(64, return_sequences=True),
                                   input_shape=(self.lookback, self.feature_count)),
                layers.Bidirectional(layers.LSTM(32)),
                layers.Dense(16, activation='relu'),
                layers.Dense(1, activation='linear')
            ])

            model.compile(
                optimizer=optimizers.Adam(learning_rate=0.001),
                loss='mean_squared_error',
                metrics=['mae']
            )

            self.model = model
            logger.info(f"Bidirectional LSTM built. Parameters: {model.count_params()}")
            return model

        except Exception as e:
            logger.error(f"Error building bidirectional LSTM: {str(e)}")
            return None

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 100, batch_size: int = 32) -> Any:
        """
        Train the bidirectional LSTM model.

        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            epochs, batch_size: Training parameters

        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()

        if self.model is None:
            return None

        try:
            early_stopping = callbacks.EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True, verbose=1
            )

            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs, batch_size=batch_size,
                callbacks=[early_stopping], verbose=1
            )

            logger.info("Bidirectional LSTM training completed")
            return history

        except Exception as e:
            logger.error(f"Error training bidirectional LSTM: {str(e)}")
            return None

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            return np.zeros(len(X))
        return self.model.predict(X, verbose=0).flatten()


class AttentionLSTM:
    """
    LSTM with attention mechanism to understand which time steps matter most.
    """

    def __init__(self, lookback: int = 30, feature_count: int = 8):
        """
        Initialize Attention LSTM.

        Args:
            lookback (int): Number of time steps to look back
            feature_count (int): Number of input features
        """
        self.lookback = lookback
        self.feature_count = feature_count
        self.model = None
        self.attention_model = None

        logger.info(f"Initialized AttentionLSTM with lookback={lookback}, features={feature_count}")

    def build_model(self) -> Any:
        """
        Build LSTM with attention mechanism.

        Returns:
            keras.Model: LSTM with attention
        """
        logger.info("Building attention LSTM model")

        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available - cannot build model")
            return None

        try:
            # Input
            inputs = layers.Input(shape=(self.lookback, self.feature_count))

            # LSTM layers
            lstm_out = layers.LSTM(64, return_sequences=True)(inputs)
            lstm_out = layers.LSTM(32, return_sequences=True)(lstm_out)

            # Attention mechanism (simplified version)
            attention = layers.Dense(1, activation='tanh')(lstm_out)
            attention = layers.Flatten()(attention)
            attention_weights = layers.Activation('softmax')(attention)
            attention_weights = layers.RepeatVector(32)(attention_weights)
            attention_weights = layers.Permute([2, 1])(attention_weights)

            # Apply attention
            attended = layers.Multiply()([lstm_out, attention_weights])
            attended = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(attended)

            # Output
            outputs = layers.Dense(16, activation='relu')(attended)
            outputs = layers.Dense(1, activation='linear')(outputs)

            model = Model(inputs=inputs, outputs=outputs)

            # Attention weights model for visualization
            attention_model = Model(inputs=inputs, outputs=attention_weights)

            model.compile(
                optimizer=optimizers.Adam(learning_rate=0.001),
                loss='mean_squared_error',
                metrics=['mae']
            )

            self.model = model
            self.attention_model = attention_model
            logger.info(f"Attention LSTM built. Parameters: {model.count_params()}")
            return model

        except Exception as e:
            logger.error(f"Error building attention LSTM: {str(e)}")
            return None

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 100, batch_size: int = 32) -> Any:
        """
        Train the attention LSTM model.

        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            epochs, batch_size: Training parameters

        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()

        if self.model is None:
            return None

        try:
            early_stopping = callbacks.EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True, verbose=1
            )

            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs, batch_size=batch_size,
                callbacks=[early_stopping], verbose=1
            )

            logger.info("Attention LSTM training completed")
            return history

        except Exception as e:
            logger.error(f"Error training attention LSTM: {str(e)}")
            return None

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions and return attention weights.

        Returns:
            Tuple[predictions, attention_weights]
        """
        if self.model is None:
            return np.zeros(len(X)), np.zeros((len(X), self.lookback))

        predictions = self.model.predict(X, verbose=0).flatten()

        if self.attention_model is not None:
            attention_weights = self.attention_model.predict(X, verbose=0)
            # Average attention across features
            attention_weights = np.mean(attention_weights, axis=-1)
        else:
            attention_weights = np.zeros((len(X), self.lookback))

        return predictions, attention_weights


class LSTMEnsemble:
    """
    Ensemble of LSTM models with different random initializations.
    Reduces overfitting and provides uncertainty estimates.
    """

    def __init__(self, lookback: int = 30, feature_count: int = 8, n_models: int = 5):
        """
        Initialize LSTM ensemble.

        Args:
            lookback (int): Number of time steps to look back
            feature_count (int): Number of input features
            n_models (int): Number of models in ensemble
        """
        self.lookback = lookback
        self.feature_count = feature_count
        self.n_models = n_models
        self.models = []

        logger.info(f"Initialized LSTMEnsemble with {n_models} models")

    def build_ensemble(self, X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray, y_val: np.ndarray) -> None:
        """
        Build and train ensemble of LSTM models.

        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
        """
        logger.info(f"Building ensemble of {self.n_models} LSTM models")

        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available - cannot build ensemble")
            return

        self.models = []

        for i in range(self.n_models):
            logger.info(f"Training model {i+1}/{self.n_models}")

            try:
                # Different random seed for each model
                tf.random.set_seed(i * 42)

                model = Sequential([
                    layers.LSTM(64, return_sequences=True, input_shape=(self.lookback, self.feature_count)),
                    layers.Dropout(0.2),
                    layers.LSTM(32, return_sequences=False),
                    layers.Dropout(0.2),
                    layers.Dense(16, activation='relu'),
                    layers.Dense(1, activation='linear')
                ])

                model.compile(
                    optimizer=optimizers.Adam(learning_rate=0.001),
                    loss='mean_squared_error',
                    metrics=['mae']
                )

                early_stopping = callbacks.EarlyStopping(
                    monitor='val_loss', patience=10, restore_best_weights=True, verbose=0
                )

                model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=50, batch_size=32,
                    callbacks=[early_stopping], verbose=0
                )

                self.models.append(model)

            except Exception as e:
                logger.warning(f"Failed to train model {i+1}: {str(e)}")

        logger.info(f"Ensemble built with {len(self.models)} models")

    def predict_ensemble(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make ensemble predictions with uncertainty.

        Args:
            X_test (np.ndarray): Test sequences

        Returns:
            Tuple[mean_predictions, std_predictions]
        """
        if not self.models:
            logger.warning("No models in ensemble")
            return np.zeros(len(X_test)), np.zeros(len(X_test))

        try:
            predictions = []

            for model in self.models:
                pred = model.predict(X_test, verbose=0).flatten()
                predictions.append(pred)

            predictions = np.array(predictions)  # Shape: (n_models, n_samples)

            mean_pred = np.mean(predictions, axis=0)
            std_pred = np.std(predictions, axis=0)

            logger.info(f"Ensemble prediction completed. Mean std: {np.mean(std_pred):.6f}")
            return mean_pred, std_pred

        except Exception as e:
            logger.error(f"Error in ensemble prediction: {str(e)}")
            return np.zeros(len(X_test)), np.zeros(len(X_test))

    def predict_with_variance_decomposition(self, X_test: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        """
        Decompose prediction error into bias and variance components.

        Args:
            X_test (np.ndarray): Test sequences
            y_true (np.ndarray): True values

        Returns:
            Dict with bias, variance, and total error estimates
        """
        if not self.models:
            return {'bias_estimate': 0.0, 'variance_estimate': 0.0, 'total_error': 0.0}

        try:
            predictions = []

            for model in self.models:
                pred = model.predict(X_test, verbose=0).flatten()
                predictions.append(pred)

            predictions = np.array(predictions)  # Shape: (n_models, n_samples)
            mean_pred = np.mean(predictions, axis=0)

            # Bias: squared difference between ensemble mean and true values
            bias = np.mean((mean_pred - y_true) ** 2)

            # Variance: average variance of individual model predictions
            variance = np.mean(np.var(predictions, axis=0))

            # Total error: bias + variance (simplified bias-variance decomposition)
            total_error = bias + variance

            result = {
                'bias_estimate': bias,
                'variance_estimate': variance,
                'total_error': total_error
            }

            logger.info(f"Variance decomposition: Bias={bias:.6f}, Variance={variance:.6f}, Total={total_error:.6f}")
            return result

        except Exception as e:
            logger.error(f"Error in variance decomposition: {str(e)}")
            return {'bias_estimate': 0.0, 'variance_estimate': 0.0, 'total_error': 0.0}


class MultiTaskLSTM:
    """
    Multi-task LSTM that predicts multiple targets simultaneously.
    Shares representations to improve generalization.
    """

    def __init__(self, lookback: int = 30, feature_count: int = 8):
        """
        Initialize Multi-Task LSTM.

        Args:
            lookback (int): Number of time steps to look back
            feature_count (int): Number of input features
        """
        self.lookback = lookback
        self.feature_count = feature_count
        self.model = None

        logger.info(f"Initialized MultiTaskLSTM with lookback={lookback}, features={feature_count}")

    def build_model(self) -> Any:
        """
        Build multi-task LSTM with shared layers and task-specific outputs.

        Returns:
            keras.Model: Multi-task LSTM model
        """
        logger.info("Building multi-task LSTM model")

        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available - cannot build model")
            return None

        try:
            # Input
            inputs = layers.Input(shape=(self.lookback, self.feature_count))

            # Shared LSTM layers
            shared = layers.LSTM(64, return_sequences=True)(inputs)
            shared = layers.Dropout(0.2)(shared)
            shared = layers.LSTM(32)(shared)
            shared = layers.Dropout(0.2)(shared)

            # Task 1: Return prediction (regression)
            return_out = layers.Dense(16, activation='relu')(shared)
            return_pred = layers.Dense(1, activation='linear', name='return_output')(return_out)

            # Task 2: Direction prediction (binary classification)
            direction_out = layers.Dense(16, activation='relu')(shared)
            direction_pred = layers.Dense(1, activation='sigmoid', name='direction_output')(direction_out)

            # Task 3: Regime prediction (multi-class classification, assuming 3 regimes)
            regime_out = layers.Dense(16, activation='relu')(shared)
            regime_pred = layers.Dense(3, activation='softmax', name='regime_output')(regime_out)

            model = Model(
                inputs=inputs,
                outputs=[return_pred, direction_pred, regime_pred]
            )

            # Multi-task loss
            losses = {
                'return_output': 'mean_squared_error',
                'direction_output': 'binary_crossentropy',
                'regime_output': 'categorical_crossentropy'
            }

            loss_weights = {
                'return_output': 0.5,
                'direction_output': 0.3,
                'regime_output': 0.2
            }

            model.compile(
                optimizer=optimizers.Adam(learning_rate=0.001),
                loss=losses,
                loss_weights=loss_weights,
                metrics={
                    'return_output': ['mae'],
                    'direction_output': ['accuracy'],
                    'regime_output': ['accuracy']
                }
            )

            self.model = model
            logger.info(f"Multi-task LSTM built. Parameters: {model.count_params()}")
            return model

        except Exception as e:
            logger.error(f"Error building multi-task LSTM: {str(e)}")
            return None

    def train(self, X_train: np.ndarray, y_train_return: np.ndarray,
              y_train_direction: np.ndarray, y_train_regime: np.ndarray,
              X_val: np.ndarray, y_val_return: np.ndarray,
              y_val_direction: np.ndarray, y_val_regime: np.ndarray,
              epochs: int = 100, batch_size: int = 32) -> Any:
        """
        Train the multi-task LSTM model.

        Args:
            X_train: Training sequences
            y_train_*: Training targets for each task
            X_val, y_val_*: Validation data
            epochs, batch_size: Training parameters

        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()

        if self.model is None:
            return None

        try:
            early_stopping = callbacks.EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True, verbose=1
            )

            history = self.model.fit(
                X_train,
                {
                    'return_output': y_train_return,
                    'direction_output': y_train_direction,
                    'regime_output': y_train_regime
                },
                validation_data=(
                    X_val,
                    {
                        'return_output': y_val_return,
                        'direction_output': y_val_direction,
                        'regime_output': y_val_regime
                    }
                ),
                epochs=epochs, batch_size=batch_size,
                callbacks=[early_stopping], verbose=1
            )

            logger.info("Multi-task LSTM training completed")
            return history

        except Exception as e:
            logger.error(f"Error training multi-task LSTM: {str(e)}")
            return None

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Make multi-task predictions.

        Returns:
            Tuple[return_pred, direction_pred, regime_pred]
        """
        if self.model is None:
            n_samples = len(X)
            return (np.zeros(n_samples), np.zeros(n_samples), np.zeros((n_samples, 3)))

        predictions = self.model.predict(X, verbose=0)
        return predictions[0].flatten(), predictions[1].flatten(), predictions[2]


def create_train_val_test_sequences(data: np.ndarray, lookback: int,
                                   train_ratio: float = 0.7, val_ratio: float = 0.15) -> Tuple:
    """
    Create train/validation/test sequences maintaining temporal order.
    """
    n_samples = len(data) - lookback
    if n_samples <= 0:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))

    X_all, y_all = [], []
    for i in range(n_samples):
        X_seq = data[i:i+lookback]
        y_val = data[i+lookback, -1]  # Assume last column is target
        X_all.append(X_seq)
        y_all.append(y_val)

    X_all = np.array(X_all)
    y_all = np.array(y_all)

    X_train = X_all[:train_end]
    y_train = y_all[:train_end]
    X_val = X_all[train_end:val_end]
    y_val = y_all[train_end:val_end]
    X_test = X_all[val_end:]
    y_test = y_all[val_end:]

    return X_train, y_train, X_val, y_val, X_test, y_test


def visualize_attention_weights(attention_weights: np.ndarray, dates: Optional[pd.Series] = None) -> Any:
    """
    Visualize attention weights as a heatmap.

    Args:
        attention_weights (np.ndarray): Attention weights (n_samples, lookback)
        dates (pd.Series, optional): Date index

    Returns:
        matplotlib figure
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Matplotlib not available")
        return None

    logger.info("Creating attention weights visualization")

    try:
        fig, ax = plt.subplots(figsize=(12, 8))

        # Average attention across samples for cleaner visualization
        avg_attention = np.mean(attention_weights, axis=0)

        # Create time lag labels
        lookback = len(avg_attention)
        lags = [f't-{i}' for i in range(lookback, 0, -1)]

        bars = ax.bar(range(len(avg_attention)), avg_attention, alpha=0.7, color='skyblue')
        ax.set_xlabel('Time Lag')
        ax.set_ylabel('Average Attention Weight')
        ax.set_title('Attention Weights: Importance of Each Time Lag')
        ax.set_xticks(range(0, len(avg_attention), max(1, len(avg_attention)//10)))
        ax.set_xticklabels([lags[i] for i in range(0, len(avg_attention), max(1, len(avg_attention)//10))])
        ax.grid(True, alpha=0.3)

        # Highlight most attended lag
        max_idx = np.argmax(avg_attention)
        bars[max_idx].set_color('red')
        bars[max_idx].set_alpha(0.9)

        plt.tight_layout()
        logger.info("Attention weights plot created")
        return fig

    except Exception as e:
        logger.error(f"Error creating attention plot: {str(e)}")
        return None


def compare_lstm_architectures(X_train: np.ndarray, y_train: np.ndarray,
                              X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
    """
    Compare different LSTM architectures on the same data.

    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data

    Returns:
        Dict with performance comparison
    """
    logger.info("Comparing LSTM architectures")

    results = {}
    lookback = X_train.shape[1]
    features = X_train.shape[2]

    # Create validation split from training data
    val_split = int(0.8 * len(X_train))
    X_train_split = X_train[:val_split]
    y_train_split = y_train[:val_split]
    X_val_split = X_train[val_split:]
    y_val_split = y_train[val_split:]

    architectures = {
        'Basic LSTM': lambda: BasicLSTM(lookback, features),
        'Bidirectional LSTM': lambda: BidirectionalLSTM(lookback, features),
        'Attention LSTM': lambda: AttentionLSTM(lookback, features),
        'LSTM Ensemble': lambda: LSTMEnsemble(lookback, features, n_models=3)
    }

    for name, arch_func in architectures.items():
        logger.info(f"Training {name}")

        try:
            model = arch_func()

            if name == 'LSTM Ensemble':
                model.build_ensemble(X_train_split, y_train_split, X_val_split, y_val_split)
                if model.models:
                    pred_mean, pred_std = model.predict_ensemble(X_test)
                    predictions = pred_mean
                else:
                    predictions = np.zeros(len(X_test))
            else:
                model.train(X_train_split, y_train_split, X_val_split, y_val_split, epochs=30)
                if hasattr(model, 'predict'):
                    if name == 'Attention LSTM':
                        predictions, _ = model.predict(X_test)
                    else:
                        predictions = model.predict(X_test)
                else:
                    predictions = np.zeros(len(X_test))

            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)

            # Direction accuracy
            actual_direction = np.sign(np.diff(y_test))
            pred_direction = np.sign(np.diff(predictions))
            valid_idx = (actual_direction != 0) & (pred_direction != 0)
            direction_acc = np.mean(actual_direction[valid_idx] == pred_direction[valid_idx]) if np.sum(valid_idx) > 0 else 0

            results[name] = {
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'direction_accuracy': direction_acc
            }

            logger.info(f"{name} - RMSE: {rmse:.6f}, R²: {r2:.4f}, Direction Acc: {direction_acc:.4f}")

        except Exception as e:
            logger.error(f"Failed to train {name}: {str(e)}")
            results[name] = {'rmse': float('inf'), 'mae': float('inf'), 'r2': 0.0, 'direction_accuracy': 0.0}

    return results


# Import BasicLSTM for comparison
try:
    from lstm_models import LSTMPredictor as BasicLSTM
except ImportError:
    class BasicLSTM:
        def __init__(self, lookback, features):
            self.lookback = lookback
            self.feature_count = features
            self.model = None

        def train(self, X_train, y_train, X_val, y_val, epochs=30):
            pass

        def predict(self, X):
            return np.zeros(len(X))


def plot_architecture_comparison(results: Dict[str, Any]) -> Any:
    """
    Plot comparison of different LSTM architectures.

    Args:
        results (Dict): Results from compare_lstm_architectures

    Returns:
        matplotlib figure
    """
    if not MATPLOTLIB_AVAILABLE or not results:
        return None

    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        architectures = list(results.keys())
        rmse_vals = [results[arch]['rmse'] for arch in architectures]
        mae_vals = [results[arch]['mae'] for arch in architectures]
        r2_vals = [results[arch]['r2'] for arch in architectures]
        dir_acc_vals = [results[arch]['direction_accuracy'] for arch in architectures]

        # RMSE
        bars1 = ax1.bar(architectures, rmse_vals, alpha=0.7, color='skyblue')
        ax1.set_title('RMSE Comparison')
        ax1.set_ylabel('RMSE (lower is better)')
        ax1.tick_params(axis='x', rotation=45)

        # MAE
        bars2 = ax2.bar(architectures, mae_vals, alpha=0.7, color='lightgreen')
        ax2.set_title('MAE Comparison')
        ax2.set_ylabel('MAE (lower is better)')
        ax2.tick_params(axis='x', rotation=45)

        # R²
        bars3 = ax3.bar(architectures, r2_vals, alpha=0.7, color='salmon')
        ax3.set_title('R² Comparison')
        ax3.set_ylabel('R² (higher is better)')
        ax3.tick_params(axis='x', rotation=45)

        # Direction Accuracy
        bars4 = ax4.bar(architectures, dir_acc_vals, alpha=0.7, color='orange')
        ax4.set_title('Direction Accuracy')
        ax4.set_ylabel('Accuracy (higher is better)')
        ax4.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        return fig

    except Exception as e:
        logger.error(f"Error creating comparison plot: {str(e)}")
        return None


def main():
    """
    Main function demonstrating advanced LSTM architectures.
    """
    logger.info("Starting advanced LSTM architectures demonstration")

    try:
        # Load or generate data
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
            np.random.seed(42)
            dates = pd.date_range('2020-01-01', periods=1000, freq='D')

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

        # Prepare data
        logger.info("Preparing data for advanced LSTM modeling")

        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data)

        lookback = 30
        X_train, y_train, X_val, y_val, X_test, y_test = create_train_val_test_sequences(
            data_scaled, lookback, train_ratio=0.7, val_ratio=0.15
        )

        if len(X_train) == 0:
            logger.error("Failed to create sequences")
            return

        # 1. Compare architectures
        logger.info("Comparing LSTM architectures")
        comparison_results = compare_lstm_architectures(X_train, y_train, X_test, y_test)

        # 2. Train Bidirectional LSTM
        logger.info("Training Bidirectional LSTM")
        bi_lstm = BidirectionalLSTM(lookback, data.shape[1])
        bi_history = bi_lstm.train(X_train, y_train, X_val, y_val, epochs=30)

        # 3. Train Attention LSTM
        logger.info("Training Attention LSTM")
        att_lstm = AttentionLSTM(lookback, data.shape[1])
        att_history = att_lstm.train(X_train, y_train, X_val, y_val, epochs=30)

        if att_history:
            att_predictions, attention_weights = att_lstm.predict(X_test[:50])  # Subset for demo

        # 4. Train Ensemble
        logger.info("Training LSTM Ensemble")
        ensemble = LSTMEnsemble(lookback, data.shape[1], n_models=3)
        ensemble.build_ensemble(X_train, y_train, X_val, y_val)

        ensemble_pred, ensemble_std = ensemble.predict_ensemble(X_test)
        variance_decomp = ensemble.predict_with_variance_decomposition(X_test, y_test)

        # 5. Multi-task LSTM (simplified - using synthetic targets)
        logger.info("Training Multi-Task LSTM")
        mt_lstm = MultiTaskLSTM(lookback, data.shape[1])

        # Create synthetic targets for multi-task learning
        y_direction = (y_train > np.median(y_train)).astype(int)
        y_val_direction = (y_val > np.median(y_val)).astype(int)
        y_regime = np.random.randint(0, 3, size=len(y_train))  # Random regimes for demo
        y_val_regime = np.random.randint(0, 3, size=len(y_val))
        y_regime_oh = tf.keras.utils.to_categorical(y_regime, 3) if TENSORFLOW_AVAILABLE else np.zeros((len(y_train), 3))
        y_val_regime_oh = tf.keras.utils.to_categorical(y_val_regime, 3) if TENSORFLOW_AVAILABLE else np.zeros((len(y_val), 3))

        mt_history = mt_lstm.train(
            X_train, y_train, y_direction, y_regime_oh,
            X_val, y_val, y_val_direction, y_val_regime_oh,
            epochs=30
        )

        # Create visualizations
        if MATPLOTLIB_AVAILABLE:
            logger.info("Creating visualizations")

            # Architecture comparison
            comp_fig = plot_architecture_comparison(comparison_results)

            # Attention weights
            if 'attention_weights' in locals():
                att_fig = visualize_attention_weights(attention_weights)

        # Save results
        logger.info("Saving results")

        # Save comparison results
        comparison_df = pd.DataFrame(comparison_results).T
        comparison_df.to_csv('lstm_advanced_comparison.csv')

        # Save ensemble results
        ensemble_df = pd.DataFrame({
            'actual': y_test,
            'ensemble_pred': ensemble_pred,
            'ensemble_std': ensemble_std
        })
        ensemble_df.to_csv('lstm_ensemble_predictions.csv')

        # Save variance decomposition
        variance_df = pd.DataFrame([variance_decomp])
        variance_df.to_csv('lstm_variance_decomposition.csv')

        # Save plots
        if MATPLOTLIB_AVAILABLE:
            if 'comp_fig' in locals() and comp_fig:
                comp_fig.savefig('lstm_architecture_comparison.png', dpi=300, bbox_inches='tight')
            if 'att_fig' in locals() and att_fig:
                att_fig.savefig('lstm_attention_weights.png', dpi=300, bbox_inches='tight')

        # Print summary
        print("\n" + "="*80)
        print("ADVANCED LSTM ARCHITECTURES COMPLETED")
        print("="*80)
        print(f"Data: {len(data)} total samples")
        print(f"Sequences: lookback={lookback}, features={data.shape[1]}")
        print(f"Train/Val/Test split: {len(X_train)}/{len(X_val)}/{len(X_test)}")
        print()

        print("ARCHITECTURE COMPARISON:")
        print("-" * 30)
        for arch, metrics in comparison_results.items():
            print(f"{arch}:")
            print(".6f")
            print(".4f")
            print(".1%")
            print()

        if variance_decomp:
            print("ENSEMBLE VARIANCE DECOMPOSITION:")
            print("-" * 35)
            print(".6f")
            print(".6f")
            print(".6f")
            print()

        print("MODELS TRAINED:")
        print("-" * 15)
        print("* Bidirectional LSTM")
        print("* Attention LSTM")
        print("* LSTM Ensemble (3 models)")
        print("* Multi-Task LSTM")
        print()

        print(f"Files saved: 4 CSV files, 2 PNG plots")
        print("="*80)

    except Exception as e:
        logger.error(f"Error in main analysis: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()