"""
Ensemble Models Module
======================

Tree-based ensemble models for financial time series forecasting.
Includes Random Forest, XGBoost, LightGBM with advanced features.

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

# Required libraries for ensemble models
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
    from sklearn.model_selection import cross_val_score, KFold
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available - Random Forest disabled")
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
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available - XGBoost models disabled")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logging.warning("LightGBM not available - LightGBM models disabled")

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logging.warning("Optuna not available - hyperparameter optimization disabled")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available - SHAP explanations disabled")

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


class RandomForestPredictor:
    """
    Random Forest ensemble for regression tasks.
    Builds multiple decision trees on bootstrap samples.
    """

    def __init__(self, n_estimators: int = 100, max_depth: int = 10,
                 min_samples_split: int = 5, random_state: int = 42):
        """
        Initialize Random Forest predictor.

        Args:
            n_estimators (int): Number of trees in the forest
            max_depth (int): Maximum depth of each tree
            min_samples_split (int): Minimum samples required to split a node
            random_state (int): Random seed for reproducibility
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state

        if SKLEARN_AVAILABLE:
            self.model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=random_state,
                n_jobs=-1  # Use all available cores
            )
        else:
            self.model = None

        logger.info(f"Initialized RandomForestPredictor with {n_estimators} trees")

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> 'RandomForestPredictor':
        """
        Fit the Random Forest model.

        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training targets

        Returns:
            RandomForestPredictor: Fitted model
        """
        if self.model is None:
            logger.warning("scikit-learn not available - cannot fit model")
            return self

        try:
            logger.info(f"Fitting Random Forest with {self.n_estimators} trees")
            self.model.fit(X_train, y_train)
            logger.info("Random Forest fitted successfully")
            return self

        except Exception as e:
            logger.error(f"Error fitting Random Forest: {str(e)}")
            return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Make predictions on test data.

        Args:
            X_test (np.ndarray): Test features

        Returns:
            np.ndarray: Predictions
        """
        if self.model is None:
            logger.warning("Model not fitted - returning zeros")
            return np.zeros(len(X_test))

        try:
            predictions = self.model.predict(X_test)
            return predictions

        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            return np.zeros(len(X_test))

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance on test data.

        Args:
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test targets

        Returns:
            Dict[str, float]: Evaluation metrics
        """
        try:
            predictions = self.predict(X_test)

            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            mae = mean_absolute_error(y_test, predictions)
            mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
            r2 = r2_score(y_test, predictions)

            # Direction accuracy
            actual_direction = np.sign(np.diff(y_test))
            pred_direction = np.sign(np.diff(predictions))
            valid_idx = (actual_direction != 0) & (pred_direction != 0)
            direction_acc = np.mean(actual_direction[valid_idx] == pred_direction[valid_idx]) if np.sum(valid_idx) > 0 else 0

            results = {
                'rmse': rmse,
                'mae': mae,
                'mape': mape,
                'r2': r2,
                'direction_accuracy': direction_acc
            }

            logger.info(f"Random Forest evaluation - RMSE: {rmse:.6f}, R²: {r2:.4f}")
            return results

        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            return {
                'rmse': float('inf'),
                'mae': float('inf'),
                'mape': float('inf'),
                'r2': 0.0,
                'direction_accuracy': 0.0
            }

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance scores.

        Returns:
            pd.DataFrame: Feature importance DataFrame
        """
        if self.model is None or not hasattr(self.model, 'feature_importances_'):
            logger.warning("Model not fitted or no feature importances available")
            return pd.DataFrame()

        try:
            importance_scores = self.model.feature_importances_
            feature_names = [f'feature_{i}' for i in range(len(importance_scores))]

            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance_scores
            }).sort_values('importance', ascending=False).reset_index(drop=True)

            logger.info(f"Feature importance calculated for {len(importance_scores)} features")
            return importance_df

        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            return pd.DataFrame()

    def plot_feature_importance(self, top_n: int = 15) -> Any:
        """
        Plot feature importance as a bar chart.

        Args:
            top_n (int): Number of top features to show

        Returns:
            matplotlib figure
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available")
            return None

        importance_df = self.get_feature_importance()
        if importance_df.empty:
            return None

        try:
            fig, ax = plt.subplots(figsize=(12, 8))

            # Take top N features
            plot_data = importance_df.head(top_n)

            bars = ax.barh(range(len(plot_data)), plot_data['importance'][::-1])
            ax.set_yticks(range(len(plot_data)))
            ax.set_yticklabels(plot_data['feature'][::-1])
            ax.set_xlabel('Feature Importance')
            ax.set_title(f'Random Forest Feature Importance (Top {top_n})')
            ax.grid(True, alpha=0.3)

            # Add value labels
            for i, (idx, row) in enumerate(plot_data[::-1].iterrows()):
                ax.text(row['importance'] + 0.001, i, '.4f', va='center')

            plt.tight_layout()
            logger.info("Feature importance plot created")
            return fig

        except Exception as e:
            logger.error(f"Error creating feature importance plot: {str(e)}")
            return None


class XGBoostPredictor:
    """
    XGBoost ensemble for regression tasks.
    Gradient boosting with advanced features and SHAP explanations.
    """

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.01,
                 max_depth: int = 5, subsample: float = 0.8):
        """
        Initialize XGBoost predictor.

        Args:
            n_estimators (int): Number of boosting rounds
            learning_rate (float): Learning rate (eta)
            max_depth (int): Maximum tree depth
            subsample (float): Subsample ratio of training instances
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample

        if XGBOOST_AVAILABLE:
            self.model = xgb.XGBRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                subsample=subsample,
                random_state=42,
                n_jobs=-1
            )
        else:
            self.model = None

        logger.info(f"Initialized XGBoostPredictor with {n_estimators} estimators")

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None,
            early_stopping_rounds: int = 10) -> 'XGBoostPredictor':
        """
        Fit the XGBoost model with optional early stopping.

        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data for early stopping
            early_stopping_rounds: Rounds to wait before stopping

        Returns:
            XGBoostPredictor: Fitted model
        """
        if self.model is None:
            logger.warning("XGBoost not available - cannot fit model")
            return self

        try:
            logger.info("Fitting XGBoost model")

            if X_val is not None and y_val is not None:
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=early_stopping_rounds,
                    verbose=False
                )
            else:
                self.model.fit(X_train, y_train, verbose=False)

            logger.info("XGBoost fitted successfully")
            return self

        except Exception as e:
            logger.error(f"Error fitting XGBoost: {str(e)}")
            return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Make predictions on test data.

        Args:
            X_test (np.ndarray): Test features

        Returns:
            np.ndarray: Predictions
        """
        if self.model is None:
            logger.warning("Model not fitted - returning zeros")
            return np.zeros(len(X_test))

        try:
            predictions = self.model.predict(X_test)
            return predictions

        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            return np.zeros(len(X_test))

    def predict_with_shap(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions and compute SHAP values for explanations.

        Args:
            X_test (np.ndarray): Test features

        Returns:
            Tuple[predictions, shap_values]
        """
        predictions = self.predict(X_test)

        if not SHAP_AVAILABLE or self.model is None:
            logger.warning("SHAP not available - returning zeros for SHAP values")
            return predictions, np.zeros_like(X_test)

        try:
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X_test)
            return predictions, shap_values

        except Exception as e:
            logger.error(f"Error computing SHAP values: {str(e)}")
            return predictions, np.zeros_like(X_test)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance on test data.

        Args:
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test targets

        Returns:
            Dict[str, float]: Evaluation metrics
        """
        try:
            predictions = self.predict(X_test)

            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            mae = mean_absolute_error(y_test, predictions)
            mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
            r2 = r2_score(y_test, predictions)

            # Direction accuracy
            actual_direction = np.sign(np.diff(y_test))
            pred_direction = np.sign(np.diff(predictions))
            valid_idx = (actual_direction != 0) & (pred_direction != 0)
            direction_acc = np.mean(actual_direction[valid_idx] == pred_direction[valid_idx]) if np.sum(valid_idx) > 0 else 0

            results = {
                'rmse': rmse,
                'mae': mae,
                'mape': mape,
                'r2': r2,
                'direction_accuracy': direction_acc
            }

            logger.info(f"XGBoost evaluation - RMSE: {rmse:.6f}, R²: {r2:.4f}")
            return results

        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            return {
                'rmse': float('inf'),
                'mae': float('inf'),
                'mape': float('inf'),
                'r2': 0.0,
                'direction_accuracy': 0.0
            }

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance scores with different types.

        Returns:
            pd.DataFrame: Feature importance DataFrame with weight, gain, cover
        """
        if self.model is None:
            logger.warning("Model not fitted")
            return pd.DataFrame()

        try:
            # Get different types of importance
            importance_types = ['weight', 'gain', 'cover']
            importance_data = {}

            for imp_type in importance_types:
                scores = self.model.get_booster().get_score(importance_type=imp_type)
                importance_data[imp_type] = scores

            # Create DataFrame
            all_features = set()
            for scores in importance_data.values():
                all_features.update(scores.keys())

            feature_names = sorted(list(all_features))
            importance_df = pd.DataFrame(index=feature_names)

            for imp_type in importance_types:
                scores = importance_data[imp_type]
                importance_df[imp_type] = [scores.get(feat, 0) for feat in feature_names]

            # Sort by gain (most informative)
            importance_df = importance_df.sort_values('gain', ascending=False).reset_index()
            importance_df.rename(columns={'index': 'feature'}, inplace=True)

            logger.info(f"Feature importance calculated for {len(feature_names)} features")
            return importance_df

        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            return pd.DataFrame()

    def plot_feature_importance(self, plot_type: str = 'weight', top_n: int = 15) -> Any:
        """
        Plot feature importance.

        Args:
            plot_type (str): Type of importance ('weight', 'gain', 'cover')
            top_n (int): Number of top features to show

        Returns:
            matplotlib figure
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available")
            return None

        importance_df = self.get_feature_importance()
        if importance_df.empty or plot_type not in importance_df.columns:
            return None

        try:
            fig, ax = plt.subplots(figsize=(12, 8))

            # Sort by the selected importance type
            plot_data = importance_df.sort_values(plot_type, ascending=False).head(top_n)

            bars = ax.barh(range(len(plot_data)), plot_data[plot_type][::-1])
            ax.set_yticks(range(len(plot_data)))
            ax.set_yticklabels(plot_data['feature'][::-1])
            ax.set_xlabel(f'Feature Importance ({plot_type})')
            ax.set_title(f'XGBoost Feature Importance ({plot_type}) - Top {top_n}')
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            logger.info(f"Feature importance plot created for {plot_type}")
            return fig

        except Exception as e:
            logger.error(f"Error creating feature importance plot: {str(e)}")
            return None

    def plot_shap_summary(self, X_sample: np.ndarray) -> Any:
        """
        Plot SHAP summary plot.

        Args:
            X_sample (np.ndarray): Sample of data for SHAP analysis

        Returns:
            matplotlib figure
        """
        if not SHAP_AVAILABLE or not MATPLOTLIB_AVAILABLE or self.model is None:
            logger.warning("SHAP or matplotlib not available")
            return None

        try:
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X_sample)

            fig, ax = plt.subplots(figsize=(10, 8))
            shap.summary_plot(shap_values, X_sample, show=False)
            plt.title('SHAP Feature Importance Summary')
            plt.tight_layout()

            logger.info("SHAP summary plot created")
            return fig

        except Exception as e:
            logger.error(f"Error creating SHAP summary plot: {str(e)}")
            return None

    def plot_shap_dependence(self, X_sample: np.ndarray, feature_name: str) -> Any:
        """
        Plot SHAP dependence plot for a specific feature.

        Args:
            X_sample (np.ndarray): Sample of data for SHAP analysis
            feature_name (str): Name of feature to plot

        Returns:
            matplotlib figure
        """
        if not SHAP_AVAILABLE or not MATPLOTLIB_AVAILABLE or self.model is None:
            logger.warning("SHAP or matplotlib not available")
            return None

        try:
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X_sample)

            # Find feature index
            if feature_name.startswith('feature_'):
                feature_idx = int(feature_name.split('_')[1])
            else:
                feature_idx = 0  # Default to first feature

            fig, ax = plt.subplots(figsize=(10, 6))
            shap.dependence_plot(feature_idx, shap_values, X_sample, show=False)
            plt.title(f'SHAP Dependence Plot: {feature_name}')
            plt.tight_layout()

            logger.info(f"SHAP dependence plot created for {feature_name}")
            return fig

        except Exception as e:
            logger.error(f"Error creating SHAP dependence plot: {str(e)}")
            return None


class LightGBMPredictor:
    """
    LightGBM ensemble for regression tasks.
    Faster and more memory-efficient than XGBoost.
    """

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.01,
                 max_depth: int = 5, subsample: float = 0.8):
        """
        Initialize LightGBM predictor.

        Args:
            n_estimators (int): Number of boosting rounds
            learning_rate (float): Learning rate
            max_depth (int): Maximum tree depth
            subsample (float): Subsample ratio
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample

        if LIGHTGBM_AVAILABLE:
            self.model = lgb.LGBMRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                subsample=subsample,
                random_state=42,
                n_jobs=-1
            )
        else:
            self.model = None

        logger.info(f"Initialized LightGBMPredictor with {n_estimators} estimators")

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None,
            early_stopping_rounds: int = 10) -> 'LightGBMPredictor':
        """
        Fit the LightGBM model with optional early stopping.

        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data for early stopping
            early_stopping_rounds: Rounds to wait before stopping

        Returns:
            LightGBMPredictor: Fitted model
        """
        if self.model is None:
            logger.warning("LightGBM not available - cannot fit model")
            return self

        try:
            logger.info("Fitting LightGBM model")

            if X_val is not None and y_val is not None:
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=early_stopping_rounds,
                    verbose=False
                )
            else:
                self.model.fit(X_train, y_train, verbose=False)

            logger.info("LightGBM fitted successfully")
            return self

        except Exception as e:
            logger.error(f"Error fitting LightGBM: {str(e)}")
            return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Make predictions on test data.

        Args:
            X_test (np.ndarray): Test features

        Returns:
            np.ndarray: Predictions
        """
        if self.model is None:
            logger.warning("Model not fitted - returning zeros")
            return np.zeros(len(X_test))

        try:
            predictions = self.model.predict(X_test)
            return predictions

        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            return np.zeros(len(X_test))

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance on test data.

        Args:
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test targets

        Returns:
            Dict[str, float]: Evaluation metrics
        """
        try:
            predictions = self.predict(X_test)

            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            mae = mean_absolute_error(y_test, predictions)
            mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
            r2 = r2_score(y_test, predictions)

            # Direction accuracy
            actual_direction = np.sign(np.diff(y_test))
            pred_direction = np.sign(np.diff(predictions))
            valid_idx = (actual_direction != 0) & (pred_direction != 0)
            direction_acc = np.mean(actual_direction[valid_idx] == pred_direction[valid_idx]) if np.sum(valid_idx) > 0 else 0

            results = {
                'rmse': rmse,
                'mae': mae,
                'mape': mape,
                'r2': r2,
                'direction_accuracy': direction_acc
            }

            logger.info(f"LightGBM evaluation - RMSE: {rmse:.6f}, R²: {r2:.4f}")
            return results

        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            return {
                'rmse': float('inf'),
                'mae': float('inf'),
                'mape': float('inf'),
                'r2': 0.0,
                'direction_accuracy': 0.0
            }

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance scores.

        Returns:
            pd.DataFrame: Feature importance DataFrame
        """
        if self.model is None or not hasattr(self.model, 'feature_importances_'):
            logger.warning("Model not fitted or no feature importances available")
            return pd.DataFrame()

        try:
            importance_scores = self.model.feature_importances_
            feature_names = [f'feature_{i}' for i in range(len(importance_scores))]

            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance_scores
            }).sort_values('importance', ascending=False).reset_index(drop=True)

            logger.info(f"Feature importance calculated for {len(importance_scores)} features")
            return importance_df

        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            return pd.DataFrame()

    def plot_feature_importance(self, top_n: int = 15) -> Any:
        """
        Plot feature importance as a bar chart.

        Args:
            top_n (int): Number of top features to show

        Returns:
            matplotlib figure
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available")
            return None

        importance_df = self.get_feature_importance()
        if importance_df.empty:
            return None

        try:
            fig, ax = plt.subplots(figsize=(12, 8))

            # Take top N features
            plot_data = importance_df.head(top_n)

            bars = ax.barh(range(len(plot_data)), plot_data['importance'][::-1])
            ax.set_yticks(range(len(plot_data)))
            ax.set_yticklabels(plot_data['feature'][::-1])
            ax.set_xlabel('Feature Importance')
            ax.set_title(f'LightGBM Feature Importance (Top {top_n})')
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            logger.info("Feature importance plot created")
            return fig

        except Exception as e:
            logger.error(f"Error creating feature importance plot: {str(e)}")
            return None


def hyperparameter_optimization(model_class: type, X_train: np.ndarray, y_train: np.ndarray,
                               X_val: np.ndarray, y_val: np.ndarray,
                               cv_folds: int = 5, n_trials: int = 100) -> Dict[str, Any]:
    """
    Perform hyperparameter optimization using Optuna.

    Args:
        model_class: The model class to optimize (XGBoostPredictor or LightGBMPredictor)
        X_train, y_train: Training data
        X_val, y_val: Validation data
        cv_folds: Number of CV folds
        n_trials: Number of optimization trials

    Returns:
        Dict with best parameters, best score, and study object
    """
    if not OPTUNA_AVAILABLE:
        logger.warning("Optuna not available - returning default parameters")
        return {
            'best_params': {'learning_rate': 0.01, 'max_depth': 5, 'subsample': 0.8},
            'best_score': float('inf'),
            'study': None
        }

    def objective(trial):
        # Define hyperparameter search space
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
        }

        try:
            # Create and fit model
            model = model_class(**params)
            model.fit(X_train, y_train, X_val, y_val, early_stopping_rounds=10)

            # Evaluate on validation set
            val_pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, val_pred))

            return rmse

        except Exception as e:
            logger.warning(f"Trial failed: {str(e)}")
            return float('inf')

    try:
        logger.info(f"Starting hyperparameter optimization with {n_trials} trials")

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_params
        best_score = study.best_value

        logger.info(f"Optimization completed. Best RMSE: {best_score:.6f}")

        return {
            'best_params': best_params,
            'best_score': best_score,
            'study': study
        }

    except Exception as e:
        logger.error(f"Error in hyperparameter optimization: {str(e)}")
        return {
            'best_params': {'learning_rate': 0.01, 'max_depth': 5, 'subsample': 0.8},
            'best_score': float('inf'),
            'study': None
        }


def cross_validation_scores(model: Any, X_train: np.ndarray, y_train: np.ndarray,
                          cv_folds: int = 5) -> Dict[str, float]:
    """
    Perform cross-validation and return scores.

    Args:
        model: Fitted model
        X_train, y_train: Training data
        cv_folds: Number of CV folds

    Returns:
        Dict with mean/std RMSE and fold scores
    """
    if not SKLEARN_AVAILABLE:
        logger.warning("scikit-learn not available - cannot perform CV")
        return {'mean_rmse': float('inf'), 'std_rmse': 0.0, 'fold_scores': []}

    try:
        logger.info(f"Performing {cv_folds}-fold cross-validation")

        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        rmse_scores = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

            # Fit model on fold
            model.fit(X_fold_train, y_fold_train)

            # Predict on validation fold
            pred = model.predict(X_fold_val)
            rmse = np.sqrt(mean_squared_error(y_fold_val, pred))
            rmse_scores.append(rmse)

            logger.info(f"Fold {fold+1} RMSE: {rmse:.6f}")

        mean_rmse = np.mean(rmse_scores)
        std_rmse = np.std(rmse_scores)

        logger.info(f"CV completed. Mean RMSE: {mean_rmse:.6f} (+/- {std_rmse:.6f})")

        return {
            'mean_rmse': mean_rmse,
            'std_rmse': std_rmse,
            'fold_scores': rmse_scores
        }

    except Exception as e:
        logger.error(f"Error in cross-validation: {str(e)}")
        return {'mean_rmse': float('inf'), 'std_rmse': 0.0, 'fold_scores': []}


class EnsembleComparison:
    """
    Compare multiple ensemble models and create ensemble predictions.
    """

    def __init__(self):
        self.models = {}
        self.model_names = ['RandomForest', 'XGBoost', 'LightGBM']

    def fit_all_models(self, X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """
        Fit all ensemble models.

        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data

        Returns:
            Dict of fitted models
        """
        logger.info("Fitting all ensemble models")

        models = {}

        # Random Forest
        if SKLEARN_AVAILABLE:
            rf = RandomForestPredictor()
            rf.fit(X_train, y_train)
            models['rf'] = rf
        else:
            models['rf'] = None

        # XGBoost
        if XGBOOST_AVAILABLE:
            xgb_model = XGBoostPredictor()
            xgb_model.fit(X_train, y_train, X_val, y_val)
            models['xgb'] = xgb_model
        else:
            models['xgb'] = None

        # LightGBM
        if LIGHTGBM_AVAILABLE:
            lgb_model = LightGBMPredictor()
            lgb_model.fit(X_train, y_train, X_val, y_val)
            models['lgb'] = lgb_model
        else:
            models['lgb'] = None

        self.models = models
        logger.info(f"Fitted {len([m for m in models.values() if m is not None])} models")
        return models

    def compare_performance(self, X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
        """
        Compare performance of all models.

        Args:
            X_test, y_test: Test data

        Returns:
            pd.DataFrame: Performance comparison
        """
        logger.info("Comparing model performance")

        results = []

        for name, model in self.models.items():
            if model is None:
                continue

            try:
                metrics = model.evaluate(X_test, y_test)
                metrics['model'] = name
                results.append(metrics)

            except Exception as e:
                logger.warning(f"Failed to evaluate {name}: {str(e)}")

        comparison_df = pd.DataFrame(results)
        logger.info(f"Performance comparison completed for {len(results)} models")
        return comparison_df

    def plot_model_comparison(self, comparison_df: pd.DataFrame) -> Any:
        """
        Plot model comparison across metrics.

        Args:
            comparison_df: Performance comparison DataFrame

        Returns:
            matplotlib figure
        """
        if not MATPLOTLIB_AVAILABLE or comparison_df.empty:
            logger.warning("Matplotlib not available or no comparison data")
            return None

        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            axes = axes.flatten()

            metrics = ['rmse', 'mae', 'r2', 'direction_accuracy']
            metric_names = ['RMSE', 'MAE', 'R²', 'Direction Accuracy']

            for i, (metric, name) in enumerate(zip(metrics, metric_names)):
                if metric in comparison_df.columns:
                    ax = axes[i]
                    bars = ax.bar(comparison_df['model'], comparison_df[metric])
                    ax.set_title(f'{name} Comparison')
                    ax.set_ylabel(name)
                    ax.tick_params(axis='x', rotation=45)

                    # Add value labels
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               '.3f', ha='center', va='bottom', fontsize=8)

            # Hide empty subplots
            for i in range(len(metrics), len(axes)):
                axes[i].set_visible(False)

            plt.tight_layout()
            logger.info("Model comparison plot created")
            return fig

        except Exception as e:
            logger.error(f"Error creating comparison plot: {str(e)}")
            return None

    def predict_ensemble(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create ensemble predictions by averaging all models.

        Args:
            X_test (np.ndarray): Test features

        Returns:
            Tuple[ensemble_predictions, ensemble_std]
        """
        logger.info("Creating ensemble predictions")

        predictions = []

        for name, model in self.models.items():
            if model is not None:
                try:
                    pred = model.predict(X_test)
                    predictions.append(pred)
                except Exception as e:
                    logger.warning(f"Failed to get predictions from {name}: {str(e)}")

        if not predictions:
            logger.warning("No models available for ensemble")
            return np.zeros(len(X_test)), np.zeros(len(X_test))

        predictions = np.array(predictions)  # Shape: (n_models, n_samples)

        ensemble_pred = np.mean(predictions, axis=0)
        ensemble_std = np.std(predictions, axis=0)

        logger.info(f"Ensemble prediction completed. Mean std: {np.mean(ensemble_std):.6f}")
        return ensemble_pred, ensemble_std


def fit_regime_ensembles(X_train: np.ndarray, y_train: np.ndarray,
                        regime_labels_train: np.ndarray) -> Dict[int, Dict[str, Any]]:
    """
    Fit separate ensemble models for each regime.

    Args:
        X_train, y_train: Training data
        regime_labels_train: Regime labels for training data

    Returns:
        Dict of regime-specific models
    """
    logger.info("Fitting regime-specific ensemble models")

    unique_regimes = np.unique(regime_labels_train)
    regime_models = {}

    for regime in unique_regimes:
        logger.info(f"Fitting models for regime {regime}")

        # Get data for this regime
        regime_mask = regime_labels_train == regime
        X_regime = X_train[regime_mask]
        y_regime = y_train[regime_mask]

        if len(X_regime) < 10:  # Skip if too few samples
            logger.warning(f"Regime {regime} has too few samples ({len(X_regime)})")
            continue

        # Fit ensemble comparison for this regime
        ensemble_comp = EnsembleComparison()
        models = ensemble_comp.fit_all_models(X_regime, y_regime, X_regime, y_regime)  # Use same data for val

        regime_models[regime] = models

    logger.info(f"Regime-specific models fitted for {len(regime_models)} regimes")
    return regime_models


def predict_regime_aware(X_test: np.ndarray, regime_labels_test: np.ndarray,
                        regime_models: Dict[int, Dict[str, Any]]) -> np.ndarray:
    """
    Make regime-aware predictions.

    Args:
        X_test: Test features
        regime_labels_test: Regime labels for test data
        regime_models: Regime-specific models

    Returns:
        np.ndarray: Predictions
    """
    logger.info("Making regime-aware predictions")

    predictions = np.zeros(len(X_test))

    for regime, models in regime_models.items():
        regime_mask = regime_labels_test == regime

        if np.sum(regime_mask) == 0:
            continue

        X_regime = X_test[regime_mask]

        # Get ensemble prediction for this regime
        ensemble_comp = EnsembleComparison()
        ensemble_comp.models = models
        regime_pred, _ = ensemble_comp.predict_ensemble(X_regime)

        predictions[regime_mask] = regime_pred

    logger.info("Regime-aware predictions completed")
    return predictions


def main():
    """
    Main function demonstrating ensemble model training and comparison.
    """
    logger.info("Starting ensemble models demonstration")

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

            # Generate synthetic financial data
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

        # Prepare data for ensemble models
        logger.info("Preparing data for ensemble modeling")

        # For tree-based models, we can use flattened sequences or raw features
        # Let's use a simple approach: use raw features for demonstration
        from sklearn.preprocessing import StandardScaler

        # Create lag features
        n_lags = 5
        feature_data = []

        for i in range(n_lags, len(data)):
            features = []
            for col in data.columns:
                # Add current value and lags
                features.extend(data[col].iloc[i-n_lags:i+1].values)
            feature_data.append(features)

        X = np.array(feature_data)
        y = data.iloc[n_lags:, -1].values  # Predict last column (returns)

        # Split data
        train_size = int(0.7 * len(X))
        val_size = int(0.15 * len(X))

        X_train = X[:train_size]
        y_train = y[:train_size]
        X_val = X[train_size:train_size+val_size]
        y_val = y[train_size:train_size+val_size]
        X_test = X[train_size+val_size:]
        y_test = y[train_size+val_size:]

        logger.info(f"Data prepared: Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}")

        # 1. Fit and compare all models
        logger.info("Fitting and comparing ensemble models")
        ensemble_comp = EnsembleComparison()
        models = ensemble_comp.fit_all_models(X_train, y_train, X_val, y_val)

        # Compare performance
        comparison_df = ensemble_comp.compare_performance(X_test, y_test)

        # Create ensemble predictions
        ensemble_pred, ensemble_std = ensemble_comp.predict_ensemble(X_test)

        # 2. Hyperparameter optimization (for XGBoost if available)
        if XGBOOST_AVAILABLE and OPTUNA_AVAILABLE:
            logger.info("Performing hyperparameter optimization")
            opt_results = hyperparameter_optimization(
                XGBoostPredictor, X_train, y_train, X_val, y_val,
                cv_folds=3, n_trials=20
            )
        else:
            opt_results = None

        # 3. Cross-validation
        if models.get('rf') is not None:
            logger.info("Performing cross-validation")
            cv_scores = cross_validation_scores(models['rf'], X_train, y_train, cv_folds=5)
        else:
            cv_scores = None

        # 4. Feature importance analysis
        feature_importance_data = {}
        if models.get('rf') is not None:
            feature_importance_data['rf'] = models['rf'].get_feature_importance()
        if models.get('xgb') is not None:
            feature_importance_data['xgb'] = models['xgb'].get_feature_importance()

        # 5. Create visualizations
        if MATPLOTLIB_AVAILABLE:
            logger.info("Creating visualizations")

            # Model comparison plot
            comp_fig = ensemble_comp.plot_model_comparison(comparison_df)

            # Feature importance plots
            fi_figs = {}
            if models.get('rf') is not None:
                fi_figs['rf'] = models['rf'].plot_feature_importance(top_n=10)
            if models.get('xgb') is not None:
                fi_figs['xgb'] = models['xgb'].plot_feature_importance(plot_type='gain', top_n=10)

        # Save results
        logger.info("Saving results")

        # Save comparison results
        comparison_df.to_csv('ensemble_model_comparison.csv', index=False)

        # Save ensemble predictions
        ensemble_results = pd.DataFrame({
            'actual': y_test,
            'ensemble_pred': ensemble_pred,
            'ensemble_std': ensemble_std
        })
        ensemble_results.to_csv('ensemble_predictions.csv', index=False)

        # Save feature importance
        for model_name, fi_df in feature_importance_data.items():
            if not fi_df.empty:
                fi_df.to_csv(f'{model_name}_feature_importance.csv', index=False)

        # Save CV results
        if cv_scores:
            cv_df = pd.DataFrame({
                'fold': range(1, len(cv_scores['fold_scores'])+1),
                'rmse': cv_scores['fold_scores']
            })
            cv_df.to_csv('cross_validation_scores.csv', index=False)

        # Save plots
        if MATPLOTLIB_AVAILABLE:
            if 'comp_fig' in locals() and comp_fig:
                comp_fig.savefig('ensemble_model_comparison.png', dpi=300, bbox_inches='tight')

            for model_name, fig in fi_figs.items():
                if fig:
                    fig.savefig(f'{model_name}_feature_importance.png', dpi=300, bbox_inches='tight')

        # Print summary
        print("\n" + "="*80)
        print("ENSEMBLE MODELS COMPLETED")
        print("="*80)
        print(f"Data: {len(data)} total samples, {X.shape[1]} features")
        print(f"Train/Val/Test split: {len(X_train)}/{len(X_val)}/{len(X_test)}")
        print()

        if not comparison_df.empty:
            print("MODEL PERFORMANCE COMPARISON:")
            print("-" * 35)
            for _, row in comparison_df.iterrows():
                print(f"{row['model'].upper()}:")
                print(".6f")
                print(".4f")
                print(".1%")
                print()

        if cv_scores:
            print("CROSS-VALIDATION RESULTS:")
            print("-" * 25)
            print(".6f")
            print(".6f")
            print()

        print("MODELS TRAINED:")
        print("-" * 15)
        available_models = [name for name, model in models.items() if model is not None]
        for model in available_models:
            print(f"* {model}")
        print()

        print(f"Files saved: {3 + len(feature_importance_data) + (1 if cv_scores else 0)} CSV files, {1 + len(fi_figs)} PNG plots")
        print("="*80)

    except Exception as e:
        logger.error(f"Error in main analysis: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()