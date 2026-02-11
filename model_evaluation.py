"""
Model Evaluation Module
=======================

Comprehensive evaluation framework for financial time series forecasting models.
Includes statistical significance testing, robustness analysis, and diagnostic visualizations.

Author: GitHub Copilot
Date: February 8, 2026
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings
warnings.filterwarnings('ignore')

# Required libraries for evaluation
try:
    from sklearn.metrics import (
        mean_absolute_error, mean_squared_error, r2_score,
        median_absolute_error, accuracy_score, precision_score,
        recall_score, f1_score, confusion_matrix
    )
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.utils import resample
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available - basic metrics disabled")

try:
    import scipy.stats as stats
    from scipy.stats import ttest_rel, normaltest, shapiro
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("scipy not available - statistical tests disabled")

try:
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import acf
    from statsmodels.stats.diagnostic import acorr_ljungbox
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    logging.warning("statsmodels not available - time series analysis disabled")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
    plt.style.use('default')
    sns.set_palette("husl")
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logging.warning("matplotlib/seaborn not available - plotting disabled")

try:
    import json
    JSON_AVAILABLE = True
except ImportError:
    JSON_AVAILABLE = False
    logging.warning("json not available - JSON output disabled")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EvaluationMetrics:
    """
    Comprehensive evaluation metrics for financial forecasting models.
    """

    def __init__(self):
        """Initialize the evaluation metrics calculator."""
        logger.info("Initialized EvaluationMetrics")

    def compute_all_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Compute comprehensive set of evaluation metrics.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Dictionary of all computed metrics
        """
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available - returning basic metrics")
            return self._compute_basic_metrics(y_true, y_pred)

        try:
            logger.info("Computing comprehensive evaluation metrics")

            # Basic regression metrics
            mae = mean_absolute_error(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true, y_pred)
            medae = median_absolute_error(y_true, y_pred)

            # MAPE (Mean Absolute Percentage Error)
            # Avoid division by zero
            mask = y_true != 0
            if mask.any():
                mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            else:
                mape = np.nan

            # Direction accuracy (for financial forecasting)
            direction_accuracy = self.direction_accuracy(y_true, y_pred)['accuracy']

            metrics = {
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'mape': mape,
                'r2_score': r2,
                'direction_accuracy': direction_accuracy,
                'mean_absolute_error': mae,  # Alias for compatibility
                'median_absolute_error': medae,
                'explained_variance': 1 - np.var(y_true - y_pred) / np.var(y_true),
                'mean_error': np.mean(y_pred - y_true),  # Bias
                'std_error': np.std(y_pred - y_true),  # Error variability
            }

            logger.info(f"Computed {len(metrics)} evaluation metrics")
            return metrics

        except Exception as e:
            logger.error(f"Error computing metrics: {str(e)}")
            return self._compute_basic_metrics(y_true, y_pred)

    def _compute_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Fallback basic metrics computation without sklearn."""
        try:
            errors = y_pred - y_true

            return {
                'mae': np.mean(np.abs(errors)),
                'mse': np.mean(errors ** 2),
                'rmse': np.sqrt(np.mean(errors ** 2)),
                'mean_error': np.mean(errors),
                'std_error': np.std(errors),
                'direction_accuracy': self._direction_accuracy_basic(y_true, y_pred)
            }
        except Exception as e:
            logger.error(f"Error in basic metrics: {str(e)}")
            return {}

    def regime_specific_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                               regime_labels: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Compute metrics separately for each regime.

        Args:
            y_true: True values
            y_pred: Predicted values
            regime_labels: Array of regime labels (0, 1, 2, etc.)

        Returns:
            Dictionary with metrics for each regime
        """
        try:
            logger.info("Computing regime-specific metrics")

            unique_regimes = np.unique(regime_labels)
            regime_metrics = {}

            for regime in unique_regimes:
                mask = regime_labels == regime
                if mask.sum() > 0:  # Only compute if we have data for this regime
                    regime_y_true = y_true[mask]
                    regime_y_pred = y_pred[mask]

                    metrics = self.compute_all_metrics(regime_y_true, regime_y_pred)
                    regime_metrics[f'regime_{regime}'] = metrics

                    logger.info(f"Regime {regime}: {mask.sum()} samples, RMSE={metrics.get('rmse', 'N/A'):.4f}")

            logger.info(f"Computed metrics for {len(regime_metrics)} regimes")
            return regime_metrics

        except Exception as e:
            logger.error(f"Error computing regime-specific metrics: {str(e)}")
            return {}

    def temporal_consistency(self, y_true: np.ndarray, y_pred: np.ndarray,
                           dates: Optional[np.ndarray] = None, window: int = 30) -> pd.DataFrame:
        """
        Compute rolling metrics over time to assess temporal consistency.

        Args:
            y_true: True values
            y_pred: Predicted values
            dates: Date array (if None, uses indices)
            window: Rolling window size

        Returns:
            DataFrame with rolling metrics over time
        """
        try:
            logger.info(f"Computing temporal consistency with {window}-period rolling window")

            errors = y_pred - y_true

            # Create date index if not provided
            if dates is None:
                dates = pd.date_range(start='2020-01-01', periods=len(y_true), freq='D')
            elif not isinstance(dates, pd.DatetimeIndex):
                dates = pd.to_datetime(dates)

            # Create DataFrame
            df = pd.DataFrame({
                'date': dates,
                'y_true': y_true,
                'y_pred': y_pred,
                'error': errors,
                'abs_error': np.abs(errors)
            })

            # Rolling metrics
            df['rmse_rolling'] = df['error'].rolling(window=window).apply(
                lambda x: np.sqrt(np.mean(x**2)) if len(x) >= window//2 else np.nan
            )

            df['mae_rolling'] = df['abs_error'].rolling(window=window).mean()

            # Direction accuracy rolling
            df['direction_correct'] = np.sign(y_pred[1:] - y_pred[:-1]) == np.sign(y_true[1:] - y_true[:-1])
            df['direction_accuracy_rolling'] = df['direction_correct'].rolling(window=window).mean()

            # Fill first window-1 values with cumulative metrics
            for i in range(1, window):
                df.loc[i-1, 'rmse_rolling'] = np.sqrt(np.mean(df.loc[:i-1, 'error']**2))
                df.loc[i-1, 'mae_rolling'] = np.mean(df.loc[:i-1, 'abs_error'])
                if i > 1:
                    df.loc[i-1, 'direction_accuracy_rolling'] = np.mean(df.loc[:i-1, 'direction_correct'])

            result_df = df[['date', 'rmse_rolling', 'mae_rolling', 'direction_accuracy_rolling']].copy()
            result_df.columns = ['Date', 'RMSE_30d', 'MAE_30d', 'Direction_Acc_30d']

            logger.info("Temporal consistency analysis completed")
            return result_df

        except Exception as e:
            logger.error(f"Error computing temporal consistency: {str(e)}")
            return pd.DataFrame()

    def direction_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Union[float, int]]:
        """
        Compute direction accuracy and related classification metrics.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Dictionary with direction accuracy metrics
        """
        try:
            # Convert to direction changes (1 = up, 0 = down)
            if len(y_true) < 2 or len(y_pred) < 2:
                return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}

            # Direction changes
            true_direction = np.sign(np.diff(y_true))
            pred_direction = np.sign(np.diff(y_pred))

            # Convert to binary (1 = positive change, 0 = negative change)
            y_true_binary = (true_direction > 0).astype(int)
            y_pred_binary = (pred_direction > 0).astype(int)

            if not SKLEARN_AVAILABLE:
                return self._direction_accuracy_basic(y_true, y_pred)

            # Compute metrics
            accuracy = accuracy_score(y_true_binary, y_pred_binary)
            precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
            recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
            f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)

            # Confusion matrix
            cm = confusion_matrix(y_true_binary, y_pred_binary)
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

            return {
                'accuracy': accuracy,
                'true_positives': int(tp),
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }

        except Exception as e:
            logger.error(f"Error computing direction accuracy: {str(e)}")
            return self._direction_accuracy_basic(y_true, y_pred)

    def _direction_accuracy_basic(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Basic direction accuracy without sklearn."""
        try:
            if len(y_true) < 2 or len(y_pred) < 2:
                return {'accuracy': 0.0}

            true_direction = np.sign(np.diff(y_true))
            pred_direction = np.sign(np.diff(y_pred))

            correct = np.sum(true_direction == pred_direction)
            total = len(true_direction)

            return {
                'accuracy': correct / total if total > 0 else 0.0,
                'true_positives': 0,
                'true_negatives': 0,
                'false_positives': 0,
                'false_negatives': 0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0
            }
        except Exception as e:
            return {'accuracy': 0.0}


def diebold_mariano_test(errors_model1: np.ndarray, errors_model2: np.ndarray,
                        horizon: int = 1) -> Dict[str, Union[float, bool]]:
    """
    Diebold-Mariano test for comparing forecast accuracy.

    Tests whether model1 has significantly different forecast accuracy than model2.

    Args:
        errors_model1: Forecast errors from model 1
        errors_model2: Forecast errors from model 2
        horizon: Forecast horizon (default: 1)

    Returns:
        Dictionary with test results
    """
    if not SCIPY_AVAILABLE or not STATSMODELS_AVAILABLE:
        logger.warning("scipy or statsmodels not available - DM test disabled")
        return {'statistic': 0.0, 'p_value': 1.0, 'model1_better': False}

    try:
        logger.info("Performing Diebold-Mariano test")

        # Compute loss differential (squared errors)
        d = errors_model1**2 - errors_model2**2

        # Newey-West standard errors for autocorrelation
        n = len(d)
        if n < 10:
            logger.warning("Insufficient data for DM test")
            return {'statistic': 0.0, 'p_value': 1.0, 'model1_better': False}

        # Simple DM statistic (assuming no autocorrelation for simplicity)
        dm_stat = np.mean(d) / (np.std(d) / np.sqrt(n))

        # Two-tailed p-value
        p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))

        # Determine if model1 is significantly better (lower errors)
        model1_better = (p_value < 0.05) and (np.mean(errors_model1**2) < np.mean(errors_model2**2))

        result = {
            'statistic': dm_stat,
            'p_value': p_value,
            'model1_better': model1_better,
            'mean_error_diff': np.mean(d),
            'significant': p_value < 0.05
        }

        logger.info(f"DM test completed: statistic={dm_stat:.4f}, p-value={p_value:.4f}")
        return result

    except Exception as e:
        logger.error(f"Error in Diebold-Mariano test: {str(e)}")
        return {'statistic': 0.0, 'p_value': 1.0, 'model1_better': False}


def paired_t_test(errors_model1: np.ndarray, errors_model2: np.ndarray) -> Dict[str, Union[float, bool]]:
    """
    Paired t-test for comparing forecast errors.

    Tests whether the mean difference between model errors is statistically significant.

    Args:
        errors_model1: Forecast errors from model 1
        errors_model2: Forecast errors from model 2

    Returns:
        Dictionary with test results
    """
    if not SCIPY_AVAILABLE:
        logger.warning("scipy not available - paired t-test disabled")
        return {'t_statistic': 0.0, 'p_value': 1.0, 'significant': False}

    try:
        logger.info("Performing paired t-test")

        # Paired t-test on the error differences
        t_stat, p_value = ttest_rel(errors_model1, errors_model2)

        # Check if errors are significantly different
        significant = p_value < 0.05

        # Determine which model is better (lower absolute errors)
        model1_better = np.mean(np.abs(errors_model1)) < np.mean(np.abs(errors_model2))

        result = {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': significant,
            'model1_better': model1_better,
            'mean_error_1': np.mean(np.abs(errors_model1)),
            'mean_error_2': np.mean(np.abs(errors_model2))
        }

        logger.info(f"Paired t-test completed: t={t_stat:.4f}, p-value={p_value:.4f}")
        return result

    except Exception as e:
        logger.error(f"Error in paired t-test: {str(e)}")
        return {'t_statistic': 0.0, 'p_value': 1.0, 'significant': False}


def model_comparison_test(models_dict: Dict[str, Any], X_test: np.ndarray,
                         y_test: np.ndarray) -> pd.DataFrame:
    """
    Perform pairwise comparison of all models using statistical tests.

    Args:
        models_dict: Dictionary of trained models {name: model}
        X_test: Test features
        y_test: Test targets

    Returns:
        DataFrame with pairwise comparison results
    """
    try:
        logger.info("Performing pairwise model comparison")

        model_names = list(models_dict.keys())
        results = []

        # Get predictions for each model
        predictions = {}
        errors = {}

        for name, model in models_dict.items():
            try:
                if hasattr(model, 'predict'):
                    pred = model.predict(X_test)
                    predictions[name] = pred
                    errors[name] = pred - y_test
                else:
                    logger.warning(f"Model {name} has no predict method")
                    continue
            except Exception as e:
                logger.warning(f"Error getting predictions for {name}: {str(e)}")
                continue

        # Pairwise comparisons
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names):
                if i >= j:  # Skip self-comparison and duplicates
                    continue

                if model1 not in errors or model2 not in errors:
                    continue

                # Diebold-Mariano test
                dm_result = diebold_mariano_test(errors[model1], errors[model2])

                # Paired t-test
                ttest_result = paired_t_test(errors[model1], errors[model2])

                # Compute basic metrics
                rmse1 = np.sqrt(np.mean(errors[model1]**2))
                rmse2 = np.sqrt(np.mean(errors[model2]**2))

                results.append({
                    'model_1': model1,
                    'model_2': model2,
                    'rmse_1': rmse1,
                    'rmse_2': rmse2,
                    'dm_statistic': dm_result['statistic'],
                    'dm_p_value': dm_result['p_value'],
                    'dm_significant': dm_result['significant'],
                    'dm_model1_better': dm_result['model1_better'],
                    'ttest_statistic': ttest_result['t_statistic'],
                    'ttest_p_value': ttest_result['p_value'],
                    'ttest_significant': ttest_result['significant'],
                    'ttest_model1_better': ttest_result['model1_better']
                })

        df = pd.DataFrame(results)
        logger.info(f"Model comparison completed: {len(results)} pairwise comparisons")
        return df

    except Exception as e:
        logger.error(f"Error in model comparison: {str(e)}")
        return pd.DataFrame()


def analyze_prediction_errors(y_true: np.ndarray, y_pred: np.ndarray,
                            dates: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Perform statistical analysis of prediction errors (residuals).

    Args:
        y_true: True values
        y_pred: Predicted values
        dates: Date array for time series analysis

    Returns:
        Dictionary with error analysis results
    """
    try:
        logger.info("Analyzing prediction errors")

        errors = y_pred - y_true

        # Basic statistics
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        skewness = stats.skew(errors) if SCIPY_AVAILABLE else 0.0
        kurtosis = stats.kurtosis(errors) if SCIPY_AVAILABLE else 0.0

        # Autocorrelation analysis
        autocorr_lag1 = 0.0
        ljung_box_p = 1.0

        if STATSMODELS_AVAILABLE and len(errors) > 10:
            try:
                # Autocorrelation at lag 1
                autocorr = acf(errors, nlags=1, fft=True)
                autocorr_lag1 = autocorr[1] if len(autocorr) > 1 else 0.0

                # Ljung-Box test for autocorrelation
                lb_test = acorr_ljungbox(errors, lags=[1], return_df=False)
                ljung_box_p = lb_test[1][0] if len(lb_test[1]) > 0 else 1.0
            except Exception as e:
                logger.warning(f"Error computing autocorrelation: {str(e)}")

        # Normality tests
        normality_p = 1.0
        if SCIPY_AVAILABLE and len(errors) > 3:
            try:
                # Shapiro-Wilk test for normality
                _, normality_p = shapiro(errors[:5000])  # Limit for test
            except Exception as e:
                logger.warning(f"Error in normality test: {str(e)}")

        result = {
            'mean_error': mean_error,
            'std_error': std_error,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'autocorr_lag1': autocorr_lag1,
            'ljung_box_p': ljung_box_p,
            'normality_p': normality_p,
            'autocorrelated': ljung_box_p < 0.05,
            'normally_distributed': normality_p > 0.05
        }

        logger.info(f"Error analysis completed: mean={mean_error:.4f}, std={std_error:.4f}")
        return result

    except Exception as e:
        logger.error(f"Error analyzing prediction errors: {str(e)}")
        return {}


def forecast_skill_score(y_true: np.ndarray, y_pred: np.ndarray,
                        baseline_pred: np.ndarray) -> float:
    """
    Compute forecast skill score relative to baseline.

    Formula: (MSE_baseline - MSE_model) / MSE_baseline
    Range: -inf to 1, where 1 = perfect, 0 = same as baseline

    Args:
        y_true: True values
        y_pred: Model predictions
        baseline_pred: Baseline predictions

    Returns:
        Skill score (float)
    """
    try:
        logger.info("Computing forecast skill score")

        # Compute MSE for model and baseline
        mse_model = np.mean((y_true - y_pred) ** 2)
        mse_baseline = np.mean((y_true - baseline_pred) ** 2)

        if mse_baseline == 0:
            logger.warning("Baseline MSE is zero - cannot compute skill score")
            return 0.0

        # Skill score
        skill_score = (mse_baseline - mse_model) / mse_baseline

        logger.info(f"Skill score computed: {skill_score:.4f}")
        return skill_score

    except Exception as e:
        logger.error(f"Error computing skill score: {str(e)}")
        return 0.0


def hit_rate_analysis(y_true: np.ndarray, y_pred: np.ndarray,
                     percentile_threshold: float = 75) -> Dict[str, float]:
    """
    Analyze model's ability to predict extreme movements.

    Args:
        y_true: True values
        y_pred: Predicted values
        percentile_threshold: Percentile threshold for "extreme" movements

    Returns:
        Dictionary with hit rate analysis
    """
    try:
        logger.info(f"Analyzing hit rates with {percentile_threshold}th percentile threshold")

        # Compute absolute changes
        true_changes = np.abs(np.diff(y_true))
        pred_changes = np.abs(np.diff(y_pred))

        # Define extreme movements
        threshold = np.percentile(true_changes, percentile_threshold)
        extreme_mask = true_changes >= threshold

        # Hit rate: how often model predicts extreme movement when it occurs
        if extreme_mask.sum() > 0:
            # Model predictions of extreme movements
            pred_extreme = pred_changes >= threshold
            hits = np.sum(pred_extreme[extreme_mask])
            total_extreme = extreme_mask.sum()

            hit_rate = hits / total_extreme
        else:
            hit_rate = 0.0

        # Overall direction accuracy
        direction_accuracy = EvaluationMetrics().direction_accuracy(y_true, y_pred)['accuracy']

        # High volatility periods accuracy
        high_vol_mask = true_changes >= np.percentile(true_changes, 75)
        if high_vol_mask.sum() > 0:
            high_vol_true = y_true[1:][high_vol_mask]
            high_vol_pred = y_pred[1:][high_vol_mask]
            high_vol_accuracy = EvaluationMetrics().direction_accuracy(high_vol_true, high_vol_pred)['accuracy']
        else:
            high_vol_accuracy = 0.0

        result = {
            'overall_accuracy': direction_accuracy,
            'high_volatility_accuracy': high_vol_accuracy,
            'extreme_movement_hit_rate': hit_rate,
            'extreme_threshold': threshold,
            'n_extreme_movements': int(extreme_mask.sum())
        }

        logger.info(f"Hit rate analysis completed: hit_rate={hit_rate:.4f}")
        return result

    except Exception as e:
        logger.error(f"Error in hit rate analysis: {str(e)}")
        return {}


def test_model_stability(model: Any, X_data: np.ndarray, y_data: np.ndarray,
                        n_splits: int = 10) -> Dict[str, Union[float, bool]]:
    """
    Test model stability using bootstrap resampling.

    Args:
        model: Trained model
        X_data: Feature data
        y_data: Target data
        n_splits: Number of bootstrap samples

    Returns:
        Dictionary with stability analysis results
    """
    try:
        logger.info(f"Testing model stability with {n_splits} bootstrap samples")

        performances = []

        n_samples = len(X_data)
        sample_size = int(0.8 * n_samples)  # 80% of data for training

        for i in range(n_splits):
            # Bootstrap sample
            indices = np.random.choice(n_samples, size=sample_size, replace=True)
            X_boot = X_data[indices]
            y_boot = y_data[indices]

            # Train model on bootstrap sample
            try:
                if hasattr(model, 'fit'):
                    boot_model = type(model)()  # Create new instance
                    boot_model.fit(X_boot, y_boot)

                    # Test on remaining data
                    remaining_indices = np.setdiff1d(np.arange(n_samples), indices)
                    if len(remaining_indices) > 0:
                        X_test_boot = X_data[remaining_indices]
                        y_test_boot = y_data[remaining_indices]

                        y_pred_boot = boot_model.predict(X_test_boot)
                        rmse = np.sqrt(np.mean((y_test_boot - y_pred_boot) ** 2))
                        performances.append(rmse)
            except Exception as e:
                logger.warning(f"Error in bootstrap iteration {i}: {str(e)}")
                continue

        if not performances:
            return {'mean_performance': 0.0, 'std_performance': 0.0, 'stable': False}

        mean_perf = np.mean(performances)
        std_perf = np.std(performances)
        cv = std_perf / mean_perf if mean_perf > 0 else float('inf')

        # Consider stable if coefficient of variation < 0.2
        stable = cv < 0.2

        result = {
            'mean_performance': mean_perf,
            'std_performance': std_perf,
            'coefficient_of_variation': cv,
            'stable': stable,
            'n_successful_bootstraps': len(performances)
        }

        logger.info(f"Stability test completed: CV={cv:.4f}, stable={stable}")
        return result

    except Exception as e:
        logger.error(f"Error in stability test: {str(e)}")
        return {'mean_performance': 0.0, 'std_performance': 0.0, 'stable': False}


def walk_forward_backtest(model: Any, X_full: np.ndarray, y_full: np.ndarray,
                         train_size: float = 0.7, step: int = 30) -> Dict[str, Any]:
    """
    Perform walk-forward backtesting for time series models.

    Args:
        model: Model to backtest
        X_full: Full feature dataset
        y_full: Full target dataset
        train_size: Initial training set size (fraction)
        step: Number of steps to move forward each iteration

    Returns:
        Dictionary with backtest results
    """
    try:
        logger.info(f"Performing walk-forward backtest with train_size={train_size}, step={step}")

        n_samples = len(X_full)
        train_end = int(n_samples * train_size)

        performances = []
        predictions = []
        actuals = []

        current_train_end = train_end

        while current_train_end + step <= n_samples:
            # Training data
            X_train = X_full[:current_train_end]
            y_train = y_full[:current_train_end]

            # Test data
            test_start = current_train_end
            test_end = min(current_train_end + step, n_samples)
            X_test = X_full[test_start:test_end]
            y_test = y_full[test_start:test_end]

            try:
                # Train model
                if hasattr(model, 'fit'):
                    backtest_model = type(model)()  # Create new instance
                    backtest_model.fit(X_train, y_train)

                    # Make predictions
                    y_pred = backtest_model.predict(X_test)

                    # Compute metrics
                    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
                    mae = np.mean(np.abs(y_test - y_pred))
                    direction_acc = EvaluationMetrics().direction_accuracy(y_test, y_pred)['accuracy']

                    period_result = {
                        'period_start': test_start,
                        'period_end': test_end,
                        'rmse': rmse,
                        'mae': mae,
                        'direction_accuracy': direction_acc,
                        'n_samples': len(y_test)
                    }

                    performances.append(period_result)
                    predictions.extend(y_pred)
                    actuals.extend(y_test)

            except Exception as e:
                logger.warning(f"Error in backtest period {current_train_end}: {str(e)}")

            # Move forward
            current_train_end += step

        if not performances:
            return {'performance_by_period': [], 'average_performance': {}}

        # Compute average performance
        avg_rmse = np.mean([p['rmse'] for p in performances])
        avg_mae = np.mean([p['mae'] for p in performances])
        avg_direction_acc = np.mean([p['direction_accuracy'] for p in performances])

        # Overall metrics on all predictions
        overall_rmse = np.sqrt(np.mean((np.array(actuals) - np.array(predictions)) ** 2))
        overall_direction_acc = EvaluationMetrics().direction_accuracy(
            np.array(actuals), np.array(predictions)
        )['accuracy']

        result = {
            'performance_by_period': performances,
            'average_performance': {
                'rmse': avg_rmse,
                'mae': avg_mae,
                'direction_accuracy': avg_direction_acc
            },
            'overall_performance': {
                'rmse': overall_rmse,
                'direction_accuracy': overall_direction_acc
            },
            'n_periods': len(performances)
        }

        logger.info(f"Walk-forward backtest completed: {len(performances)} periods, avg RMSE={avg_rmse:.4f}")
        return result

    except Exception as e:
        logger.error(f"Error in walk-forward backtest: {str(e)}")
        return {'performance_by_period': [], 'average_performance': {}}


def analyze_out_of_distribution(model: Any, X_normal: np.ndarray, y_normal: np.ndarray,
                              X_extreme: np.ndarray, y_extreme: np.ndarray) -> Dict[str, float]:
    """
    Analyze model performance in normal vs extreme conditions.

    Args:
        model: Trained model
        X_normal: Normal condition features
        y_normal: Normal condition targets
        X_extreme: Extreme condition features
        y_extreme: Extreme condition targets

    Returns:
        Dictionary with OOD analysis results
    """
    try:
        logger.info("Analyzing out-of-distribution performance")

        # Performance on normal data
        y_pred_normal = model.predict(X_normal)
        rmse_normal = np.sqrt(np.mean((y_normal - y_pred_normal) ** 2))
        mae_normal = np.mean(np.abs(y_normal - y_pred_normal))
        direction_acc_normal = EvaluationMetrics().direction_accuracy(y_normal, y_pred_normal)['accuracy']

        # Performance on extreme data
        y_pred_extreme = model.predict(X_extreme)
        rmse_extreme = np.sqrt(np.mean((y_extreme - y_pred_extreme) ** 2))
        mae_extreme = np.mean(np.abs(y_extreme - y_pred_extreme))
        direction_acc_extreme = EvaluationMetrics().direction_accuracy(y_extreme, y_pred_extreme)['accuracy']

        # Degradation analysis
        rmse_degradation = (rmse_extreme - rmse_normal) / rmse_normal * 100
        mae_degradation = (mae_extreme - mae_normal) / mae_normal * 100
        direction_degradation = (direction_acc_normal - direction_acc_extreme) * 100

        result = {
            'normal_rmse': rmse_normal,
            'normal_mae': mae_normal,
            'normal_direction_accuracy': direction_acc_normal,
            'extreme_rmse': rmse_extreme,
            'extreme_mae': mae_extreme,
            'extreme_direction_accuracy': direction_acc_extreme,
            'rmse_degradation_percent': rmse_degradation,
            'mae_degradation_percent': mae_degradation,
            'direction_degradation_percent': direction_degradation,
            'n_normal_samples': len(y_normal),
            'n_extreme_samples': len(y_extreme)
        }

        logger.info(f"OOD analysis completed: normal RMSE={rmse_normal:.4f}, extreme RMSE={rmse_extreme:.4f}")
        return result

    except Exception as e:
        logger.error(f"Error in OOD analysis: {str(e)}")
        return {}


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray,
                  dates: Optional[np.ndarray] = None) -> Any:
    """
    Create comprehensive residual analysis plots.

    Args:
        y_true: True values
        y_pred: Predicted values
        dates: Date array for time series

    Returns:
        matplotlib figure
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("matplotlib not available")
        return None

    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Residual Analysis', fontsize=16)

        errors = y_pred - y_true

        # Create date index if not provided
        if dates is None:
            dates = np.arange(len(errors))
        elif not isinstance(dates, (pd.DatetimeIndex, np.ndarray)):
            dates = pd.to_datetime(dates)

        # 1. Residuals over time
        axes[0, 0].plot(dates, errors, 'b-', alpha=0.7)
        axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[0, 0].set_title('Residuals Over Time')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Residual')
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Residual distribution
        axes[0, 1].hist(errors, bins=50, alpha=0.7, color='blue', density=True)
        axes[0, 1].axvline(x=0, color='r', linestyle='--', alpha=0.5)

        # Add normal distribution fit
        if SCIPY_AVAILABLE:
            try:
                mu, std = stats.norm.fit(errors)
                xmin, xmax = axes[0, 1].get_xlim()
                x = np.linspace(xmin, xmax, 100)
                p = stats.norm.pdf(x, mu, std)
                axes[0, 1].plot(x, p, 'r-', linewidth=2, alpha=0.7, label='Normal fit')
                axes[0, 1].legend()
            except:
                pass

        axes[0, 1].set_title('Residual Distribution')
        axes[0, 1].set_xlabel('Residual')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Q-Q plot
        if SCIPY_AVAILABLE:
            try:
                stats.probplot(errors, dist="norm", plot=axes[1, 0])
                axes[1, 0].set_title('Q-Q Plot')
                axes[1, 0].grid(True, alpha=0.3)
            except Exception as e:
                axes[1, 0].text(0.5, 0.5, f'Q-Q plot failed:\n{str(e)}',
                               ha='center', va='center', transform=axes[1, 0].transAxes)
        else:
            axes[1, 0].text(0.5, 0.5, 'scipy not available\nfor Q-Q plot',
                           ha='center', va='center', transform=axes[1, 0].transAxes)

        # 4. Autocorrelation function
        if STATSMODELS_AVAILABLE and len(errors) > 10:
            try:
                lags = min(20, len(errors) - 1)
                autocorr = acf(errors, nlags=lags, fft=True)
                axes[1, 1].bar(range(len(autocorr)), autocorr, alpha=0.7, color='green')
                axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
                axes[1, 1].axhline(y=1.96/np.sqrt(len(errors)), color='r', linestyle='--', alpha=0.7)
                axes[1, 1].axhline(y=-1.96/np.sqrt(len(errors)), color='r', linestyle='--', alpha=0.7)
                axes[1, 1].set_title('Residual Autocorrelation')
                axes[1, 1].set_xlabel('Lag')
                axes[1, 1].set_ylabel('Autocorrelation')
                axes[1, 1].grid(True, alpha=0.3)
            except Exception as e:
                axes[1, 1].text(0.5, 0.5, f'ACF failed:\n{str(e)}',
                               ha='center', va='center', transform=axes[1, 1].transAxes)
        else:
            axes[1, 1].text(0.5, 0.5, 'statsmodels not available\nfor ACF',
                           ha='center', va='center', transform=axes[1, 1].transAxes)

        plt.tight_layout()
        logger.info("Residual analysis plots created")
        return fig

    except Exception as e:
        logger.error(f"Error creating residual plots: {str(e)}")
        return None


def plot_prediction_bands(y_true: np.ndarray, y_pred: np.ndarray,
                         uncertainty: Optional[np.ndarray] = None,
                         dates: Optional[np.ndarray] = None) -> Any:
    """
    Plot predictions with uncertainty bands.

    Args:
        y_true: True values
        y_pred: Predicted values
        uncertainty: Prediction uncertainty (standard deviation)
        dates: Date array

    Returns:
        matplotlib figure
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("matplotlib not available")
        return None

    try:
        fig, ax = plt.subplots(figsize=(15, 8))

        # Create date index if not provided
        if dates is None:
            dates = np.arange(len(y_true))
        elif not isinstance(dates, (pd.DatetimeIndex, np.ndarray)):
            dates = pd.to_datetime(dates)

        # Plot actual vs predicted
        ax.plot(dates, y_true, 'b-', linewidth=2, label='Actual', alpha=0.8)
        ax.plot(dates, y_pred, 'r-', linewidth=2, label='Predicted', alpha=0.8)

        # Add uncertainty bands if provided
        if uncertainty is not None:
            ax.fill_between(dates, y_pred - 2*uncertainty, y_pred + 2*uncertainty,
                          alpha=0.3, color='red', label='±2σ Confidence')
            ax.fill_between(dates, y_pred - uncertainty, y_pred + uncertainty,
                          alpha=0.5, color='red', label='±1σ Confidence')

        ax.set_title('Predictions with Uncertainty Bands', fontsize=16)
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        logger.info("Prediction bands plot created")
        return fig

    except Exception as e:
        logger.error(f"Error creating prediction bands plot: {str(e)}")
        return None


def plot_regime_metrics(regime_metrics_dict: Dict[str, Dict[str, float]]) -> Any:
    """
    Plot metrics comparison across regimes.

    Args:
        regime_metrics_dict: Results from regime_specific_metrics

    Returns:
        matplotlib figure
    """
    if not MATPLOTLIB_AVAILABLE or not regime_metrics_dict:
        logger.warning("matplotlib not available or no regime data")
        return None

    try:
        metrics_to_plot = ['rmse', 'mae', 'direction_accuracy', 'r2_score']
        regimes = list(regime_metrics_dict.keys())
        n_regimes = len(regimes)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Performance Metrics by Regime', fontsize=16)
        axes = axes.flatten()

        for i, metric in enumerate(metrics_to_plot):
            ax = axes[i]

            values = []
            for regime in regimes:
                if metric in regime_metrics_dict[regime]:
                    values.append(regime_metrics_dict[regime][metric])
                else:
                    values.append(0.0)

            bars = ax.bar(regimes, values, alpha=0.7, color=['blue', 'green', 'red', 'orange'][:n_regimes])
            ax.set_title(f'{metric.upper()} by Regime')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.grid(True, alpha=0.3)

            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                       '.3f', ha='center', va='bottom')

        plt.tight_layout()
        logger.info("Regime metrics comparison plot created")
        return fig

    except Exception as e:
        logger.error(f"Error creating regime metrics plot: {str(e)}")
        return None


def plot_metric_evolution(temporal_metrics: pd.DataFrame, metric_name: str) -> Any:
    """
    Plot temporal evolution of a specific metric.

    Args:
        temporal_metrics: Results from temporal_consistency
        metric_name: Name of metric to plot

    Returns:
        matplotlib figure
    """
    if not MATPLOTLIB_AVAILABLE or temporal_metrics.empty:
        logger.warning("matplotlib not available or no temporal data")
        return None

    try:
        fig, ax = plt.subplots(figsize=(15, 6))

        if 'Date' in temporal_metrics.columns and metric_name in temporal_metrics.columns:
            ax.plot(temporal_metrics['Date'], temporal_metrics[metric_name],
                   'b-', linewidth=2, alpha=0.8)

            # Add trend line
            y_values = temporal_metrics[metric_name].dropna()
            if len(y_values) > 10:
                x_values = np.arange(len(y_values))
                z = np.polyfit(x_values, y_values, 1)
                p = np.poly1d(z)
                ax.plot(temporal_metrics['Date'].iloc[:len(y_values)],
                       p(x_values), 'r--', linewidth=2, alpha=0.7, label='Trend')

            ax.set_title(f'{metric_name} Evolution Over Time', fontsize=16)
            ax.set_xlabel('Time')
            ax.set_ylabel(metric_name)
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        logger.info(f"Metric evolution plot created for {metric_name}")
        return fig

    except Exception as e:
        logger.error(f"Error creating metric evolution plot: {str(e)}")
        return None


def plot_prediction_error_distribution(errors_by_regime: Dict[str, np.ndarray]) -> Any:
    """
    Plot error distribution comparison across regimes.

    Args:
        errors_by_regime: Dictionary of errors by regime

    Returns:
        matplotlib figure
    """
    if not MATPLOTLIB_AVAILABLE or not errors_by_regime:
        logger.warning("matplotlib not available or no error data")
        return None

    try:
        fig, ax = plt.subplots(figsize=(12, 6))

        # Prepare data for boxplot
        data = []
        labels = []

        for regime, errors in errors_by_regime.items():
            if len(errors) > 0:
                data.append(errors)
                labels.append(regime)

        if data:
            bp = ax.boxplot(data, labels=labels, patch_artist=True)

            # Color the boxes
            colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
            for patch, color in zip(bp['boxes'], colors[:len(data)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            ax.set_title('Prediction Error Distribution by Regime', fontsize=16)
            ax.set_ylabel('Prediction Error')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        logger.info("Error distribution plot created")
        return fig

    except Exception as e:
        logger.error(f"Error creating error distribution plot: {str(e)}")
        return None


def main():
    """
    Main function demonstrating comprehensive model evaluation.
    """
    logger.info("Starting comprehensive model evaluation")

    try:
        # Generate synthetic financial data for demonstration
        logger.info("Generating synthetic financial data")
        np.random.seed(42)

        # Simulate financial time series with regimes
        n_samples = 2000
        dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')

        # Create features (technical indicators)
        X = np.random.randn(n_samples, 18)

        # Create target with regime-dependent behavior
        regime_labels = np.zeros(n_samples, dtype=int)

        # Regime 0: Normal market (first 800 days)
        regime_labels[:800] = 0
        y_normal = X[:800, -1] + 0.1 * np.random.randn(800)

        # Regime 1: High volatility (next 600 days)
        regime_labels[800:1400] = 1
        y_volatile = X[800:1400, -1] + 0.3 * np.random.randn(600) + 0.2 * X[800:1400, 4]

        # Regime 2: Crisis period (last 600 days)
        regime_labels[1400:] = 2
        y_crisis = X[1400:, -1] + 0.5 * np.random.randn(600) - 0.1 * X[1400:, 0]

        y = np.concatenate([y_normal, y_volatile, y_crisis])

        # Split data
        train_size = int(0.7 * n_samples)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        dates_test = dates[train_size:]
        regime_test = regime_labels[train_size:]

        # Train models for comparison
        models_dict = {}

        # Random Forest
        try:
            from sklearn.ensemble import RandomForestRegressor
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
            rf_model.fit(X_train, y_train)
            models_dict['RandomForest'] = rf_model
            logger.info("Random Forest model trained")
        except Exception as e:
            logger.warning(f"Could not train Random Forest: {str(e)}")

        # Linear model as baseline
        try:
            from sklearn.linear_model import LinearRegression
            lr_model = LinearRegression()
            lr_model.fit(X_train, y_train)
            models_dict['LinearRegression'] = lr_model
            logger.info("Linear Regression model trained")
        except Exception as e:
            logger.warning(f"Could not train Linear Regression: {str(e)}")

        if not models_dict:
            logger.warning("No models available for evaluation")
            return

        # Initialize evaluation metrics
        evaluator = EvaluationMetrics()

        # 1. Basic metrics evaluation
        logger.info("Computing basic evaluation metrics")
        all_metrics = {}
        for name, model in models_dict.items():
            try:
                y_pred = model.predict(X_test)
                metrics = evaluator.compute_all_metrics(y_test, y_pred)
                all_metrics[name] = metrics
                logger.info(f"{name} metrics computed")
            except Exception as e:
                logger.warning(f"Error computing metrics for {name}: {str(e)}")

        # 2. Regime-specific analysis
        logger.info("Performing regime-specific analysis")
        regime_metrics = {}
        for name, model in models_dict.items():
            try:
                y_pred = model.predict(X_test)
                regime_results = evaluator.regime_specific_metrics(y_test, y_pred, regime_test)
                regime_metrics[name] = regime_results
                logger.info(f"Regime analysis completed for {name}")
            except Exception as e:
                logger.warning(f"Error in regime analysis for {name}: {str(e)}")

        # 3. Temporal consistency
        logger.info("Analyzing temporal consistency")
        temporal_consistency = {}
        for name, model in models_dict.items():
            try:
                y_pred = model.predict(X_test)
                temporal_df = evaluator.temporal_consistency(y_test, y_pred, dates_test, window=30)
                temporal_consistency[name] = temporal_df
                logger.info(f"Temporal consistency analyzed for {name}")
            except Exception as e:
                logger.warning(f"Error in temporal analysis for {name}: {str(e)}")

        # 4. Statistical significance testing
        logger.info("Performing statistical significance tests")
        model_comparison_df = model_comparison_test(models_dict, X_test, y_test)

        # 5. Forecasting performance analysis
        logger.info("Analyzing forecasting performance")
        error_analysis = {}
        skill_scores = {}
        hit_rates = {}

        baseline_pred = np.mean(y_train) * np.ones_like(y_test)  # Naive baseline

        for name, model in models_dict.items():
            try:
                y_pred = model.predict(X_test)

                # Error analysis
                error_stats = analyze_prediction_errors(y_test, y_pred, dates_test)
                error_analysis[name] = error_stats

                # Skill score
                skill = forecast_skill_score(y_test, y_pred, baseline_pred)
                skill_scores[name] = skill

                # Hit rate analysis
                hit_analysis = hit_rate_analysis(y_test, y_pred, percentile_threshold=75)
                hit_rates[name] = hit_analysis

                logger.info(f"Performance analysis completed for {name}")
            except Exception as e:
                logger.warning(f"Error in performance analysis for {name}: {str(e)}")

        # 6. Robustness testing
        logger.info("Testing model robustness")
        stability_results = {}
        backtest_results = {}

        for name, model in models_dict.items():
            try:
                # Stability test
                stability = test_model_stability(model, X_train, y_train, n_splits=5)
                stability_results[name] = stability

                # Walk-forward backtest
                backtest = walk_forward_backtest(model, X, y, train_size=0.7, step=50)
                backtest_results[name] = backtest

                logger.info(f"Robustness testing completed for {name}")
            except Exception as e:
                logger.warning(f"Error in robustness testing for {name}: {str(e)}")

        # 7. Out-of-distribution analysis
        logger.info("Analyzing out-of-distribution performance")
        ood_results = {}

        # Define normal vs extreme periods
        normal_mask = regime_test == 0
        extreme_mask = regime_test == 2

        if normal_mask.sum() > 0 and extreme_mask.sum() > 0:
            X_normal = X_test[normal_mask]
            y_normal = y_test[normal_mask]
            X_extreme = X_test[extreme_mask]
            y_extreme = y_test[extreme_mask]

            for name, model in models_dict.items():
                try:
                    ood_analysis = analyze_out_of_distribution(model, X_normal, y_normal,
                                                             X_extreme, y_extreme)
                    ood_results[name] = ood_analysis
                    logger.info(f"OOD analysis completed for {name}")
                except Exception as e:
                    logger.warning(f"Error in OOD analysis for {name}: {str(e)}")

        # Create visualizations
        if MATPLOTLIB_AVAILABLE:
            logger.info("Creating evaluation visualizations")

            # Residual analysis
            residual_plots = {}
            for name, model in models_dict.items():
                try:
                    y_pred = model.predict(X_test)
                    plot = plot_residuals(y_test, y_pred, dates_test)
                    if plot:
                        residual_plots[name] = plot
                        logger.info(f"Residual plot created for {name}")
                except Exception as e:
                    logger.warning(f"Error creating residual plot for {name}: {str(e)}")

            # Prediction bands (without uncertainty for now)
            prediction_plots = {}
            for name, model in models_dict.items():
                try:
                    y_pred = model.predict(X_test)
                    plot = plot_prediction_bands(y_test, y_pred, None, dates_test)
                    if plot:
                        prediction_plots[name] = plot
                        logger.info(f"Prediction plot created for {name}")
                except Exception as e:
                    logger.warning(f"Error creating prediction plot for {name}: {str(e)}")

            # Regime metrics comparison
            regime_plot = None
            if regime_metrics:
                # Use first model's regime results
                first_model = list(regime_metrics.keys())[0]
                if regime_metrics[first_model]:
                    regime_plot = plot_regime_metrics(regime_metrics[first_model])

            # Temporal evolution
            temporal_plots = {}
            if temporal_consistency:
                first_model = list(temporal_consistency.keys())[0]
                if not temporal_consistency[first_model].empty:
                    temp_plot = plot_metric_evolution(temporal_consistency[first_model], 'RMSE_30d')
                    if temp_plot:
                        temporal_plots['RMSE'] = temp_plot

            # Error distribution by regime
            error_distributions = {}
            for regime in [0, 1, 2]:
                regime_mask = regime_test == regime
                if regime_mask.sum() > 0:
                    error_distributions[f'regime_{regime}'] = []
                    for name, model in models_dict.items():
                        try:
                            y_pred = model.predict(X_test[regime_mask])
                            errors = y_pred - y_test[regime_mask]
                            error_distributions[f'regime_{regime}'].extend(errors)
                        except:
                            continue

            error_plot = plot_prediction_error_distribution(error_distributions)

        # Save results
        logger.info("Saving evaluation results")

        # Save metrics
        if all_metrics:
            metrics_df = pd.DataFrame(all_metrics).T
            metrics_df.to_csv('model_evaluation_metrics.csv')
            logger.info("Model metrics saved")

        # Save regime analysis
        if regime_metrics:
            with open('regime_analysis_results.json', 'w') as f:
                # Convert numpy types to native Python types for JSON
                json_compatible = {}
                for model, regimes in regime_metrics.items():
                    json_compatible[model] = {}
                    for regime, metrics in regimes.items():
                        json_compatible[model][regime] = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                                                        for k, v in metrics.items()}
                json.dump(json_compatible, f, indent=2)
            logger.info("Regime analysis saved")

        # Save model comparison
        if not model_comparison_df.empty:
            model_comparison_df.to_csv('model_comparison_significance.csv', index=False)
            logger.info("Model comparison results saved")

        # Save robustness results
        robustness_summary = {
            'stability': stability_results,
            'backtest': {k: v.get('average_performance', {}) for k, v in backtest_results.items()},
            'ood': ood_results
        }

        with open('robustness_analysis.json', 'w') as f:
            json.dump(robustness_summary, f, indent=2, default=str)
        logger.info("Robustness analysis saved")

        # Save plots
        if MATPLOTLIB_AVAILABLE:
            plot_files = []

            # Residual plots
            for name, plot in residual_plots.items():
                filename = f'residual_analysis_{name.lower()}.png'
                plot.savefig(filename, dpi=150, bbox_inches='tight')
                plot_files.append(filename)

            # Prediction plots
            for name, plot in prediction_plots.items():
                filename = f'prediction_bands_{name.lower()}.png'
                plot.savefig(filename, dpi=150, bbox_inches='tight')
                plot_files.append(filename)

            # Other plots
            if regime_plot:
                regime_plot.savefig('regime_metrics_comparison.png', dpi=150, bbox_inches='tight')
                plot_files.append('regime_metrics_comparison.png')

            if temporal_plots:
                for metric, plot in temporal_plots.items():
                    filename = f'temporal_evolution_{metric.lower()}.png'
                    plot.savefig(filename, dpi=150, bbox_inches='tight')
                    plot_files.append(filename)

            if error_plot:
                error_plot.savefig('error_distribution_by_regime.png', dpi=150, bbox_inches='tight')
                plot_files.append('error_distribution_by_regime.png')

            logger.info(f"Plots saved: {plot_files}")

        # Generate comprehensive evaluation report
        print("\n" + "="*100)
        print("COMPREHENSIVE MODEL EVALUATION REPORT")
        print("="*100)
        print(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Test Data: {len(X_test)} samples, {X_test.shape[1]} features")
        print(f"Models Evaluated: {len(models_dict)}")
        print(f"Regimes Identified: {len(np.unique(regime_test))}")
        print()

        if all_metrics:
            print("MODEL PERFORMANCE METRICS:")
            print("-" * 40)
            for model, metrics in all_metrics.items():
                print(f"\n{model}:")
                print(".4f")
                print(".4f")
                print(".4f")
                print(".1%")
                if 'r2_score' in metrics:
                    print(".4f")
            print()

        if regime_metrics:
            print("REGIME-SPECIFIC PERFORMANCE:")
            print("-" * 35)
            first_model = list(regime_metrics.keys())[0]
            if regime_metrics[first_model]:
                for regime, metrics in regime_metrics[first_model].items():
                    print(f"\n{regime}:")
                    print(".4f")
                    print(".1%")
            print()

        if skill_scores:
            print("FORECAST SKILL SCORES (vs Naive Baseline):")
            print("-" * 45)
            for model, skill in skill_scores.items():
                print(".1%")
            print()

        if stability_results:
            print("MODEL STABILITY ANALYSIS:")
            print("-" * 30)
            for model, stability in stability_results.items():
                stable_status = "STABLE" if stability.get('stable', False) else "UNSTABLE"
                print(".4f")
            print()

        if backtest_results:
            print("WALK-FORWARD BACKTEST RESULTS:")
            print("-" * 35)
            for model, backtest in backtest_results.items():
                avg_perf = backtest.get('average_performance', {})
                if avg_perf:
                    print(f"\n{model}:")
                    print(".4f")
                    print(".1%")
            print()

        if ood_results:
            print("OUT-OF-DISTRIBUTION ANALYSIS:")
            print("-" * 35)
            for model, ood in ood_results.items():
                print(f"\n{model}:")
                print(".1%")
                print(".1%")
            print()

        if not model_comparison_df.empty:
            print("STATISTICAL SIGNIFICANCE TESTING:")
            print("-" * 40)
            significant_comparisons = model_comparison_df[
                (model_comparison_df['dm_significant']) |
                (model_comparison_df['ttest_significant'])
            ]
            if not significant_comparisons.empty:
                for _, row in significant_comparisons.iterrows():
                    print(f"{row['model_1']} vs {row['model_2']}: Significant difference detected")
            else:
                print("No statistically significant differences detected between models")
            print()

        print("EVALUATION SUMMARY:")
        print("-" * 20)
        print(f"✓ Basic metrics computed for {len(all_metrics)} models")
        print(f"✓ Regime analysis completed for {len(regime_metrics)} models")
        print(f"✓ Temporal consistency analyzed for {len(temporal_consistency)} models")
        print(f"✓ Statistical significance tests: {len(model_comparison_df)} comparisons")
        print(f"✓ Robustness testing completed for {len(stability_results)} models")
        print(f"✓ Backtesting performed for {len(backtest_results)} models")
        print(f"✓ Out-of-distribution analysis: {len(ood_results)} models")
        print(f"✓ Visualizations created: {len(residual_plots) + len(prediction_plots) + (1 if regime_plot else 0)} plots")
        print()

        print("FILES GENERATED:")
        print("-" * 20)
        files_generated = [
            'model_evaluation_metrics.csv',
            'regime_analysis_results.json',
            'model_comparison_significance.csv',
            'robustness_analysis.json'
        ]

        if MATPLOTLIB_AVAILABLE:
            plot_files = [f'residual_analysis_{name.lower()}.png' for name in models_dict.keys()]
            plot_files.extend([f'prediction_bands_{name.lower()}.png' for name in models_dict.keys()])
            plot_files.extend(['regime_metrics_comparison.png', 'error_distribution_by_regime.png'])
            files_generated.extend(plot_files)

        for file in files_generated:
            print(f"  • {file}")

        print()
        print("="*100)

    except Exception as e:
        logger.error(f"Error in main evaluation: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()