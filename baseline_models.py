"""
Baseline Linear Models Module
=============================

Implements linear baseline models for financial time series forecasting.
Serves as a benchmark for comparing against advanced AI models.

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

# Required libraries for statistical modeling
try:
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.stats.diagnostic import het_white
    from statsmodels.stats.stattools import durbin_watson, jarque_bera
    from scipy import stats
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    logging.warning("statsmodels not available - linear modeling disabled")

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
    plt.style.use('default')
    sns.set_palette("husl")
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    sns = None
except Exception as e:
    # Handle any other matplotlib initialization issues
    MATPLOTLIB_AVAILABLE = False
    plt = None
    sns = None
    logging.warning(f"Matplotlib initialization failed: {e}")
    logging.warning("matplotlib/seaborn not available - plotting disabled")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LinearBaselineModel:
    """
    Linear baseline model for financial time series forecasting.
    """

    def __init__(self, model_type: str = 'static'):
        """
        Initialize linear baseline model.

        Args:
            model_type (str): 'static' or 'lagged'
        """
        if model_type not in ['static', 'lagged']:
            raise ValueError("model_type must be 'static' or 'lagged'")

        self.model_type = model_type
        self.model = None
        self.fitted_model = None
        self.feature_names = None
        self.target_name = None

        logger.info(f"Initialized LinearBaselineModel with type: {model_type}")

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> 'LinearBaselineModel':
        """
        Fit OLS regression model.

        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target

        Returns:
            LinearBaselineModel: Fitted model
        """
        logger.info(f"Fitting {self.model_type} OLS model")

        if not STATSMODELS_AVAILABLE:
            logger.warning("statsmodels not available - cannot fit model")
            return self

        try:
            # Store feature and target names
            self.feature_names = X_train.columns.tolist()
            self.target_name = y_train.name if y_train.name else 'target'

            # Add constant
            X_train_const = sm.add_constant(X_train)

            # Fit OLS model
            self.fitted_model = sm.OLS(y_train, X_train_const).fit()

            logger.info(f"Model fitted successfully. R² = {self.fitted_model.rsquared:.4f}")
            return self

        except Exception as e:
            logger.error(f"Error fitting model: {str(e)}")
            return self

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on test data.

        Args:
            X_test (pd.DataFrame): Test features

        Returns:
            np.ndarray: Predictions
        """
        if self.fitted_model is None:
            logger.warning("Model not fitted - returning zeros")
            return np.zeros(len(X_test))

        try:
            X_test_const = sm.add_constant(X_test)
            predictions = self.fitted_model.predict(X_test_const)
            return predictions.values

        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            return np.zeros(len(X_test))

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance on test data.

        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target

        Returns:
            Dict[str, float]: Evaluation metrics
        """
        logger.info("Evaluating model performance")

        if self.fitted_model is None:
            logger.warning("Model not fitted - returning default metrics")
            return {
                'r2': 0.0, 'adjusted_r2': 0.0, 'rmse': float('inf'),
                'mae': float('inf'), 'mape': float('inf'),
                'durbin_watson': float('nan'), 'f_statistic': 0.0, 'f_pvalue': 1.0
            }

        try:
            predictions = self.predict(X_test)

            # Basic metrics
            r2 = self.fitted_model.rsquared
            adjusted_r2 = self.fitted_model.rsquared_adj

            # Error metrics
            residuals = y_test.values - predictions
            rmse = np.sqrt(np.mean(residuals**2))
            mae = np.mean(np.abs(residuals))
            mape = np.mean(np.abs(residuals / y_test.values)) * 100

            # Statistical tests
            durbin_watson_stat = durbin_watson(residuals)

            # F-statistic from fitted model
            f_statistic = self.fitted_model.fvalue
            f_pvalue = self.fitted_model.f_pvalue

            results = {
                'r2': r2,
                'adjusted_r2': adjusted_r2,
                'rmse': rmse,
                'mae': mae,
                'mape': mape,
                'durbin_watson': durbin_watson_stat,
                'f_statistic': f_statistic,
                'f_pvalue': f_pvalue
            }

            logger.info(f"Evaluation complete. R² = {r2:.4f}, RMSE = {rmse:.4f}")
            return results

        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            return {
                'r2': 0.0, 'adjusted_r2': 0.0, 'rmse': float('inf'),
                'mae': float('inf'), 'mape': float('inf'),
                'durbin_watson': float('nan'), 'f_statistic': 0.0, 'f_pvalue': 1.0
            }

    def get_coefficients(self) -> pd.DataFrame:
        """
        Get model coefficients with statistical significance.

        Returns:
            pd.DataFrame: Coefficient summary
        """
        if self.fitted_model is None:
            logger.warning("Model not fitted - returning empty DataFrame")
            return pd.DataFrame()

        try:
            # Extract coefficients and statistics
            coef_data = []

            for i, param_name in enumerate(self.fitted_model.params.index):
                if param_name == 'const':
                    variable = 'Constant'
                else:
                    variable = param_name

                coef = self.fitted_model.params[param_name]
                std_err = self.fitted_model.bse[param_name] if param_name in self.fitted_model.bse.index else 0
                t_stat = self.fitted_model.tvalues[param_name] if param_name in self.fitted_model.tvalues.index else 0
                p_value = self.fitted_model.pvalues[param_name] if param_name in self.fitted_model.pvalues.index else 1

                coef_data.append({
                    'Variable': variable,
                    'Coefficient': coef,
                    'Std_Error': std_err,
                    't_stat': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                })

            result_df = pd.DataFrame(coef_data)
            logger.info(f"Coefficients extracted: {len(result_df)} parameters")
            return result_df

        except Exception as e:
            logger.error(f"Error extracting coefficients: {str(e)}")
            return pd.DataFrame()

    def test_assumptions(self) -> Dict[str, Dict[str, Any]]:
        """
        Test classical linear regression assumptions.

        Returns:
            Dict[str, Dict[str, Any]]: Test results
        """
        logger.info("Testing linear regression assumptions")

        if self.fitted_model is None:
            logger.warning("Model not fitted - returning empty results")
            return {}

        try:
            results = {}

            # Get residuals
            residuals = self.fitted_model.resid
            fitted_values = self.fitted_model.fittedvalues

            # 1. Durbin-Watson test for autocorrelation
            dw_stat = durbin_watson(residuals)
            # DW test: 1.5-2.5 suggests no autocorrelation
            dw_passed = 1.5 <= dw_stat <= 2.5

            results['durbin_watson'] = {
                'statistic': dw_stat,
                'p_value': None,  # DW doesn't have p-value
                'passed': dw_passed,
                'interpretation': f"DW = {dw_stat:.3f} (1.5-2.5 = no autocorrelation)"
            }

            # 2. White test for heteroscedasticity
            try:
                exog = self.fitted_model.model.exog
                white_test = het_white(residuals, exog)
                white_stat = white_test[0]
                white_pvalue = white_test[1]
                white_passed = white_pvalue > 0.05

                results['white_test'] = {
                    'statistic': white_stat,
                    'p_value': white_pvalue,
                    'passed': white_passed,
                    'interpretation': f"White test p = {white_pvalue:.4f} (>0.05 = homoscedastic)"
                }
            except Exception as e:
                logger.warning(f"White test failed: {str(e)}")
                results['white_test'] = {'statistic': None, 'p_value': None, 'passed': None}

            # 3. Jarque-Bera test for normality
            try:
                jb_stat, jb_pvalue, _, _ = jarque_bera(residuals)
                jb_passed = jb_pvalue > 0.05

                results['jarque_bera'] = {
                    'statistic': jb_stat,
                    'p_value': jb_pvalue,
                    'passed': jb_passed,
                    'interpretation': f"JB test p = {jb_pvalue:.4f} (>0.05 = normal)"
                }
            except Exception as e:
                logger.warning(f"Jarque-Bera test failed: {str(e)}")
                results['jarque_bera'] = {'statistic': None, 'p_value': None, 'passed': None}

            # 4. Variance Inflation Factor (VIF) for multicollinearity
            try:
                exog = self.fitted_model.model.exog
                vif_data = []

                for i in range(1, exog.shape[1]):  # Skip constant
                    vif = variance_inflation_factor(exog, i)
                    vif_data.append(vif)

                max_vif = max(vif_data) if vif_data else 0
                vif_passed = max_vif < 5  # VIF < 5 is generally acceptable

                results['vif'] = {
                    'statistic': max_vif,
                    'p_value': None,  # VIF doesn't have p-value
                    'passed': vif_passed,
                    'interpretation': f"Max VIF = {max_vif:.2f} (<5 = no multicollinearity)"
                }
            except Exception as e:
                logger.warning(f"VIF calculation failed: {str(e)}")
                results['vif'] = {'statistic': None, 'p_value': None, 'passed': None}

            logger.info("Assumption testing completed")
            return results

        except Exception as e:
            logger.error(f"Error testing assumptions: {str(e)}")
            return {}

    def plot_diagnostics(self) -> Any:
        """
        Create diagnostic plots for regression assumptions.

        Returns:
            matplotlib figure object
        """
        if not MATPLOTLIB_AVAILABLE or self.fitted_model is None:
            logger.warning("Matplotlib not available or model not fitted - skipping diagnostics")
            return None

        logger.info("Creating diagnostic plots")

        try:
            if plt is not None:
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            residuals = self.fitted_model.resid
            fitted_values = self.fitted_model.fittedvalues
            std_residuals = residuals / np.std(residuals)

            # 1. Residuals vs Fitted
            axes[0, 0].scatter(fitted_values, residuals, alpha=0.6)
            axes[0, 0].axhline(y=0, color='red', linestyle='--')
            axes[0, 0].set_xlabel('Fitted Values')
            axes[0, 0].set_ylabel('Residuals')
            axes[0, 0].set_title('Residuals vs Fitted')
            axes[0, 0].grid(True, alpha=0.3)

            # 2. Q-Q Plot
            sm.qqplot(residuals, line='45', ax=axes[0, 1])
            axes[0, 1].set_title('Q-Q Plot')

            # 3. Scale-Location Plot
            sqrt_std_residuals = np.sqrt(np.abs(std_residuals))
            axes[1, 0].scatter(fitted_values, sqrt_std_residuals, alpha=0.6)
            axes[1, 0].set_xlabel('Fitted Values')
            axes[1, 0].set_ylabel('√|Standardized Residuals|')
            axes[1, 0].set_title('Scale-Location Plot')
            axes[1, 0].grid(True, alpha=0.3)

            # 4. Residuals vs Leverage
            try:
                influence = self.fitted_model.get_influence()
                leverage = influence.hat_matrix_diag
                axes[1, 1].scatter(leverage, std_residuals, alpha=0.6)
                axes[1, 1].axhline(y=0, color='red', linestyle='--')
                axes[1, 1].set_xlabel('Leverage')
                axes[1, 1].set_ylabel('Standardized Residuals')
                axes[1, 1].set_title('Residuals vs Leverage')
                axes[1, 1].grid(True, alpha=0.3)
            except Exception as e:
                logger.warning(f"Could not create leverage plot: {str(e)}")
                axes[1, 1].text(0.5, 0.5, 'Leverage plot\nunavailable',
                               ha='center', va='center', transform=axes[1, 1].transAxes)

            if plt is not None:
                plt.tight_layout()
            logger.info("Diagnostic plots created")
            return fig

        except Exception as e:
            logger.error(f"Error creating diagnostic plots: {str(e)}")
            return None

    def plot_predictions_vs_actual(self, X_test: pd.DataFrame, y_test: pd.Series,
                                 dates: Optional[pd.Series] = None) -> Any:
        """
        Plot predictions vs actual values over time.

        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            dates (pd.Series, optional): Date index for x-axis

        Returns:
            matplotlib figure object
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available - skipping predictions plot")
            return None

        logger.info("Creating predictions vs actual plot")

        try:
            if plt is not None:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

            predictions = self.predict(X_test)

            # Use dates if provided, otherwise use index
            if dates is not None:
                x_axis = dates
            else:
                x_axis = range(len(y_test))

            # Plot predictions vs actual
            ax1.plot(x_axis, y_test.values, label='Actual', linewidth=2, alpha=0.8)
            ax1.plot(x_axis, predictions, label='Predicted', linewidth=2, alpha=0.8)
            ax1.set_ylabel(self.target_name if self.target_name else 'Value')
            ax1.set_title(f'{self.model_type.capitalize()} OLS: Predictions vs Actual')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Plot residuals
            residuals = y_test.values - predictions
            ax2.plot(x_axis, residuals, color='red', alpha=0.7)
            ax2.axhline(y=0, color='black', linestyle='--')
            ax2.set_xlabel('Time' if dates is not None else 'Sample')
            ax2.set_ylabel('Residuals')
            ax2.set_title('Residuals Over Time')
            ax2.grid(True, alpha=0.3)

            if plt is not None:
                plt.tight_layout()
            logger.info("Predictions vs actual plot created")
            return fig

        except Exception as e:
            logger.error(f"Error creating predictions plot: {str(e)}")
            return None

    def fit_regime_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                         regime_labels: np.ndarray) -> Dict[int, 'LinearBaselineModel']:
        """
        Fit separate OLS models for each regime.

        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            regime_labels (np.ndarray): Regime labels for each sample

        Returns:
            Dict[int, LinearBaselineModel]: Models for each regime
        """
        logger.info("Fitting regime-specific models")

        unique_regimes = np.unique(regime_labels)
        regime_models = {}

        for regime in unique_regimes:
            try:
                # Filter data for this regime
                regime_mask = regime_labels == regime
                X_regime = X_train[regime_mask]
                y_regime = y_train[regime_mask]

                if len(X_regime) < 10:  # Minimum sample size
                    logger.warning(f"Insufficient data for regime {regime}: {len(X_regime)} samples")
                    continue

                # Fit model for this regime
                regime_model = LinearBaselineModel(model_type=self.model_type)
                regime_model.fit(X_regime, y_regime)
                regime_models[int(regime)] = regime_model

                rsq = getattr(regime_model.fitted_model, 'rsquared', None)
                if rsq is not None:
                    logger.info(f"Regime {regime} model fitted: {len(X_regime)} samples, R² = {rsq:.4f}")
                else:
                    logger.info(f"Regime {regime} model fitted: {len(X_regime)} samples, R² not available")

            except Exception as e:
                logger.error(f"Error fitting regime {regime} model: {str(e)}")

        logger.info(f"Regime-specific modeling completed: {len(regime_models)} models fitted")
        return regime_models

    def predict_regime_aware(self, X_test: pd.DataFrame, regime_labels: np.ndarray) -> np.ndarray:
        """
        Make predictions using regime-specific models.

        Args:
            X_test (pd.DataFrame): Test features
            regime_labels (np.ndarray): Regime labels for each test sample

        Returns:
            np.ndarray: Predictions
        """
        logger.info("Making regime-aware predictions")

        # This would require regime_models to be stored as instance variable
        # For now, return zeros as placeholder
        logger.warning("Regime-aware prediction not implemented - use fit_regime_models separately")
        return np.zeros(len(X_test))

    def plot_coefficients(self) -> Any:
        """
        Plot model coefficients with error bars.

        Returns:
            matplotlib figure object
        """
        if not MATPLOTLIB_AVAILABLE or self.fitted_model is None:
            logger.warning("Matplotlib not available or model not fitted - skipping coefficient plot")
            return None

        logger.info("Creating coefficient plot")

        try:
            coef_df = self.get_coefficients()

            if coef_df.empty:
                logger.warning("No coefficients to plot")
                return None

            if plt is not None:
                fig, ax = plt.subplots(figsize=(10, 6))

            # Filter out constant for better visualization
            plot_data = coef_df[coef_df['Variable'] != 'Constant'].copy()

            if plot_data.empty:
                logger.warning("No non-constant coefficients to plot")
                return None

            # Plot coefficients with error bars
            y_pos = range(len(plot_data))
            coefficients = plot_data['Coefficient']
            errors = plot_data['Std_Error']

            ax.barh(y_pos, coefficients, xerr=errors, capsize=5, alpha=0.7)
            ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)

            # Color significant coefficients
            colors = ['red' if sig else 'blue' for sig in plot_data['significant']]
            for i, bar in enumerate(ax.patches):
                bar.set_color(colors[i])

            ax.set_yticks(y_pos)
            ax.set_yticklabels(plot_data['Variable'])
            ax.set_xlabel('Coefficient Value')
            ax.set_title(f'{self.model_type.capitalize()} OLS Coefficients')

            # Add significance stars
            for i, (coef, p_val) in enumerate(zip(plot_data['Coefficient'], plot_data['p_value'])):
                if p_val < 0.001:
                    star = '***'
                elif p_val < 0.01:
                    star = '**'
                elif p_val < 0.05:
                    star = '*'
                else:
                    star = ''

                ax.text(coef + (errors.iloc[i] if coef > 0 else -errors.iloc[i]),
                       i, star, ha='left' if coef > 0 else 'right', va='center', fontweight='bold')

            if plt is not None:
                plt.tight_layout()
            logger.info("Coefficient plot created")
            return fig

        except Exception as e:
            logger.error(f"Error creating coefficient plot: {str(e)}")
            return None

    def plot_regime_comparison(self, regime_models: Dict[int, 'LinearBaselineModel']) -> Any:
        """
        Compare coefficients across regimes.

        Args:
            regime_models (Dict[int, LinearBaselineModel]): Models for each regime

        Returns:
            matplotlib figure object
        """
        if not MATPLOTLIB_AVAILABLE or not regime_models:
            logger.warning("Matplotlib not available or no regime models - skipping comparison plot")
            return None

        logger.info("Creating regime comparison plot")

        try:
            # Collect coefficients from all regime models
            all_coef_data = []

            for regime, model in regime_models.items():
                coef_df = model.get_coefficients()
                if not coef_df.empty:
                    coef_df = coef_df.copy()
                    coef_df['regime'] = regime
                    all_coef_data.append(coef_df)

            if not all_coef_data:
                logger.warning("No coefficient data to compare")
                return None

            combined_df = pd.concat(all_coef_data, ignore_index=True)

            # Filter to common variables across regimes
            variable_counts = combined_df['Variable'].value_counts()
            common_vars = variable_counts[variable_counts == len(regime_models)].index

            if len(common_vars) == 0:
                logger.warning("No common variables across regimes")
                return None

            plot_data = combined_df[combined_df['Variable'].isin(common_vars)]

            # Create comparison plot
            fig, ax = plt.subplots(figsize=(12, 8))

            # Group by variable and plot
            variables = plot_data['Variable'].unique()
            x_pos = range(len(variables))

            for i, regime in enumerate(sorted(regime_models.keys())):
                regime_data = plot_data[plot_data['regime'] == regime]
                coefficients = []

                for var in variables:
                    coef_row = regime_data[regime_data['Variable'] == var]
                    coef = coef_row['Coefficient'].iloc[0] if len(coef_row) > 0 else 0
                    coefficients.append(coef)

                ax.bar([x + i*0.25 for x in x_pos], coefficients, width=0.25,
                      label=f'Regime {regime}', alpha=0.7)

            ax.set_xticks([x + 0.25 for x in x_pos])
            ax.set_xticklabels(variables, rotation=45, ha='right')
            ax.set_ylabel('Coefficient Value')
            ax.set_title('Coefficient Comparison Across Regimes')
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            logger.info("Regime comparison plot created")
            return fig

        except Exception as e:
            logger.error(f"Error creating regime comparison plot: {str(e)}")
            return None


def create_static_features(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Create features for static OLS model.

    Args:
        data (pd.DataFrame): Input dataframe with VIX, Returns, Unemployment

    Returns:
        Tuple[pd.DataFrame, pd.Series]: X, y for modeling
    """
    logger.info("Creating static OLS features")

    required_cols = ['VIX', 'Returns', 'Unemployment']
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Static model: Returns ~ VIX + Unemployment
    X = data[['VIX', 'Unemployment']].copy()
    y = data['Returns'].copy()

    # Remove any NaN values
    valid_idx = ~(X.isna().any(axis=1) | y.isna())
    X = X[valid_idx]
    y = y[valid_idx]

    logger.info(f"Static features created: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y


def create_lagged_features(data: pd.DataFrame, lags: List[int] = [1, 5]) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Create features for lagged OLS model.

    Args:
        data (pd.DataFrame): Input dataframe
        lags (List[int]): Lag periods to include

    Returns:
        Tuple[pd.DataFrame, pd.Series]: X, y for modeling
    """
    logger.info(f"Creating lagged OLS features with lags: {lags}")

    required_cols = ['VIX', 'Returns', 'Unemployment']
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Create lagged features
    feature_data = []

    for col in required_cols:
        # Current values
        feature_data.append(data[col].rename(f'{col}(t)'))

        # Lagged values
        for lag in lags:
            if col == 'Returns':  # Include lagged returns
                feature_data.append(data[col].shift(lag).rename(f'{col}(t-{lag})'))
            elif col == 'VIX':  # Include one lag for VIX
                if lag == 1:
                    feature_data.append(data[col].shift(lag).rename(f'{col}(t-{lag})'))

    # Combine features
    X = pd.concat(feature_data, axis=1)

    # Target: current returns
    y = data['Returns'].copy()

    # Remove rows with NaN (due to lagging)
    valid_idx = ~(X.isna().any(axis=1) | y.isna())
    X = X[valid_idx]
    y = y[valid_idx]

    logger.info(f"Lagged features created: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y


def main():
    """
    Main function demonstrating baseline linear models.
    """
    logger.info("Starting baseline linear models demonstration")

    try:
        # Load preprocessed data (try multiple sources)
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
            dates = pd.date_range('2020-01-01', periods=500, freq='D')

            # Create correlated time series
            vix_base = 20 + np.random.normal(0, 2, 500)
            vix = pd.Series(np.maximum(5, vix_base), index=dates, name='VIX')

            returns_noise = np.random.normal(0, 0.02, 500)
            returns = pd.Series(returns_noise - 0.5 * (vix - 20) / 100, index=dates, name='Returns')

            unemployment_base = 6.0
            unemployment = []
            for i in range(500):
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

        # Split data: 70% train, 30% test
        train_size = int(len(data) * 0.7)
        train_data = data[:train_size]
        test_data = data[train_size:]

        logger.info(f"Data split: {len(train_data)} train, {len(test_data)} test samples")

        # Create static model
        logger.info("Fitting static OLS model")
        X_train_static, y_train_static = create_static_features(train_data)
        X_test_static, y_test_static = create_static_features(test_data)

        static_model = LinearBaselineModel(model_type='static')
        static_model.fit(X_train_static, y_train_static)

        # Create lagged model
        logger.info("Fitting lagged OLS model")
        X_train_lagged, y_train_lagged = create_lagged_features(train_data)
        X_test_lagged, y_test_lagged = create_lagged_features(test_data)

        lagged_model = LinearBaselineModel(model_type='lagged')
        lagged_model.fit(X_train_lagged, y_train_lagged)

        # Evaluate both models
        logger.info("Evaluating models")
        static_eval = static_model.evaluate(X_test_static, y_test_static)
        lagged_eval = lagged_model.evaluate(X_test_lagged, y_test_lagged)

        # Test assumptions
        logger.info("Testing model assumptions")
        static_assumptions = static_model.test_assumptions()
        lagged_assumptions = lagged_model.test_assumptions()

        # Create diagnostic plots
        if MATPLOTLIB_AVAILABLE:
            logger.info("Creating diagnostic plots")

            # Static model diagnostics
            static_diag_fig = static_model.plot_diagnostics()
            static_pred_fig = static_model.plot_predictions_vs_actual(
                X_test_static, y_test_static, pd.Series(test_data.index[-len(X_test_static):])
            )
            static_coef_fig = static_model.plot_coefficients()

            # Lagged model diagnostics
            lagged_diag_fig = lagged_model.plot_diagnostics()
            lagged_pred_fig = lagged_model.plot_predictions_vs_actual(
                X_test_lagged, y_test_lagged, pd.Series(test_data.index[-len(X_test_lagged):])
            )
            lagged_coef_fig = lagged_model.plot_coefficients()

        # Regime-specific modeling
        logger.info("Fitting regime-specific models")
        # Create simple regimes based on VIX levels
        train_regimes = np.where(train_data['VIX'] > train_data['VIX'].quantile(0.67), 2,
                                np.where(train_data['VIX'] > train_data['VIX'].quantile(0.33), 1, 0))

        # Align regimes with training data
        train_regime_mask = ~X_train_static.isna().any(axis=1)
        train_regimes_aligned = train_regimes[:len(train_regime_mask)][train_regime_mask]

        static_regime_models = static_model.fit_regime_models(
            X_train_static, y_train_static, train_regime_mask.to_numpy()
        )

        if MATPLOTLIB_AVAILABLE and static_regime_models:
            regime_comp_fig = static_model.plot_regime_comparison(static_regime_models)

        # Save results
        logger.info("Saving results")

        # Save model coefficients
        static_coefs = static_model.get_coefficients()
        lagged_coefs = lagged_model.get_coefficients()

        static_coefs.to_csv('static_model_coefficients.csv', index=False)
        lagged_coefs.to_csv('lagged_model_coefficients.csv', index=False)

        # Save evaluation results
        eval_results = pd.DataFrame({
            'Metric': list(static_eval.keys()),
            'Static_OLS': list(static_eval.values()),
            'Lagged_OLS': list(lagged_eval.values())
        })
        eval_results.to_csv('baseline_model_evaluation.csv', index=False)

        # Save plots
        if MATPLOTLIB_AVAILABLE:
            if 'static_diag_fig' in locals() and static_diag_fig:
                static_diag_fig.savefig('static_model_diagnostics.png', dpi=300, bbox_inches='tight')
            if 'static_pred_fig' in locals() and static_pred_fig:
                static_pred_fig.savefig('static_model_predictions.png', dpi=300, bbox_inches='tight')
            if 'static_coef_fig' in locals() and static_coef_fig:
                static_coef_fig.savefig('static_model_coefficients.png', dpi=300, bbox_inches='tight')

            if 'lagged_diag_fig' in locals() and lagged_diag_fig:
                lagged_diag_fig.savefig('lagged_model_diagnostics.png', dpi=300, bbox_inches='tight')
            if 'lagged_pred_fig' in locals() and lagged_pred_fig:
                lagged_pred_fig.savefig('lagged_model_predictions.png', dpi=300, bbox_inches='tight')
            if 'lagged_coef_fig' in locals() and lagged_coef_fig:
                lagged_coef_fig.savefig('lagged_model_coefficients.png', dpi=300, bbox_inches='tight')

            if 'regime_comp_fig' in locals() and regime_comp_fig:
                regime_comp_fig.savefig('regime_coefficient_comparison.png', dpi=300, bbox_inches='tight')

        # Print summary
        print("\n" + "="*80)
        print("BASELINE LINEAR MODELS ANALYSIS COMPLETED")
        print("="*80)
        print(f"Data: {len(data)} total samples")
        print(f"Train/Test split: {len(train_data)}/{len(test_data)}")
        print()

        print("MODEL PERFORMANCE COMPARISON:")
        print("-" * 40)
        print("<15")
        print(".4f")
        print(".4f")
        print(".4f")
        print(".4f")
        print(".4f")
        print(".4f")
        print()

        print("STATIC OLS COEFFICIENTS:")
        print("-" * 30)
        if not static_coefs.empty:
            for _, row in static_coefs.iterrows():
                sig_star = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
                print(f"{row['Variable']:<15} {row['Coefficient']:+.4f} ({row['p_value']:.4f}) {sig_star}")
        print()

        print("LAGGED OLS COEFFICIENTS:")
        print("-" * 30)
        if not lagged_coefs.empty:
            for _, row in lagged_coefs.iterrows():
                sig_star = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
                print(f"{row['Variable']:<15} {row['Coefficient']:+.4f} ({row['p_value']:.4f}) {sig_star}")
        print()

        print("ASSUMPTION TESTS:")
        print("-" * 20)
        print("Static Model:")
        for test_name, result in static_assumptions.items():
            status = "✓ PASS" if result.get('passed') else "✗ FAIL"
            stat = ".3f" if result.get('statistic') else "N/A"
            print(f"  {test_name}: {status} ({stat})")

        print("Lagged Model:")
        for test_name, result in lagged_assumptions.items():
            status = "✓ PASS" if result.get('passed') else "✗ FAIL"
            stat = ".3f" if result.get('statistic') else "N/A"
            print(f"  {test_name}: {status} ({stat})")
        print()

        print(f"Files saved: 3 CSV files, 6 PNG plots")
        print("="*80)

    except Exception as e:
        logger.error(f"Error in main analysis: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()