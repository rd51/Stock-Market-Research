"""
Comparison Report Generator
===========================

Comprehensive model comparison and reporting system for financial forecasting models.
Generates professional reports, visualizations, and dashboard-ready data.

Author: GitHub Copilot
Date: February 8, 2026
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings
import os
warnings.filterwarnings('ignore')

# Required libraries for reporting
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
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logging.warning("plotly not available - interactive plots disabled")

try:
    from sklearn.metrics import (
        mean_absolute_error, mean_squared_error, r2_score
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available - basic metrics disabled")

try:
    import json
    JSON_AVAILABLE = True
except ImportError:
    JSON_AVAILABLE = False
    logging.warning("json not available - JSON output disabled")

try:
    import tabulate
    TABULATE_AVAILABLE = True
except ImportError:
    TABULATE_AVAILABLE = False
    logging.warning("tabulate not available - markdown tables disabled")

try:
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False
    logging.warning("openpyxl not available - Excel export disabled")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ComparisonReportGenerator:
    """
    Comprehensive model comparison and reporting system.
    """

    def __init__(self, models_dict: Optional[Dict[str, Any]] = None,
                 model_names: List[str] = ['OLS', 'LSTM', 'RF', 'XGBoost', 'LightGBM', 'Ensemble']):
        """
        Initialize the comparison report generator.

        Args:
            models_dict: Dictionary of trained models {name: model}
            model_names: List of model names for consistent ordering
        """
        self.models_dict = models_dict or {}
        self.model_names = model_names
        self.comparison_data = {}
        self.plots = {}

        logger.info(f"Initialized ComparisonReportGenerator with {len(self.models_dict)} models")

    def create_summary_table(self, X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
        """
        Create comprehensive comparison table with key metrics.

        Args:
            X_test: Test features
            y_test: Test targets

        Returns:
            DataFrame with model comparison summary
        """
        logger.info("Creating summary comparison table")

        results = []

        for model_name in self.model_names:
            if model_name not in self.models_dict:
                logger.warning(f"Model {model_name} not found in models_dict")
                continue

            model = self.models_dict[model_name]

            try:
                # Time the prediction
                start_time = datetime.now()
                y_pred = model.predict(X_test)
                prediction_time = (datetime.now() - start_time).total_seconds()

                # Calculate metrics
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)

                # MAPE (Mean Absolute Percentage Error)
                mask = y_test != 0
                mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100 if mask.any() else np.nan

                r2 = r2_score(y_test, y_pred)

                # Direction accuracy
                true_direction = np.sign(np.diff(y_test))
                pred_direction = np.sign(np.diff(y_pred))
                direction_acc = np.mean(true_direction == pred_direction) * 100

                # Training time (placeholder - would need to be tracked during training)
                training_time = getattr(model, '_training_time', np.nan)

                results.append({
                    'Model': model_name,
                    'RMSE': rmse,
                    'MAE': mae,
                    'MAPE': mape,
                    'R²': r2,
                    'Direction_Acc': direction_acc,
                    'Training_Time_Sec': training_time,
                    'Prediction_Time_Sec': prediction_time
                })

                logger.info(f"Evaluated {model_name}: RMSE={rmse:.4f}, Direction_Acc={direction_acc:.1f}%")

            except Exception as e:
                logger.warning(f"Error evaluating {model_name}: {str(e)}")
                results.append({
                    'Model': model_name,
                    'RMSE': np.nan,
                    'MAE': np.nan,
                    'MAPE': np.nan,
                    'R²': np.nan,
                    'Direction_Acc': np.nan,
                    'Training_Time_Sec': np.nan,
                    'Prediction_Time_Sec': np.nan
                })

        df = pd.DataFrame(results)

        # Sort by RMSE (best to worst)
        if not df.empty and 'RMSE' in df.columns:
            df = df.sort_values('RMSE').reset_index(drop=True)

        logger.info(f"Created summary table with {len(df)} models")
        return df

    def create_regime_breakdown(self, X_test: np.ndarray, y_test: np.ndarray,
                               regime_labels: np.ndarray) -> Dict[str, pd.DataFrame]:
        """
        Create separate comparison tables for each regime.

        Args:
            X_test: Test features
            y_test: Test targets
            regime_labels: Array of regime labels

        Returns:
            Dictionary with comparison tables for each regime
        """
        logger.info("Creating regime-specific comparison tables")

        unique_regimes = np.unique(regime_labels)
        regime_comparisons = {}

        for regime in unique_regimes:
            mask = regime_labels == regime
            if mask.sum() == 0:
                continue

            X_regime = X_test[mask]
            y_regime = y_test[mask]

            regime_table = self.create_summary_table(X_regime, y_regime)
            regime_comparisons[f'regime_{regime}'] = regime_table

            logger.info(f"Created comparison for regime {regime}: {mask.sum()} samples")

        logger.info(f"Created regime breakdowns for {len(regime_comparisons)} regimes")
        return regime_comparisons

    def create_detailed_comparison(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Create extended metrics for each model.

        Args:
            X_test: Test features
            y_test: Test targets

        Returns:
            Dictionary with detailed metrics for each model
        """
        logger.info("Creating detailed model comparison")

        detailed_metrics = {}

        for model_name in self.model_names:
            if model_name not in self.models_dict:
                continue

            model = self.models_dict[model_name]

            try:
                y_pred = model.predict(X_test)
                errors = y_pred - y_test

                # Basic metrics
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                # MAPE
                mask = y_test != 0
                mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100 if mask.any() else np.nan

                # Direction accuracy
                if len(y_test) > 1:
                    true_direction = np.sign(np.diff(y_test))
                    pred_direction = np.sign(np.diff(y_pred))
                    direction_acc = np.mean(true_direction == pred_direction) * 100
                else:
                    direction_acc = np.nan

                # Additional metrics
                bias = np.mean(errors)  # Mean error (bias)
                error_std = np.std(errors)  # Error variability
                max_error = np.max(np.abs(errors))  # Worst case error

                # Skill score (assuming naive baseline of mean prediction)
                baseline_pred = np.full_like(y_test, np.mean(y_test))
                baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))
                skill_score = (baseline_rmse - rmse) / baseline_rmse if baseline_rmse > 0 else 0

                # Theil's U statistic (forecast accuracy relative to naive forecast)
                naive_errors = np.abs(np.diff(y_test))  # Naive forecast errors
                forecast_errors = np.abs(errors[1:])  # Model forecast errors
                theil_u = np.sqrt(np.mean(forecast_errors**2)) / np.sqrt(np.mean(naive_errors**2)) if len(naive_errors) > 0 else np.nan

                detailed_metrics[model_name] = {
                    'RMSE': rmse,
                    'MAE': mae,
                    'MAPE': mape,
                    'R²': r2,
                    'Direction_Accuracy': direction_acc,
                    'Skill_Score': skill_score,
                    'Bias': bias,
                    'Error_Std': error_std,
                    'Max_Error': max_error,
                    'Theil_U': theil_u,
                    'Sample_Size': len(y_test)
                }

                logger.info(f"Detailed metrics for {model_name}: RMSE={rmse:.4f}, Skill={skill_score:.3f}")

            except Exception as e:
                logger.warning(f"Error in detailed comparison for {model_name}: {str(e)}")
                detailed_metrics[model_name] = {}

        logger.info(f"Created detailed comparison for {len(detailed_metrics)} models")
        return detailed_metrics

    def statistical_significance_matrix(self, X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
        """
        Create pairwise statistical significance matrix.

        Args:
            X_test: Test features
            y_test: Test targets

        Returns:
            DataFrame with p-values for pairwise comparisons
        """
        try:
            from model_evaluation import diebold_mariano_test
            DM_AVAILABLE = True
        except ImportError:
            DM_AVAILABLE = False
            logger.warning("Diebold-Mariano test not available")

        if not DM_AVAILABLE:
            logger.warning("Cannot create significance matrix - DM test unavailable")
            return pd.DataFrame()

        logger.info("Creating statistical significance matrix")

        # Get predictions for all models
        predictions = {}
        for model_name in self.model_names:
            if model_name in self.models_dict:
                try:
                    predictions[model_name] = self.models_dict[model_name].predict(X_test)
                except Exception as e:
                    logger.warning(f"Error getting predictions for {model_name}: {str(e)}")

        if len(predictions) < 2:
            logger.warning("Need at least 2 models for significance testing")
            return pd.DataFrame()

        # Create matrix
        model_list = list(predictions.keys())
        n_models = len(model_list)
        p_matrix = np.full((n_models, n_models), np.nan)

        for i in range(n_models):
            for j in range(n_models):
                if i != j:
                    errors_i = predictions[model_list[i]] - y_test
                    errors_j = predictions[model_list[j]] - y_test

                    dm_result = diebold_mariano_test(errors_i, errors_j)
                    p_matrix[i, j] = dm_result.get('p_value', np.nan)

        # Create DataFrame
        df = pd.DataFrame(p_matrix, index=model_list, columns=model_list)

        logger.info(f"Created significance matrix for {n_models} models")
        return df

    def create_all_comparison_plots(self, comparison_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate all comparison plots.

        Args:
            comparison_data: Dictionary with comparison results

        Returns:
            Dictionary of plot figures
        """
        logger.info("Creating all comparison plots")

        plots = {}

        if 'summary_table' in comparison_data:
            summary_table = comparison_data['summary_table']

            # Basic comparison plots
            plots['rmse_comparison'] = self.plot_rmse_comparison(summary_table)
            plots['mae_comparison'] = self.plot_mae_comparison(summary_table)
            plots['direction_accuracy_comparison'] = self.plot_direction_accuracy_comparison(summary_table)
            plots['all_metrics_radar'] = self.plot_all_metrics_comparison(summary_table)

        if 'regime_breakdown' in comparison_data:
            plots['regime_breakdown'] = self.plot_regime_breakdown(comparison_data['regime_breakdown'])

        if 'predictions' in comparison_data and 'dates' in comparison_data:
            plots['prediction_overlay'] = self.create_prediction_overlay_plot(
                comparison_data.get('y_true'),
                comparison_data['predictions'],
                comparison_data['dates'],
                comparison_data.get('regime_labels')
            )

        logger.info(f"Created {len(plots)} comparison plots")
        return plots

    def plot_rmse_comparison(self, summary_table: pd.DataFrame, figsize: Tuple[int, int] = (10, 6)) -> Any:
        """
        Create RMSE comparison bar chart.

        Args:
            summary_table: Summary comparison table
            figsize: Figure size

        Returns:
            matplotlib figure
        """
        if not MATPLOTLIB_AVAILABLE or summary_table.empty:
            logger.warning("Cannot create RMSE plot - matplotlib or data unavailable")
            return None

        try:
            fig, ax = plt.subplots(figsize=figsize)

            # Sort by RMSE for better visualization
            df_plot = summary_table.dropna(subset=['RMSE']).sort_values('RMSE')

            # Create color gradient (green to red)
            n_models = len(df_plot)
            colors = plt.cm.RdYlGn_r(np.linspace(0, 1, n_models))

            bars = ax.barh(df_plot['Model'], df_plot['RMSE'], color=colors, alpha=0.8)

            ax.set_title('Model RMSE Comparison', fontsize=16, fontweight='bold')
            ax.set_xlabel('RMSE (Lower is Better)', fontsize=12)
            ax.set_ylabel('Model', fontsize=12)
            ax.grid(True, alpha=0.3)

            # Add value labels
            for bar, value in zip(bars, df_plot['RMSE']):
                ax.text(bar.get_width() + max(df_plot['RMSE']) * 0.01,
                       bar.get_y() + bar.get_height()/2,
                       '.4f', ha='left', va='center', fontsize=10)

            plt.tight_layout()
            logger.info("Created RMSE comparison plot")
            return fig

        except Exception as e:
            logger.error(f"Error creating RMSE plot: {str(e)}")
            return None

    def plot_mae_comparison(self, summary_table: pd.DataFrame) -> Any:
        """
        Create MAE comparison bar chart.

        Args:
            summary_table: Summary comparison table

        Returns:
            matplotlib figure
        """
        if not MATPLOTLIB_AVAILABLE or summary_table.empty:
            return None

        try:
            fig, ax = plt.subplots(figsize=(10, 6))

            df_plot = summary_table.dropna(subset=['MAE']).sort_values('MAE')
            n_models = len(df_plot)
            colors = plt.cm.RdYlGn_r(np.linspace(0, 1, n_models))

            bars = ax.barh(df_plot['Model'], df_plot['MAE'], color=colors, alpha=0.8)

            ax.set_title('Model MAE Comparison', fontsize=16, fontweight='bold')
            ax.set_xlabel('MAE (Lower is Better)', fontsize=12)
            ax.set_ylabel('Model', fontsize=12)
            ax.grid(True, alpha=0.3)

            for bar, value in zip(bars, df_plot['MAE']):
                ax.text(bar.get_width() + max(df_plot['MAE']) * 0.01,
                       bar.get_y() + bar.get_height()/2,
                       '.4f', ha='left', va='center', fontsize=10)

            plt.tight_layout()
            return fig

        except Exception as e:
            logger.error(f"Error creating MAE plot: {str(e)}")
            return None

    def plot_direction_accuracy_comparison(self, summary_table: pd.DataFrame) -> Any:
        """
        Create direction accuracy comparison bar chart.

        Args:
            summary_table: Summary comparison table

        Returns:
            matplotlib figure
        """
        if not MATPLOTLIB_AVAILABLE or summary_table.empty:
            return None

        try:
            fig, ax = plt.subplots(figsize=(10, 6))

            df_plot = summary_table.dropna(subset=['Direction_Acc']).sort_values('Direction_Acc', ascending=False)
            n_models = len(df_plot)
            colors = plt.cm.RdYlGn(np.linspace(0, 1, n_models))  # Green to red, higher accuracy = greener

            bars = ax.barh(df_plot['Model'], df_plot['Direction_Acc'], color=colors, alpha=0.8)

            ax.set_title('Model Direction Accuracy Comparison', fontsize=16, fontweight='bold')
            ax.set_xlabel('Direction Accuracy (%)', fontsize=12)
            ax.set_ylabel('Model', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 100)

            for bar, value in zip(bars, df_plot['Direction_Acc']):
                ax.text(bar.get_width() + 1,
                       bar.get_y() + bar.get_height()/2,
                       '.1f', ha='left', va='center', fontsize=10)

            plt.tight_layout()
            return fig

        except Exception as e:
            logger.error(f"Error creating direction accuracy plot: {str(e)}")
            return None

    def plot_all_metrics_comparison(self, summary_table: pd.DataFrame) -> Any:
        """
        Create radar chart comparing all metrics.

        Args:
            summary_table: Summary comparison table

        Returns:
            matplotlib figure
        """
        if not MATPLOTLIB_AVAILABLE or summary_table.empty:
            return None

        try:
            # Prepare data for radar chart
            df_radar = summary_table.dropna().copy()

            if df_radar.empty or len(df_radar) < 3:
                logger.warning("Not enough data for radar chart")
                return None

            # Normalize metrics (0-1 scale, where 1 is best)
            metrics_to_plot = ['RMSE', 'MAE', 'Direction_Acc', 'R²']
            df_normalized = df_radar.copy()

            # For error metrics (lower is better), invert the scale
            for metric in ['RMSE', 'MAE']:
                if metric in df_normalized.columns:
                    min_val = df_normalized[metric].min()
                    max_val = df_normalized[metric].max()
                    if max_val > min_val:
                        df_normalized[metric] = 1 - (df_normalized[metric] - min_val) / (max_val - min_val)

            # For R² (higher is better), keep as is
            if 'R²' in df_normalized.columns:
                min_val = df_normalized['R²'].min()
                max_val = df_normalized['R²'].max()
                if max_val > min_val:
                    df_normalized['R²'] = (df_normalized['R²'] - min_val) / (max_val - min_val)

            # For Direction_Acc (higher is better), normalize to 0-1
            if 'Direction_Acc' in df_normalized.columns:
                df_normalized['Direction_Acc'] = df_normalized['Direction_Acc'] / 100

            # Create radar chart
            fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))

            angles = np.linspace(0, 2 * np.pi, len(metrics_to_plot), endpoint=False).tolist()
            angles += angles[:1]  # Close the loop

            for _, row in df_normalized.iterrows():
                values = [row[metric] for metric in metrics_to_plot if metric in row.index]
                if len(values) == len(metrics_to_plot):
                    values += values[:1]  # Close the loop
                    ax.plot(angles, values, 'o-', linewidth=2, label=row['Model'])
                    ax.fill(angles, values, alpha=0.25)

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metrics_to_plot)
            ax.set_ylim(0, 1)
            ax.set_title('Model Performance Profile (Radar Chart)', fontsize=16, fontweight='bold', pad=20)
            ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
            ax.grid(True)

            plt.tight_layout()
            return fig

        except Exception as e:
            logger.error(f"Error creating radar plot: {str(e)}")
            return None

    def plot_regime_breakdown(self, regime_comparison: Dict[str, pd.DataFrame]) -> Any:
        """
        Create faceted plot showing performance by regime.

        Args:
            regime_comparison: Dictionary of regime-specific comparison tables

        Returns:
            matplotlib figure
        """
        if not MATPLOTLIB_AVAILABLE or not regime_comparison:
            return None

        try:
            n_regimes = len(regime_comparison)
            if n_regimes == 0:
                return None

            fig, axes = plt.subplots(n_regimes, 1, figsize=(12, 4*n_regimes))
            if n_regimes == 1:
                axes = [axes]

            for i, (regime_name, df) in enumerate(regime_comparison.items()):
                ax = axes[i]

                if df.empty:
                    ax.text(0.5, 0.5, f'No data for {regime_name}',
                           ha='center', va='center', transform=ax.transAxes)
                    continue

                df_plot = df.dropna(subset=['RMSE']).head(5)  # Top 5 models
                if df_plot.empty:
                    continue

                x = np.arange(len(df_plot))
                width = 0.35

                ax.bar(x - width/2, df_plot['RMSE'], width, label='RMSE', alpha=0.8)
                ax.bar(x + width/2, df_plot['MAE'], width, label='MAE', alpha=0.8)

                ax.set_title(f'{regime_name.replace("_", " ").title()} - Model Performance', fontsize=14)
                ax.set_xticks(x)
                ax.set_xticklabels(df_plot['Model'], rotation=45, ha='right')
                ax.set_ylabel('Error')
                ax.legend()
                ax.grid(True, alpha=0.3)

            plt.tight_layout()
            return fig

        except Exception as e:
            logger.error(f"Error creating regime breakdown plot: {str(e)}")
            return None

    def create_prediction_overlay_plot(self, y_true: np.ndarray,
                                     predictions_dict: Dict[str, np.ndarray],
                                     dates: Optional[np.ndarray] = None,
                                     regime_labels: Optional[np.ndarray] = None,
                                     figsize: Tuple[int, int] = (16, 8)) -> Any:
        """
        Create overlay plot showing predictions from all models.

        Args:
            y_true: True values
            predictions_dict: Dictionary of predictions {model_name: predictions}
            dates: Date array for x-axis
            regime_labels: Regime labels for background shading
            figsize: Figure size

        Returns:
            matplotlib figure
        """
        if not MATPLOTLIB_AVAILABLE:
            return None

        try:
            fig, ax = plt.subplots(figsize=figsize)

            # Create x-axis
            if dates is None:
                x = np.arange(len(y_true))
            else:
                x = dates

            # Plot actual values
            ax.plot(x, y_true, 'k-', linewidth=3, label='Actual', alpha=0.9)

            # Plot predictions from each model
            colors = plt.cm.tab10(np.linspace(0, 1, len(predictions_dict)))
            for i, (model_name, y_pred) in enumerate(predictions_dict.items()):
                if len(y_pred) == len(y_true):
                    ax.plot(x, y_pred, color=colors[i], linewidth=2,
                           label=model_name, alpha=0.8)

            # Add regime background shading if available
            if regime_labels is not None:
                unique_regimes = np.unique(regime_labels)
                regime_colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']

                for i, regime in enumerate(unique_regimes):
                    mask = regime_labels == regime
                    if mask.any():
                        # Find contiguous segments
                        diff = np.diff(np.r_[False, mask, False].astype(int))
                        starts = np.where(diff == 1)[0]
                        ends = np.where(diff == -1)[0]

                        for start, end in zip(starts, ends):
                            ax.axvspan(x[start], x[min(end, len(x)-1)],
                                     alpha=0.2, color=regime_colors[i % len(regime_colors)],
                                     label=f'Regime {regime}' if i == 0 else "")

            ax.set_title('Model Predictions Comparison', fontsize=16, fontweight='bold')
            ax.set_xlabel('Time' if dates is None else 'Date')
            ax.set_ylabel('Value')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            return fig

        except Exception as e:
            logger.error(f"Error creating prediction overlay plot: {str(e)}")
            return None

    def generate_markdown_report(self, comparison_data: Dict[str, Any]) -> str:
        """
        Generate markdown-formatted comparison report.

        Args:
            comparison_data: Dictionary with all comparison results

        Returns:
            Markdown string
        """
        logger.info("Generating markdown report")

        report_lines = []

        # Header
        report_lines.extend([
            "# Financial Forecasting Model Comparison Report",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "---",
            ""
        ])

        # Summary table
        if 'summary_table' in comparison_data:
            summary_table = comparison_data['summary_table']

            report_lines.extend([
                "## Model Performance Summary",
                "",
                "Key metrics comparison across all models:",
                "",
            ])

            if TABULATE_AVAILABLE:
                report_lines.append(summary_table.to_markdown(index=False))
            else:
                # Fallback: simple text table
                report_lines.append("| Model | RMSE | MAE | MAPE | R² | Direction_Acc |")
                report_lines.append("|-------|------|-----|------|----|---------------|")
                for _, row in summary_table.iterrows():
                    report_lines.append(f"| {row['Model']} | {row['RMSE']:.4f} | {row['MAE']:.4f} | {row['MAPE']:.2f} | {row['R²']:.4f} | {row['Direction_Acc']:.1f}% |")

            report_lines.extend([
                "",
                "---",
                ""
            ])

        # Key findings
        report_lines.extend([
            "## Key Findings",
            "",
        ])

        if 'summary_table' in comparison_data:
            df = comparison_data['summary_table']
            if not df.empty:
                # Best RMSE
                best_rmse = df.loc[df['RMSE'].idxmin()]
                report_lines.append(f"- **Best RMSE:** {best_rmse['Model']} ({best_rmse['RMSE']:.4f})")

                # Best direction accuracy
                best_dir = df.loc[df['Direction_Acc'].idxmax()]
                report_lines.append(f"- **Best Direction Accuracy:** {best_dir['Model']} ({best_dir['Direction_Acc']:.1f}%)")

                # Best R²
                best_r2 = df.loc[df['R²'].idxmax()]
                report_lines.append(f"- **Best R² Score:** {best_r2['Model']} ({best_r2['R²']:.4f})")

        report_lines.extend([
            "",
            "---",
            ""
        ])

        # Model rankings
        report_lines.extend([
            "## Model Rankings",
            "",
            "### By RMSE (Lower is Better)",
            ""
        ])

        if 'summary_table' in comparison_data:
            df = comparison_data['summary_table'].sort_values('RMSE')
            for i, (_, row) in enumerate(df.iterrows(), 1):
                report_lines.append(f"{i}. **{row['Model']}** - RMSE: {row['RMSE']:.4f}")

        report_lines.extend([
            "",
            "### By Direction Accuracy (Higher is Better)",
            ""
        ])

        if 'summary_table' in comparison_data:
            df = comparison_data['summary_table'].sort_values('Direction_Acc', ascending=False)
            for i, (_, row) in enumerate(df.iterrows(), 1):
                report_lines.append(f"{i}. **{row['Model']}** - Direction Acc: {row['Direction_Acc']:.1f}%")

        report_lines.extend([
            "",
            "---",
            ""
        ])

        # Recommendations
        report_lines.extend([
            "## Recommendations",
            "",
        ])

        if 'summary_table' in comparison_data:
            df = comparison_data['summary_table']
            if not df.empty:
                # Overall recommendation
                best_overall = self.recommend_best_model(comparison_data, criteria='balanced')
                report_lines.append(f"- **Recommended Model:** {best_overall}")
                report_lines.append("- **Rationale:** Best balance of accuracy and directional prediction")

                # Ensemble weights
                weights = self.recommend_ensemble_weights(comparison_data)
                if weights:
                    report_lines.append("- **Ensemble Weights:**")
                    for model, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
                        report_lines.append(f"  - {model}: {weight:.3f}")

        report_lines.extend([
            "",
            "---",
            ""
        ])

        # Caveats
        report_lines.extend([
            "## Important Caveats",
            "",
            "- Results are based on the specific test dataset and time period",
            "- Model performance may vary with different market conditions",
            "- Direction accuracy is critical for trading applications",
            "- Consider transaction costs and slippage in real trading",
            "- Regular model retraining is recommended",
            "",
            "---",
            "",
            "*Report generated automatically by ComparisonReportGenerator*"
        ])

        report = "\n".join(report_lines)
        logger.info("Generated markdown report")
        return report

    def generate_html_report(self, comparison_data: Dict[str, Any], output_path: str) -> None:
        """
        Generate HTML-formatted report with inline plots.

        Args:
            comparison_data: Dictionary with all comparison results
            output_path: Path to save HTML report
        """
        logger.info(f"Generating HTML report: {output_path}")

        try:
            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Financial Forecasting Model Comparison Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 40px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-left: 4px solid #3498db;
            padding-left: 15px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #f8f9fa;
            font-weight: bold;
        }}
        tr:nth-child(even) {{
            background-color: #f8f9fa;
        }}
        .metric-box {{
            background-color: #ecf0f1;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
            border-left: 4px solid #3498db;
        }}
        .recommendation {{
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
            padding: 15px;
            margin: 20px 0;
            border-radius: 5px;
        }}
        .plot-container {{
            margin: 30px 0;
            text-align: center;
        }}
        .plot-container img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Financial Forecasting Model Comparison Report</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
"""

            # Summary table
            if 'summary_table' in comparison_data:
                html_content += """
        <h2>Model Performance Summary</h2>
        <table>
            <thead>
                <tr>
                    <th>Model</th>
                    <th>RMSE</th>
                    <th>MAE</th>
                    <th>MAPE</th>
                    <th>R²</th>
                    <th>Direction Accuracy</th>
                </tr>
            </thead>
            <tbody>
"""

                for _, row in comparison_data['summary_table'].iterrows():
                    html_content += f"""
                <tr>
                    <td>{row['Model']}</td>
                    <td>{row['RMSE']:.4f}</td>
                    <td>{row['MAE']:.4f}</td>
                    <td>{row['MAPE']:.2f}%</td>
                    <td>{row['R²']:.4f}</td>
                    <td>{row['Direction_Acc']:.1f}%</td>
                </tr>
"""

                html_content += """
            </tbody>
        </table>
"""

            # Key findings
            html_content += """
        <h2>Key Findings</h2>
        <div class="metric-box">
"""

            if 'summary_table' in comparison_data:
                df = comparison_data['summary_table']
                if not df.empty:
                    best_rmse = df.loc[df['RMSE'].idxmin()]
                    best_dir = df.loc[df['Direction_Acc'].idxmax()]
                    best_r2 = df.loc[df['R²'].idxmax()]

                    html_content += f"""
            <p><strong>Best RMSE:</strong> {best_rmse['Model']} ({best_rmse['RMSE']:.4f})</p>
            <p><strong>Best Direction Accuracy:</strong> {best_dir['Model']} ({best_dir['Direction_Acc']:.1f}%)</p>
            <p><strong>Best R² Score:</strong> {best_r2['Model']} ({best_r2['R²']:.4f})</p>
"""

            html_content += """
        </div>
"""

            # Recommendations
            html_content += """
        <h2>Recommendations</h2>
        <div class="recommendation">
"""

            if 'summary_table' in comparison_data:
                best_model = self.recommend_best_model(comparison_data, criteria='balanced')
                html_content += f"""
            <p><strong>Recommended Model:</strong> {best_model}</p>
            <p><em>Rationale: Best balance of accuracy and directional prediction</em></p>
"""

                weights = self.recommend_ensemble_weights(comparison_data)
                if weights:
                    html_content += "<p><strong>Ensemble Weights:</strong></p><ul>"
                    for model, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
                        html_content += f"<li>{model}: {weight:.3f}</li>"
                    html_content += "</ul>"

            html_content += """
        </div>
"""

            # Placeholder for plots (would need to save plots separately and embed)
            html_content += """
        <h2>Performance Visualizations</h2>
        <p>Plots would be embedded here in a full implementation.</p>
"""

            # Footer
            html_content += """
        <hr>
        <p><em>Report generated automatically by ComparisonReportGenerator</em></p>
    </div>
</body>
</html>
"""

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

            logger.info(f"HTML report saved to {output_path}")

        except Exception as e:
            logger.error(f"Error generating HTML report: {str(e)}")

    def generate_latex_table(self, comparison_table: pd.DataFrame) -> str:
        """
        Generate LaTeX-formatted table for research papers.

        Args:
            comparison_table: Comparison table DataFrame

        Returns:
            LaTeX code string
        """
        logger.info("Generating LaTeX table")

        try:
            latex_code = """
\\begin{table}[h!]
\\centering
\\caption{Financial Forecasting Model Comparison}
\\label{tab:model_comparison}
\\begin{tabular}{lcccccc}
\\hline
Model & RMSE & MAE & MAPE & R² & Direction Acc & Training Time \\\\
 &  &  & (\%) &  & (\%) & (sec) \\\\
\\hline
"""

            for _, row in comparison_table.iterrows():
                latex_code += f"{row['Model']} & {row['RMSE']:.4f} & {row['MAE']:.4f} & {row['MAPE']:.2f} & {row['R²']:.4f} & {row['Direction_Acc']:.1f} & {row.get('Training_Time_Sec', 'N/A')} \\\\\n"

            latex_code += """\\hline
\\end{tabular}
\\end{table}
"""

            logger.info("Generated LaTeX table")
            return latex_code

        except Exception as e:
            logger.error(f"Error generating LaTeX table: {str(e)}")
            return ""

    def recommend_best_model(self, comparison_data: Dict[str, Any], criteria: str = 'rmse') -> str:
        """
        Recommend the best model based on specified criteria.

        Args:
            comparison_data: Dictionary with comparison results
            criteria: Selection criteria ('rmse', 'direction_acc', 'balanced')

        Returns:
            Recommended model name
        """
        if 'summary_table' not in comparison_data:
            return "No data available"

        df = comparison_data['summary_table']
        if df.empty:
            return "No models evaluated"

        try:
            if criteria == 'rmse':
                best_model = df.loc[df['RMSE'].idxmin(), 'Model']
            elif criteria == 'direction_acc':
                best_model = df.loc[df['Direction_Acc'].idxmax(), 'Model']
            elif criteria == 'balanced':
                # Normalize metrics and create composite score
                df_norm = df.copy()
                df_norm['RMSE_norm'] = 1 - (df_norm['RMSE'] - df_norm['RMSE'].min()) / (df_norm['RMSE'].max() - df_norm['RMSE'].min())
                df_norm['Dir_Acc_norm'] = (df_norm['Direction_Acc'] - df_norm['Direction_Acc'].min()) / (df_norm['Direction_Acc'].max() - df_norm['Direction_Acc'].min())
                df_norm['composite_score'] = (df_norm['RMSE_norm'] + df_norm['Dir_Acc_norm']) / 2
                best_model = df_norm.loc[df_norm['composite_score'].idxmax(), 'Model']
            else:
                best_model = df.loc[df['RMSE'].idxmin(), 'Model']  # Default to RMSE

            logger.info(f"Recommended model ({criteria}): {best_model}")
            return best_model

        except Exception as e:
            logger.error(f"Error recommending model: {str(e)}")
            return df.iloc[0]['Model'] if not df.empty else "Error"

    def recommend_ensemble_weights(self, comparison_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Recommend ensemble weights based on inverse RMSE.

        Args:
            comparison_data: Dictionary with comparison results

        Returns:
            Dictionary of model weights
        """
        if 'summary_table' not in comparison_data:
            return {}

        df = comparison_data['summary_table']
        if df.empty:
            return {}

        try:
            # Calculate weights based on inverse RMSE (better models get higher weights)
            df_weights = df.dropna(subset=['RMSE']).copy()
            if df_weights.empty:
                return {}

            # Inverse RMSE weights
            df_weights['inv_rmse'] = 1 / df_weights['RMSE']
            total_inv_rmse = df_weights['inv_rmse'].sum()
            df_weights['weight'] = df_weights['inv_rmse'] / total_inv_rmse

            weights = dict(zip(df_weights['Model'], df_weights['weight']))

            logger.info(f"Recommended ensemble weights: {weights}")
            return weights

        except Exception as e:
            logger.error(f"Error calculating ensemble weights: {str(e)}")
            return {}

    def prepare_dashboard_data(self, comparison_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare data structures for Streamlit dashboard.

        Args:
            comparison_data: Dictionary with comparison results

        Returns:
            Dictionary ready for dashboard consumption
        """
        logger.info("Preparing dashboard data")

        dashboard_data = {
            'timestamp': datetime.now().isoformat(),
            'status': 'success',
            'metrics': {},
            'tables': {},
            'recommendations': {},
            'plots_data': {}
        }

        try:
            # Summary metrics
            if 'summary_table' in comparison_data:
                df = comparison_data['summary_table']
                dashboard_data['metrics'] = {
                    'total_models': len(df),
                    'best_rmse': df.loc[df['RMSE'].idxmin(), 'RMSE'] if not df.empty else None,
                    'best_direction_acc': df.loc[df['Direction_Acc'].idxmax(), 'Direction_Acc'] if not df.empty else None,
                    'avg_rmse': df['RMSE'].mean() if not df.empty else None
                }

            # Tables
            dashboard_data['tables'] = {
                'summary': comparison_data.get('summary_table', pd.DataFrame()).to_dict('records'),
                'detailed': comparison_data.get('detailed_comparison', {}),
                'significance': comparison_data.get('significance_matrix', pd.DataFrame()).to_dict() if 'significance_matrix' in comparison_data else {}
            }

            # Recommendations
            dashboard_data['recommendations'] = {
                'best_model': self.recommend_best_model(comparison_data, 'balanced'),
                'ensemble_weights': self.recommend_ensemble_weights(comparison_data)
            }

            # Plot data (simplified for dashboard)
            if 'summary_table' in comparison_data:
                df = comparison_data['summary_table']
                dashboard_data['plots_data'] = {
                    'rmse_data': df[['Model', 'RMSE']].dropna().to_dict('records'),
                    'direction_acc_data': df[['Model', 'Direction_Acc']].dropna().to_dict('records')
                }

            logger.info("Dashboard data prepared")
            return dashboard_data

        except Exception as e:
            logger.error(f"Error preparing dashboard data: {str(e)}")
            dashboard_data['status'] = 'error'
            dashboard_data['error'] = str(e)
            return dashboard_data

    def save_comparison_results(self, comparison_data: Dict[str, Any], output_dir: str) -> None:
        """
        Save all comparison results to files.

        Args:
            comparison_data: Dictionary with all comparison results
            output_dir: Directory to save results
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"Saving comparison results to {output_dir}")

        try:
            # Save summary table
            if 'summary_table' in comparison_data:
                summary_path = os.path.join(output_dir, 'model_comparison_summary.csv')
                comparison_data['summary_table'].to_csv(summary_path, index=False)
                logger.info(f"Saved summary table: {summary_path}")

            # Save detailed comparison
            if 'detailed_comparison' in comparison_data:
                detailed_path = os.path.join(output_dir, 'detailed_model_metrics.json')
                with open(detailed_path, 'w') as f:
                    json.dump(comparison_data['detailed_comparison'], f, indent=2, default=str)
                logger.info(f"Saved detailed metrics: {detailed_path}")

            # Save significance matrix
            if 'significance_matrix' in comparison_data:
                sig_path = os.path.join(output_dir, 'statistical_significance.csv')
                comparison_data['significance_matrix'].to_csv(sig_path)
                logger.info(f"Saved significance matrix: {sig_path}")

            # Save plots
            if 'plots' in comparison_data:
                plots_dir = os.path.join(output_dir, 'plots')
                os.makedirs(plots_dir, exist_ok=True)

                for plot_name, plot_fig in comparison_data['plots'].items():
                    if plot_fig is not None and MATPLOTLIB_AVAILABLE:
                        plot_path = os.path.join(plots_dir, f'{plot_name}.png')
                        plot_fig.savefig(plot_path, dpi=150, bbox_inches='tight')
                        plt.close(plot_fig)
                        logger.info(f"Saved plot: {plot_path}")

            # Save markdown report
            markdown_report = self.generate_markdown_report(comparison_data)
            report_path = os.path.join(output_dir, 'model_comparison_report.md')
            with open(report_path, 'w') as f:
                f.write(markdown_report)
            logger.info(f"Saved markdown report: {report_path}")

            logger.info("All comparison results saved successfully")

        except Exception as e:
            logger.error(f"Error saving comparison results: {str(e)}")

    def export_to_excel(self, comparison_data: Dict[str, Any], output_path: str) -> None:
        """
        Export comparison results to Excel workbook.

        Args:
            comparison_data: Dictionary with all comparison results
            output_path: Path to save Excel file
        """
        if not EXCEL_AVAILABLE:
            logger.warning("openpyxl not available - Excel export disabled")
            return

        logger.info(f"Exporting to Excel: {output_path}")

        try:
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # Summary sheet
                if 'summary_table' in comparison_data:
                    comparison_data['summary_table'].to_excel(writer, sheet_name='Summary', index=False)

                # Detailed metrics sheet
                if 'detailed_comparison' in comparison_data:
                    detailed_df = pd.DataFrame(comparison_data['detailed_comparison']).T
                    detailed_df.to_excel(writer, sheet_name='Detailed_Metrics')

                # Significance matrix sheet
                if 'significance_matrix' in comparison_data:
                    comparison_data['significance_matrix'].to_excel(writer, sheet_name='Significance_Matrix')

                # Regime analysis sheets
                if 'regime_breakdown' in comparison_data:
                    for regime_name, df in comparison_data['regime_breakdown'].items():
                        sheet_name = f'Regime_{regime_name.replace("regime_", "")}'
                        df.to_excel(writer, sheet_name=sheet_name, index=False)

            logger.info(f"Excel export completed: {output_path}")

        except Exception as e:
            logger.error(f"Error exporting to Excel: {str(e)}")


def main():
    """
    Main function demonstrating comprehensive model comparison.
    """
    logger.info("Starting comprehensive model comparison")

    try:
        # Generate synthetic data for demonstration
        logger.info("Generating synthetic financial data")
        np.random.seed(42)

        n_samples = 1000
        n_features = 18

        # Generate features and target
        X = np.random.randn(n_samples, n_features)
        y = X[:, -1] + 0.1 * np.random.randn(n_samples)

        # Create regime labels
        regime_labels = np.zeros(n_samples, dtype=int)
        regime_labels[300:600] = 1  # Volatile period
        regime_labels[600:] = 2    # Crisis period

        # Split data
        train_size = int(0.7 * n_samples)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        regime_test = regime_labels[train_size:]

        # Create dates
        dates = pd.date_range(start='2020-01-01', periods=len(X_test), freq='D')

        # Initialize models (simplified for demo)
        models_dict = {}

        # Linear Regression
        try:
            from sklearn.linear_model import LinearRegression
            lr_model = LinearRegression()
            lr_model.fit(X_train, y_train)
            models_dict['OLS'] = lr_model
            logger.info("Trained OLS model")
        except Exception as e:
            logger.warning(f"Could not train OLS: {str(e)}")

        # Random Forest
        try:
            from sklearn.ensemble import RandomForestRegressor
            rf_model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10)
            rf_model.fit(X_train, y_train)
            models_dict['RF'] = rf_model
            logger.info("Trained Random Forest model")
        except Exception as e:
            logger.warning(f"Could not train RF: {str(e)}")

        if not models_dict:
            logger.warning("No models available for comparison")
            return

        # Initialize comparison generator
        comparator = ComparisonReportGenerator(models_dict)

        # Generate all comparison data
        logger.info("Generating comparison data")

        comparison_data = {}

        # Summary table
        comparison_data['summary_table'] = comparator.create_summary_table(X_test, y_test)

        # Regime breakdown
        comparison_data['regime_breakdown'] = comparator.create_regime_breakdown(X_test, y_test, regime_test)

        # Detailed comparison
        comparison_data['detailed_comparison'] = comparator.create_detailed_comparison(X_test, y_test)

        # Statistical significance
        comparison_data['significance_matrix'] = comparator.statistical_significance_matrix(X_test, y_test)

        # Get predictions for plotting
        predictions = {}
        for name, model in models_dict.items():
            try:
                predictions[name] = model.predict(X_test)
            except Exception as e:
                logger.warning(f"Error getting predictions for {name}: {str(e)}")

        comparison_data['predictions'] = predictions
        comparison_data['y_true'] = y_test
        comparison_data['dates'] = dates
        comparison_data['regime_labels'] = regime_test

        # Generate plots
        comparison_data['plots'] = comparator.create_all_comparison_plots(comparison_data)

        # Save results
        output_dir = "comparison_results"
        comparator.save_comparison_results(comparison_data, output_dir)

        # Generate HTML report
        html_path = os.path.join(output_dir, "model_comparison_report.html")
        comparator.generate_html_report(comparison_data, html_path)

        # Export to Excel
        excel_path = os.path.join(output_dir, "model_comparison.xlsx")
        comparator.export_to_excel(comparison_data, excel_path)

        # Prepare dashboard data
        dashboard_data = comparator.prepare_dashboard_data(comparison_data)

        # Save dashboard data
        dashboard_path = os.path.join(output_dir, "dashboard_data.json")
        with open(dashboard_path, 'w') as f:
            json.dump(dashboard_data, f, indent=2, default=str)

        # Print summary
        print("\n" + "="*80)
        print("MODEL COMPARISON COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"Models Compared: {len(models_dict)}")
        print(f"Test Samples: {len(X_test)}")
        print(f"Regimes Identified: {len(np.unique(regime_test))}")
        print()

        if 'summary_table' in comparison_data:
            df = comparison_data['summary_table']
            print("TOP 3 MODELS BY RMSE:")
            for i, (_, row) in enumerate(df.head(3).iterrows(), 1):
                print(f"{i}. {row['Model']}: RMSE={row['RMSE']:.4f}, Direction_Acc={row['Direction_Acc']:.1f}%")

        print()
        print("RECOMMENDATIONS:")
        best_model = comparator.recommend_best_model(comparison_data, 'balanced')
        print(f"Best Overall Model: {best_model}")

        weights = comparator.recommend_ensemble_weights(comparison_data)
        if weights:
            print("Ensemble Weights:")
            for model, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
                print(".3f")

        print()
        print("FILES GENERATED:")
        print(f"• Results Directory: {output_dir}/")
        print("  - model_comparison_summary.csv")
        print("  - detailed_model_metrics.json")
        print("  - statistical_significance.csv")
        print("  - model_comparison_report.md")
        print("  - model_comparison_report.html")
        print("  - model_comparison.xlsx")
        print("  - dashboard_data.json")
        print("  - plots/ (multiple PNG files)")

        print()
        print("="*80)

    except Exception as e:
        logger.error(f"Error in main comparison: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()