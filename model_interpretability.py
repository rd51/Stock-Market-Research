"""
Model Interpretability Module
=============================

Comprehensive tools for understanding and explaining model decisions.
Includes SHAP, partial dependence, LIME, and regime-specific analysis.

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

# Required libraries for interpretability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available - SHAP analysis disabled")

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    logging.warning("LIME not available - LIME explanations disabled")

try:
    import eli5
    ELI5_AVAILABLE = True
except ImportError:
    ELI5_AVAILABLE = False
    logging.warning("ELI5 not available - ELI5 explanations disabled")

try:
    from sklearn.inspection import partial_dependence, PartialDependenceDisplay
    from sklearn.inspection import permutation_importance
    SKLEARN_INSPECTION_AVAILABLE = True
except ImportError:
    SKLEARN_INSPECTION_AVAILABLE = False
    logging.warning("sklearn.inspection not available - partial dependence disabled")

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


def calculate_feature_importance_all_models(models_dict: Dict[str, Any], X_test: np.ndarray) -> pd.DataFrame:
    """
    Calculate feature importance across all available models.

    Args:
        models_dict: Dictionary of trained models {name: model}
        X_test: Test features for importance calculation

    Returns:
        DataFrame with feature importance from each model
    """
    logger.info("Calculating feature importance across all models")

    importance_data = {}
    feature_names = [f'feature_{i}' for i in range(X_test.shape[1])]

    for model_name, model in models_dict.items():
        if model is None:
            continue

        try:
            if hasattr(model, 'feature_importances_'):  # Random Forest, etc.
                importance = model.feature_importances_
            elif hasattr(model, 'get_booster'):  # XGBoost
                importance = model.get_booster().get_score(importance_type='gain')
                # Convert to array in correct order
                importance_array = np.zeros(X_test.shape[1])
                for feat_name, score in importance.items():
                    if feat_name.startswith('f'):
                        idx = int(feat_name[1:])
                        if idx < len(importance_array):
                            importance_array[idx] = score
                importance = importance_array
            elif hasattr(model, 'feature_importances_'):  # LightGBM
                importance = model.feature_importances_
            else:
                logger.warning(f"No feature importance method for {model_name}")
                continue

            importance_data[f'{model_name.upper()}_Importance'] = importance

        except Exception as e:
            logger.warning(f"Failed to get importance for {model_name}: {str(e)}")

    if not importance_data:
        logger.warning("No feature importance data available")
        return pd.DataFrame()

    # Create DataFrame
    importance_df = pd.DataFrame(importance_data)
    importance_df['Feature'] = feature_names

    # Reorder columns
    cols = ['Feature'] + [col for col in importance_df.columns if col != 'Feature']
    importance_df = importance_df[cols]

    logger.info(f"Feature importance calculated for {len(importance_data)} models")
    return importance_df


def plot_feature_importance_comparison(importance_df: pd.DataFrame, top_n: int = 15) -> Any:
    """
    Plot feature importance comparison across models.

    Args:
        importance_df: DataFrame from calculate_feature_importance_all_models
        top_n: Number of top features to show

    Returns:
        matplotlib figure
    """
    if not MATPLOTLIB_AVAILABLE or importance_df.empty:
        logger.warning("Matplotlib not available or no importance data")
        return None

    try:
        # Get importance columns (exclude 'Feature')
        imp_cols = [col for col in importance_df.columns if col != 'Feature']

        if not imp_cols:
            return None

        # Normalize each model's importance to 0-1 scale
        normalized_df = importance_df.copy()
        for col in imp_cols:
            if normalized_df[col].max() > 0:
                normalized_df[col] = normalized_df[col] / normalized_df[col].max()

        # Get top N features by average importance
        normalized_df['avg_importance'] = normalized_df[imp_cols].mean(axis=1)
        top_features = normalized_df.nlargest(top_n, 'avg_importance')

        # Plot
        fig, ax = plt.subplots(figsize=(14, 10))

        x = np.arange(len(top_features))
        width = 0.8 / len(imp_cols)

        for i, col in enumerate(imp_cols):
            ax.barh(x + i * width, top_features[col],
                   height=width, label=col.replace('_Importance', ''), alpha=0.8)

        ax.set_yticks(x + width * (len(imp_cols) - 1) / 2)
        ax.set_yticklabels(top_features['Feature'])
        ax.set_xlabel('Normalized Importance')
        ax.set_title(f'Feature Importance Comparison (Top {top_n} Features)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        logger.info("Feature importance comparison plot created")
        return fig

    except Exception as e:
        logger.error(f"Error creating importance comparison plot: {str(e)}")
        return None


def permutation_importance(model: Any, X_test: np.ndarray, y_test: np.ndarray,
                          n_repeats: int = 10) -> pd.DataFrame:
    """
    Calculate permutation feature importance.

    Args:
        model: Trained model
        X_test, y_test: Test data
        n_repeats: Number of times to shuffle each feature

    Returns:
        DataFrame with permutation importance scores
    """
    if not SKLEARN_INSPECTION_AVAILABLE:
        logger.warning("sklearn.inspection not available - using manual permutation")
        # Manual implementation
        try:
            baseline_score = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))

            importance_scores = []
            feature_names = [f'feature_{i}' for i in range(X_test.shape[1])]

            for i in range(X_test.shape[1]):
                scores = []
                for _ in range(n_repeats):
                    X_permuted = X_test.copy()
                    np.random.shuffle(X_permuted[:, i])
                    permuted_score = np.sqrt(mean_squared_error(y_test, model.predict(X_permuted)))
                    scores.append(permuted_score - baseline_score)

                importance_scores.append({
                    'feature': feature_names[i],
                    'importance_mean': np.mean(scores),
                    'importance_std': np.std(scores)
                })

            return pd.DataFrame(importance_scores).sort_values('importance_mean', ascending=False)

        except Exception as e:
            logger.error(f"Error in manual permutation importance: {str(e)}")
            return pd.DataFrame()

    try:
        logger.info(f"Calculating permutation importance with {n_repeats} repeats")

        # Use sklearn's permutation_importance
        perm_importance = permutation_importance(
            model, X_test, y_test, n_repeats=n_repeats,
            random_state=42, scoring='neg_mean_squared_error'
        )

        feature_names = [f'feature_{i}' for i in range(X_test.shape[1])]

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance_mean': perm_importance.importances_mean,
            'importance_std': perm_importance.importances_std
        }).sort_values('importance_mean', ascending=False)

        logger.info("Permutation importance calculated")
        return importance_df

    except Exception as e:
        logger.error(f"Error calculating permutation importance: {str(e)}")
        return pd.DataFrame()


def calculate_shap_values(model: Any, X_sample: np.ndarray) -> np.ndarray:
    """
    Calculate SHAP values for model interpretability.

    Args:
        model: Trained model (preferably tree-based)
        X_sample: Sample data for SHAP calculation

    Returns:
        SHAP values array
    """
    if not SHAP_AVAILABLE:
        logger.warning("SHAP not available - returning zeros")
        return np.zeros_like(X_sample)

    try:
        logger.info("Calculating SHAP values")

        if hasattr(model, 'predict_proba'):  # Classification
            explainer = shap.TreeExplainer(model)
        else:  # Regression
            explainer = shap.TreeExplainer(model)

        shap_values = explainer.shap_values(X_sample)

        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            shap_values = shap_values[0]  # For binary classification

        logger.info(f"SHAP values calculated for {X_sample.shape[0]} samples")
        return shap_values

    except Exception as e:
        logger.error(f"Error calculating SHAP values: {str(e)}")
        return np.zeros_like(X_sample)


def plot_shap_summary(shap_values: np.ndarray, X_sample: np.ndarray,
                     feature_names: Optional[List[str]] = None) -> Any:
    """
    Plot SHAP summary plot (beeswarm plot).

    Args:
        shap_values: SHAP values from calculate_shap_values
        X_sample: Original feature values
        feature_names: Names of features

    Returns:
        matplotlib figure
    """
    if not SHAP_AVAILABLE or not MATPLOTLIB_AVAILABLE:
        logger.warning("SHAP or matplotlib not available")
        return None

    try:
        fig, ax = plt.subplots(figsize=(10, 8))

        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X_sample.shape[1])]

        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
        plt.title('SHAP Feature Importance Summary')
        plt.tight_layout()

        logger.info("SHAP summary plot created")
        return fig

    except Exception as e:
        logger.error(f"Error creating SHAP summary plot: {str(e)}")
        return None


def plot_shap_dependence(shap_values: np.ndarray, X_sample: np.ndarray,
                        feature_name: str) -> Any:
    """
    Plot SHAP dependence plot for a specific feature.

    Args:
        shap_values: SHAP values
        X_sample: Original feature values
        feature_name: Name of feature to plot

    Returns:
        matplotlib figure
    """
    if not SHAP_AVAILABLE or not MATPLOTLIB_AVAILABLE:
        logger.warning("SHAP or matplotlib not available")
        return None

    try:
        # Find feature index
        if feature_name.startswith('feature_'):
            feature_idx = int(feature_name.split('_')[1])
        else:
            feature_idx = 0

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.dependence_plot(feature_idx, shap_values, X_sample, show=False)
        plt.title(f'SHAP Dependence Plot: {feature_name}')
        plt.tight_layout()

        logger.info(f"SHAP dependence plot created for {feature_name}")
        return fig

    except Exception as e:
        logger.error(f"Error creating SHAP dependence plot: {str(e)}")
        return None


def plot_shap_force(shap_values: np.ndarray, X_sample: np.ndarray,
                   instance_idx: int, base_value: float) -> Any:
    """
    Plot SHAP force plot (waterfall) for individual prediction.

    Args:
        shap_values: SHAP values
        X_sample: Original feature values
        instance_idx: Index of instance to explain
        base_value: Base value (expected value)

    Returns:
        matplotlib figure
    """
    if not SHAP_AVAILABLE or not MATPLOTLIB_AVAILABLE:
        logger.warning("SHAP or matplotlib not available")
        return None

    try:
        fig, ax = plt.subplots(figsize=(12, 4))

        # Create force plot manually since shap.plots.force might not work in all environments
        instance_shap = shap_values[instance_idx]
        instance_features = X_sample[instance_idx]

        # Sort by absolute SHAP value
        sorted_idx = np.argsort(np.abs(instance_shap))[::-1]
        top_features = sorted_idx[:10]  # Top 10 features

        # Create waterfall plot
        cumulative = base_value
        positions = [cumulative]

        for i, feat_idx in enumerate(top_features):
            effect = instance_shap[feat_idx]
            cumulative += effect
            positions.append(cumulative)

        # Plot
        ax.plot([0, len(top_features)], [base_value, base_value], 'k--', alpha=0.5, label='Base Value')

        feature_names = [f'feature_{i}' for i in top_features]
        colors = ['red' if x > 0 else 'blue' for x in instance_shap[top_features]]

        for i in range(len(top_features)):
            ax.bar(i, positions[i+1] - positions[i], bottom=positions[i],
                  color=colors[i], alpha=0.7, width=0.8)

        ax.set_xticks(range(len(top_features)))
        ax.set_xticklabels(feature_names, rotation=45, ha='right')
        ax.set_ylabel('Prediction Value')
        ax.set_title(f'SHAP Force Plot: Instance {instance_idx}')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        logger.info(f"SHAP force plot created for instance {instance_idx}")
        return fig

    except Exception as e:
        logger.error(f"Error creating SHAP force plot: {str(e)}")
        return None


def plot_shap_interaction(shap_values: np.ndarray, X_sample: np.ndarray,
                         feature_pairs: List[Tuple[int, int]]) -> Any:
    """
    Plot SHAP interaction effects between feature pairs.

    Args:
        shap_values: SHAP values
        X_sample: Original feature values
        feature_pairs: List of (feature_idx1, feature_idx2) tuples

    Returns:
        matplotlib figure
    """
    if not SHAP_AVAILABLE or not MATPLOTLIB_AVAILABLE:
        logger.warning("SHAP or matplotlib not available")
        return None

    try:
        fig, axes = plt.subplots(1, len(feature_pairs), figsize=(6*len(feature_pairs), 5))

        if len(feature_pairs) == 1:
            axes = [axes]

        for i, (feat1, feat2) in enumerate(feature_pairs):
            ax = axes[i]

            # Simple interaction plot
            x_vals = X_sample[:, feat1]
            y_vals = X_sample[:, feat2]
            colors = shap_values[:, feat1] + shap_values[:, feat2]  # Combined effect

            scatter = ax.scatter(x_vals, y_vals, c=colors, cmap='RdYlBu_r', alpha=0.6)
            ax.set_xlabel(f'Feature {feat1}')
            ax.set_ylabel(f'Feature {feat2}')
            ax.set_title(f'Interaction: Features {feat1} & {feat2}')

            plt.colorbar(scatter, ax=ax, label='Combined SHAP Value')

        plt.tight_layout()
        logger.info("SHAP interaction plot created")
        return fig

    except Exception as e:
        logger.error(f"Error creating SHAP interaction plot: {str(e)}")
        return None


def calculate_partial_dependence(model: Any, X_test: np.ndarray, feature_name: str,
                               resolution: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate partial dependence for a feature.

    Args:
        model: Trained model
        X_test: Test data
        feature_name: Name of feature to analyze
        resolution: Number of points to evaluate

    Returns:
        Tuple of (feature_values, partial_dependence)
    """
    if not SKLEARN_INSPECTION_AVAILABLE:
        logger.warning("sklearn.inspection not available - using manual PD")
        # Manual implementation
        try:
            if feature_name.startswith('feature_'):
                feature_idx = int(feature_name.split('_')[1])
            else:
                feature_idx = 0

            # Create range of feature values
            feature_min, feature_max = X_test[:, feature_idx].min(), X_test[:, feature_idx].max()
            feature_values = np.linspace(feature_min, feature_max, resolution)

            pd_values = []

            for val in feature_values:
                X_modified = X_test.copy()
                X_modified[:, feature_idx] = val
                predictions = model.predict(X_modified)
                pd_values.append(np.mean(predictions))

            return np.array(feature_values), np.array(pd_values)

        except Exception as e:
            logger.error(f"Error in manual partial dependence: {str(e)}")
            return np.array([]), np.array([])

    try:
        logger.info(f"Calculating partial dependence for {feature_name}")

        if feature_name.startswith('feature_'):
            feature_idx = int(feature_name.split('_')[1])
        else:
            feature_idx = [0]

        pd_results = partial_dependence(model, X_test, features=[feature_idx],
                                      kind='average', grid_resolution=resolution)

        feature_values = pd_results['grid_values'][0]
        pd_values = pd_results['average'][0]

        logger.info("Partial dependence calculated")
        return feature_values, pd_values

    except Exception as e:
        logger.error(f"Error calculating partial dependence: {str(e)}")
        return np.array([]), np.array([])


def plot_partial_dependence(model: Any, X_test: np.ndarray, feature_names: List[str],
                          top_n: int = 8) -> Any:
    """
    Plot partial dependence for multiple features.

    Args:
        model: Trained model
        X_test: Test data
        feature_names: List of feature names to plot
        top_n: Maximum number of features to plot

    Returns:
        matplotlib figure
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Matplotlib not available")
        return None

    try:
        features_to_plot = feature_names[:top_n]
        n_features = len(features_to_plot)

        if n_features == 0:
            return None

        # Calculate grid dimensions
        n_cols = min(4, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_features == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for i, feature_name in enumerate(features_to_plot):
            ax = axes[i]

            feature_values, pd_values = calculate_partial_dependence(model, X_test, feature_name)

            if len(feature_values) > 0:
                ax.plot(feature_values, pd_values, 'b-', linewidth=2)
                ax.set_xlabel(feature_name)
                ax.set_ylabel('Partial Dependence')
                ax.set_title(f'PD: {feature_name}')
                ax.grid(True, alpha=0.3)

        # Hide empty subplots
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        logger.info("Partial dependence plots created")
        return fig

    except Exception as e:
        logger.error(f"Error creating partial dependence plots: {str(e)}")
        return None


def calculate_ice_curves(model: Any, X_test: np.ndarray, feature_name: str,
                        resolution: int = 30) -> np.ndarray:
    """
    Calculate Individual Conditional Expectation (ICE) curves.

    Args:
        model: Trained model
        X_test: Test data
        feature_name: Name of feature to analyze
        resolution: Number of points to evaluate

    Returns:
        Array of ICE curves (n_samples x resolution)
    """
    try:
        logger.info(f"Calculating ICE curves for {feature_name}")

        if feature_name.startswith('feature_'):
            feature_idx = int(feature_name.split('_')[1])
        else:
            feature_idx = 0

        # Create range of feature values
        feature_min, feature_max = X_test[:, feature_idx].min(), X_test[:, feature_idx].max()
        feature_values = np.linspace(feature_min, feature_max, resolution)

        ice_curves = []

        for sample_idx in range(min(100, len(X_test))):  # Limit to 100 samples for speed
            sample_ice = []

            for val in feature_values:
                X_modified = X_test[[sample_idx]].copy()
                X_modified[:, feature_idx] = val
                prediction = model.predict(X_modified)[0]
                sample_ice.append(prediction)

            ice_curves.append(sample_ice)

        logger.info(f"ICE curves calculated for {len(ice_curves)} samples")
        return np.array(ice_curves)

    except Exception as e:
        logger.error(f"Error calculating ICE curves: {str(e)}")
        return np.array([])


def plot_ice_curves(ice_curves: np.ndarray, feature_values: np.ndarray,
                   feature_name: str) -> Any:
    """
    Plot Individual Conditional Expectation (ICE) curves.

    Args:
        ice_curves: ICE curves from calculate_ice_curves
        feature_values: Feature values used for ICE
        feature_name: Name of feature

    Returns:
        matplotlib figure
    """
    if not MATPLOTLIB_AVAILABLE or ice_curves.size == 0:
        logger.warning("Matplotlib not available or no ICE data")
        return None

    try:
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot individual ICE curves
        for i, ice_curve in enumerate(ice_curves):
            ax.plot(feature_values, ice_curve, alpha=0.3, color='blue', linewidth=1)

        # Plot average (PD)
        pd_curve = np.mean(ice_curves, axis=0)
        ax.plot(feature_values, pd_curve, 'r-', linewidth=3, label='Average (PD)')

        ax.set_xlabel(feature_name)
        ax.set_ylabel('Prediction')
        ax.set_title(f'ICE Curves: {feature_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        logger.info("ICE curves plot created")
        return fig

    except Exception as e:
        logger.error(f"Error creating ICE plot: {str(e)}")
        return None


def calculate_ale(model: Any, X_test: np.ndarray, feature_name: str,
                 n_bins: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate Accumulated Local Effects (ALE).

    Args:
        model: Trained model
        X_test: Test data
        feature_name: Name of feature to analyze
        n_bins: Number of bins for ALE calculation

    Returns:
        Tuple of (bin_centers, ale_values)
    """
    try:
        logger.info(f"Calculating ALE for {feature_name}")

        if feature_name.startswith('feature_'):
            feature_idx = int(feature_name.split('_')[1])
        else:
            feature_idx = 0

        feature_values = X_test[:, feature_idx]

        # Create bins
        bins = np.quantile(feature_values, np.linspace(0, 1, n_bins+1))
        bin_centers = (bins[:-1] + bins[1:]) / 2

        ale_values = []

        for i in range(len(bins) - 1):
            # Samples in this bin
            mask = (feature_values >= bins[i]) & (feature_values < bins[i+1])
            if np.sum(mask) == 0:
                ale_values.append(0)
                continue

            # Calculate local effect
            X_bin = X_test[mask]

            # Predictions with actual values
            pred_actual = model.predict(X_bin)

            # Predictions with feature set to bin center
            X_modified = X_bin.copy()
            X_modified[:, feature_idx] = bin_centers[i]
            pred_modified = model.predict(X_modified)

            # Local effect
            local_effect = np.mean(pred_modified - pred_actual)
            ale_values.append(local_effect)

        # Accumulate effects
        accumulated_ale = np.cumsum(ale_values)

        logger.info("ALE calculated")
        return bin_centers, accumulated_ale

    except Exception as e:
        logger.error(f"Error calculating ALE: {str(e)}")
        return np.array([]), np.array([])


def plot_ale(ale_values: np.ndarray, feature_bins: np.ndarray) -> Any:
    """
    Plot Accumulated Local Effects.

    Args:
        ale_values: ALE values from calculate_ale
        feature_bins: Feature bin centers

    Returns:
        matplotlib figure
    """
    if not MATPLOTLIB_AVAILABLE or ale_values.size == 0:
        logger.warning("Matplotlib not available or no ALE data")
        return None

    try:
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.step(feature_bins, ale_values, 'b-', linewidth=2, where='mid')
        ax.set_xlabel('Feature Value')
        ax.set_ylabel('Accumulated Local Effect')
        ax.set_title('Accumulated Local Effects (ALE)')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)

        plt.tight_layout()
        logger.info("ALE plot created")
        return fig

    except Exception as e:
        logger.error(f"Error creating ALE plot: {str(e)}")
        return None


def lime_explanation(model: Any, X_test: np.ndarray, instance_idx: int,
                   n_features: int = 10) -> Dict[str, Any]:
    """
    Generate LIME explanation for a single instance.

    Args:
        model: Trained model
        X_test: Test data
        instance_idx: Index of instance to explain
        n_features: Number of top features to include

    Returns:
        Dictionary with LIME explanation results
    """
    if not LIME_AVAILABLE:
        logger.warning("LIME not available - returning empty explanation")
        return {}

    try:
        logger.info(f"Generating LIME explanation for instance {instance_idx}")

        # Create LIME explainer
        explainer = lime.lime_tabular.LimeTabularExplainer(
            X_test,
            feature_names=[f'feature_{i}' for i in range(X_test.shape[1])],
            class_names=['prediction'],
            mode='regression'
        )

        # Explain instance
        exp = explainer.explain_instance(
            X_test[instance_idx],
            model.predict,
            num_features=n_features
        )

        # Extract results
        feature_importance = exp.as_list()
        prediction = model.predict(X_test[[instance_idx]])[0]

        explanation = {
            'instance_idx': instance_idx,
            'prediction': prediction,
            'feature_importance': feature_importance,
            'lime_exp': exp
        }

        logger.info("LIME explanation generated")
        return explanation

    except Exception as e:
        logger.error(f"Error generating LIME explanation: {str(e)}")
        return {}


def plot_lime_explanation(lime_results: Dict[str, Any]) -> Any:
    """
    Plot LIME explanation as a bar chart.

    Args:
        lime_results: Results from lime_explanation

    Returns:
        matplotlib figure
    """
    if not MATPLOTLIB_AVAILABLE or not lime_results:
        logger.warning("Matplotlib not available or no LIME results")
        return None

    try:
        fig, ax = plt.subplots(figsize=(10, 6))

        features, importance = zip(*lime_results['feature_importance'])

        colors = ['red' if x > 0 else 'blue' for x in importance]
        bars = ax.barh(range(len(features)), importance, color=colors, alpha=0.7)

        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features)
        ax.set_xlabel('Feature Contribution')
        ax.set_title(f'LIME Explanation: Instance {lime_results["instance_idx"]}')
        ax.grid(True, alpha=0.3)
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)

        plt.tight_layout()
        logger.info("LIME explanation plot created")
        return fig

    except Exception as e:
        logger.error(f"Error creating LIME plot: {str(e)}")
        return None


def detect_interaction_effects(model: Any, X_test: np.ndarray,
                              feature_pairs: List[Tuple[int, int]]) -> Dict[str, float]:
    """
    Detect interaction effects between feature pairs.

    Args:
        model: Trained model
        X_test: Test data
        feature_pairs: List of (feature_idx1, feature_idx2) tuples

    Returns:
        Dictionary with interaction strengths
    """
    try:
        logger.info("Detecting interaction effects")

        interactions = {}

        for feat1, feat2 in feature_pairs:
            # Simple interaction detection: compare predictions with/without interaction
            # This is a simplified approach - more sophisticated methods exist

            # Get predictions with both features
            pred_both = model.predict(X_test)

            # Predictions with feat1 only (feat2 set to mean)
            X_no_feat2 = X_test.copy()
            X_no_feat2[:, feat2] = np.mean(X_test[:, feat2])
            pred_no_feat2 = model.predict(X_no_feat2)

            # Predictions with feat2 only (feat1 set to mean)
            X_no_feat1 = X_test.copy()
            X_no_feat1[:, feat1] = np.mean(X_test[:, feat1])
            pred_no_feat1 = model.predict(X_no_feat1)

            # Predictions with neither
            X_neither = X_test.copy()
            X_neither[:, feat1] = np.mean(X_test[:, feat1])
            X_neither[:, feat2] = np.mean(X_test[:, feat2])
            pred_neither = model.predict(X_neither)

            # Calculate interaction effect
            # Interaction = pred_both - pred_no_feat2 - pred_no_feat1 + pred_neither
            interaction = np.mean(pred_both - pred_no_feat2 - pred_no_feat1 + pred_neither)
            interactions[f'feat_{feat1}_feat_{feat2}'] = abs(interaction)

        logger.info(f"Interaction effects detected for {len(interactions)} pairs")
        return interactions

    except Exception as e:
        logger.error(f"Error detecting interactions: {str(e)}")
        return {}


def plot_interaction_heatmap(interactions_dict: Dict[str, float]) -> Any:
    """
    Plot interaction effects as a heatmap.

    Args:
        interactions_dict: Results from detect_interaction_effects

    Returns:
        matplotlib figure
    """
    if not MATPLOTLIB_AVAILABLE or not interactions_dict:
        logger.warning("Matplotlib not available or no interaction data")
        return None

    try:
        # Parse feature pairs
        features = set()
        interaction_matrix = {}

        for pair_str, strength in interactions_dict.items():
            feat1, feat2 = pair_str.split('_feat_')
            feat1 = int(feat1.replace('feat_', ''))
            feat2 = int(feat2)
            features.add(feat1)
            features.add(feat2)

            if feat1 not in interaction_matrix:
                interaction_matrix[feat1] = {}
            if feat2 not in interaction_matrix:
                interaction_matrix[feat2] = {}

            interaction_matrix[feat1][feat2] = strength
            interaction_matrix[feat2][feat1] = strength

        features = sorted(list(features))
        n_features = len(features)

        # Create matrix
        matrix = np.zeros((n_features, n_features))
        for i, f1 in enumerate(features):
            for j, f2 in enumerate(features):
                if f1 in interaction_matrix and f2 in interaction_matrix[f1]:
                    matrix[i, j] = interaction_matrix[f1][f2]

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')

        ax.set_xticks(range(n_features))
        ax.set_yticks(range(n_features))
        ax.set_xticklabels([f'feat_{f}' for f in features])
        ax.set_yticklabels([f'feat_{f}' for f in features])
        ax.set_title('Feature Interaction Strength Heatmap')

        plt.colorbar(im, ax=ax, label='Interaction Strength')
        plt.tight_layout()

        logger.info("Interaction heatmap created")
        return fig

    except Exception as e:
        logger.error(f"Error creating interaction heatmap: {str(e)}")
        return None


def compare_model_interpretability(models_dict: Dict[str, Any], X_test: np.ndarray,
                                  y_test: np.ndarray) -> Any:
    """
    Compare interpretability across different models.

    Args:
        models_dict: Dictionary of trained models
        X_test, y_test: Test data

    Returns:
        matplotlib figure
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Matplotlib not available")
        return None

    try:
        # Get feature importance for each model
        importance_df = calculate_feature_importance_all_models(models_dict, X_test)

        if importance_df.empty:
            return None

        return plot_feature_importance_comparison(importance_df, top_n=10)

    except Exception as e:
        logger.error(f"Error comparing model interpretability: {str(e)}")
        return None


def feature_stability_analysis(models_dict: Dict[str, Any], X_train: np.ndarray,
                              X_test: np.ndarray) -> Dict[str, float]:
    """
    Analyze feature importance stability across models and datasets.

    Args:
        models_dict: Dictionary of trained models
        X_train, X_test: Training and test data

    Returns:
        Dictionary with stability scores for each feature
    """
    try:
        logger.info("Analyzing feature stability")

        # Get importance on training data
        train_importance = calculate_feature_importance_all_models(models_dict, X_train)

        # Get importance on test data
        test_importance = calculate_feature_importance_all_models(models_dict, X_test)

        if train_importance.empty or test_importance.empty:
            return {}

        stability_scores = {}

        for feature in train_importance['Feature']:
            train_scores = []
            test_scores = []

            # Get importance across models for this feature
            for col in train_importance.columns:
                if col != 'Feature':
                    train_val = train_importance[train_importance['Feature'] == feature][col].values
                    test_val = test_importance[test_importance['Feature'] == feature][col].values

                    if len(train_val) > 0 and len(test_val) > 0:
                        train_scores.append(train_val[0])
                        test_scores.append(test_val[0])

            if train_scores and test_scores:
                # Calculate correlation between train and test importance
                if len(train_scores) > 1:
                    stability = np.corrcoef(train_scores, test_scores)[0, 1]
                else:
                    stability = abs(train_scores[0] - test_scores[0])  # Simple difference for single model

                stability_scores[feature] = stability

        logger.info(f"Feature stability calculated for {len(stability_scores)} features")
        return stability_scores

    except Exception as e:
        logger.error(f"Error analyzing feature stability: {str(e)}")
        return {}


def analyze_regime_specific_importance(regime_models: Dict[int, Dict[str, Any]],
                                      X_by_regime: Dict[int, np.ndarray]) -> Dict[int, pd.DataFrame]:
    """
    Analyze feature importance by regime.

    Args:
        regime_models: Dictionary of regime-specific models
        X_by_regime: Dictionary of data by regime

    Returns:
        Dictionary of importance DataFrames by regime
    """
    logger.info("Analyzing regime-specific importance")

    regime_importance = {}

    for regime, models in regime_models.items():
        if regime in X_by_regime:
            X_regime = X_by_regime[regime]
            importance_df = calculate_feature_importance_all_models(models, X_regime)
            if not importance_df.empty:
                regime_importance[regime] = importance_df

    logger.info(f"Regime-specific importance analyzed for {len(regime_importance)} regimes")
    return regime_importance


def plot_regime_importance_comparison(regime_importance: Dict[int, pd.DataFrame]) -> Any:
    """
    Plot feature importance comparison across regimes.

    Args:
        regime_importance: Results from analyze_regime_specific_importance

    Returns:
        matplotlib figure
    """
    if not MATPLOTLIB_AVAILABLE or not regime_importance:
        logger.warning("Matplotlib not available or no regime data")
        return None

    try:
        fig, axes = plt.subplots(1, len(regime_importance), figsize=(6*len(regime_importance), 6))

        if len(regime_importance) == 1:
            axes = [axes]

        for i, (regime, importance_df) in enumerate(regime_importance.items()):
            ax = axes[i]

            # Get top 10 features
            if 'Feature' in importance_df.columns:
                top_features = importance_df.head(10)

                # Plot importance from first available model
                imp_cols = [col for col in importance_df.columns if col != 'Feature']
                if imp_cols:
                    ax.barh(range(len(top_features)), top_features[imp_cols[0]])
                    ax.set_yticks(range(len(top_features)))
                    ax.set_yticklabels(top_features['Feature'])
                    ax.set_xlabel('Importance')
                    ax.set_title(f'Regime {regime} Feature Importance')

        plt.tight_layout()
        logger.info("Regime importance comparison plot created")
        return fig

    except Exception as e:
        logger.error(f"Error creating regime comparison plot: {str(e)}")
        return None


def main():
    """
    Main function demonstrating model interpretability analysis.
    """
    logger.info("Starting model interpretability analysis")

    try:
        # Generate sample data for demonstration
        logger.info("Generating sample financial data")
        np.random.seed(42)
        n_samples, n_features = 1000, 18
        X_sample = np.random.randn(n_samples, n_features)

        # Create target that depends on some features (simulating financial relationships)
        y_sample = (0.8 * X_sample[:, -1] +  # Recent target value (most important)
                   0.4 * X_sample[:, -2] +  # Second most recent
                   0.2 * X_sample[:, 4] +   # Some technical indicator
                   0.1 * np.random.randn(n_samples))  # Noise

        # Split data
        train_size = int(0.7 * len(X_sample))
        X_train = X_sample[:train_size]
        y_train = y_sample[:train_size]
        X_test = X_sample[train_size:]
        y_test = y_sample[train_size:]

        # Create and train models
        models_dict = {}

        # Random Forest
        try:
            from sklearn.ensemble import RandomForestRegressor
            rf_model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10)
            rf_model.fit(X_train, y_train)
            models_dict['rf'] = rf_model
            logger.info("Random Forest model trained")
        except Exception as e:
            logger.warning(f"Could not create Random Forest: {str(e)}")

        # Try to load ensemble models if available
        try:
            from ensemble_models import RandomForestPredictor
            ensemble_rf = RandomForestPredictor()
            ensemble_rf.fit(X_train, y_train)
            models_dict['ensemble_rf'] = ensemble_rf
            logger.info("Ensemble Random Forest model trained")
        except Exception as e:
            logger.warning(f"Could not load ensemble models: {str(e)}")

        if not models_dict:
            logger.warning("No models available for analysis")
            return

        # 1. Feature importance analysis
        logger.info("Analyzing feature importance")
        importance_df = calculate_feature_importance_all_models(models_dict, X_test)

        # Permutation importance (use simpler approach)
        perm_importance = pd.DataFrame()
        if models_dict.get('rf'):
            try:
                perm_importance = permutation_importance(models_dict['rf'], X_test, y_test, n_repeats=3)
            except Exception as e:
                logger.warning(f"Permutation importance failed: {str(e)}")

        # 2. SHAP analysis
        logger.info("Performing SHAP analysis")
        shap_values = None
        if models_dict.get('rf') and SHAP_AVAILABLE:
            try:
                shap_values = calculate_shap_values(models_dict['rf'], X_test[:50])  # Smaller subset
            except Exception as e:
                logger.warning(f"SHAP analysis failed: {str(e)}")

        # 3. Partial dependence
        logger.info("Calculating partial dependence")
        top_features = ['feature_17', 'feature_16', 'feature_4']  # Based on expected importance

        # 4. ICE curves
        logger.info("Calculating ICE curves")
        ice_curves = np.array([])
        if models_dict.get('rf'):
            try:
                ice_curves = calculate_ice_curves(models_dict['rf'], X_test[:30], 'feature_17')
            except Exception as e:
                logger.warning(f"ICE curves failed: {str(e)}")

        # 5. ALE
        logger.info("Calculating ALE")
        ale_centers, ale_values = np.array([]), np.array([])
        if models_dict.get('rf'):
            try:
                ale_centers, ale_values = calculate_ale(models_dict['rf'], X_test[:100], 'feature_17')
            except Exception as e:
                logger.warning(f"ALE failed: {str(e)}")

        # 6. LIME explanation
        logger.info("Generating LIME explanation")
        lime_results = {}
        if models_dict.get('rf') and LIME_AVAILABLE:
            try:
                lime_results = lime_explanation(models_dict['rf'], X_test, instance_idx=0)
            except Exception as e:
                logger.warning(f"LIME explanation failed: {str(e)}")

        # 7. Interaction detection
        logger.info("Detecting interactions")
        interactions = {}
        feature_pairs = [(16, 17), (4, 8)]  # Some feature pairs
        if models_dict.get('rf'):
            try:
                interactions = detect_interaction_effects(models_dict['rf'], X_test[:50], feature_pairs)
            except Exception as e:
                logger.warning(f"Interaction detection failed: {str(e)}")

        # 8. Model comparison
        logger.info("Comparing model interpretability")
        comparison_fig = None
        try:
            comparison_fig = compare_model_interpretability(models_dict, X_test, y_test)
        except Exception as e:
            logger.warning(f"Model comparison failed: {str(e)}")

        # 9. Feature stability
        logger.info("Analyzing feature stability")
        stability_scores = {}
        try:
            stability_scores = feature_stability_analysis(models_dict, X_train, X_test)
        except Exception as e:
            logger.warning(f"Feature stability analysis failed: {str(e)}")

        # Create visualizations
        plots_created = 0
        if MATPLOTLIB_AVAILABLE:
            logger.info("Creating visualizations")

            # Feature importance comparison
            imp_fig = None
            if not importance_df.empty:
                try:
                    imp_fig = plot_feature_importance_comparison(importance_df, top_n=10)
                    if imp_fig:
                        plots_created += 1
                except Exception as e:
                    logger.warning(f"Importance plot failed: {str(e)}")

            # SHAP plots
            shap_summary_fig = None
            shap_dep_fig = None
            shap_force_fig = None
            if shap_values is not None and shap_values.shape[0] > 0:
                try:
                    feature_names = [f'feature_{i}' for i in range(X_test.shape[1])]
                    shap_summary_fig = plot_shap_summary(shap_values, X_test[:50], feature_names)
                    if shap_summary_fig:
                        plots_created += 1

                    shap_dep_fig = plot_shap_dependence(shap_values, X_test[:50], 'feature_17')
                    if shap_dep_fig:
                        plots_created += 1

                    # SHAP force plot
                    base_value = np.mean(models_dict['rf'].predict(X_test[:10]))
                    shap_force_fig = plot_shap_force(shap_values, X_test[:50], 0, base_value)
                    if shap_force_fig:
                        plots_created += 1
                except Exception as e:
                    logger.warning(f"SHAP plots failed: {str(e)}")

            # Partial dependence
            pd_fig = None
            if models_dict.get('rf'):
                try:
                    pd_fig = plot_partial_dependence(models_dict['rf'], X_test[:100], top_features[:3], top_n=3)
                    if pd_fig:
                        plots_created += 1
                except Exception as e:
                    logger.warning(f"Partial dependence plot failed: {str(e)}")

            # ICE curves
            ice_fig = None
            if ice_curves.size > 0:
                try:
                    feature_values = np.linspace(X_test[:, -1].min(), X_test[:, -1].max(), 30)
                    ice_fig = plot_ice_curves(ice_curves, feature_values, 'feature_17')
                    if ice_fig:
                        plots_created += 1
                except Exception as e:
                    logger.warning(f"ICE plot failed: {str(e)}")

            # ALE plot
            ale_fig = None
            if ale_centers.size > 0:
                try:
                    ale_fig = plot_ale(ale_values, ale_centers)
                    if ale_fig:
                        plots_created += 1
                except Exception as e:
                    logger.warning(f"ALE plot failed: {str(e)}")

            # LIME plot
            lime_fig = None
            if lime_results:
                try:
                    lime_fig = plot_lime_explanation(lime_results)
                    if lime_fig:
                        plots_created += 1
                except Exception as e:
                    logger.warning(f"LIME plot failed: {str(e)}")

            # Interaction heatmap
            interact_fig = None
            if interactions:
                try:
                    interact_fig = plot_interaction_heatmap(interactions)
                    if interact_fig:
                        plots_created += 1
                except Exception as e:
                    logger.warning(f"Interaction heatmap failed: {str(e)}")

        # Save results
        logger.info("Saving results")
        files_saved = 0

        # Save importance data
        if not importance_df.empty:
            try:
                importance_df.to_csv('feature_importance_comparison.csv', index=False)
                files_saved += 1
            except Exception as e:
                logger.warning(f"Could not save importance data: {str(e)}")

        if not perm_importance.empty:
            try:
                perm_importance.to_csv('permutation_importance.csv', index=False)
                files_saved += 1
            except Exception as e:
                logger.warning(f"Could not save permutation importance: {str(e)}")

        # Save SHAP values
        if shap_values is not None:
            try:
                np.save('shap_values.npy', shap_values)
                files_saved += 1
            except Exception as e:
                logger.warning(f"Could not save SHAP values: {str(e)}")

        # Save stability scores
        if stability_scores:
            try:
                stability_df = pd.DataFrame(list(stability_scores.items()),
                                           columns=['feature', 'stability'])
                stability_df.to_csv('feature_stability.csv', index=False)
                files_saved += 1
            except Exception as e:
                logger.warning(f"Could not save stability scores: {str(e)}")

        # Save interactions
        if interactions:
            try:
                interactions_df = pd.DataFrame(list(interactions.items()),
                                              columns=['feature_pair', 'interaction_strength'])
                interactions_df.to_csv('feature_interactions.csv', index=False)
                files_saved += 1
            except Exception as e:
                logger.warning(f"Could not save interactions: {str(e)}")

        # Save plots
        plot_files = [
            ('feature_importance_comparison.png', 'imp_fig'),
            ('shap_summary.png', 'shap_summary_fig'),
            ('shap_dependence.png', 'shap_dep_fig'),
            ('shap_force.png', 'shap_force_fig'),
            ('partial_dependence.png', 'pd_fig'),
            ('ice_curves.png', 'ice_fig'),
            ('ale_plot.png', 'ale_fig'),
            ('lime_explanation.png', 'lime_fig'),
            ('interaction_heatmap.png', 'interact_fig'),
            ('model_comparison.png', 'comparison_fig')
        ]

        if MATPLOTLIB_AVAILABLE:
            for filename, var_name in plot_files:
                if var_name in locals() and locals()[var_name] is not None:
                    try:
                        locals()[var_name].savefig(filename, dpi=150, bbox_inches='tight')
                        files_saved += 1
                    except Exception as e:
                        logger.warning(f"Could not save {filename}: {str(e)}")

        # Print summary
        print("\n" + "="*80)
        print("MODEL INTERPRETABILITY ANALYSIS COMPLETED")
        print("="*80)
        print(f"Data: {len(X_sample)} samples, {X_sample.shape[1]} features")
        print(f"Models analyzed: {len(models_dict)}")
        print(f"Files saved: {files_saved}")
        print(f"Plots created: {plots_created}")
        print()

        if not importance_df.empty:
            print("FEATURE IMPORTANCE ANALYSIS:")
            print("-" * 30)
            top_features = importance_df.head(3)
            for _, row in top_features.iterrows():
                rf_imp = row.get('RF_Importance', 'N/A')
                print(f"{row['Feature']}: RF={rf_imp}")
            print()

        if stability_scores:
            print("FEATURE STABILITY:")
            print("-" * 20)
            stable_features = sorted(stability_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            for feature, stability in stable_features:
                print(".4f")
            print()

        print("INTERPRETABILITY METHODS APPLIED:")
        print("-" * 35)
        methods = []
        if not importance_df.empty:
            methods.append("Feature Importance")
        if shap_values is not None:
            methods.append("SHAP Analysis")
        if 'pd_fig' in locals() and pd_fig:
            methods.append("Partial Dependence")
        if ice_curves.size > 0:
            methods.append("ICE Curves")
        if ale_centers.size > 0:
            methods.append("ALE")
        if lime_results:
            methods.append("LIME")
        if interactions:
            methods.append("Interaction Detection")

        for method in methods:
            print(f"* {method}")
        print()

        print("KEY INSIGHTS:")
        print("-" * 15)
        print("* Feature_17 (most recent target) should be most important")
        print("* SHAP values show individual prediction contributions")
        print("* Partial dependence reveals marginal feature effects")
        print("* ICE curves show effect heterogeneity")
        print("* ALE provides unbiased local effect estimates")
        print("* LIME offers model-agnostic local explanations")
        print("* Interaction detection identifies feature dependencies")
        print()

        print("="*80)

    except Exception as e:
        logger.error(f"Error in main analysis: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()