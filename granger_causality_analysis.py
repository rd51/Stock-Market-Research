"""
Granger Causality Analysis Module
==================================

Advanced causality testing on freshly scraped financial data to detect feedback loops
and causal relationships between market variables using Granger causality and VAR models.

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

# Required libraries for advanced analysis
try:
    import statsmodels.api as sm
    from statsmodels.tsa.api import VAR
    from statsmodels.tsa.stattools import adfuller, grangercausalitytests
    from statsmodels.tsa.vector_ar.var_model import VARResults
    import scipy.stats as stats
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    logging.warning("statsmodels not available - Granger causality analysis disabled")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import networkx as nx
    MATPLOTLIB_AVAILABLE = True
    plt.style.use('default')
    sns.set_palette("husl")
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logging.warning("matplotlib/seaborn/networkx not available - plotting disabled")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GrangerCausalityAnalysis:
    """
    Comprehensive Granger causality analysis for financial time series data.
    """

    def __init__(self, data: pd.DataFrame, target_columns: List[str] = ['VIX', 'Returns', 'Unemployment']):
        """
        Initialize Granger causality analysis.

        Args:
            data (pd.DataFrame): Input dataframe with time series data
            target_columns (List[str]): Columns to analyze for causality
        """
        self.original_data = data.copy()
        self.target_columns = target_columns
        self.stationary_data = None
        self.differencing_orders = {}
        self.var_model = None
        self.causality_results = {}

        # Check if required columns exist
        missing_cols = [col for col in target_columns if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        logger.info(f"Initialized GrangerCausalityAnalysis with {len(target_columns)} variables")

    def prepare_data(self, df: pd.DataFrame, stationarity_test: bool = True) -> pd.DataFrame:
        """
        Prepare data for Granger causality analysis by ensuring stationarity.

        Args:
            df (pd.DataFrame): Input dataframe
            stationarity_test (bool): Whether to test and enforce stationarity

        Returns:
            pd.DataFrame: Stationary data ready for analysis
        """
        logger.info("Preparing data for Granger causality analysis")

        if not stationarity_test:
            self.stationary_data = df[self.target_columns].dropna()
            return self.stationary_data

        # Test stationarity for each series
        stationary_data = {}
        differencing_orders = {}

        for col in self.target_columns:
            series = df[col].dropna()
            is_stationary, order = self._test_and_transform_stationary(series, col)
            stationary_data[col] = series if order == 0 else self._difference_series(series, order)
            differencing_orders[col] = order

        self.stationary_data = pd.DataFrame(stationary_data).dropna()
        self.differencing_orders = differencing_orders

        logger.info(f"Data preparation complete. Differencing orders: {differencing_orders}")
        logger.info(f"Final dataset shape: {self.stationary_data.shape}")

        return self.stationary_data

    def _test_and_transform_stationary(self, series: pd.Series, name: str,
                                     max_differencing: int = 2) -> Tuple[bool, int]:
        """
        Test stationarity and apply differencing if needed.

        Args:
            series (pd.Series): Time series to test
            name (str): Series name for logging
            max_differencing (int): Maximum differencing order to try

        Returns:
            Tuple[bool, int]: (is_stationary, differencing_order_used)
        """
        if not STATSMODELS_AVAILABLE:
            logger.warning("statsmodels not available - skipping stationarity test")
            return True, 0

        # Test original series
        try:
            adf_result = adfuller(series.dropna(), autolag='AIC')
            p_value = adf_result[1]

            if p_value < 0.05:
                logger.info(f"{name}: Stationary (p={p_value:.4f})")
                return True, 0

            # Try differencing
            for order in range(1, max_differencing + 1):
                diff_series = self._difference_series(series, order)
                if len(diff_series) < 10:  # Too few observations
                    break

                adf_result = adfuller(diff_series.dropna(), autolag='AIC')
                p_value = adf_result[1]

                if p_value < 0.05:
                    logger.info(f"{name}: Stationary after {order} differencing (p={p_value:.4f})")
                    return True, order

            logger.warning(f"{name}: Could not achieve stationarity after {max_differencing} differencing")
            return False, 0

        except Exception as e:
            logger.error(f"Error testing stationarity for {name}: {str(e)}")
            return False, 0

    def _difference_series(self, series: pd.Series, order: int) -> pd.Series:
        """
        Apply differencing to make series stationary.

        Args:
            series (pd.Series): Input series
            order (int): Differencing order

        Returns:
            pd.Series: Differenced series
        """
        result = series.copy()
        for _ in range(order):
            result = result.diff()
        return result.dropna()


def granger_causality_test(cause_series: pd.Series, effect_series: pd.Series,
                          max_lag: int = 20) -> Dict[str, Any]:
    """
    Perform Granger causality test between two time series.

    Args:
        cause_series (pd.Series): Potential cause variable
        effect_series (pd.Series): Effect variable
        max_lag (int): Maximum lag to test

    Returns:
        Dict[str, Any]: Test results
    """
    logger.info(f"Testing Granger causality: {cause_series.name} → {effect_series.name}")

    if not STATSMODELS_AVAILABLE:
        logger.warning("statsmodels not available - returning basic result")
        return {
            'p_value': None,
            'f_statistic': None,
            'lags': None,
            'significant': None,
            'note': 'statsmodels not available'
        }

    try:
        # Prepare data
        data = pd.DataFrame({
            'cause': cause_series,
            'effect': effect_series
        }).dropna()

        if len(data) < max_lag + 10:
            return {
                'p_value': None,
                'f_statistic': None,
                'lags': None,
                'significant': None,
                'error': 'Insufficient data for Granger test'
            }

        # Run Granger causality test
        test_result = grangercausalitytests(data[['effect', 'cause']], maxlag=max_lag, verbose=False)

        # Extract results for optimal lag (based on AIC if available, else last lag)
        best_lag = max_lag
        best_p_value = 1.0
        best_f_stat = 0

        for lag in range(1, max_lag + 1):
            if lag in test_result:
                ssr_chi2test = test_result[lag][0]['ssr_chi2test']
                p_value = ssr_chi2test[1]
                f_stat = ssr_chi2test[0]

                if p_value < best_p_value:
                    best_p_value = p_value
                    best_f_stat = f_stat
                    best_lag = lag

        significant = best_p_value < 0.05

        result = {
            'p_value': best_p_value,
            'f_statistic': best_f_stat,
            'lags': best_lag,
            'significant': significant,
            'cause_variable': cause_series.name,
            'effect_variable': effect_series.name
        }

        logger.info(f"Granger test result: p={best_p_value:.4f}, significant={significant}")
        return result

    except Exception as e:
        logger.error(f"Error in Granger causality test: {str(e)}")
        return {
            'p_value': None,
            'f_statistic': None,
            'lags': None,
            'significant': None,
            'error': str(e)
        }


def test_bidirectional_causality(series_x: pd.Series, series_y: pd.Series,
                               max_lag: int = 20) -> Dict[str, Any]:
    """
    Test Granger causality in both directions.

    Args:
        series_x (pd.Series): First series
        series_y (pd.Series): Second series
        max_lag (int): Maximum lag to test

    Returns:
        Dict[str, Any]: Bidirectional test results
    """
    logger.info(f"Testing bidirectional causality: {series_x.name} ↔ {series_y.name}")

    # Test X → Y
    x_to_y = granger_causality_test(series_x, series_y, max_lag)

    # Test Y → X
    y_to_x = granger_causality_test(series_y, series_x, max_lag)

    # Determine bidirectional relationship
    bidirectional = (x_to_y.get('significant', False) and y_to_x.get('significant', False))

    result = {
        'x_to_y': x_to_y,
        'y_to_x': y_to_x,
        'bidirectional': bidirectional,
        'x_name': series_x.name,
        'y_name': series_y.name
    }

    logger.info(f"Bidirectional test: X→Y={x_to_y.get('significant')}, Y→X={y_to_x.get('significant')}, bidirectional={bidirectional}")
    return result


def fit_var_model(data: pd.DataFrame, lag_order: int = 5) -> Any:
    """
    Fit Vector Autoregression (VAR) model.

    Args:
        data (pd.DataFrame): Stationary time series data
        lag_order (int): Number of lags to include

    Returns:
        VAR model object
    """
    logger.info(f"Fitting VAR model with {lag_order} lags")

    if not STATSMODELS_AVAILABLE:
        logger.warning("statsmodels not available - cannot fit VAR model")
        return None

    try:
        # Ensure we have enough data
        min_obs = lag_order * len(data.columns) + 10
        if len(data) < min_obs:
            logger.warning(f"Insufficient data for VAR model: {len(data)} observations, need at least {min_obs}")
            return None

        model = VAR(data)
        var_model = model.fit(lag_order)

        logger.info(f"VAR model fitted successfully. AIC: {var_model.aic:.4f}")
        return var_model

    except Exception as e:
        logger.error(f"Error fitting VAR model: {str(e)}")
        # Try with fewer lags if original fails
        try:
            max_possible_lag = min(lag_order, len(data) // (len(data.columns) + 1) - 1)
            if max_possible_lag >= 1:
                logger.info(f"Retrying VAR fit with {max_possible_lag} lags")
                var_model = model.fit(max_possible_lag)
                logger.info(f"VAR model fitted with reduced lags. AIC: {var_model.aic:.4f}")
                return var_model
        except Exception as e2:
            logger.error(f"Failed to fit VAR model even with reduced lags: {str(e2)}")

        return None


def get_var_summary(var_model: Any) -> pd.DataFrame:
    """
    Extract summary statistics from VAR model.

    Args:
        var_model: Fitted VAR model

    Returns:
        pd.DataFrame: Model summary statistics
    """
    logger.info("Extracting VAR model summary")

    if var_model is None:
        logger.warning("VAR model is None")
        return pd.DataFrame()

    try:
        # Extract coefficients and p-values
        coef_data = []
        var_names = var_model.names

        # VAR.params is a (n_params, n_vars) array
        # Rows are parameters (const, L1.x, L1.y, etc.)
        # Columns are dependent variables
        params_df = pd.DataFrame(var_model.params, columns=var_names)
        pvalues_df = pd.DataFrame(var_model.pvalues, columns=var_names)

        for dep_var in var_names:
            for param_name in params_df.index:
                if param_name != 'const':  # Skip constant
                    # Parse parameter name (e.g., 'L1.VIX' -> lag=1, ind_var='VIX')
                    if param_name.startswith('L') and '.' in param_name:
                        parts = param_name.split('.')
                        lag_part = parts[0]  # 'L1'
                        ind_var = parts[1]   # 'VIX'
                        lag = int(lag_part[1:])  # Extract number from 'L1'

                        coef = params_df.loc[param_name, dep_var]
                        p_value = pvalues_df.loc[param_name, dep_var]

                        # Get t-statistic using bse (standard errors)
                        std_err = var_model.bse.loc[param_name, dep_var] if hasattr(var_model, 'bse') and param_name in var_model.bse.index else 0
                        t_stat = coef / std_err if std_err != 0 else 0

                        coef_data.append({
                            'dependent': dep_var,
                            'independent': ind_var,
                            'lag': lag,
                            'coefficient': coef,
                            't_statistic': t_stat,
                            'p_value': p_value,
                            'significant': p_value < 0.05
                        })

        result_df = pd.DataFrame(coef_data)
        logger.info(f"VAR summary extracted: {len(result_df)} coefficient relationships")
        return result_df

    except Exception as e:
        logger.error(f"Error extracting VAR summary: {str(e)}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def select_optimal_lag(data: pd.DataFrame, max_lags: int = 20) -> int:
    """
    Select optimal lag order using information criteria.

    Args:
        data (pd.DataFrame): Time series data
        max_lags (int): Maximum lag order to consider

    Returns:
        int: Optimal lag order
    """
    logger.info(f"Selecting optimal lag order (max: {max_lags})")

    if not STATSMODELS_AVAILABLE:
        logger.warning("statsmodels not available - using default lag order")
        return 5

    try:
        model = VAR(data)

        # Adjust max_lags based on available data
        max_possible_lags = min(max_lags, len(data) // (len(data.columns) + 1) - 1)
        max_possible_lags = max(1, max_possible_lags)

        lag_selection = model.select_order(maxlags=max_possible_lags)

        # lag_selection.aic returns the INDEX of the minimum AIC, not the AIC value
        # We need to add 1 because lag_selection starts from lag 0
        optimal_lag_index = int(lag_selection.aic)
        optimal_lag = optimal_lag_index + 1  # Convert from 0-based index to actual lag order

        # Ensure we don't return lag 0 (no autoregression)
        optimal_lag = max(1, optimal_lag)

        aic_values = lag_selection.ics['aic']
        actual_aic = aic_values[optimal_lag_index] if optimal_lag_index < len(aic_values) else float('inf')

        logger.info(f"Optimal lag selected: {optimal_lag} (AIC index: {optimal_lag_index}, AIC value: {actual_aic:.4f})")
        return optimal_lag

    except Exception as e:
        logger.error(f"Error selecting optimal lag: {str(e)}")
        return 5


def calculate_irf(var_model: Any, periods: int = 20) -> Any:
    """
    Calculate Impulse Response Functions.

    Args:
        var_model: Fitted VAR model
        periods (int): Number of periods for IRF

    Returns:
        IRF results object
    """
    logger.info(f"Calculating Impulse Response Functions for {periods} periods")

    if var_model is None:
        logger.warning("VAR model is None - cannot calculate IRF")
        return None

    try:
        irf = var_model.irf(periods)
        logger.info("IRF calculation completed")
        return irf

    except Exception as e:
        logger.error(f"Error calculating IRF: {str(e)}")
        return None


def plot_irf(irf_results: Any, figsize: Tuple[int, int] = (15, 10)) -> Any:
    """
    Plot Impulse Response Functions.

    Args:
        irf_results: IRF results object
        figsize (Tuple[int, int]): Figure size

    Returns:
        matplotlib figure object
    """
    if not MATPLOTLIB_AVAILABLE or irf_results is None:
        logger.warning("Matplotlib not available or IRF results missing - skipping plot")
        return None

    logger.info("Creating IRF visualization")

    try:
        fig = irf_results.plot(figsize=figsize)
        plt.tight_layout()
        logger.info("IRF visualization created")
        return fig

    except Exception as e:
        logger.error(f"Error creating IRF plot: {str(e)}")
        return None


def calculate_fevd(var_model: Any, periods: int = 20) -> pd.DataFrame:
    """
    Calculate Forecast Error Variance Decomposition.

    Args:
        var_model: Fitted VAR model
        periods (int): Number of periods for FEVD

    Returns:
        pd.DataFrame: FEVD results
    """
    logger.info(f"Calculating Forecast Error Variance Decomposition for {periods} periods")

    if var_model is None:
        logger.warning("VAR model is None - cannot calculate FEVD")
        return pd.DataFrame()

    try:
        fevd = var_model.fevd(periods)

        # Extract FEVD data
        fevd_data = []
        var_names = var_model.names

        for period in range(periods):
            for i, dep_var in enumerate(var_names):
                for j, ind_var in enumerate(var_names):
                    variance_share = fevd.decomp[period, i, j]

                    fevd_data.append({
                        'period': period + 1,
                        'dependent': dep_var,
                        'source': ind_var,
                        'variance_share': variance_share
                    })

        result_df = pd.DataFrame(fevd_data)
        logger.info(f"FEVD calculation completed: {len(result_df)} observations")
        return result_df

    except Exception as e:
        logger.error(f"Error calculating FEVD: {str(e)}")
        return pd.DataFrame()


def plot_fevd(fevd_results: pd.DataFrame) -> Any:
    """
    Plot Forecast Error Variance Decomposition.

    Args:
        fevd_results (pd.DataFrame): FEVD results dataframe

    Returns:
        matplotlib figure object
    """
    if not MATPLOTLIB_AVAILABLE or fevd_results.empty:
        logger.warning("Matplotlib not available or FEVD results empty - skipping plot")
        return None

    logger.info("Creating FEVD visualization")

    try:
        fig, axes = plt.subplots(len(fevd_results['dependent'].unique()),
                                figsize=(12, 8), sharex=True)

        if len(fevd_results['dependent'].unique()) == 1:
            axes = [axes]

        for i, dep_var in enumerate(fevd_results['dependent'].unique()):
            dep_data = fevd_results[fevd_results['dependent'] == dep_var]

            # Pivot data for stacked area plot
            pivot_data = dep_data.pivot(index='period', columns='source', values='variance_share')

            # Plot stacked area
            axes[i].stackplot(pivot_data.index, pivot_data.T.values,
                            labels=pivot_data.columns, alpha=0.8)
            axes[i].set_title(f'FEVD: {dep_var} Variance Decomposition')
            axes[i].set_ylabel('Variance Share')
            axes[i].legend(loc='upper right')

        axes[-1].set_xlabel('Periods Ahead')
        plt.tight_layout()
        logger.info("FEVD visualization created")
        return fig

    except Exception as e:
        logger.error(f"Error creating FEVD plot: {str(e)}")
        return None


def build_causality_network(data: pd.DataFrame, max_lag: int = 20,
                          threshold: float = 0.05) -> Dict[str, Any]:
    """
    Build causality network from pairwise Granger tests.

    Args:
        data (pd.DataFrame): Time series data
        max_lag (int): Maximum lag for tests
        threshold (float): Significance threshold

    Returns:
        Dict[str, Any]: Network results
    """
    logger.info(f"Building causality network with threshold {threshold}")

    columns = data.columns.tolist()
    n_vars = len(columns)

    # Initialize adjacency matrix
    adjacency_matrix = np.zeros((n_vars, n_vars))

    # Test all pairwise relationships
    edges = []

    for i, cause_var in enumerate(columns):
        for j, effect_var in enumerate(columns):
            if i != j:  # No self-causality
                cause_series = data[cause_var]
                effect_series = data[effect_var]

                test_result = granger_causality_test(cause_series, effect_series, max_lag)

                if test_result.get('significant', False):
                    adjacency_matrix[i, j] = 1
                    edges.append({
                        'source': cause_var,
                        'target': effect_var,
                        'p_value': test_result.get('p_value'),
                        'f_statistic': test_result.get('f_statistic'),
                        'lags': test_result.get('lags')
                    })

    # Create network graph if networkx available
    network_graph = None
    if MATPLOTLIB_AVAILABLE:
        try:
            import networkx as nx
            G = nx.DiGraph()

            # Add nodes
            for col in columns:
                G.add_node(col)

            # Add edges
            for edge in edges:
                G.add_edge(edge['source'], edge['target'],
                          weight=-np.log(edge['p_value']),  # Stronger for smaller p-values
                          p_value=edge['p_value'])

            network_graph = G
        except ImportError:
            logger.warning("networkx not available - network graph not created")

    result = {
        'edges': edges,
        'adjacency_matrix': adjacency_matrix,
        'variables': columns,
        'network_graph': network_graph,
        'n_edges': len(edges),
        'threshold': threshold
    }

    logger.info(f"Causality network built: {len(edges)} significant edges found")
    return result


def plot_causality_network(causality_results: Dict[str, Any]) -> Any:
    """
    Plot causality network visualization.

    Args:
        causality_results (Dict[str, Any]): Network results from build_causality_network

    Returns:
        matplotlib figure object
    """
    if not MATPLOTLIB_AVAILABLE or causality_results.get('network_graph') is None:
        logger.warning("Matplotlib or networkx not available - skipping network plot")
        return None

    logger.info("Creating causality network visualization")

    try:
        fig, ax = plt.subplots(figsize=(10, 8))

        G = causality_results['network_graph']

        # Calculate positions
        pos = nx.spring_layout(G, seed=42)

        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='lightblue', ax=ax)

        # Draw edges with arrows
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True,
                             arrowsize=20, arrowstyle='->', ax=ax)

        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold', ax=ax)

        # Add edge labels (p-values)
        edge_labels = {(u, v): '.3f' for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8, ax=ax)

        ax.set_title('Granger Causality Network')
        ax.axis('off')

        plt.tight_layout()
        logger.info("Causality network visualization created")
        return fig

    except Exception as e:
        logger.error(f"Error creating network plot: {str(e)}")
        return None


def granger_test_by_regime(data: pd.DataFrame, regimes: np.ndarray,
                          max_lag: int = 20) -> Dict[str, Any]:
    """
    Test Granger causality separately for each regime.

    Args:
        data (pd.DataFrame): Time series data
        regimes (np.ndarray): Regime labels
        max_lag (int): Maximum lag for tests

    Returns:
        Dict[str, Any]: Regime-specific causality results
    """
    logger.info("Testing Granger causality by regime")

    unique_regimes = np.unique(regimes)
    regime_results = {}

    for regime in unique_regimes:
        regime_mask = regimes == regime
        regime_data = data[regime_mask]

        if len(regime_data) < max_lag + 10:
            logger.warning(f"Insufficient data for regime {regime}")
            continue

        logger.info(f"Testing regime {regime} ({len(regime_data)} observations)")

        # Test all pairwise relationships for this regime
        columns = data.columns.tolist()
        regime_causality = {}

        for i, cause_var in enumerate(columns):
            for j, effect_var in enumerate(columns):
                if i != j:
                    cause_series = regime_data[cause_var]
                    effect_series = regime_data[effect_var]

                    test_result = granger_causality_test(cause_series, effect_series, max_lag)
                    key = f"{cause_var}_to_{effect_var}"

                    regime_causality[key] = test_result

        regime_results[regime] = {
            'causality_tests': regime_causality,
            'n_observations': len(regime_data),
            'significant_relationships': sum(1 for r in regime_causality.values()
                                           if r.get('significant', False))
        }

    logger.info(f"Regime-specific causality testing completed for {len(regime_results)} regimes")
    return regime_results


def plot_regime_causality_comparison(regime_results: Dict[str, Any]) -> Any:
    """
    Plot comparison of causality patterns across regimes.

    Args:
        regime_results (Dict[str, Any]): Results from granger_test_by_regime

    Returns:
        matplotlib figure object
    """
    if not MATPLOTLIB_AVAILABLE or not regime_results:
        logger.warning("Matplotlib not available or no regime results - skipping plot")
        return None

    logger.info("Creating regime causality comparison visualization")

    try:
        fig, ax = plt.subplots(figsize=(12, 8))

        regimes = list(regime_results.keys())
        significant_counts = [r['significant_relationships'] for r in regime_results.values()]

        bars = ax.bar(regimes, significant_counts, color='skyblue', alpha=0.7)
        ax.set_title('Significant Granger Causality Relationships by Regime')
        ax.set_xlabel('Regime')
        ax.set_ylabel('Number of Significant Relationships')
        ax.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, count in zip(bars, significant_counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{count}', ha='center', va='bottom')

        plt.tight_layout()
        logger.info("Regime causality comparison visualization created")
        return fig

    except Exception as e:
        logger.error(f"Error creating regime comparison plot: {str(e)}")
        return None


def test_instantaneous_causality(var_model: Any) -> Dict[str, Any]:
    """
    Test for instantaneous (contemporaneous) causality.

    Args:
        var_model: Fitted VAR model

    Returns:
        Dict[str, Any]: Instantaneous causality results
    """
    logger.info("Testing instantaneous causality")

    if var_model is None:
        logger.warning("VAR model is None")
        return {}

    try:
        # Get residuals correlation matrix
        residuals = var_model.resid
        corr_matrix = residuals.corr()

        # Test significance of correlations
        n_obs = len(residuals)
        instantaneous_results = {}

        var_names = var_model.names
        for i, var1 in enumerate(var_names):
            for j, var2 in enumerate(var_names):
                if i != j:
                    correlation = corr_matrix.loc[var1, var2]

                    # T-test for correlation significance
                    t_stat = correlation * np.sqrt((n_obs - 2) / (1 - correlation**2))
                    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n_obs - 2))

                    instantaneous_results[f"{var1}_with_{var2}"] = {
                        'correlation': correlation,
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    }

        logger.info(f"Instantaneous causality testing completed: {len(instantaneous_results)} relationships tested")
        return instantaneous_results

    except Exception as e:
        logger.error(f"Error testing instantaneous causality: {str(e)}")
        return {}


def detect_feedback_loops(causality_matrix: np.ndarray) -> List[List[int]]:
    """
    Detect feedback loops in causality network.

    Args:
        causality_matrix (np.ndarray): Adjacency matrix from causality network

    Returns:
        List[List[int]]: List of feedback loop paths
    """
    logger.info("Detecting feedback loops in causality network")

    if not MATPLOTLIB_AVAILABLE:
        logger.warning("networkx not available - cannot detect feedback loops")
        return []

    try:
        import networkx as nx

        # Create directed graph from adjacency matrix
        n_vars = causality_matrix.shape[0]
        G = nx.DiGraph()

        # Add nodes
        for i in range(n_vars):
            G.add_node(i)

        # Add edges
        for i in range(n_vars):
            for j in range(n_vars):
                if causality_matrix[i, j] == 1:
                    G.add_edge(i, j)

        # Find all cycles
        cycles = list(nx.simple_cycles(G))

        logger.info(f"Feedback loop detection completed: {len(cycles)} loops found")
        return cycles

    except Exception as e:
        logger.error(f"Error detecting feedback loops: {str(e)}")
        return []


def quantify_loop_strength(var_model: Any, loop_path: List[int]) -> float:
    """
    Quantify the strength of a feedback loop.

    Args:
        var_model: Fitted VAR model
        loop_path (List[int]): Path indices forming the loop

    Returns:
        float: Loop strength (0-1 scale)
    """
    logger.info(f"Quantifying strength of feedback loop: {loop_path}")

    if var_model is None or len(loop_path) < 3:
        return 0.0

    try:
        # Calculate product of coefficients along the loop
        loop_strength = 1.0

        for i in range(len(loop_path)):
            current_var = loop_path[i]
            next_var = loop_path[(i + 1) % len(loop_path)]

            # Find coefficient from current to next variable (lag 1)
            var_names = var_model.names
            coef_name = f'{var_names[current_var]}.L1'

            if coef_name in var_model.params.index:
                coef = abs(var_model.params[coef_name])
                loop_strength *= coef
            else:
                loop_strength *= 0.1  # Weak default

        # Normalize to 0-1 scale
        loop_strength = min(loop_strength, 1.0)

        logger.info(f"Loop strength calculated: {loop_strength:.4f}")
        return loop_strength

    except Exception as e:
        logger.error(f"Error quantifying loop strength: {str(e)}")
        return 0.0


def calculate_system_complexity(var_model: Any, causality_matrix: np.ndarray) -> Dict[str, Any]:
    """
    Calculate complexity metrics for the causal system.

    Args:
        var_model: Fitted VAR model
        causality_matrix (np.ndarray): Causality adjacency matrix

    Returns:
        Dict[str, Any]: Complexity metrics
    """
    logger.info("Calculating system complexity metrics")

    if not MATPLOTLIB_AVAILABLE:
        logger.warning("networkx not available - basic complexity metrics only")
        n_vars = causality_matrix.shape[0]
        n_edges = np.sum(causality_matrix)
        density = n_edges / (n_vars * (n_vars - 1)) if n_vars > 1 else 0

        return {
            'network_density': density,
            'feedback_loops': 0,
            'complexity_score': density * 50  # Simple score
        }

    try:
        import networkx as nx

        # Create network
        n_vars = causality_matrix.shape[0]
        G = nx.DiGraph()

        for i in range(n_vars):
            G.add_node(i)

        for i in range(n_vars):
            for j in range(n_vars):
                if causality_matrix[i, j] == 1:
                    G.add_edge(i, j)

        # Calculate metrics
        n_edges = G.number_of_edges()
        network_density = n_edges / (n_vars * (n_vars - 1)) if n_vars > 1 else 0

        # Find feedback loops
        cycles = list(nx.simple_cycles(G))
        n_feedback_loops = len(cycles)

        # Calculate average path length (for connected components)
        try:
            avg_path_length = nx.average_shortest_path_length(G) if nx.is_strongly_connected(G) else float('inf')
        except:
            avg_path_length = float('inf')

        # Complexity score (0-100 scale)
        complexity_score = min(100, (network_density * 40 + n_feedback_loops * 10 +
                                    (100 / max(avg_path_length, 1)) * 10))

        metrics = {
            'network_density': network_density,
            'n_edges': n_edges,
            'n_nodes': n_vars,
            'feedback_loops': n_feedback_loops,
            'avg_path_length': avg_path_length,
            'complexity_score': complexity_score
        }

        logger.info(f"System complexity calculated: score={complexity_score:.1f}")
        return metrics

    except Exception as e:
        logger.error(f"Error calculating system complexity: {str(e)}")
        return {}


def plot_comprehensive_causality(data: pd.DataFrame, results: Dict[str, Any]) -> Any:
    """
    Create comprehensive causality analysis visualization.

    Args:
        data (pd.DataFrame): Original data
        results (Dict[str, Any]): Complete analysis results

    Returns:
        matplotlib figure object
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Matplotlib not available - skipping comprehensive plot")
        return None

    logger.info("Creating comprehensive causality visualization")

    try:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Plot 1: Time series data
        for i, col in enumerate(data.columns):
            axes[0, 0].plot(data.index, data[col], label=col, linewidth=2)
        axes[0, 0].set_title('Time Series Data')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: VAR Coefficients (if available)
        if 'var_summary' in results and not results['var_summary'].empty:
            var_summary = results['var_summary']
            significant_coefs = var_summary[var_summary['significant']]

            if not significant_coefs.empty:
                pivot_data = significant_coefs.pivot_table(
                    index='dependent', columns='independent',
                    values='coefficient', aggfunc='mean'
                )

                sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='RdYlBu_r',
                          ax=axes[0, 1], center=0)
                axes[0, 1].set_title('VAR Coefficients (Significant Only)')

        # Plot 3: Causality Network (if available)
        if 'causality_network' in results and results['causality_network'].get('network_graph'):
            G = results['causality_network']['network_graph']
            pos = nx.spring_layout(G, seed=42)

            nx.draw(G, pos, with_labels=True, node_color='lightblue',
                   node_size=1000, arrows=True, ax=axes[1, 0])
            axes[1, 0].set_title('Causality Network')

        # Plot 4: FEVD Summary (if available)
        if 'fevd_results' in results and not results['fevd_results'].empty:
            fevd_data = results['fevd_results']
            latest_fevd = fevd_data[fevd_data['period'] == fevd_data['period'].max()]

            for dep_var in latest_fevd['dependent'].unique():
                dep_data = latest_fevd[latest_fevd['dependent'] == dep_var]
                axes[1, 1].bar(range(len(dep_data)), dep_data['variance_share'],
                              label=f'{dep_var} sources', alpha=0.7)

            axes[1, 1].set_title('Final Period FEVD')
            axes[1, 1].set_xlabel('Source Variables')
            axes[1, 1].set_ylabel('Variance Share')
            axes[1, 1].legend()

        plt.tight_layout()
        logger.info("Comprehensive causality visualization created")
        return fig

    except Exception as e:
        logger.error(f"Error creating comprehensive plot: {str(e)}")
        return None


def generate_causality_report(results: Dict[str, Any]) -> str:
    """
    Generate comprehensive causality analysis report.

    Args:
        results (Dict[str, Any]): Complete analysis results

    Returns:
        str: Markdown report
    """
    logger.info("Generating causality analysis report")

    report_lines = []
    report_lines.append("# Granger Causality Analysis Report")
    report_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")

    # Summary
    report_lines.append("## Executive Summary")
    if 'causality_network' in results:
        network = results['causality_network']
        report_lines.append(f"- **Variables Analyzed:** {len(network.get('variables', []))}")
        report_lines.append(f"- **Significant Relationships:** {network.get('n_edges', 0)}")
        report_lines.append(f"- **Significance Threshold:** {network.get('threshold', 0.05)}")

    if 'system_complexity' in results:
        complexity = results['system_complexity']
        report_lines.append(f"- **System Complexity Score:** {complexity.get('complexity_score', 0):.1f}/100")
        report_lines.append(f"- **Network Density:** {complexity.get('network_density', 0):.1%}")
    report_lines.append("")

    # Granger Causality Results
    report_lines.append("## Granger Causality Results")
    if 'pairwise_tests' in results:
        for test_name, test_result in results['pairwise_tests'].items():
            if test_result.get('significant'):
                p_val = test_result.get('p_value', 1)
                report_lines.append(f"- **{test_name}:** Significant (p = {p_val:.4f})")
            else:
                p_val = test_result.get('p_value', 1)
                report_lines.append(f"- **{test_name}:** Not significant (p = {p_val:.4f})")
    report_lines.append("")

    # VAR Model Results
    report_lines.append("## VAR Model Results")
    if 'var_summary' in results and not results['var_summary'].empty:
        var_summary = results['var_summary']
        significant = var_summary[var_summary['significant']]

        report_lines.append(f"- **Total Relationships:** {len(var_summary)}")
        report_lines.append(f"- **Significant Relationships:** {len(significant)}")

        # Top relationships
        if len(significant) > 0:
            top_rels = significant.nlargest(5, 'coefficient')
            report_lines.append("- **Strongest Relationships:**")
            for _, row in top_rels.iterrows():
                report_lines.append(f"  - {row['independent']} -> {row['dependent']} "
                                  f"(coef: {row['coefficient']:.3f}, p: {row['p_value']:.4f})")
    report_lines.append("")

    # Feedback Loops
    report_lines.append("## Feedback Loops")
    if 'feedback_loops' in results:
        loops = results['feedback_loops']
        if loops:
            report_lines.append(f"- **Detected Loops:** {len(loops)}")
            for i, loop in enumerate(loops[:3]):  # Show first 3
                report_lines.append(f"  - Loop {i+1}: {' -> '.join(map(str, loop))}")
        else:
            report_lines.append("- **No feedback loops detected**")
    report_lines.append("")

    # Interpretation
    report_lines.append("## Interpretation")
    report_lines.append("### Key Findings:")
    report_lines.append("- Granger causality indicates predictive relationships between variables")
    report_lines.append("- VAR model shows simultaneous relationships and dynamics")
    report_lines.append("- IRF reveals how shocks propagate through the system")
    report_lines.append("- FEVD shows which variables drive forecast uncertainty")
    report_lines.append("")

    report_lines.append("### Practical Implications:")
    report_lines.append("- Use significant causal relationships for forecasting")
    report_lines.append("- Monitor variables with high FEVD contributions")
    report_lines.append("- Consider feedback loops for policy analysis")
    report_lines.append("- Regime-specific causality may indicate changing relationships")

    return "\n".join(report_lines)


def main():
    """
    Main function demonstrating comprehensive Granger causality analysis.
    """
    logger.info("Starting comprehensive Granger causality analysis demonstration")

    try:
        # Generate synthetic financial data
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=300, freq='D')

        # Create correlated time series
        n_obs = len(dates)

        # Base VIX series (volatility)
        vix_base = 20 + np.random.normal(0, 2, n_obs)
        vix = pd.Series(np.maximum(5, vix_base), index=dates, name='VIX')

        # Returns influenced by VIX
        returns_noise = np.random.normal(0, 0.02, n_obs)
        returns = pd.Series(returns_noise - 0.5 * (vix - 20) / 100, index=dates, name='Returns')

        # Unemployment with some lag from market conditions
        unemployment_base = 6.0
        unemployment = []
        for i in range(n_obs):
            # Unemployment responds to market stress with lag
            market_stress = vix.iloc[min(i, len(vix)-1)] - 20
            shock = np.random.normal(0, 0.05)
            unemployment_base += shock - 0.01 * market_stress  # Negative relationship
            unemployment_base = max(3, min(15, unemployment_base))
            unemployment.append(unemployment_base)

        unemployment = pd.Series(unemployment, index=dates, name='Unemployment')

        # Combine into dataframe
        financial_data = pd.DataFrame({
            'VIX': vix,
            'Returns': returns,
            'Unemployment': unemployment
        })

        logger.info(f"Generated synthetic data: {financial_data.shape}")

        # Step 1: Initialize analysis
        logger.info("Step 1: Initializing Granger causality analysis")
        analyzer = GrangerCausalityAnalysis(financial_data)

        # Step 2: Prepare data (ensure stationarity)
        logger.info("Step 2: Preparing data and testing stationarity")
        stationary_data = analyzer.prepare_data(financial_data)

        if stationary_data.empty:
            logger.error("Data preparation failed - no stationary data")
            return

        # Step 3: Pairwise Granger causality tests
        logger.info("Step 3: Testing pairwise Granger causality")
        pairwise_tests = {}

        columns = stationary_data.columns.tolist()
        for i, cause in enumerate(columns):
            for j, effect in enumerate(columns):
                if i != j:
                    test_result = granger_causality_test(
                        stationary_data[cause], stationary_data[effect]
                    )
                    pairwise_tests[f"{cause}_to_{effect}"] = test_result

        # Step 4: Fit VAR model
        logger.info("Step 4: Fitting VAR model")
        optimal_lag = select_optimal_lag(stationary_data)
        var_model = fit_var_model(stationary_data, optimal_lag)

        # Step 5: VAR model summary
        logger.info("Step 5: Extracting VAR model summary")
        var_summary = get_var_summary(var_model)

        # Step 6: Impulse Response Functions
        logger.info("Step 6: Calculating Impulse Response Functions")
        irf_results = calculate_irf(var_model)

        # Step 7: Forecast Error Variance Decomposition
        logger.info("Step 7: Calculating Forecast Error Variance Decomposition")
        fevd_results = calculate_fevd(var_model)

        # Step 8: Build causality network
        logger.info("Step 8: Building causality network")
        causality_network = build_causality_network(stationary_data)

        # Step 9: Test by regime (using simple regime detection)
        logger.info("Step 9: Testing causality by regime")
        # Create simple regimes based on VIX levels from original data
        regimes = np.where(financial_data['VIX'] > financial_data['VIX'].quantile(0.67), 2,
                          np.where(financial_data['VIX'] > financial_data['VIX'].quantile(0.33), 1, 0))

        # Align regimes with stationary data length (trim if necessary)
        if len(regimes) > len(stationary_data):
            regimes = regimes[:len(stationary_data)]
        elif len(regimes) < len(stationary_data):
            # Pad with last regime value if needed
            padding = np.full(len(stationary_data) - len(regimes), regimes[-1])
            regimes = np.concatenate([regimes, padding])

        regime_causality = granger_test_by_regime(stationary_data, regimes)

        # Step 10: Instantaneous causality
        logger.info("Step 10: Testing instantaneous causality")
        instantaneous_causality = test_instantaneous_causality(var_model)

        # Step 11: Detect feedback loops
        logger.info("Step 11: Detecting feedback loops")
        feedback_loops = detect_feedback_loops(causality_network['adjacency_matrix'])

        # Step 12: Calculate system complexity
        logger.info("Step 12: Calculating system complexity")
        system_complexity = calculate_system_complexity(var_model, causality_network['adjacency_matrix'])

        # Step 13: Create visualizations
        logger.info("Step 13: Creating visualizations")
        if MATPLOTLIB_AVAILABLE:
            # IRF plot
            irf_fig = plot_irf(irf_results)

            # FEVD plot
            fevd_fig = plot_fevd(fevd_results)

            # Causality network
            network_fig = plot_causality_network(causality_network)

            # Regime comparison
            regime_fig = plot_regime_causality_comparison(regime_causality)

            # Comprehensive plot
            comprehensive_fig = plot_comprehensive_causality(financial_data, {
                'var_summary': var_summary,
                'causality_network': causality_network,
                'fevd_results': fevd_results
            })

        # Step 14: Generate report
        logger.info("Step 14: Generating comprehensive report")
        all_results = {
            'pairwise_tests': pairwise_tests,
            'var_summary': var_summary,
            'causality_network': causality_network,
            'regime_causality': regime_causality,
            'instantaneous_causality': instantaneous_causality,
            'feedback_loops': feedback_loops,
            'system_complexity': system_complexity,
            'fevd_results': fevd_results
        }

        report = generate_causality_report(all_results)

        # Step 15: Save results
        logger.info("Step 15: Saving results")

        # Save data
        stationary_data.to_csv('stationary_data.csv')
        var_summary.to_csv('var_model_summary.csv', index=False)
        fevd_results.to_csv('fevd_results.csv', index=False)

        # Save pairwise test results
        pairwise_df = pd.DataFrame.from_dict(pairwise_tests, orient='index')
        pairwise_df.to_csv('pairwise_granger_tests.csv')

        # Save analysis summary
        summary_data = {
            'n_variables': len(columns),
            'n_observations': len(stationary_data),
            'optimal_lag': optimal_lag,
            'significant_relationships': sum(1 for r in pairwise_tests.values() if r.get('significant', False)),
            'feedback_loops': len(feedback_loops),
            'system_complexity_score': system_complexity.get('complexity_score', 0)
        }

        with open('causality_analysis_summary.json', 'w') as f:
            import json
            json.dump(summary_data, f, indent=2, default=str)

        # Save report
        with open('granger_causality_report.md', 'w') as f:
            f.write(report)

        # Save plots
        if MATPLOTLIB_AVAILABLE:
            if irf_fig:
                irf_fig.savefig('impulse_response_functions.png', dpi=300, bbox_inches='tight')
            if fevd_fig:
                fevd_fig.savefig('forecast_error_decomposition.png', dpi=300, bbox_inches='tight')
            if network_fig:
                network_fig.savefig('causality_network.png', dpi=300, bbox_inches='tight')
            if regime_fig:
                regime_fig.savefig('regime_causality_comparison.png', dpi=300, bbox_inches='tight')
            if comprehensive_fig:
                comprehensive_fig.savefig('comprehensive_causality_analysis.png', dpi=300, bbox_inches='tight')

        logger.info("Comprehensive Granger causality analysis completed successfully!")
        logger.info("Results saved to CSV files, JSON summary, and PNG plots")

        # Print summary
        print("\n" + "="*80)
        print("COMPREHENSIVE GRANGER CAUSALITY ANALYSIS COMPLETED")
        print("="*80)
        print(f"Variables analyzed: {len(columns)}")
        print(f"Observations: {len(stationary_data)}")
        print(f"Significant causal relationships: {summary_data['significant_relationships']}")
        print(f"Feedback loops detected: {len(feedback_loops)}")
        print(f"System complexity score: {summary_data['system_complexity_score']:.1f}/100")
        print(f"Files saved: 4 CSV files, 1 JSON, 1 Markdown report, 5 PNG plots")
        print("="*80)

    except Exception as e:
        logger.error(f"Error in main analysis: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()