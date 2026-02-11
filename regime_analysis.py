"""
Regime Analysis Module
======================

Comprehensive volatility regime analysis for freshly scraped financial market data.
Implements advanced statistical methods for regime identification and analysis.

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

# Required libraries for advanced analysis
try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available - k-means clustering disabled")

try:
    from hmmlearn import hmm
    HMMLEARN_AVAILABLE = True
except ImportError:
    HMMLEARN_AVAILABLE = False
    logging.warning("hmmlearn not available - HMM regime detection disabled")

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
    import scipy.stats as stats
    from statsmodels.tsa.stattools import acf
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    logging.warning("statsmodels not available - some statistical tests disabled")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RegimeIdentification:
    """
    Advanced regime identification using multiple statistical methods.
    """

    def __init__(self, vix_series: pd.Series, method: str = 'percentile'):
        """
        Initialize regime identification.

        Args:
            vix_series (pd.Series): VIX time series data
            method (str): Default identification method
        """
        self.vix_series = vix_series.dropna()
        self.method = method
        self.regimes = None

        if len(self.vix_series) == 0:
            raise ValueError("VIX series is empty after removing NaN values")

        logger.info(f"Initialized RegimeIdentification with {len(self.vix_series)} observations")

    def identify_regimes_percentile(self, percentiles: List[float] = [33, 67]) -> np.ndarray:
        """
        Identify regimes using percentile-based thresholds.

        Args:
            percentiles (List[float]): Percentile thresholds for regime boundaries

        Returns:
            np.ndarray: Regime labels (0=Low, 1=Medium, 2=High)
        """
        logger.info(f"Identifying regimes using percentiles: {percentiles}")

        if len(percentiles) != 2:
            raise ValueError("Must provide exactly 2 percentile values")

        # Calculate percentile thresholds
        thresholds = np.percentile(self.vix_series, percentiles)

        # Create regime labels
        regimes = np.zeros(len(self.vix_series), dtype=int)

        # Low volatility (below first percentile)
        regimes[self.vix_series <= thresholds[0]] = 0

        # Medium volatility (between percentiles)
        regimes[(self.vix_series > thresholds[0]) & (self.vix_series <= thresholds[1])] = 1

        # High volatility (above second percentile)
        regimes[self.vix_series > thresholds[1]] = 2

        self.regimes = regimes
        logger.info(f"Identified regimes: {np.bincount(regimes)} observations in each regime")
        return regimes

    def identify_regimes_kmeans(self, n_clusters: int = 3) -> np.ndarray:
        """
        Identify regimes using K-means clustering.

        Args:
            n_clusters (int): Number of clusters/regimes

        Returns:
            np.ndarray: Cluster labels representing regimes
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for k-means clustering")

        logger.info(f"Identifying regimes using K-means with {n_clusters} clusters")

        # Prepare data for clustering
        vix_reshaped = self.vix_series.values.reshape(-1, 1)

        # Standardize the data
        scaler = StandardScaler()
        vix_scaled = scaler.fit_transform(vix_reshaped)

        # Perform k-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        regimes = kmeans.fit_predict(vix_scaled)

        # Ensure consistent labeling (low to high volatility)
        centroids = kmeans.cluster_centers_.flatten()
        sorted_indices = np.argsort(centroids)
        regime_mapping = {old_label: new_label for new_label, old_label in enumerate(sorted_indices)}
        regimes = np.array([regime_mapping[label] for label in regimes])

        self.regimes = regimes
        logger.info(f"K-means clustering complete: {np.bincount(regimes)} observations in each cluster")
        return regimes

    def identify_regimes_hmm(self, n_states: int = 3) -> np.ndarray:
        """
        Identify regimes using Hidden Markov Model.

        Args:
            n_states (int): Number of hidden states/regimes

        Returns:
            np.ndarray: Most likely state sequence
        """
        if not HMMLEARN_AVAILABLE:
            raise ImportError("hmmlearn required for HMM regime detection")

        logger.info(f"Identifying regimes using HMM with {n_states} states")

        # Prepare data
        vix_reshaped = self.vix_series.values.reshape(-1, 1)

        # Fit HMM
        model = hmm.GaussianHMM(n_components=n_states, covariance_type="full", random_state=42)
        model.fit(vix_reshaped)

        # Get most likely state sequence
        regimes = model.predict(vix_reshaped)

        # Ensure consistent labeling (low to high volatility based on state means)
        state_means = model.means_.flatten()
        sorted_indices = np.argsort(state_means)
        regime_mapping = {old_label: new_label for new_label, old_label in enumerate(sorted_indices)}
        regimes = np.array([regime_mapping[label] for label in regimes])

        self.regimes = regimes
        logger.info(f"HMM regime detection complete: {np.bincount(regimes)} observations in each state")
        return regimes

    def smooth_regimes(self, regime_array: np.ndarray, window: int = 5) -> np.ndarray:
        """
        Smooth regime transitions using majority voting in sliding window.

        Args:
            regime_array (np.ndarray): Raw regime labels
            window (int): Size of smoothing window

        Returns:
            np.ndarray: Smoothed regime labels
        """
        logger.info(f"Smoothing regimes with window size {window}")

        if window < 3 or window % 2 == 0:
            raise ValueError("Window size must be odd and >= 3")

        smoothed = regime_array.copy()

        for i in range(len(regime_array)):
            start = max(0, i - window // 2)
            end = min(len(regime_array), i + window // 2 + 1)

            window_regimes = regime_array[start:end]
            majority_regime = stats.mode(window_regimes, keepdims=True)[0][0]

            smoothed[i] = majority_regime

        logger.info("Regime smoothing complete")
        return smoothed


class RegimeStatistics:
    """
    Comprehensive statistical analysis of identified regimes.
    """

    def __init__(self):
        """Initialize regime statistics calculator."""
        pass

    def regime_durations(self, regimes: np.ndarray, dates: Union[pd.Series, pd.DatetimeIndex]) -> pd.DataFrame:
        """
        Calculate duration statistics for each regime.

        Args:
            regimes (np.ndarray): Regime labels
            dates (Union[pd.Series, pd.DatetimeIndex]): Corresponding dates

        Returns:
            pd.DataFrame: Duration statistics per regime
        """
        logger.info("Calculating regime duration statistics")

        # Convert DatetimeIndex to Series if needed
        if isinstance(dates, pd.DatetimeIndex):
            dates = pd.Series(dates)

        # Find regime change points
        regime_changes = np.where(regimes[:-1] != regimes[1:])[0] + 1
        change_indices = np.concatenate([[0], regime_changes, [len(regimes)]])

        durations = []
        for i in range(len(change_indices) - 1):
            start_idx = change_indices[i]
            end_idx = change_indices[i + 1]
            regime_label = regimes[start_idx]
            duration_days = (dates.iloc[end_idx - 1] - dates.iloc[start_idx]).days

            durations.append({
                'regime': regime_label,
                'start_date': dates.iloc[start_idx],
                'end_date': dates.iloc[end_idx - 1],
                'duration_days': duration_days
            })

        df_durations = pd.DataFrame(durations)

        # Calculate summary statistics
        summary_stats = []
        for regime in np.unique(regimes):
            regime_data = df_durations[df_durations['regime'] == regime]['duration_days']

            if len(regime_data) > 0:
                summary_stats.append({
                    'Regime': regime,
                    'Count': len(regime_data),
                    'Mean_Days': regime_data.mean(),
                    'Min_Days': regime_data.min(),
                    'Max_Days': regime_data.max(),
                    'Std_Days': regime_data.std()
                })

        result_df = pd.DataFrame(summary_stats)
        logger.info(f"Calculated duration statistics for {len(result_df)} regimes")
        return result_df

    def regime_frequency(self, regimes: np.ndarray) -> pd.DataFrame:
        """
        Calculate the percentage of time spent in each regime.

        Args:
            regimes (np.ndarray): Regime labels

        Returns:
            pd.DataFrame: Frequency statistics
        """
        logger.info("Calculating regime frequency statistics")

        total_observations = len(regimes)
        unique_regimes, counts = np.unique(regimes, return_counts=True)

        frequencies = []
        for regime, count in zip(unique_regimes, counts):
            frequency_pct = (count / total_observations) * 100
            frequencies.append({
                'Regime': regime,
                'Count': count,
                'Frequency_Percent': frequency_pct
            })

        result_df = pd.DataFrame(frequencies)
        logger.info(f"Calculated frequencies for {len(result_df)} regimes")
        return result_df

    def regime_returns_analysis(self, returns: pd.Series, regimes: np.ndarray) -> pd.DataFrame:
        """
        Analyze returns characteristics for each regime.

        Args:
            returns (pd.Series): Return series
            regimes (np.ndarray): Regime labels

        Returns:
            pd.DataFrame: Returns analysis per regime
        """
        logger.info("Analyzing returns by regime")

        # Align data
        aligned_data = pd.DataFrame({'returns': returns, 'regime': regimes})
        aligned_data = aligned_data.dropna()

        analysis_results = []
        for regime in np.unique(regimes):
            regime_returns = aligned_data[aligned_data['regime'] == regime]['returns']

            if len(regime_returns) > 0:
                mean_return = regime_returns.mean()
                std_return = regime_returns.std()
                sharpe_ratio = mean_return / std_return if std_return > 0 else 0

                analysis_results.append({
                    'Regime': regime,
                    'Mean_Return': mean_return,
                    'Std_Return': std_return,
                    'Sharpe_Ratio': sharpe_ratio,
                    'Observations': len(regime_returns)
                })

        result_df = pd.DataFrame(analysis_results)
        logger.info(f"Completed returns analysis for {len(result_df)} regimes")
        return result_df

    def transition_matrix(self, regimes: np.ndarray) -> pd.DataFrame:
        """
        Calculate regime transition probability matrix.

        Args:
            regimes (np.ndarray): Regime labels

        Returns:
            pd.DataFrame: Transition probability matrix
        """
        logger.info("Calculating transition probability matrix")

        unique_regimes = np.unique(regimes)
        n_regimes = len(unique_regimes)

        # Initialize transition matrix
        transition_matrix = np.zeros((n_regimes, n_regimes))

        # Count transitions
        for i in range(len(regimes) - 1):
            from_regime = regimes[i]
            to_regime = regimes[i + 1]

            from_idx = np.where(unique_regimes == from_regime)[0][0]
            to_idx = np.where(unique_regimes == to_regime)[0][0]

            transition_matrix[from_idx, to_idx] += 1

        # Convert to probabilities
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        transition_probabilities = transition_matrix / row_sums

        # Create DataFrame
        result_df = pd.DataFrame(
            transition_probabilities,
            index=[f'From_Regime_{r}' for r in unique_regimes],
            columns=[f'To_Regime_{r}' for r in unique_regimes]
        )

        logger.info(f"Calculated transition matrix for {n_regimes} regimes")
        return result_df

    def plot_regime_timeline(self, dates: pd.Series, regimes: np.ndarray,
                           prices: Optional[pd.Series] = None) -> Any:
        """
        Create timeline visualization of regimes.

        Args:
            dates (pd.Series): Date series
            regimes (np.ndarray): Regime labels
            prices (Optional[pd.Series]): Price series to overlay

        Returns:
            matplotlib figure object
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available - skipping plot")
            return None

        logger.info("Creating regime timeline visualization")

        fig, ax1 = plt.subplots(figsize=(15, 8))

        # Plot regimes as colored background
        unique_regimes = np.unique(regimes)
        colors = ['green', 'yellow', 'red']  # Low, Medium, High volatility

        for i, regime in enumerate(unique_regimes):
            regime_mask = regimes == regime
            if regime_mask.any():
                color = colors[i] if i < len(colors) else 'gray'
                ax1.fill_between(dates, 0, 1, where=regime_mask,
                               color=color, alpha=0.3, label=f'Regime {regime}')

        ax1.set_ylim(0, 1)
        ax1.set_ylabel('Regime Indicator')
        ax1.set_title('Market Regime Timeline')
        ax1.legend()

        # Overlay price data if provided
        if prices is not None:
            ax2 = ax1.twinx()
            ax2.plot(dates, prices, color='blue', linewidth=1, alpha=0.7)
            ax2.set_ylabel('Price', color='blue')
            ax2.tick_params(axis='y', labelcolor='blue')

        plt.tight_layout()
        logger.info("Regime timeline visualization created")
        return fig


def rolling_correlation(vix: pd.Series, returns: pd.Series, unemployment: pd.Series,
                       window: int = 30) -> pd.DataFrame:
    """
    Calculate rolling correlations between VIX, returns, and unemployment.

    Args:
        vix (pd.Series): VIX series
        returns (pd.Series): Returns series
        unemployment (pd.Series): Unemployment series
        window (int): Rolling window size

    Returns:
        pd.DataFrame: Rolling correlations over time
    """
    logger.info(f"Calculating rolling correlations with window {window}")

    # Combine all series
    combined_df = pd.DataFrame({
        'vix': vix,
        'returns': returns,
        'unemployment': unemployment
    }).dropna()

    if len(combined_df) < window:
        raise ValueError(f"Insufficient data for rolling correlation: {len(combined_df)} < {window}")

    # Calculate rolling correlations
    correlations = []

    for i in range(window, len(combined_df) + 1):
        window_data = combined_df.iloc[i-window:i]

        corr_matrix = window_data.corr()

        correlations.append({
            'date': combined_df.index[i-1],
            'vix_returns_corr': corr_matrix.loc['vix', 'returns'],
            'vix_unemployment_corr': corr_matrix.loc['vix', 'unemployment'],
            'returns_unemployment_corr': corr_matrix.loc['returns', 'unemployment']
        })

    result_df = pd.DataFrame(correlations)
    logger.info(f"Calculated {len(result_df)} rolling correlation observations")
    return result_df


def plot_rolling_correlation(corr_df: pd.DataFrame, figsize: Tuple[int, int] = (15, 8)) -> Any:
    """
    Visualize rolling correlations over time.

    Args:
        corr_df (pd.DataFrame): Rolling correlation data
        figsize (Tuple[int, int]): Figure size

    Returns:
        matplotlib figure object
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Matplotlib not available - skipping plot")
        return None

    logger.info("Creating rolling correlation visualization")

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    # VIX vs Returns correlation
    axes[0].plot(corr_df['date'], corr_df['vix_returns_corr'], color='blue', linewidth=2)
    axes[0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[0].set_title('VIX vs Returns Rolling Correlation')
    axes[0].set_ylabel('Correlation')
    axes[0].grid(True, alpha=0.3)

    # VIX vs Unemployment correlation
    axes[1].plot(corr_df['date'], corr_df['vix_unemployment_corr'], color='red', linewidth=2)
    axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1].set_title('VIX vs Unemployment Rolling Correlation')
    axes[1].set_ylabel('Correlation')
    axes[1].grid(True, alpha=0.3)

    # Returns vs Unemployment correlation
    axes[2].plot(corr_df['date'], corr_df['returns_unemployment_corr'], color='green', linewidth=2)
    axes[2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[2].set_title('Returns vs Unemployment Rolling Correlation')
    axes[2].set_ylabel('Correlation')
    axes[2].set_xlabel('Date')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    logger.info("Rolling correlation visualization created")
    return fig


def correlation_by_regime(vix: pd.Series, returns: pd.Series, unemployment: pd.Series,
                        regimes: np.ndarray) -> Dict[int, pd.DataFrame]:
    """
    Calculate correlation matrices separately for each regime.

    Args:
        vix (pd.Series): VIX series
        returns (pd.Series): Returns series
        unemployment (pd.Series): Unemployment series
        regimes (np.ndarray): Regime labels

    Returns:
        Dict[int, pd.DataFrame]: Correlation matrices by regime
    """
    logger.info("Calculating correlations by regime")

    # Combine data
    combined_df = pd.DataFrame({
        'vix': vix,
        'returns': returns,
        'unemployment': unemployment,
        'regime': regimes
    }).dropna()

    correlation_matrices = {}

    for regime in np.unique(regimes):
        regime_data = combined_df[combined_df['regime'] == regime][['vix', 'returns', 'unemployment']]

        if len(regime_data) > 3:  # Need at least 3 observations for correlation
            corr_matrix = regime_data.corr()
            correlation_matrices[regime] = corr_matrix
            logger.info(f"Calculated correlation matrix for regime {regime} ({len(regime_data)} observations)")
        else:
            logger.warning(f"Insufficient data for regime {regime} correlation analysis")

    return correlation_matrices


def detect_volatility_clustering(returns: pd.Series, window: int = 20) -> Dict[str, Any]:
    """
    Detect volatility clustering using ARCH test.

    Args:
        returns (pd.Series): Returns series
        window (int): Window size for analysis

    Returns:
        Dict[str, Any]: ARCH test results
    """
    logger.info(f"Detecting volatility clustering with window {window}")

    if not STATSMODELS_AVAILABLE:
        logger.warning("statsmodels not available - returning basic analysis")
        return {
            'test_statistic': None,
            'p_value': None,
            'is_clustered': None,
            'note': 'statsmodels not available'
        }

    try:
        from statsmodels.stats.diagnostic import het_arch

        # Calculate squared returns (volatility proxy)
        squared_returns = returns ** 2

        # Perform ARCH test
        arch_test = het_arch(squared_returns.dropna().values, nlags=window)

        test_statistic = arch_test[0]
        p_value = arch_test[1]
        is_clustered = p_value < 0.05  # Significant ARCH effect indicates clustering

        result = {
            'test_statistic': test_statistic,
            'p_value': p_value,
            'is_clustered': is_clustered,
            'significance_level': 0.05
        }

        logger.info(f"ARCH test completed: statistic={test_statistic:.4f}, p-value={p_value:.4f}, clustered={is_clustered}")
        return result

    except Exception as e:
        logger.error(f"Error in ARCH test: {str(e)}")
        return {
            'test_statistic': None,
            'p_value': None,
            'is_clustered': None,
            'error': str(e)
        }


def visualize_volatility_clusters(returns: pd.Series, dates: pd.Series) -> Any:
    """
    Visualize volatility clustering patterns.

    Args:
        returns (pd.Series): Returns series
        dates (pd.Series): Date series

    Returns:
        matplotlib figure object
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Matplotlib not available - skipping plot")
        return None

    logger.info("Creating volatility clustering visualization")

    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    # Plot returns
    axes[0].plot(dates, returns, color='blue', linewidth=1, alpha=0.7)
    axes[0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[0].set_title('Returns Time Series')
    axes[0].set_ylabel('Returns')
    axes[0].grid(True, alpha=0.3)

    # Plot squared returns (volatility)
    squared_returns = returns ** 2
    axes[1].plot(dates, squared_returns, color='red', linewidth=1, alpha=0.7)
    axes[1].set_title('Squared Returns (Volatility Clustering)')
    axes[1].set_ylabel('Squared Returns')
    axes[1].set_xlabel('Date')
    axes[1].grid(True, alpha=0.3)

    # Highlight high volatility periods
    high_vol_threshold = squared_returns.quantile(0.95)
    high_vol_mask = squared_returns > high_vol_threshold

    for ax in axes:
        ax.fill_between(dates, ax.get_ylim()[0], ax.get_ylim()[1],
                       where=high_vol_mask, color='red', alpha=0.1)

    plt.tight_layout()
    logger.info("Volatility clustering visualization created")
    return fig


def analyze_asymmetric_response(returns: pd.Series, vix: pd.Series,
                              regimes: np.ndarray) -> Dict[str, Any]:
    """
    Analyze asymmetric response of returns to VIX shocks.

    Args:
        returns (pd.Series): Returns series
        vix (pd.Series): VIX series
        regimes (np.ndarray): Regime labels

    Returns:
        Dict[str, Any]: Asymmetric response analysis
    """
    logger.info("Analyzing asymmetric response to VIX shocks")

    # Calculate VIX changes
    vix_changes = vix.pct_change()

    # Combine data
    analysis_df = pd.DataFrame({
        'returns': returns,
        'vix_changes': vix_changes,
        'regime': regimes
    }).dropna()

    results = {}

    # Overall analysis
    positive_shocks = analysis_df[analysis_df['vix_changes'] > 0]
    negative_shocks = analysis_df[analysis_df['vix_changes'] < 0]

    results['overall'] = {
        'positive_shock_avg_return': positive_shocks['returns'].mean(),
        'negative_shock_avg_return': negative_shocks['returns'].mean(),
        'positive_shock_count': len(positive_shocks),
        'negative_shock_count': len(negative_shocks),
        'asymmetry_index': abs(positive_shocks['returns'].mean()) / abs(negative_shocks['returns'].mean())
                         if negative_shocks['returns'].mean() != 0 else 0
    }

    # By regime analysis
    results['by_regime'] = {}
    for regime in np.unique(regimes):
        regime_data = analysis_df[analysis_df['regime'] == regime]

        if len(regime_data) > 10:  # Sufficient data
            pos_shocks = regime_data[regime_data['vix_changes'] > 0]
            neg_shocks = regime_data[regime_data['vix_changes'] < 0]

            results['by_regime'][regime] = {
                'positive_shock_avg_return': pos_shocks['returns'].mean() if len(pos_shocks) > 0 else 0,
                'negative_shock_avg_return': neg_shocks['returns'].mean() if len(neg_shocks) > 0 else 0,
                'positive_shock_count': len(pos_shocks),
                'negative_shock_count': len(neg_shocks),
                'asymmetry_index': abs(pos_shocks['returns'].mean()) / abs(neg_shocks['returns'].mean())
                                 if len(pos_shocks) > 0 and len(neg_shocks) > 0 and neg_shocks['returns'].mean() != 0 else 0
            }

    logger.info(f"Asymmetric response analysis completed for {len(results['by_regime'])} regimes")
    return results


def plot_asymmetry(shock_analysis: Dict[str, Any]) -> Any:
    """
    Visualize asymmetric response analysis.

    Args:
        shock_analysis (Dict[str, Any]): Results from analyze_asymmetric_response

    Returns:
        matplotlib figure object
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Matplotlib not available - skipping plot")
        return None

    logger.info("Creating asymmetry visualization")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Overall asymmetry
    overall = shock_analysis['overall']
    axes[0].bar(['Positive Shocks', 'Negative Shocks'],
                [overall['positive_shock_avg_return'], overall['negative_shock_avg_return']],
                color=['green', 'red'], alpha=0.7)
    axes[0].set_title('Overall Asymmetric Response to VIX Shocks')
    axes[0].set_ylabel('Average Return')
    axes[0].grid(True, alpha=0.3)

    # By regime
    if shock_analysis['by_regime']:
        regimes = list(shock_analysis['by_regime'].keys())
        pos_returns = [shock_analysis['by_regime'][r]['positive_shock_avg_return'] for r in regimes]
        neg_returns = [shock_analysis['by_regime'][r]['negative_shock_avg_return'] for r in regimes]

        x = np.arange(len(regimes))
        width = 0.35

        axes[1].bar(x - width/2, pos_returns, width, label='Positive Shocks', color='green', alpha=0.7)
        axes[1].bar(x + width/2, neg_returns, width, label='Negative Shocks', color='red', alpha=0.7)

        axes[1].set_title('Asymmetric Response by Regime')
        axes[1].set_xlabel('Regime')
        axes[1].set_ylabel('Average Return')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([f'Regime {r}' for r in regimes])
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    logger.info("Asymmetry visualization created")
    return fig


def calculate_lagged_correlations(vix: pd.Series, returns: pd.Series, unemployment: pd.Series,
                                max_lags: int = 20) -> pd.DataFrame:
    """
    Calculate correlations between VIX and future returns at different lags.

    Args:
        vix (pd.Series): VIX series
        returns (pd.Series): Returns series
        unemployment (pd.Series): Unemployment series
        max_lags (int): Maximum number of lags to analyze

    Returns:
        pd.DataFrame: Lagged correlation analysis
    """
    logger.info(f"Calculating lagged correlations with max_lags {max_lags}")

    if not STATSMODELS_AVAILABLE:
        logger.warning("statsmodels not available - returning basic correlations")
        return pd.DataFrame()

    # Combine and clean data
    combined_df = pd.DataFrame({
        'vix': vix,
        'returns': returns,
        'unemployment': unemployment
    }).dropna()

    lagged_results = []

    for lag in range(1, max_lags + 1):
        # Shift returns backward (VIX at t correlated with returns at t+lag)
        lagged_returns = combined_df['returns'].shift(-lag)
        lagged_unemployment = combined_df['unemployment'].shift(-lag)

        # Calculate correlations
        vix_returns_corr = combined_df['vix'].corr(lagged_returns)
        vix_unemployment_corr = combined_df['vix'].corr(lagged_unemployment)
        returns_unemployment_corr = combined_df['returns'].corr(lagged_unemployment)

        lagged_results.append({
            'lag': lag,
            'vix_returns_corr': vix_returns_corr,
            'vix_unemployment_corr': vix_unemployment_corr,
            'returns_unemployment_corr': returns_unemployment_corr
        })

    result_df = pd.DataFrame(lagged_results)
    logger.info(f"Calculated lagged correlations for {len(result_df)} lag periods")
    return result_df


def plot_lag_correlations(lag_corr_df: pd.DataFrame) -> Any:
    """
    Visualize lagged correlation analysis.

    Args:
        lag_corr_df (pd.DataFrame): Lagged correlation data

    Returns:
        matplotlib figure object
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Matplotlib not available - skipping plot")
        return None

    logger.info("Creating lag correlation visualization")

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # VIX vs Returns correlation
    axes[0].plot(lag_corr_df['lag'], lag_corr_df['vix_returns_corr'], 'o-', color='blue', linewidth=2)
    axes[0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[0].set_title('VIX vs Future Returns Correlation by Lag')
    axes[0].set_ylabel('Correlation')
    axes[0].grid(True, alpha=0.3)

    # VIX vs Unemployment correlation
    axes[1].plot(lag_corr_df['lag'], lag_corr_df['vix_unemployment_corr'], 'o-', color='red', linewidth=2)
    axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1].set_title('VIX vs Future Unemployment Correlation by Lag')
    axes[1].set_ylabel('Correlation')
    axes[1].grid(True, alpha=0.3)

    # Returns vs Unemployment correlation
    axes[2].plot(lag_corr_df['lag'], lag_corr_df['returns_unemployment_corr'], 'o-', color='green', linewidth=2)
    axes[2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[2].set_title('Returns vs Future Unemployment Correlation by Lag')
    axes[2].set_ylabel('Correlation')
    axes[2].set_xlabel('Lag (periods)')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    logger.info("Lag correlation visualization created")
    return fig


def unemployment_lag_correlation(unemployment: pd.Series, returns: pd.Series,
                               max_lags: int = 12) -> pd.DataFrame:
    """
    Analyze how unemployment lags affect market returns.

    Args:
        unemployment (pd.Series): Unemployment series
        returns (pd.Series): Returns series
        max_lags (int): Maximum lag periods to analyze

    Returns:
        pd.DataFrame: Unemployment lag correlation analysis
    """
    logger.info(f"Analyzing unemployment lag correlation with max_lags {max_lags}")

    # Combine and clean data
    combined_df = pd.DataFrame({
        'unemployment': unemployment,
        'returns': returns
    }).dropna()

    lag_results = []

    for lag in range(max_lags + 1):
        # Shift unemployment backward (unemployment at t correlated with returns at t+lag)
        lagged_unemployment = combined_df['unemployment'].shift(-lag)

        # Calculate correlation
        correlation = combined_df['returns'].corr(lagged_unemployment)

        lag_results.append({
            'lag_months': lag,
            'correlation': correlation,
            'abs_correlation': abs(correlation)
        })

    result_df = pd.DataFrame(lag_results)

    # Find optimal lag
    if len(result_df) > 0:
        optimal_lag = result_df.loc[result_df['abs_correlation'].idxmax()]
        logger.info(f"Optimal unemployment lag: {optimal_lag['lag_months']} months (correlation: {optimal_lag['correlation']:.4f})")

    logger.info(f"Unemployment lag analysis completed for {len(result_df)} lag periods")
    return result_df


def estimate_transition_probabilities(regimes: np.ndarray) -> pd.DataFrame:
    """
    Estimate regime transition probabilities using Markov chain theory.

    Args:
        regimes (np.ndarray): Regime labels

    Returns:
        pd.DataFrame: Transition probability matrix
    """
    logger.info("Estimating transition probabilities")

    # Use the RegimeStatistics method
    stats_calculator = RegimeStatistics()
    transition_matrix = stats_calculator.transition_matrix(regimes)

    logger.info("Transition probability estimation complete")
    return transition_matrix


def plot_transition_network(transition_matrix: pd.DataFrame) -> Any:
    """
    Create network visualization of regime transitions.

    Args:
        transition_matrix (pd.DataFrame): Transition probability matrix

    Returns:
        matplotlib figure object
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Matplotlib not available - skipping plot")
        return None

    logger.info("Creating transition network visualization")

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create heatmap
    sns.heatmap(transition_matrix, annot=True, fmt='.3f', cmap='Blues',
                ax=ax, square=True, cbar_kws={'label': 'Transition Probability'})

    ax.set_title('Regime Transition Probability Matrix')
    ax.set_xlabel('To Regime')
    ax.set_ylabel('From Regime')

    plt.tight_layout()
    logger.info("Transition network visualization created")
    return fig


def correlation_significance_test(x: pd.Series, y: pd.Series, lag: int = 0) -> Dict[str, Any]:
    """
    Test statistical significance of correlation coefficients.

    Args:
        x (pd.Series): First series
        y (pd.Series): Second series
        lag (int): Lag for correlation

    Returns:
        Dict[str, Any]: Significance test results
    """
    logger.info(f"Testing correlation significance with lag {lag}")

    # Align and clean data
    combined_df = pd.DataFrame({'x': x, 'y': y}).dropna()

    if lag != 0:
        combined_df['y'] = combined_df['y'].shift(lag)
        combined_df = combined_df.dropna()

    if len(combined_df) < 3:
        return {'error': 'Insufficient data for correlation test'}

    # Calculate correlation
    correlation = combined_df['x'].corr(combined_df['y'])

    # T-test for correlation significance
    n = len(combined_df)
    t_statistic = correlation * np.sqrt((n - 2) / (1 - correlation**2))
    p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), n - 2))

    result = {
        'correlation': correlation,
        't_statistic': t_statistic,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'sample_size': n,
        'lag': lag
    }

    logger.info(f"Correlation test: r={correlation:.4f}, p={p_value:.4f}, significant={result['significant']}")
    return result


def regime_difference_test(variable_regime1: pd.Series, variable_regime2: pd.Series) -> Dict[str, Any]:
    """
    Test if a variable is significantly different across regimes.

    Args:
        variable_regime1 (pd.Series): Variable values in regime 1
        variable_regime2 (pd.Series): Variable values in regime 2

    Returns:
        Dict[str, Any]: Difference test results
    """
    logger.info("Testing regime difference significance")

    # Clean data
    data1 = variable_regime1.dropna()
    data2 = variable_regime2.dropna()

    if len(data1) < 3 or len(data2) < 3:
        return {'error': 'Insufficient data for difference test'}

    # Perform t-test
    t_statistic, p_value = stats.ttest_ind(data1, data2, equal_var=False)

    result = {
        'mean_regime1': data1.mean(),
        'mean_regime2': data2.mean(),
        't_statistic': t_statistic,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'sample_size_regime1': len(data1),
        'sample_size_regime2': len(data2)
    }

    logger.info(f"Regime difference test: t={t_statistic:.4f}, p={p_value:.4f}, significant={result['significant']}")
    return result


def plot_regime_summary(regime_results: Dict[str, Any]) -> Any:
    """
    Create comprehensive regime summary visualization.

    Args:
        regime_results (Dict[str, Any]): Complete regime analysis results

    Returns:
        matplotlib figure object
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Matplotlib not available - skipping plot")
        return None

    logger.info("Creating comprehensive regime summary visualization")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Regime timeline (if available)
    if 'regime_data' in regime_results and 'dates' in regime_results:
        dates = regime_results['dates']
        regimes = regime_results['regimes']

        unique_regimes = np.unique(regimes)
        colors = ['green', 'yellow', 'red']

        for i, regime in enumerate(unique_regimes):
            regime_mask = regimes == regime
            if regime_mask.any():
                color = colors[i] if i < len(colors) else 'gray'
                axes[0, 0].fill_between(dates, 0, 1, where=regime_mask,
                                       color=color, alpha=0.3, label=f'Regime {regime}')

        axes[0, 0].set_title('Regime Timeline')
        axes[0, 0].set_ylabel('Regime')
        axes[0, 0].legend()

    # Duration statistics (if available)
    if 'duration_stats' in regime_results:
        duration_df = regime_results['duration_stats']
        regimes = duration_df['Regime']
        durations = duration_df['Mean_Days']

        axes[0, 1].bar(regimes.astype(str), durations, color='skyblue', alpha=0.7)
        axes[0, 1].set_title('Average Regime Duration')
        axes[0, 1].set_xlabel('Regime')
        axes[0, 1].set_ylabel('Days')
        axes[0, 1].grid(True, alpha=0.3)

    # Transition matrix heatmap (if available)
    if 'transition_matrix' in regime_results:
        trans_matrix = regime_results['transition_matrix']
        sns.heatmap(trans_matrix, annot=True, fmt='.3f', cmap='Blues', ax=axes[1, 0])
        axes[1, 0].set_title('Transition Probabilities')

    # Return distributions by regime (if available)
    if 'returns_analysis' in regime_results:
        returns_df = regime_results['returns_analysis']
        regimes = returns_df['Regime']
        means = returns_df['Mean_Return']
        stds = returns_df['Std_Return']

        axes[1, 1].errorbar(regimes.astype(str), means, yerr=stds, fmt='o-', capsize=5)
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 1].set_title('Returns by Regime (Mean ± Std)')
        axes[1, 1].set_xlabel('Regime')
        axes[1, 1].set_ylabel('Returns')
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    logger.info("Comprehensive regime summary visualization created")
    return fig


def main():
    """
    Main function demonstrating comprehensive regime analysis.
    """
    logger.info("Starting comprehensive regime analysis demonstration")

    try:
        # Load sample data (in real usage, this would come from scraped data)
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=500, freq='D')

        # Generate synthetic VIX data with regime-like behavior
        vix_base = 20
        vix_data = []
        for i in range(500):
            if 100 <= i < 200:  # High volatility period
                shock = np.random.normal(0, 3)
            elif 300 <= i < 400:  # Extreme volatility period
                shock = np.random.normal(0, 5)
            else:  # Normal volatility
                shock = np.random.normal(0, 1)
            vix_base = max(5, min(80, vix_base + shock))
            vix_data.append(vix_base)

        # Generate correlated returns and unemployment data
        returns_data = []
        unemployment_data = []
        base_unemployment = 6.0

        for vix in vix_data:
            vol_factor = vix / 20
            ret = np.random.normal(0.0005, 0.02 * vol_factor)
            returns_data.append(ret)

            # Unemployment lags market (with some noise)
            unemployment_shock = np.random.normal(0, 0.1)
            base_unemployment += unemployment_shock * 0.01
            base_unemployment = max(3, min(15, base_unemployment))
            unemployment_data.append(base_unemployment)

        # Create DataFrames
        market_data = pd.DataFrame({
            'date': dates,
            'vix': vix_data,
            'returns': returns_data,
            'price': 100 + np.cumsum(returns_data)
        })

        economic_data = pd.DataFrame({
            'date': dates,
            'unemployment': unemployment_data
        })

        # Step 1: Identify regimes using percentile method
        logger.info("Step 1: Identifying regimes")
        regime_identifier = RegimeIdentification(market_data['vix'], method='percentile')
        regimes = regime_identifier.identify_regimes_percentile()
        smoothed_regimes = regime_identifier.smooth_regimes(regimes)

        # Step 2: Calculate regime statistics
        logger.info("Step 2: Calculating regime statistics")
        stats_calculator = RegimeStatistics()

        duration_stats = stats_calculator.regime_durations(smoothed_regimes, market_data['date'])
        frequency_stats = stats_calculator.regime_frequency(smoothed_regimes)
        returns_analysis = stats_calculator.regime_returns_analysis(market_data['returns'], smoothed_regimes)
        transition_matrix = stats_calculator.transition_matrix(smoothed_regimes)

        # Step 3: Rolling correlation analysis
        logger.info("Step 3: Rolling correlation analysis")
        rolling_corr = rolling_correlation(
            market_data.set_index('date')['vix'],
            market_data.set_index('date')['returns'],
            economic_data.set_index('date')['unemployment']
        )

        # Step 4: Volatility clustering detection
        logger.info("Step 4: Volatility clustering detection")
        clustering_results = detect_volatility_clustering(market_data['returns'])

        # Step 5: Asymmetric response analysis
        logger.info("Step 5: Asymmetric response analysis")
        asymmetry_results = analyze_asymmetric_response(
            market_data['returns'], market_data['vix'], smoothed_regimes
        )

        # Step 6: Lag analysis
        logger.info("Step 6: Lag analysis")
        lagged_corr = calculate_lagged_correlations(
            market_data.set_index('date')['vix'],
            market_data.set_index('date')['returns'],
            economic_data.set_index('date')['unemployment']
        )

        unemployment_lags = unemployment_lag_correlation(
            economic_data.set_index('date')['unemployment'],
            market_data.set_index('date')['returns']
        )

        # Step 7: Create summary visualizations
        logger.info("Step 7: Creating visualizations")
        if MATPLOTLIB_AVAILABLE:
            # Regime timeline
            timeline_fig = stats_calculator.plot_regime_timeline(
                market_data['date'], smoothed_regimes, market_data['price']
            )

            # Rolling correlations
            rolling_fig = plot_rolling_correlation(rolling_corr)

            # Volatility clustering
            clustering_fig = visualize_volatility_clusters(market_data['returns'], market_data['date'])

            # Asymmetry analysis
            asymmetry_fig = plot_asymmetry(asymmetry_results)

            # Lag correlations
            lag_fig = plot_lag_correlations(lagged_corr)

            # Transition network
            transition_fig = plot_transition_network(transition_matrix)

            # Comprehensive summary
            regime_results = {
                'dates': market_data['date'],
                'regimes': smoothed_regimes,
                'duration_stats': duration_stats,
                'transition_matrix': transition_matrix,
                'returns_analysis': returns_analysis
            }
            summary_fig = plot_regime_summary(regime_results)

        # Step 8: Save results
        logger.info("Step 8: Saving results")

        # Save statistics to CSV
        duration_stats.to_csv('regime_duration_stats.csv', index=False)
        frequency_stats.to_csv('regime_frequency_stats.csv', index=False)
        returns_analysis.to_csv('regime_returns_analysis.csv', index=False)
        transition_matrix.to_csv('regime_transition_matrix.csv')
        rolling_corr.to_csv('rolling_correlations.csv', index=False)
        lagged_corr.to_csv('lagged_correlations.csv', index=False)
        unemployment_lags.to_csv('unemployment_lags.csv', index=False)

        # Save analysis results as JSON
        analysis_summary = {
            'volatility_clustering': clustering_results,
            'asymmetric_response': asymmetry_results,
            'regime_summary': {
                'total_regimes': len(np.unique(smoothed_regimes)),
                'total_observations': len(smoothed_regimes),
                'avg_regime_duration': duration_stats['Mean_Days'].mean()
            }
        }

        # Convert numpy types to Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {str(k): convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj

        analysis_summary = convert_numpy_types(analysis_summary)

        import json
        with open('regime_analysis_summary.json', 'w') as f:
            json.dump(analysis_summary, f, indent=2, default=str)

        # Save plots if matplotlib available
        if MATPLOTLIB_AVAILABLE:
            timeline_fig.savefig('regime_timeline.png', dpi=300, bbox_inches='tight')
            rolling_fig.savefig('rolling_correlations.png', dpi=300, bbox_inches='tight')
            clustering_fig.savefig('volatility_clustering.png', dpi=300, bbox_inches='tight')
            asymmetry_fig.savefig('asymmetric_response.png', dpi=300, bbox_inches='tight')
            lag_fig.savefig('lag_correlations.png', dpi=300, bbox_inches='tight')
            transition_fig.savefig('transition_network.png', dpi=300, bbox_inches='tight')
            summary_fig.savefig('regime_summary.png', dpi=300, bbox_inches='tight')

        logger.info("Comprehensive regime analysis completed successfully!")
        logger.info("Results saved to CSV files and PNG plots")

        # Print summary
        print("\n" + "="*80)
        print("COMPREHENSIVE REGIME ANALYSIS COMPLETED")
        print("="*80)
        print(f"Total observations: {len(market_data)}")
        print(f"Regimes identified: {len(np.unique(smoothed_regimes))}")
        print(f"Volatility clustering detected: {clustering_results.get('is_clustered', 'N/A')}")
        print(f"Files saved: 7 CSV files, 7 PNG plots, 1 JSON summary")
        print("="*80)

    except Exception as e:
        logger.error(f"Error in main analysis: {str(e)}")
        raise


if __name__ == "__main__":
    main()