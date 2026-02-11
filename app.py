"""
Stock Market AI Analytics Dashboard
===================================

Main Streamlit application entry point for comprehensive financial analysis.
Provides multi-page navigation, real-time monitoring, and AI-powered insights.

Author: GitHub Copilot
Date: February 8, 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

import html
from io import BytesIO
import warnings

# Optional plotting libraries — initialize safely for headless environments
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for servers/containers
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except Exception as e:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    warnings.warn(f"matplotlib not available; plotting disabled: {e}")

# Configure page
st.set_page_config(
    page_title="Stock Market AI Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/stock-market-analysis',
        'Report a bug': 'https://github.com/your-repo/stock-market-analysis/issues',
        'About': '''
        # Stock Market AI Analytics Dashboard

        A comprehensive financial analysis platform featuring:
        - Real-time market data monitoring
        - AI-powered regime detection
        - Multi-model prediction systems
        - Interactive data visualization
        - Automated alert generation

        Built with Streamlit and machine learning.
        '''
    }
)

# Custom CSS for professional styling
def load_css():
    """Load custom CSS for enhanced styling."""
    css = """
    <style>
    /* Main container styling */
    .main {
        background-color: #f8f9fa;
    }

    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }

    .header-title {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }

    .header-subtitle {
        font-size: 1.2rem;
        opacity: 0.9;
    }

    /* Metric cards styling */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #333;
    }

    .metric-label {
        font-size: 0.9rem;
        color: #666;
        margin-top: 0.5rem;
    }

    /* Sidebar styling */
    .sidebar-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        text-align: center;
    }

    .sidebar-title {
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }

    .sidebar-subtitle {
        font-size: 0.8rem;
        opacity: 0.9;
    }

    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }

    /* Status indicators */
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 0.5rem;
    }

    .status-active {
        background-color: #28a745;
    }

    .status-inactive {
        background-color: #dc3545;
    }

    .status-loading {
        background-color: #ffc107;
    }

    /* Footer styling */
    .footer {
        background: #343a40;
        color: white;
        padding: 1rem;
        text-align: center;
        border-radius: 10px;
        margin-top: 2rem;
        font-size: 0.9rem;
    }

    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
        .main {
            background-color: #1a1a1a;
        }

        .metric-card {
            background: #2d2d2d;
            color: white;
        }

        .metric-value {
            color: #e9ecef;
        }

        .metric-label {
            color: #adb5bd;
        }
    }

    /* Responsive design */
    @media (max-width: 768px) {
        .header-title {
            font-size: 2rem;
        }

        .metric-card {
            padding: 1rem;
        }

        .metric-value {
            font-size: 1.5rem;
        }
    }

    /* Animation for loading states */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }

    .loading {
        animation: pulse 2s infinite;
    }

    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #f1f1f1;
    }

    ::-webkit-scrollbar-thumb {
        background: #667eea;
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #5a67d8;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Load custom CSS
load_css()

# Import our custom modules
try:
    from real_time_monitor import RealtimeDataFeed
    MONITOR_AVAILABLE = True
except ImportError:
    MONITOR_AVAILABLE = False
    st.warning("⚠️ Real-time monitor not available")

try:
    from preprocessing_pipeline import DataPreprocessingPipeline as DataPreprocessor
    PREPROCESSOR_AVAILABLE = True
except ImportError:
    PREPROCESSOR_AVAILABLE = False

# Model training imports (optional) - commented out due to matplotlib import issues
# try:
#     from baseline_models import ModelTrainer
#     MODELS_AVAILABLE = True
# except (ImportError, AttributeError):
#     MODELS_AVAILABLE = False
#     ModelTrainer = None

MODELS_AVAILABLE = False
ModelTrainer = None

try:
    from prediction_updater import RealtimePredictorUpdater
    PREDICTOR_AVAILABLE = True
except ImportError:
    PREDICTOR_AVAILABLE = False

# Set up logging
import os
os.makedirs('logs', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/dashboard.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_DATE_RANGE = 365 * 7  # 7 years to include historical data
CACHE_TTL = 3600  # 1 hour in seconds
CONFIG_FILE = "config/dashboard_config.json"

# Create necessary directories
os.makedirs("data/cache", exist_ok=True)
os.makedirs("data/models", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("pages", exist_ok=True)

# Session state initialization
def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'models_loaded' not in st.session_state:
        st.session_state.models_loaded = False
    if 'predictions_updated' not in st.session_state:
        st.session_state.predictions_updated = False
    if 'last_update' not in st.session_state:
        st.session_state.last_update = datetime.now()
    if 'config' not in st.session_state:
        st.session_state.config = load_config()
    if 'data_cache' not in st.session_state:
        st.session_state.data_cache = {}
    if 'model_cache' not in st.session_state:
        st.session_state.model_cache = {}

def load_config() -> Dict[str, Any]:
    """Load dashboard configuration from file."""
    default_config = {
        "date_range_days": DEFAULT_DATE_RANGE,
        "auto_refresh_interval": 300,  # 5 minutes
        "theme": "light",
        "default_regime": "all",
        "cache_enabled": True,
        "max_cache_age": CACHE_TTL,
        "data_sources": ["NSE", "Investing.com", "data.gov.in"],
        "alert_thresholds": {
            "volatility_high": 0.05,
            "prediction_confidence_low": 0.6,
            "regime_change": True
        }
    }

    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                loaded_config = json.load(f)
                default_config.update(loaded_config)
                logger.info("Configuration loaded from file")
        else:
            # Save default config
            os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
            with open(CONFIG_FILE, 'w') as f:
                json.dump(default_config, f, indent=2)
            logger.info("Default configuration saved")
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        st.error(f"⚠️ Configuration error: {e}")

    return default_config

@st.cache_data(ttl=CACHE_TTL)
def load_market_data(start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """Load market data with caching."""
    try:
        # Load actual data from processed datasets
        data_path = "stationary_data.csv"

        if os.path.exists(data_path):
            # Load the stationary data
            df = pd.read_csv(data_path, index_col=0, parse_dates=True)
            df = df.reset_index()
            df = df.rename(columns={'index': 'Date'})

            # Filter by date range if provided and if data exists in that range
            if start_date and end_date:
                df['Date'] = pd.to_datetime(df['Date'])
                filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

                # If no data in the filtered range, show all available data
                if len(filtered_df) == 0:
                    logger.warning(f"No data found in date range {start_date} to {end_date}, showing all available data")
                    df_filtered = df.copy()
                else:
                    df_filtered = filtered_df
            else:
                df_filtered = df.copy()

            # Add market regime column if it doesn't exist
            if 'MarketRegime' not in df_filtered.columns:
                # Simple regime detection based on returns
                if 'Returns' in df_filtered.columns:
                    df_filtered['MarketRegime'] = df_filtered['Returns'].apply(
                        lambda x: 'bull' if x > 0.005 else ('bear' if x < -0.005 else 'sideways')
                    )
                else:
                    df_filtered['MarketRegime'] = 'unknown'

            logger.info(f"Market data loaded from {data_path}: {len(df_filtered)} records")
            return df_filtered

        else:
            # Fallback to sample data if file doesn't exist
            logger.warning(f"Data file {data_path} not found, using sample data")
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            np.random.seed(42)

            data = {
                'Date': dates,
                'NIFTY_Close': 18000 + np.cumsum(np.random.normal(0, 50, len(dates))),
                'VIX_Close': 15 + np.random.normal(0, 3, len(dates)),
                'UnemploymentRate': 7.5 + np.random.normal(0, 0.5, len(dates)),
                'MarketRegime': np.random.choice(['bull', 'bear', 'sideways'], len(dates))
            }

            df = pd.DataFrame(data)
            df['Date'] = pd.to_datetime(df['Date'])
            logger.info(f"Sample market data generated: {len(df)} records")
            return df

    except Exception as e:
        logger.error(f"Error loading market data: {e}")
        return None

@st.cache_resource
def load_trained_models():
    """Load trained models with caching."""
    try:
        models = {}
        scalers = {}

        # Try to load actual trained models
        model_files = {
            'ols_static': 'models/ols_static.joblib',
            'random_forest': 'models/random_forest.joblib',
            'xgboost': 'models/xgboost.joblib',
            'lightgbm': 'models/lightgbm.joblib',
            'ensemble': 'models/ensemble.joblib'
        }

        scaler_file = 'models/scalers.joblib'

        # Load models
        for model_name, model_path in model_files.items():
            if os.path.exists(model_path):
                try:
                    import joblib
                    models[model_name] = joblib.load(model_path)
                    logger.info(f"Loaded model: {model_name}")
                except Exception as e:
                    logger.warning(f"Failed to load model {model_name}: {e}")
            else:
                logger.warning(f"Model file not found: {model_path}")

        # Load scalers
        if os.path.exists(scaler_file):
            try:
                import joblib
                scalers = joblib.load(scaler_file)
                logger.info("Loaded scalers")
            except Exception as e:
                logger.warning(f"Failed to load scalers: {e}")
                scalers = {}
        else:
            logger.warning(f"Scaler file not found: {scaler_file}")

        # If no models loaded, create dummy models for demo
        if not models:
            logger.warning("No models loaded, creating dummy models")
            from sklearn.linear_model import LinearRegression
            from sklearn.preprocessing import StandardScaler

            models = {
                'ols_static': LinearRegression(),
                'random_forest': LinearRegression(),  # Placeholder
                'xgboost': LinearRegression(),  # Placeholder
                'lightgbm': LinearRegression(),  # Placeholder
                'ensemble': LinearRegression()  # Placeholder
            }

            scalers = {
                'VIX': StandardScaler(),
                'UnemploymentRate': StandardScaler(),
                'MarketReturn': StandardScaler()
            }

        logger.info(f"Models loaded: {len(models)} models, {len(scalers)} scalers")
        return models, scalers

    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return {}, {}

@st.cache_data
def load_prediction_history() -> Optional[pd.DataFrame]:
    """Load prediction history with caching."""
    try:
        # Load from prediction history CSV
        history_file = "data/cache/prediction_history.csv"
        if os.path.exists(history_file):
            df = pd.read_csv(history_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            # Rename columns to match expected format
            column_mapping = {
                'consensus_mean': 'prediction',
                'agreement_strength': 'confidence',
                'market_return': 'regime'
            }
            df = df.rename(columns=column_mapping)
            # Add default values for missing columns
            if 'prediction' not in df.columns:
                df['prediction'] = df.get('consensus_mean', 0)
            if 'confidence' not in df.columns:
                df['confidence'] = df.get('agreement_strength', 0.5)
            if 'regime' not in df.columns:
                df['regime'] = 'unknown'
            return df
        else:
            # Return empty DataFrame if no history
            return pd.DataFrame(columns=['timestamp', 'prediction', 'confidence', 'regime'])
    except Exception as e:
        logger.error(f"Error loading prediction history: {e}")
        return pd.DataFrame()


def load_model_comparison_data():
    """Attempt to load precomputed model comparison data into session state.

    This function searches common output paths (CSV/JSON) produced by the
    project's evaluation scripts and will populate `st.session_state['model_comparison_data']`
    with a dict containing at least `predictions`, `actual_values`, and optionally `dates`.
    """
    # Do nothing if already present
    if st.session_state.get('model_comparison_data'):
        return

    candidates = [
        os.path.join('comparison_results', 'model_comparison_summary.csv'),
        os.path.join('comparison_results', 'model_comparison_summary.json'),
        'model_comparison_summary.csv',
        os.path.join('comparison_results', 'detailed_model_metrics.json'),
        os.path.join('comparison_results', 'dashboard_data.json')
    ]

    for path in candidates:
        if not os.path.exists(path):
            continue

        try:
            if path.endswith('.csv'):
                df = pd.read_csv(path)

                # Heuristic mapping: look for an `actual` or `y_true` column
                possible_actual_cols = [c for c in df.columns if c.lower() in ('actual', 'y_true', 'y_true_values', 'ground_truth')]
                if possible_actual_cols:
                    actual_col = possible_actual_cols[0]
                    y_true = df[actual_col].values
                else:
                    # If not present, assume second column is actual
                    y_true = df.iloc[:, 1].values if df.shape[1] >= 2 else None

                # Predictions are any remaining numeric columns
                pred_cols = [c for c in df.columns if c not in (['Date', 'date', possible_actual_cols[0]] if possible_actual_cols else ['Date', 'date'])]
                predictions = {}
                for col in pred_cols:
                    # Skip non-numeric columns
                    try:
                        predictions[col] = pd.to_numeric(df[col]).values
                    except Exception:
                        continue

                dates = None
                if 'Date' in df.columns or 'date' in df.columns:
                    date_col = 'Date' if 'Date' in df.columns else 'date'
                    try:
                        dates = pd.to_datetime(df[date_col]).tolist()
                    except Exception:
                        dates = None

                st.session_state['model_comparison_data'] = {
                    'predictions': predictions,
                    'actual_values': y_true,
                    'dates': dates
                }
                logger.info(f"Loaded model comparison data from {path}")
                return

            else:
                # JSON or other structured file
                with open(path, 'r') as f:
                    obj = json.load(f)

                # Basic validation and assignment
                if isinstance(obj, dict) and ('predictions' in obj or 'actual_values' in obj or 'summary' in obj):
                    st.session_state['model_comparison_data'] = obj
                    logger.info(f"Loaded model comparison data from {path}")
                    return

        except Exception as e:
            logger.warning(f"Failed to load model comparison file {path}: {e}")

    # If we reach here nothing was loaded; leave key unset or empty
    logger.debug("No model comparison files found; session state not populated.")

# Utility functions
def get_status_indicator(status: bool) -> str:
    """Get HTML for status indicator."""
    color_class = "status-active" if status else "status-inactive"
    return f'<span class="status-indicator {color_class}"></span>'

def format_metric_value(value: Any, format_type: str = "number") -> str:
    """Format metric values for display."""
    if value is None:
        return "N/A"

    try:
        if format_type == "currency":
            return f"₹{float(value):,.0f}"
        elif format_type == "percentage":
            return f"{float(value):.1f}%"
        elif format_type == "number":
            if isinstance(value, float):
                return f"{value:,.2f}"
            else:
                return str(value)
        elif format_type == "date":
            if isinstance(value, (str, pd.Timestamp)):
                return pd.to_datetime(value).strftime("%Y-%m-%d")
            return str(value)
        else:
            return str(value)
    except:
        return str(value)

def create_metric_card(title: str, value: Any, subtitle: str = "",
                      format_type: str = "number", delta: Optional[float] = None):
    """Create a styled metric card."""
    delta_html = ""
    if delta is not None:
        delta_color = "green" if delta >= 0 else "red"
        delta_sign = "+" if delta >= 0 else ""
        # escape delta display as text
        delta_html = f'<div style="color: {delta_color}; font-size: 0.8rem;">{delta_sign}{html.escape(f"{delta:.2f}%")}</div>'

    # Escape subtitle to avoid raw HTML injection and render consistently
    subtitle_html = ""
    if subtitle:
        subtitle_html = f'<div style="font-size: 0.8rem; color: #666; margin-top: 0.5rem;">{html.escape(str(subtitle))}</div>'

    card_html = (
        f'<div class="metric-card">'
        f'<div class="metric-label">{html.escape(str(title))}</div>'
        f'<div class="metric-value">{html.escape(str(format_metric_value(value, format_type)))}</div>'
        f'{delta_html}'
        f'{subtitle_html}'
        f'</div>'
    )

    st.markdown(card_html, unsafe_allow_html=True)

def export_visualization():
    """Render a simple market plot and provide a PNG download button."""
    if not MATPLOTLIB_AVAILABLE:
        st.error("Plotting disabled: matplotlib is not available in this environment.")
        return

    # Additional runtime guard for static type checkers (plt may be None at import failure)
    if plt is None:
        st.error("Plotting disabled: matplotlib is not available in this environment.")
        return

    data = None
    if 'data_cache' in st.session_state:
        data = st.session_state.data_cache.get('market_data')
    if data is None or len(data) == 0:
        st.info("No market data available to plot.")
        return

    try:
        df = data.copy()
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            x = df['Date']
        else:
            x = df.index

        if 'NIFTY_Close' in df.columns:
            y = df['NIFTY_Close']
            ylabel = 'NIFTY_Close'
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                y = df[numeric_cols[0]]
                ylabel = numeric_cols[0]
            else:
                st.info("No numeric data found to plot.")
                return

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(x, y, color='#2b6cb0')
        ax.set_xlabel('Date')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{ylabel} over time')
        fig.autofmt_xdate()

        st.pyplot(fig)

        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)

        st.download_button(
            label='Download plot (PNG)',
            data=buf,
            file_name='market_plot.png',
            mime='image/png'
        )
    except Exception as e:
        st.error(f"Failed to generate visualization: {e}")

# Sidebar components
def create_sidebar():
    """Create the sidebar with controls and information."""
    with st.sidebar:
        # Header
        st.markdown("""
        <div class="sidebar-header">
            <div class="sidebar-title">📈 AI Analytics</div>
            <div class="sidebar-subtitle">Stock Market Intelligence</div>
        </div>
        """, unsafe_allow_html=True)

        # General dashboard filters
        st.subheader("🔎 General Filters")

        config = st.session_state.get('config', {})
        default_days = config.get('date_range_days', DEFAULT_DATE_RANGE)
        end_default = datetime.now().date()
        start_default = (datetime.now() - timedelta(days=default_days)).date()

        date_range = st.date_input(
            "Date range",
            value=(start_default, end_default)
        )

        # `st.date_input` may return a single date or a (start, end) tuple/list.
        if isinstance(date_range, (tuple, list)):
            if len(date_range) >= 2:
                start_date, end_date = date_range[0], date_range[1]
            elif len(date_range) == 1:
                start_date = end_date = date_range[0]
            else:
                start_date, end_date = start_default, end_default
        else:
            start_date = end_date = date_range

        regime = st.selectbox(
            "Market regime",
            options=["all", "bull", "bear", "sideways"],
            index=["all", "bull", "bear", "sideways"].index(config.get('default_regime', 'all'))
        )

        # Models multiselect
        available_models = []
        if st.session_state.get('models_loaded') and 'models' in st.session_state.get('model_cache', {}):
            try:
                available_models = list(st.session_state.model_cache['models'].keys())
            except Exception:
                available_models = []
        if not available_models:
            available_models = ['ols_static', 'random_forest', 'xgboost', 'lightgbm', 'ensemble']

        selected_models = st.multiselect(
            "Models",
            options=available_models,
            default=available_models
        )

        show_predictions = st.checkbox("Show model predictions", value=True)

        if st.button("Apply Filters"):
            st.session_state['filters'] = {
                'start_date': start_date.strftime('%Y-%m-%d') if hasattr(start_date, 'strftime') else str(start_date),
                'end_date': end_date.strftime('%Y-%m-%d') if hasattr(end_date, 'strftime') else str(end_date),
                'regime': regime,
                'models': selected_models,
                'show_predictions': show_predictions
            }
            st.success("Filters applied")

        st.divider()

        # Hypothesis (kept concise)
        st.subheader("🧪 Hypothesis")
        st.markdown("""
        - Market regimes meaningfully affect short-term prediction error.
        - Volatility spikes increase model disagreement and lower confidence.
        - Combining macro indicators improves regime detection accuracy.
        """)

        st.divider()

        # Data availability status
        st.subheader("📦 Data Availability")
        data_points = 0
        if st.session_state.data_loaded and 'market_data' in st.session_state.data_cache:
            data = st.session_state.data_cache.get('market_data')
            if data is not None:
                data_points = len(data)

        status_html = get_status_indicator(data_points > 0)
        st.markdown(f"{status_html} {data_points} records available", unsafe_allow_html=True)

        st.divider()

        # Model status
        st.subheader("🤖 Model Status")
        models_count = 0
        if st.session_state.models_loaded and 'models' in st.session_state.model_cache:
            models = st.session_state.model_cache['models']
            if models:
                models_count = len(models)

        model_html = get_status_indicator(models_count > 0)
        st.markdown(f"{model_html} {models_count} models loaded", unsafe_allow_html=True)

        st.divider()

        # Last updated
        st.caption(f"Last updated: {st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S')}")

    return None

def refresh_data():
    """Refresh all data and clear caches."""
    try:
        # Clear caches
        load_market_data.clear()
        load_trained_models.clear()
        load_prediction_history.clear()

        # Reset session state
        st.session_state.data_loaded = False
        st.session_state.models_loaded = False
        st.session_state.predictions_updated = False
        st.session_state.data_cache = {}
        st.session_state.model_cache = {}

        # Reload data
        initialize_data()

        st.success("✅ Data refreshed successfully!")
        st.rerun()

    except Exception as e:
        logger.error(f"Error refreshing data: {e}")
        st.error(f"❌ Failed to refresh data: {e}")

def show_settings_modal():
    """Show settings modal dialog."""
    with st.container():
        st.subheader("⚙️ Dashboard Settings")

        config = st.session_state.config.copy()

        # Theme settings
        theme = st.selectbox(
            "Theme",
            options=["light", "dark", "auto"],
            index=["light", "dark", "auto"].index(config.get('theme', 'light'))
        )

        # Auto refresh settings
        auto_refresh = st.slider(
            "Auto-refresh interval (minutes)",
            min_value=1,
            max_value=60,
            value=config.get('auto_refresh_interval', 300) // 60
        )

        # Cache settings
        cache_enabled = st.checkbox(
            "Enable caching",
            value=config.get('cache_enabled', True)
        )

        # Alert thresholds
        st.subheader("Alert Thresholds")
        volatility_threshold = st.slider(
            "Volatility threshold (%)",
            min_value=1.0,
            max_value=10.0,
            value=config.get('alert_thresholds', {}).get('volatility_high', 0.05) * 100
        ) / 100

        confidence_threshold = st.slider(
            "Min prediction confidence (%)",
            min_value=50.0,
            max_value=95.0,
            value=config.get('alert_thresholds', {}).get('prediction_confidence_low', 0.6) * 100
        ) / 100

        if st.button("💾 Save Settings"):
            # Update config
            config.update({
                'theme': theme,
                'auto_refresh_interval': auto_refresh * 60,
                'cache_enabled': cache_enabled,
                'alert_thresholds': {
                    'volatility_high': volatility_threshold,
                    'prediction_confidence_low': confidence_threshold,
                    'regime_change': config.get('alert_thresholds', {}).get('regime_change', True)
                }
            })

            # Save to file
            try:
                with open(CONFIG_FILE, 'w') as f:
                    json.dump(config, f, indent=2)

                st.session_state.config = config
                st.success("✅ Settings saved successfully!")

                # Apply theme if changed
                if theme != st.session_state.config.get('theme'):
                    st.rerun()

            except Exception as e:
                st.error(f"❌ Failed to save settings: {e}")

def initialize_data():
    """Initialize all data and models."""
    try:
        with st.spinner("Loading market data..."):
            # Load market data
            config = st.session_state.config
            start_date = (datetime.now() - timedelta(days=config.get('date_range_days', DEFAULT_DATE_RANGE))).strftime('%Y-%m-%d')
            end_date = datetime.now().strftime('%Y-%m-%d')

            market_data = load_market_data(start_date, end_date)
            if market_data is not None:
                st.session_state.data_cache['market_data'] = market_data
                st.session_state.data_loaded = True
                logger.info("Market data loaded successfully")
            else:
                st.warning("⚠️ Failed to load market data")

        with st.spinner("Loading trained models..."):
            # Load models
            models, scalers = load_trained_models()
            if models:
                st.session_state.model_cache['models'] = models
                st.session_state.model_cache['scalers'] = scalers
                st.session_state.models_loaded = True
                logger.info("Models loaded successfully")
            else:
                st.warning("⚠️ Failed to load trained models")

        with st.spinner("Loading prediction history..."):
            # Load prediction history
            prediction_history = load_prediction_history()
            if prediction_history is not None:
                st.session_state.data_cache['prediction_history'] = prediction_history
                st.session_state.predictions_updated = True
                logger.info("Prediction history loaded successfully")
            else:
                st.warning("⚠️ Failed to load prediction history")


        # Update last update timestamp
        st.session_state.last_update = datetime.now()

    except Exception as e:
        logger.error(f"Error initializing data: {e}")
        st.error(f"❌ Initialization error: {e}")

def create_navigation():
    """Create multi-page navigation."""
    pages = [
        st.Page("pages/01_home.py", title="🏠 Home", icon=":material/home:"),
        st.Page("pages/02_data_explorer.py", title="📊 Data Explorer", icon=":material/analytics:"),
        st.Page("pages/03_regime_analysis.py", title="🎯 Regime Analysis", icon=":material/trending_up:"),
        st.Page("pages/05_real_time_monitor.py", title="📈 Real-time Monitor", icon=":material/monitoring:"),
        st.Page("pages/06_complex_systems.py", title="🧠 Complex Systems", icon=":material/psychology:"),
        st.Page("pages/07_documentation.py", title="📚 Documentation", icon=":material/library_books:"),
    ]
    

def create_footer():
    """Create the footer section."""
    st.markdown("""
    <div class="footer">
        <strong>Stock Market AI Analytics Dashboard</strong> | Version 1.0.0<br>
        Built with ❤️ using Streamlit & Machine Learning | Last updated: February 8, 2026
    </div>
    """, unsafe_allow_html=True)

def create_header():
    """Create the main header section."""
    st.markdown("""
    <div class="header-container">
        <div class="header-title">📈 Stock Market AI Analytics Dashboard</div>
        <div class="header-subtitle">Real-time Financial Intelligence & AI-Powered Insights</div>
    </div>
    """, unsafe_allow_html=True)

def create_status_metrics():
    """Create status metrics row."""
    col1, col2, col3 = st.columns(3)

    with col1:
        data_points = 0
        if st.session_state.data_loaded and 'market_data' in st.session_state.data_cache:
            data = st.session_state.data_cache.get('market_data')
            if data is not None:
                try:
                    data_points = len(data)
                except Exception:
                    data_points = 0

        create_metric_card(
            "Data Points Loaded",
            data_points,
            "Market data records",
            "number"
        )

    with col2:
        models_count = 0
        if st.session_state.models_loaded and 'models' in st.session_state.model_cache:
            models = st.session_state.model_cache['models']
            if models:
                try:
                    models_count = len(models)
                except Exception:
                    models_count = 0

        create_metric_card(
            "Active Models",
            models_count,
            "Trained prediction models",
            "number"
        )

    with col3:
        last_update = st.session_state.get('last_update', datetime.now())
        try:
            time_since_update = (datetime.now() - last_update).total_seconds() / 60  # minutes
            last_val = f"{time_since_update:.1f}"
        except Exception:
            last_val = "N/A"

        create_metric_card(
            "Last Update",
            last_val,
            "Minutes ago",
            "number"
        )

def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()

    # Initialize data if not loaded
    if not st.session_state.data_loaded:
        initialize_data()
    # Create sidebar
    sidebar_data = create_sidebar()

    # Always render main/home content so visuals appear reliably
    create_header()

    # Status metrics
    st.subheader("📊 System Overview")
    create_status_metrics()

    # Horizontal tabs (Home + other pages) placed on Home page
    pages = {
        "🏠 Home": "pages/01_home.py",
        "📊 Data Explorer": "pages/02_data_explorer.py",
        "🎯 Regime Analysis": "pages/03_regime_analysis.py",
        # Model Comparison removed
        "📈 Real-time Monitor": "pages/05_real_time_monitor.py",
        "🧠 Complex Systems": "pages/06_complex_systems.py",
        "📚 Documentation": "pages/07_documentation.py"
    }

    tab_labels = list(pages.keys())
    tabs = st.tabs(tab_labels)

    # Render each tab's content; for non-Home tabs attempt to open the page
    for label, tab in zip(tab_labels, tabs):
        with tab:
            if label == "🏠 Home":
                # Home content
                st.markdown("""
                ## Welcome to Stock Market AI Analytics! 🚀

                This comprehensive dashboard provides real-time financial analysis and AI-powered insights for informed trading decisions.

                ### Key Features:
                - **📊 Data Explorer**: Interactive market data visualization
                - **🎯 Regime Analysis**: AI-powered market regime detection
                - **🤖 Model Comparison**: Multi-model prediction performance analysis
                - **📈 Real-time Monitor**: Live market data and predictions
                - **🧠 Complex Systems**: Advanced financial modeling
                - **📚 Documentation**: Complete system documentation

                ### Getting Started:
                1. Use the sidebar to configure date ranges and filters
                2. Navigate through different analysis modules using the tabs above
                3. Monitor real-time predictions and alerts
                4. Explore interactive visualizations and insights

                ---
                """)

                # Visualizations (exportable)
                st.subheader("Visualizations")
                export_visualization()

                # Footer for Home
                create_footer()
            else:
                # Non-home tabs: render the page inline instead of auto-switching.
                # Calling `st.switch_page` here caused immediate navigation when the
                # app rendered the second tab (Data Explorer). Instead we run the
                # page script so its Streamlit UI appears inside the tab context.
                target = pages[label]
                try:
                    import runpy
                    runpy.run_path(target, run_name="__main__")
                except FileNotFoundError:
                    st.error(f"Page not found: {target}")
                except Exception as e:
                    st.error(f"Failed to load page {target}: {e}")

if __name__ == "__main__":
    main()
