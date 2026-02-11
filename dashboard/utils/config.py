"""
Dashboard Configuration Constants
=================================

Centralized configuration for the financial analytics dashboard.

Author: GitHub Copilot
Date: February 8, 2026
"""

from typing import Dict, List, Any

# Color schemes for consistent theming
COLORS = {
    # Regime colors
    'regime': {
        'bull': '#2E8B57',      # Sea Green
        'bear': '#DC143C',      # Crimson
        'sideways': '#4682B4',  # Steel Blue
        'volatile': '#FF6347',  # Tomato
        'calm': '#32CD32',      # Lime Green
        'unknown': '#808080'    # Gray
    },

    # Performance colors
    'performance': {
        'excellent': '#228B22',  # Forest Green
        'good': '#32CD32',       # Lime Green
        'average': '#FFD700',    # Gold
        'poor': '#FF6347',       # Tomato
        'terrible': '#DC143C'    # Crimson
    },

    # Metric colors
    'metric': {
        'positive': '#2E8B57',   # Sea Green
        'negative': '#DC143C',   # Crimson
        'neutral': '#4682B4',    # Steel Blue
        'warning': '#FFA500',    # Orange
        'success': '#32CD32',    # Lime Green
        'error': '#FF0000'       # Red
    },

    # Chart colors
    'chart': {
        'primary': '#1f77b4',    # Blue
        'secondary': '#ff7f0e',  # Orange
        'tertiary': '#2ca02c',   # Green
        'quaternary': '#d62728', # Red
        'quinary': '#9467bd',    # Purple
        'senary': '#8c564b',     # Brown
        'septenary': '#e377c2',  # Pink
        'octonary': '#7f7f7f'    # Gray
    },

    # Status colors
    'status': {
        'online': '#32CD32',     # Lime Green
        'offline': '#DC143C',    # Crimson
        'warning': '#FFA500',    # Orange
        'maintenance': '#FFD700', # Gold
        'unknown': '#808080'     # Gray
    }
}

# Page configuration
PAGE_CONFIG = {
    'home': {
        'title': '🏠 Home',
        'icon': '🏠',
        'description': 'Executive summary and project overview'
    },
    'data_explorer': {
        'title': '📊 Data Explorer',
        'icon': '📊',
        'description': 'Interactive data exploration and analysis'
    },
    'regime_analysis': {
        'title': '🎯 Regime Analysis',
        'icon': '🎯',
        'description': 'Market regime detection and analysis'
    },
    'model_comparison': {
        'title': '🤖 Model Comparison',
        'icon': '🤖',
        'description': 'AI model performance evaluation'
    },
    'real_time_monitor': {
        'title': '📈 Real-Time Monitor',
        'icon': '📈',
        'description': 'Live market monitoring and predictions'
    },
    'complex_systems': {
        'title': '🌀 Complex Systems',
        'icon': '🌀',
        'description': 'Theoretical foundations and complexity analysis'
    },
    'documentation': {
        'title': '📚 Documentation',
        'icon': '📚',
        'description': 'Research documentation and guides'
    }
}

# Model names and configurations
MODEL_NAMES = [
    'OLS Linear Regression',
    'LSTM Neural Network',
    'Random Forest',
    'XGBoost',
    'LightGBM',
    'Ensemble (RF+XGB+LGB)',
    'Complex Systems Model'
]

# Metric definitions and descriptions
METRIC_DEFINITIONS = {
    'mse': {
        'name': 'Mean Squared Error',
        'description': 'Average of squared prediction errors',
        'unit': 'points²',
        'direction': 'minimize',
        'range': (0, float('inf'))
    },
    'rmse': {
        'name': 'Root Mean Squared Error',
        'description': 'Square root of MSE, in same units as target',
        'unit': 'points',
        'direction': 'minimize',
        'range': (0, float('inf'))
    },
    'mae': {
        'name': 'Mean Absolute Error',
        'description': 'Average of absolute prediction errors',
        'unit': 'points',
        'direction': 'minimize',
        'range': (0, float('inf'))
    },
    'mape': {
        'name': 'Mean Absolute Percentage Error',
        'description': 'Average absolute percentage error',
        'unit': '%',
        'direction': 'minimize',
        'range': (0, 100)
    },
    'r2': {
        'name': 'R² Score',
        'description': 'Proportion of variance explained by model',
        'unit': 'coefficient',
        'direction': 'maximize',
        'range': (-float('inf'), 1)
    },
    'accuracy': {
        'name': 'Accuracy',
        'description': 'Proportion of correct predictions',
        'unit': '%',
        'direction': 'maximize',
        'range': (0, 100)
    },
    'precision': {
        'name': 'Precision',
        'description': 'True positives / (True positives + False positives)',
        'unit': 'ratio',
        'direction': 'maximize',
        'range': (0, 1)
    },
    'recall': {
        'name': 'Recall',
        'description': 'True positives / (True positives + False negatives)',
        'unit': 'ratio',
        'direction': 'maximize',
        'range': (0, 1)
    },
    'f1': {
        'name': 'F1 Score',
        'description': 'Harmonic mean of precision and recall',
        'unit': 'score',
        'direction': 'maximize',
        'range': (0, 1)
    },
    'sharpe': {
        'name': 'Sharpe Ratio',
        'description': 'Risk-adjusted return measure',
        'unit': 'ratio',
        'direction': 'maximize',
        'range': (-float('inf'), float('inf'))
    },
    'sortino': {
        'name': 'Sortino Ratio',
        'description': 'Downside risk-adjusted return',
        'unit': 'ratio',
        'direction': 'maximize',
        'range': (-float('inf'), float('inf'))
    },
    'max_drawdown': {
        'name': 'Maximum Drawdown',
        'description': 'Largest peak-to-trough decline',
        'unit': '%',
        'direction': 'minimize',
        'range': (-100, 0)
    },
    'volatility': {
        'name': 'Volatility',
        'description': 'Standard deviation of returns',
        'unit': '%',
        'direction': 'context_dependent',
        'range': (0, float('inf'))
    },
    'correlation': {
        'name': 'Correlation',
        'description': 'Linear relationship strength between variables',
        'unit': 'coefficient',
        'direction': 'context_dependent',
        'range': (-1, 1)
    },
    'adf_statistic': {
        'name': 'ADF Statistic',
        'description': 'Augmented Dickey-Fuller test statistic',
        'unit': 'statistic',
        'direction': 'minimize',
        'range': (-float('inf'), float('inf'))
    },
    'p_value': {
        'name': 'P-Value',
        'description': 'Probability of observing data under null hypothesis',
        'unit': 'probability',
        'direction': 'minimize',
        'range': (0, 1)
    }
}

# Update frequencies for real-time components
UPDATE_FREQUENCIES = {
    'realtime': {
        'name': 'Real-Time',
        'seconds': 1,
        'description': 'Continuous updates every second'
    },
    'high_frequency': {
        'name': 'High Frequency',
        'seconds': 30,
        'description': 'Updates every 30 seconds'
    },
    'medium_frequency': {
        'name': 'Medium Frequency',
        'seconds': 300,  # 5 minutes
        'description': 'Updates every 5 minutes'
    },
    'low_frequency': {
        'name': 'Low Frequency',
        'seconds': 1800,  # 30 minutes
        'description': 'Updates every 30 minutes'
    },
    'daily': {
        'name': 'Daily',
        'seconds': 86400,  # 24 hours
        'description': 'Daily updates'
    },
    'manual': {
        'name': 'Manual',
        'seconds': None,
        'description': 'Manual refresh only'
    }
}

# Cache time-to-live settings
CACHE_TTL = {
    'data': {
        'raw_data': 3600,        # 1 hour
        'processed_data': 1800,  # 30 minutes
        'summary_stats': 900,    # 15 minutes
        'correlations': 1800,    # 30 minutes
        'regime_data': 3600,     # 1 hour
    },
    'models': {
        'model_predictions': 300,  # 5 minutes
        'model_metrics': 1800,     # 30 minutes
        'feature_importance': 3600, # 1 hour
        'model_comparison': 1800,   # 30 minutes
    },
    'visualizations': {
        'charts': 900,           # 15 minutes
        'plots': 900,            # 15 minutes
        'dashboards': 300,       # 5 minutes
    },
    'api_responses': {
        'market_data': 60,       # 1 minute
        'news_data': 300,        # 5 minutes
        'economic_indicators': 3600, # 1 hour
    },
    'computations': {
        'statistical_tests': 1800, # 30 minutes
        'complex_calculations': 3600, # 1 hour
        'simulations': 7200,     # 2 hours
    }
}

# Data quality thresholds
DATA_QUALITY_THRESHOLDS = {
    'missing_data': {
        'warning': 0.05,   # 5% missing
        'critical': 0.15   # 15% missing
    },
    'outliers': {
        'warning': 0.10,   # 10% outliers
        'critical': 0.25   # 25% outliers
    },
    'stationarity': {
        'adf_threshold': -2.86,  # 5% significance level
        'p_value_threshold': 0.05
    },
    'correlation': {
        'high_correlation': 0.8,
        'low_correlation': 0.3
    }
}

# Performance thresholds for model evaluation
PERFORMANCE_THRESHOLDS = {
    'excellent': {
        'r2': 0.8,
        'mape': 5.0,
        'sharpe': 2.0
    },
    'good': {
        'r2': 0.6,
        'mape': 10.0,
        'sharpe': 1.0
    },
    'average': {
        'r2': 0.4,
        'mape': 15.0,
        'sharpe': 0.5
    },
    'poor': {
        'r2': 0.2,
        'mape': 25.0,
        'sharpe': 0.0
    }
}

# Market regime definitions
REGIME_DEFINITIONS = {
    'bull': {
        'description': 'Strong upward market movement',
        'volatility_threshold': 0.15,
        'return_threshold': 0.02,
        'duration_min_days': 20
    },
    'bear': {
        'description': 'Strong downward market movement',
        'volatility_threshold': 0.20,
        'return_threshold': -0.02,
        'duration_min_days': 15
    },
    'sideways': {
        'description': 'Range-bound market with low volatility',
        'volatility_threshold': 0.10,
        'return_threshold': 0.005,
        'duration_min_days': 30
    },
    'volatile': {
        'description': 'High volatility regardless of direction',
        'volatility_threshold': 0.25,
        'return_threshold': None,
        'duration_min_days': 10
    }
}

# Export settings
EXPORT_SETTINGS = {
    'formats': ['csv', 'excel', 'json', 'pdf'],
    'max_rows': 10000,
    'compression': 'gzip',
    'date_format': '%Y-%m-%d %H:%M:%S',
    'decimal_places': 4
}

# API rate limits and timeouts
API_SETTINGS = {
    'timeout': 30,  # seconds
    'retries': 3,
    'backoff_factor': 2,
    'rate_limit_delay': 1,  # seconds between requests
    'max_concurrent_requests': 5
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'max_file_size': 10 * 1024 * 1024,  # 10MB
    'backup_count': 5,
    'log_directory': 'logs'
}

# Session state defaults
SESSION_DEFAULTS = {
    'theme': 'light',
    'auto_refresh': True,
    'refresh_interval': 300,  # 5 minutes
    'show_warnings': True,
    'export_format': 'csv',
    'decimal_precision': 2,
    'chart_height': 400,
    'table_rows_per_page': 50
}