"""
Dashboard Layout Utilities
=========================

Reusable UI components and styling functions for the financial analytics dashboard.

Author: GitHub Copilot
Date: February 8, 2026
"""

import streamlit as st
from typing import Optional, Dict, Any
import pandas as pd

def styled_metric(label: str, value: str, delta: Optional[str] = None, color_scheme: str = "auto") -> None:
    """Create a styled metric display with custom colors and formatting.

    Args:
        label: Metric label
        value: Main value to display
        delta: Change indicator (optional)
        color_scheme: Color scheme - "auto", "positive", "negative", "neutral"
    """
    if color_scheme == "auto" and delta:
        if delta.startswith("+") or delta.startswith("↑"):
            color_scheme = "positive"
        elif delta.startswith("-") or delta.startswith("↓"):
            color_scheme = "negative"
        else:
            color_scheme = "neutral"

    # Color mapping
    color_map = {
        "positive": "#10b981",  # Green
        "negative": "#ef4444",  # Red
        "neutral": "#6b7280",   # Gray
        "warning": "#f59e0b",   # Amber
        "info": "#3b82f6"       # Blue
    }

    color = color_map.get(color_scheme, color_map["neutral"])

    # Create custom CSS for the metric
    metric_html = f"""
    <div style="
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
        margin: 8px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        transition: transform 0.2s ease;
    ">
        <div style="font-size: 14px; color: #64748b; margin-bottom: 8px; font-weight: 500;">
            {label}
        </div>
        <div style="font-size: 28px; font-weight: 700; color: #1e293b; margin-bottom: 4px;">
            {value}
        </div>
    """

    if delta:
        metric_html += f"""
        <div style="font-size: 14px; color: {color}; font-weight: 600;">
            {delta}
        </div>
        """

    metric_html += "</div>"

    st.markdown(metric_html, unsafe_allow_html=True)

def styled_table(df: pd.DataFrame, height: Optional[int] = None) -> None:
    """Create a styled data table with custom formatting.

    Args:
        df: DataFrame to display
        height: Optional height for the table container
    """
    # Convert to HTML with custom styling
    table_html = f"""
    <div style="overflow-x: auto; {'max-height: ' + str(height) + 'px; overflow-y: auto;' if height else ''}">
        <table style="
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        ">
    """

    # Header row
    table_html += "<thead><tr style='background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%); color: white;'>"
    for col in df.columns:
        table_html += f"<th style='padding: 12px 16px; text-align: left; font-weight: 600;'>{col}</th>"
    table_html += "</tr></thead>"

    # Data rows
    table_html += "<tbody>"
    for idx, row in df.iterrows():
        row_style = "background: #f8fafc;" if idx % 2 == 0 else "background: white;"
        table_html += f"<tr style='{row_style}'>"

        for col in df.columns:
            value = row[col]
            # Format numeric values
            if isinstance(value, (int, float)):
                if isinstance(value, float):
                    formatted_value = ".4f"
                else:
                    formatted_value = str(value)
            else:
                formatted_value = str(value)

            table_html += f"<td style='padding: 10px 16px; border-bottom: 1px solid #e2e8f0;'>{formatted_value}</td>"

        table_html += "</tr>"
    table_html += "</tbody></table></div>"

    st.markdown(table_html, unsafe_allow_html=True)

def styled_alert(message: str, alert_type: str = "info") -> None:
    """Create a styled alert/notification box.

    Args:
        message: Alert message
        alert_type: Type of alert - "info", "success", "warning", "error"
    """
    # Color and icon mapping
    alert_config = {
        "info": {
            "color": "#3b82f6",
            "bg_color": "#eff6ff",
            "border_color": "#dbeafe",
            "icon": "ℹ️"
        },
        "success": {
            "color": "#10b981",
            "bg_color": "#f0fdf4",
            "border_color": "#dcfce7",
            "icon": "✅"
        },
        "warning": {
            "color": "#f59e0b",
            "bg_color": "#fffbeb",
            "border_color": "#fef3c7",
            "icon": "⚠️"
        },
        "error": {
            "color": "#ef4444",
            "bg_color": "#fef2f2",
            "border_color": "#fee2e2",
            "icon": "❌"
        }
    }

    config = alert_config.get(alert_type, alert_config["info"])

    alert_html = f"""
    <div style="
        background: {config['bg_color']};
        border: 1px solid {config['border_color']};
        border-left: 4px solid {config['color']};
        border-radius: 8px;
        padding: 16px 20px;
        margin: 12px 0;
        display: flex;
        align-items: flex-start;
        gap: 12px;
    ">
        <div style="font-size: 20px; color: {config['color']}; flex-shrink: 0;">
            {config['icon']}
        </div>
        <div style="color: #374151; line-height: 1.5;">
            {message}
        </div>
    </div>
    """

    st.markdown(alert_html, unsafe_allow_html=True)

def regime_color(regime_label: str) -> str:
    """Get color for regime visualization.

    Args:
        regime_label: Regime name (bull, bear, sideways, volatile, etc.)

    Returns:
        Hex color code
    """
    regime_colors = {
        "bull": "#10b981",      # Green
        "bear": "#ef4444",      # Red
        "sideways": "#6b7280",  # Gray
        "volatile": "#f59e0b",  # Amber
        "recovery": "#8b5cf6",  # Purple
        "crisis": "#dc2626",    # Dark red
        "normal": "#3b82f6",    # Blue
        "high_volatility": "#f97316",  # Orange
        "low_volatility": "#06b6d4",   # Cyan
    }

    return regime_colors.get(regime_label.lower(), "#6b7280")

def performance_color(value: float, metric_type: str = "accuracy") -> str:
    """Get color for performance metrics.

    Args:
        value: Performance value
        metric_type: Type of metric (accuracy, returns, risk, etc.)

    Returns:
        Hex color code
    """
    if metric_type in ["accuracy", "returns", "sharpe", "win_rate"]:
        if value >= 0.8:
            return "#10b981"  # Excellent - Green
        elif value >= 0.7:
            return "#84cc16"  # Good - Light green
        elif value >= 0.6:
            return "#f59e0b"  # Fair - Amber
        elif value >= 0.5:
            return "#f97316"  # Poor - Orange
        else:
            return "#ef4444"  # Bad - Red

    elif metric_type in ["risk", "volatility", "drawdown", "loss"]:
        if value <= 0.1:
            return "#10b981"  # Low risk - Green
        elif value <= 0.2:
            return "#f59e0b"  # Moderate risk - Amber
        elif value <= 0.3:
            return "#f97316"  # High risk - Orange
        else:
            return "#ef4444"  # Very high risk - Red

    else:
        return "#6b7280"  # Default gray

def create_header(title: str, subtitle: Optional[str] = None) -> None:
    """Create a styled page header.

    Args:
        title: Main title
        subtitle: Optional subtitle
    """
    header_html = f"""
    <div style="
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        color: white;
        padding: 32px 24px;
        border-radius: 16px;
        margin-bottom: 24px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
    ">
        <h1 style="
            margin: 0;
            font-size: 32px;
            font-weight: 700;
            background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        ">
            {title}
        </h1>
    """

    if subtitle:
        header_html += f"""
        <p style="
            margin: 8px 0 0 0;
            font-size: 16px;
            color: #cbd5e1;
            opacity: 0.9;
        ">
            {subtitle}
        </p>
        """

    header_html += "</div>"

    st.markdown(header_html, unsafe_allow_html=True)

def create_footer() -> None:
    """Create a styled page footer."""
    footer_html = """
    <div style="
        background: #f8fafc;
        border-top: 1px solid #e2e8f0;
        padding: 24px;
        margin-top: 48px;
        text-align: center;
        color: #64748b;
        font-size: 14px;
    ">
        <div style="margin-bottom: 12px;">
            <strong>Stock Market AI Analytics Dashboard</strong>
        </div>
        <div style="margin-bottom: 8px;">
            Powered by Advanced Machine Learning & Real-time Data Processing
        </div>
        <div style="font-size: 12px; opacity: 0.7;">
            © 2026 Financial Intelligence Platform | Built with Streamlit & Python
        </div>
    </div>
    """

    st.markdown(footer_html, unsafe_allow_html=True)

def load_custom_css() -> str:
    """Load custom CSS for the dashboard.

    Returns:
        CSS string for custom styling
    """
    css = """
    /* Custom CSS for Financial Analytics Dashboard */

    /* Global styles */
    .main {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        min-height: 100vh;
    }

    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #1e293b 0%, #334155 100%);
        color: white;
    }

    .sidebar .sidebar-content .block-container {
        padding: 2rem 1rem;
    }

    /* Metric cards hover effect */
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(0,0,0,0.15) !important;
    }

    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
        transform: translateY(-1px);
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #f8fafc;
        border-radius: 8px;
        padding: 4px;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 6px;
        color: #64748b;
        font-weight: 500;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        font-weight: 600;
        color: #1e293b;
    }

    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #10b981 0%, #34d399 100%);
    }

    /* Dataframe styling */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }

    .dataframe thead th {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        font-weight: 600;
        border: none;
    }

    .dataframe tbody tr:nth-child(even) {
        background: #f8fafc;
    }

    /* Chart styling */
    .js-plotly-plot .plotly .modebar {
        background: rgba(255, 255, 255, 0.8) !important;
        border-radius: 4px;
    }

    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #f1f5f9;
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #cbd5e1 0%, #94a3b8 100%);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #94a3b8 0%, #64748b 100%);
    }

    /* Loading animation */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    .loading {
        animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }

    /* Responsive design */
    @media (max-width: 768px) {
        .main .block-container {
            padding: 1rem;
        }

        .metric-card {
            margin: 4px 0;
        }
    }
    """

    return css