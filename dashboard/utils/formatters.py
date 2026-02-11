"""
Dashboard Formatting Utilities
==============================

Formatting functions for consistent display of financial data and metrics.

Author: GitHub Copilot
Date: February 8, 2026
"""

import streamlit as st
from typing import Union, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def format_percentage(value: Union[float, int],
                     decimals: int = 2,
                     include_symbol: bool = True,
                     color_code: bool = False) -> str:
    """Format a value as a percentage with optional color coding.

    Args:
        value: Numeric value to format (0.05 = 5%)
        decimals: Number of decimal places
        include_symbol: Whether to include % symbol
        color_code: Whether to add color based on value

    Returns:
        Formatted percentage string
    """
    if pd.isna(value) or value is None:
        return "N/A"

    try:
        # Convert to percentage if needed (assume value is decimal)
        if abs(value) < 1 and decimals <= 2:  # Likely already a decimal
            percentage = value * 100
        else:
            percentage = float(value)

        formatted = f"{percentage:.{decimals}f}"

        if include_symbol:
            formatted += "%"

        if color_code:
            if percentage > 0:
                formatted = f"🟢 +{formatted}" if include_symbol else f"🟢 {formatted}"
            elif percentage < 0:
                formatted = f"🔴 {formatted}"
            else:
                formatted = f"⚪ {formatted}"

        return formatted

    except (ValueError, TypeError):
        return "N/A"

def format_currency(value: Union[float, int],
                   currency: str = "USD",
                   decimals: int = 2,
                   compact: bool = False) -> str:
    """Format a value as currency.

    Args:
        value: Numeric value to format
        currency: Currency code (USD, EUR, etc.)
        decimals: Number of decimal places
        compact: Use compact notation (K, M, B)

    Returns:
        Formatted currency string
    """
    if pd.isna(value) or value is None:
        return "N/A"

    try:
        value = float(value)

        if compact:
            if abs(value) >= 1e9:
                return f"${value/1e9:.{decimals}f}B"
            elif abs(value) >= 1e6:
                return f"${value/1e6:.{decimals}f}M"
            elif abs(value) >= 1e3:
                return f"${value/1e3:.{decimals}f}K"
            else:
                return f"${value:.{decimals}f}"

        # Standard currency formatting
        formatted = f"{value:,.{decimals}f}"

        if currency == "USD":
            return f"${formatted}"
        elif currency == "EUR":
            return f"€{formatted}"
        elif currency == "GBP":
            return f"£{formatted}"
        elif currency == "JPY":
            return f"¥{formatted}"
        else:
            return f"{currency} {formatted}"

    except (ValueError, TypeError):
        return "N/A"

def format_number(value: Union[float, int],
                 decimals: int = 2,
                 compact: bool = False,
                 scientific: bool = False) -> str:
    """Format a numeric value with various options.

    Args:
        value: Numeric value to format
        decimals: Number of decimal places
        compact: Use compact notation (K, M, B)
        scientific: Use scientific notation

    Returns:
        Formatted number string
    """
    if pd.isna(value) or value is None:
        return "N/A"

    try:
        value = float(value)

        if scientific:
            return f"{value:.{decimals}e}"

        if compact:
            if abs(value) >= 1e9:
                return f"{value/1e9:.{decimals}f}B"
            elif abs(value) >= 1e6:
                return f"{value/1e6:.{decimals}f}M"
            elif abs(value) >= 1e3:
                return f"{value/1e3:.{decimals}f}K"
            else:
                return f"{value:.{decimals}f}"

        # Standard number formatting with commas
        return f"{value:,.{decimals}f}"

    except (ValueError, TypeError):
        return "N/A"

def format_returns(value: Union[float, int],
                  decimals: int = 2,
                  include_symbol: bool = True,
                  color_code: bool = True) -> str:
    """Format returns with color coding and percentage symbol.

    Args:
        value: Return value (0.05 = 5%)
        decimals: Number of decimal places
        include_symbol: Whether to include % symbol
        color_code: Whether to add color based on value

    Returns:
        Formatted return string
    """
    if pd.isna(value) or value is None:
        return "N/A"

    try:
        # Convert to percentage
        percentage = float(value) * 100
        formatted = f"{percentage:.{decimals}f}"

        if include_symbol:
            formatted += "%"

        if color_code:
            if percentage > 0:
                formatted = f"🟢 +{formatted}" if include_symbol else f"🟢 {formatted}"
            elif percentage < 0:
                formatted = f"🔴 {formatted}"
            else:
                formatted = f"⚪ {formatted}"

        return formatted

    except (ValueError, TypeError):
        return "N/A"

def format_pvalue(p_value: Union[float, int],
                 decimals: int = 4,
                 significance_levels: bool = True) -> str:
    """Format p-values with significance markers.

    Args:
        p_value: P-value to format
        decimals: Number of decimal places
        significance_levels: Add significance stars/markers

    Returns:
        Formatted p-value string
    """
    if pd.isna(p_value) or p_value is None:
        return "N/A"

    try:
        p = float(p_value)

        if p < 0:
            return "Invalid"

        # Format the p-value
        if p < 0.001:
            formatted = "< 0.001"
        elif p < 0.01:
            formatted = f"{p:.{decimals}f}"
        else:
            formatted = f"{p:.{decimals}f}"

        if significance_levels:
            if p < 0.001:
                formatted += " ***"
            elif p < 0.01:
                formatted += " **"
            elif p < 0.05:
                formatted += " *"
            elif p < 0.1:
                formatted += " †"

        return formatted

    except (ValueError, TypeError):
        return "N/A"

def format_datetime(dt: Union[datetime, str, pd.Timestamp],
                   format_str: str = "%Y-%m-%d %H:%M:%S",
                   relative: bool = False) -> str:
    """Format datetime objects with optional relative time.

    Args:
        dt: Datetime to format
        format_str: Format string for absolute time
        relative: Show relative time (e.g., "2 hours ago")

    Returns:
        Formatted datetime string
    """
    if pd.isna(dt) or dt is None:
        return "N/A"

    try:
        if isinstance(dt, str):
            dt = pd.to_datetime(dt)
        elif isinstance(dt, pd.Timestamp):
            dt = dt.to_pydatetime()

        if not isinstance(dt, datetime):
            return "Invalid Date"

        if relative:
            now = datetime.now()
            diff = now - dt

            if diff.days > 365:
                years = diff.days // 365
                return f"{years} year{'s' if years != 1 else ''} ago"
            elif diff.days > 30:
                months = diff.days // 30
                return f"{months} month{'s' if months != 1 else ''} ago"
            elif diff.days > 0:
                return f"{diff.days} day{'s' if diff.days != 1 else ''} ago"
            elif diff.seconds > 3600:
                hours = diff.seconds // 3600
                return f"{hours} hour{'s' if hours != 1 else ''} ago"
            elif diff.seconds > 60:
                minutes = diff.seconds // 60
                return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
            else:
                return "Just now"

        return dt.strftime(format_str)

    except (ValueError, TypeError, AttributeError):
        return "Invalid Date"

def format_duration(seconds: Union[float, int],
                   compact: bool = False) -> str:
    """Format duration in seconds to human-readable format.

    Args:
        seconds: Duration in seconds
        compact: Use compact format (e.g., "1h 30m")

    Returns:
        Formatted duration string
    """
    if pd.isna(seconds) or seconds is None or seconds < 0:
        return "N/A"

    try:
        seconds = int(seconds)

        days = seconds // 86400
        hours = (seconds % 86400) // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60

        if compact:
            parts = []
            if days > 0:
                parts.append(f"{days}d")
            if hours > 0:
                parts.append(f"{hours}h")
            if minutes > 0:
                parts.append(f"{minutes}m")
            if secs > 0 and not parts:  # Only show seconds if no larger units
                parts.append(f"{secs}s")

            return " ".join(parts) if parts else "0s"
        else:
            if days > 0:
                return f"{days} day{'s' if days != 1 else ''}, {hours} hour{'s' if hours != 1 else ''}"
            elif hours > 0:
                return f"{hours} hour{'s' if hours != 1 else ''}, {minutes} minute{'s' if minutes != 1 else ''}"
            elif minutes > 0:
                return f"{minutes} minute{'s' if minutes != 1 else ''}, {secs} second{'s' if secs != 1 else ''}"
            else:
                return f"{secs} second{'s' if secs != 1 else ''}"

    except (ValueError, TypeError):
        return "N/A"

def format_file_size(bytes_size: Union[float, int],
                    binary: bool = True) -> str:
    """Format file size in bytes to human-readable format.

    Args:
        bytes_size: Size in bytes
        binary: Use binary (1024) or decimal (1000) units

    Returns:
        Formatted file size string
    """
    if pd.isna(bytes_size) or bytes_size is None or bytes_size < 0:
        return "N/A"

    try:
        size = float(bytes_size)

        if binary:
            units = ['B', 'KiB', 'MiB', 'GiB', 'TiB']
            divisor = 1024
        else:
            units = ['B', 'KB', 'MB', 'GB', 'TB']
            divisor = 1000

        unit_index = 0
        while size >= divisor and unit_index < len(units) - 1:
            size /= divisor
            unit_index += 1

        return f"{size:.1f} {units[unit_index]}"

    except (ValueError, TypeError):
        return "N/A"

def format_confidence_interval(lower: Union[float, int],
                             upper: Union[float, int],
                             decimals: int = 2,
                             include_brackets: bool = True) -> str:
    """Format confidence interval.

    Args:
        lower: Lower bound
        upper: Upper bound
        decimals: Number of decimal places
        include_brackets: Include square brackets

    Returns:
        Formatted confidence interval string
    """
    if pd.isna(lower) or pd.isna(upper) or lower is None or upper is None:
        return "N/A"

    try:
        lower = float(lower)
        upper = float(upper)

        formatted = f"[{lower:.{decimals}f}, {upper:.{decimals}f}]" if include_brackets else f"{lower:.{decimals}f} - {upper:.{decimals}f}"

        return formatted

    except (ValueError, TypeError):
        return "N/A"

def format_statistic_test(statistic: Union[float, int],
                         p_value: Union[float, int],
                         test_name: str = "",
                         significance_level: float = 0.05) -> str:
    """Format statistical test results.

    Args:
        statistic: Test statistic value
        p_value: P-value
        test_name: Name of the test
        significance_level: Significance threshold

    Returns:
        Formatted test result string
    """
    if pd.isna(statistic) or pd.isna(p_value):
        return "N/A"

    try:
        stat_formatted = f"{float(statistic):.4f}"
        p_formatted = format_pvalue(p_value, significance_levels=True)

        result = "Reject H₀" if float(p_value) < significance_level else "Fail to reject H₀"

        if test_name:
            return f"{test_name}: {stat_formatted}, p = {p_formatted} ({result})"
        else:
            return f"{stat_formatted}, p = {p_formatted} ({result})"

    except (ValueError, TypeError):
        return "N/A"

def format_correlation(corr: Union[float, int],
                      decimals: int = 3,
                      strength_indicator: bool = True) -> str:
    """Format correlation coefficient with strength indicator.

    Args:
        corr: Correlation coefficient (-1 to 1)
        decimals: Number of decimal places
        strength_indicator: Add strength description

    Returns:
        Formatted correlation string
    """
    if pd.isna(corr) or corr is None:
        return "N/A"

    try:
        corr = float(corr)

        if abs(corr) > 1:
            return "Invalid"

        formatted = f"{corr:.{decimals}f}"

        if strength_indicator:
            abs_corr = abs(corr)
            if abs_corr >= 0.8:
                strength = "Very Strong"
            elif abs_corr >= 0.6:
                strength = "Strong"
            elif abs_corr >= 0.4:
                strength = "Moderate"
            elif abs_corr >= 0.2:
                strength = "Weak"
            else:
                strength = "Very Weak"

            direction = "Positive" if corr > 0 else "Negative" if corr < 0 else "Zero"
            formatted += f" ({direction}, {strength})"

        return formatted

    except (ValueError, TypeError):
        return "N/A"

def format_ratio(numerator: Union[float, int],
                denominator: Union[float, int],
                decimals: int = 2,
                as_percentage: bool = False) -> str:
    """Format ratio with optional percentage conversion.

    Args:
        numerator: Numerator value
        denominator: Denominator value
        decimals: Number of decimal places
        as_percentage: Convert to percentage

    Returns:
        Formatted ratio string
    """
    if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
        return "N/A"

    try:
        ratio = float(numerator) / float(denominator)

        if as_percentage:
            ratio *= 100
            return f"{ratio:.{decimals}f}%"
        else:
            return f"{ratio:.{decimals}f}"

    except (ValueError, TypeError, ZeroDivisionError):
        return "N/A"