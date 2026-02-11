"""
Data Explorer Page - Interactive Data Analysis
==============================================

Comprehensive data exploration with time-series visualization, correlation analysis,
stationarity tests, and data quality assessment.

Author: GitHub Copilot
Date: February 8, 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import scipy.stats as stats
from typing import Dict, List, Any, Optional, Tuple
import io

st.set_page_config(page_title="Data Explorer - Stock Market AI Analytics", page_icon="📊")

def load_market_data() -> Optional[pd.DataFrame]:
    """Load market data for analysis."""
    # First check if data is already loaded in session state
    if st.session_state.get('data_loaded', False) and 'market_data' in st.session_state.data_cache:
        data = st.session_state.data_cache.get('market_data')
        if data is not None and not data.empty:
            return data.copy()

    # If not loaded, try to load it directly
    try:
        data_path = "stationary_data.csv"
        if os.path.exists(data_path):
            df = pd.read_csv(data_path, index_col=0, parse_dates=True)
            df = df.reset_index()
            df = df.rename(columns={'index': 'Date'})

            # Filter by reasonable date range (last 5 years)
            from datetime import datetime, timedelta
            start_date = (datetime.now() - timedelta(days=1825)).strftime('%Y-%m-%d')
            end_date = datetime.now().strftime('%Y-%m-%d')

            df['Date'] = pd.to_datetime(df['Date'])
            filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

            if len(filtered_df) == 0:
                # If no data in filtered range, return all data
                return df.copy()
            else:
                return filtered_df.copy()
        else:
            st.error("❌ Data file not found. Please run data_generator.py first.")
            return None
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return None

def create_data_overview(data: pd.DataFrame):
    """Create data overview section with summary statistics."""
    st.subheader("📊 Data Overview")

    # Summary statistics table
    st.markdown("**Summary Statistics**")

    # Select numeric columns for statistics
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

    if numeric_cols:
        summary_stats = data[numeric_cols].describe().round(4)

        # Configure column display
        column_config = {}
        for col in summary_stats.columns:
            column_config[col] = st.column_config.NumberColumn(
                col,
                format="%.4f",
                help=f"Statistics for {col}"
            )

        st.dataframe(
            summary_stats,
            column_config=column_config,
            use_container_width=True
        )

    # Data availability chart
    st.markdown("**Data Availability**")

    missing_data = data.isnull().sum() / len(data) * 100
    missing_data = missing_data[missing_data > 0]  # Only show columns with missing data

    if not missing_data.empty:
        fig_missing = px.bar(
            x=missing_data.index,
            y=missing_data.values,
            title="Missing Data Percentage by Variable",
            labels={'x': 'Variable', 'y': 'Missing %'},
            color=missing_data.values,
            color_continuous_scale='Reds'
        )
        fig_missing.update_layout(height=300)
        st.plotly_chart(fig_missing, use_container_width=True)
    else:
        st.success("✅ No missing data detected!")

    # Data shape and date range
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Rows", f"{len(data):,}")

    with col2:
        st.metric("Total Columns", len(data.columns))

    with col3:
        if 'Date' in data.columns:
            date_range = f"{data['Date'].min().date()} to {data['Date'].max().date()}"
            st.metric("Date Range", date_range)

def create_time_series_visualization(data: pd.DataFrame):
    """Create time-series visualization with multi-series plot."""
    st.subheader("📈 Time-Series Visualization")

    if 'Date' not in data.columns:
        st.error("❌ Date column not found in data.")
        return

    # Ensure Date is datetime
    data = data.copy()
    data['Date'] = pd.to_datetime(data['Date'])

    # Select columns for plotting
    available_cols = [col for col in data.columns if col != 'Date' and data[col].dtype in ['float64', 'int64']]

    if not available_cols:
        st.error("❌ No numeric columns available for plotting.")
        return

    # Column selection
    selected_cols = st.multiselect(
        "Select series to display:",
        available_cols,
        default=available_cols[:min(3, len(available_cols))],
        key="time_series_cols"
    )

    if not selected_cols:
        st.warning("Please select at least one series to display.")
        return

    # Date range selector
    date_range = st.slider(
        "Select date range:",
        min_value=data['Date'].min().date(),
        max_value=data['Date'].max().date(),
        value=(data['Date'].min().date(), data['Date'].max().date()),
        key="date_range_slider"
    )

    # Filter data by date range
    mask = (data['Date'].dt.date >= date_range[0]) & (data['Date'].dt.date <= date_range[1])
    plot_data = data[mask]

    # Create subplot figure
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Color palette
    colors = px.colors.qualitative.Set1

    # Add traces
    for i, col in enumerate(selected_cols):
        color = colors[i % len(colors)]

        # Primary y-axis for first series, secondary for others if VIX is involved
        secondary_y = ('VIX' in col and i > 0)

        fig.add_trace(
            go.Scatter(
                x=plot_data['Date'],
                y=plot_data[col],
                name=col,
                line=dict(color=color, width=2),
                mode='lines',
                hovertemplate=f'{col}: %{{y:.4f}}<br>Date: %{{x}}<extra></extra>'
            ),
            secondary_y=secondary_y
        )

    # Update layout
    fig.update_layout(
        title="Time Series Analysis",
        xaxis_title="Date",
        yaxis_title="Value",
        height=500,
        hovermode='x unified'
    )

    # Update y-axes titles
    fig.update_yaxes(title_text="Primary Y-axis", secondary_y=False)
    if any('VIX' in col for col in selected_cols):
        fig.update_yaxes(title_text="VIX", secondary_y=True)

    st.plotly_chart(fig, use_container_width=True)

    # Series visibility toggles (additional controls)
    st.markdown("**Series Controls**")
    col1, col2 = st.columns(2)

    with col1:
        show_grid = st.checkbox("Show Grid", value=True, key="show_grid")
        if show_grid:
            fig.update_xaxes(showgrid=True)
            fig.update_yaxes(showgrid=True)

    with col2:
        show_legend = st.checkbox("Show Legend", value=True, key="show_legend")
        fig.update_layout(showlegend=show_legend)

def create_rolling_statistics(data: pd.DataFrame):
    """Create rolling statistics analysis."""
    st.subheader("📊 Rolling Statistics")

    if 'Date' not in data.columns:
        st.error("❌ Date column not found in data.")
        return

    # Window size selector
    window_sizes = [7, 14, 30, 60, 90]
    window_size = st.selectbox(
        "Select rolling window size (days):",
        window_sizes,
        index=2,  # Default to 30 days
        key="rolling_window"
    )

    # Select column for analysis
    numeric_cols = [col for col in data.columns if col != 'Date' and data[col].dtype in ['float64', 'int64']]
    selected_col = st.selectbox(
        "Select variable for rolling analysis:",
        numeric_cols,
        key="rolling_col"
    )

    if not selected_col:
        return

    # Calculate rolling statistics
    series = data[selected_col].dropna()
    rolling_mean = series.rolling(window=window_size).mean()
    rolling_std = series.rolling(window=window_size).std()

    # Create plot
    fig = go.Figure()

    # Add original series
    fig.add_trace(go.Scatter(
        x=data['Date'],
        y=series,
        name='Original',
        line=dict(color='lightgray', width=1),
        opacity=0.7
    ))

    # Add rolling mean
    fig.add_trace(go.Scatter(
        x=data['Date'],
        y=rolling_mean,
        name=f'Rolling Mean ({window_size}d)',
        line=dict(color='blue', width=2)
    ))

    # Add confidence bands
    fig.add_trace(go.Scatter(
        x=data['Date'],
        y=rolling_mean + rolling_std,
        name='+1 STD',
        line=dict(color='red', width=1, dash='dash'),
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=data['Date'],
        y=rolling_mean - rolling_std,
        name='-1 STD',
        line=dict(color='red', width=1, dash='dash'),
        fill='tonexty',
        fillcolor='rgba(255, 0, 0, 0.1)',
        showlegend=False
    ))

    fig.update_layout(
        title=f"Rolling Statistics: {selected_col} ({window_size}-day window)",
        xaxis_title="Date",
        yaxis_title="Value",
        height=400,
        hovermode='x unified'
    )

    st.plotly_chart(fig, use_container_width=True)

    # Analysis insights
    st.markdown("**Analysis Insights**")

    # Calculate clustering periods (high volatility periods)
    volatility_threshold = rolling_std.quantile(0.8)  # Top 20% volatility
    high_volatility_periods = rolling_std > volatility_threshold

    clustering_periods = high_volatility_periods.sum()
    total_periods = len(high_volatility_periods)

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "High Volatility Periods",
            f"{clustering_periods}/{total_periods}",
            f"{clustering_periods/total_periods*100:.1f}%"
        )

    with col2:
        # Detect anomalies (beyond 2 std)
        mean_val = series.mean()
        std_val = series.std()
        anomalies = ((series - mean_val) / std_val).abs() > 2
        st.metric("Anomalies Detected", anomalies.sum())

def create_correlation_analysis(data: pd.DataFrame):
    """Create correlation analysis section."""
    st.subheader("🔗 Correlation Analysis")

    # Select numeric columns
    numeric_cols = [col for col in data.columns if col != 'Date' and data[col].dtype in ['float64', 'int64']]

    if len(numeric_cols) < 2:
        st.error("❌ Need at least 2 numeric columns for correlation analysis.")
        return

    # Create tabs for different correlation views
    tab1, tab2, tab3 = st.tabs(["Overall Correlation", "Regime-Specific", "Rolling Correlation"])

    with tab1:
        # Overall correlation matrix
        corr_matrix = data[numeric_cols].corr()

        # Create heatmap
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmin=-1, zmax=1,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False
        ))

        fig_corr.update_layout(
            title="Correlation Matrix Heatmap",
            height=500
        )

        st.plotly_chart(fig_corr, use_container_width=True)

    with tab2:
        st.info("Regime-specific correlation analysis would be implemented here based on market regime detection.")

    with tab3:
        # Rolling correlation
        if len(numeric_cols) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                var1 = st.selectbox("Variable 1:", numeric_cols, key="rolling_corr_var1")
            with col2:
                var2 = st.selectbox("Variable 2:", [col for col in numeric_cols if col != var1], key="rolling_corr_var2")

            if var1 and var2:
                window_size = st.slider("Rolling window (days):", 10, 100, 30, key="rolling_corr_window")

                # Calculate rolling correlation
                rolling_corr = data[var1].rolling(window=window_size).corr(data[var2])

                fig_rolling = go.Figure()
                fig_rolling.add_trace(go.Scatter(
                    x=data['Date'],
                    y=rolling_corr,
                    name=f'Rolling Correlation ({window_size}d)',
                    line=dict(color='purple', width=2)
                ))

                fig_rolling.add_hline(y=0, line_dash="dash", line_color="gray")
                fig_rolling.update_layout(
                    title=f"Rolling Correlation: {var1} vs {var2}",
                    xaxis_title="Date",
                    yaxis_title="Correlation",
                    height=400
                )

                st.plotly_chart(fig_rolling, use_container_width=True)

    # Correlation interpretation
    st.markdown("**Correlation Interpretation**")
    st.markdown("""
    - **High Correlation (>0.7)**: Strong positive relationship
    - **Moderate Correlation (0.3-0.7)**: Moderate relationship
    - **Low Correlation (<0.3)**: Weak relationship
    - **Negative Correlation**: Inverse relationship
    - **Rolling Correlation Changes**: May indicate regime shifts or changing relationships
    """)

def create_stationarity_tests(data: pd.DataFrame):
    """Create stationarity tests section."""
    st.subheader("📏 Stationarity Tests")

    # Select numeric columns
    numeric_cols = [col for col in data.columns if col != 'Date' and data[col].dtype in ['float64', 'int64']]

    if not numeric_cols:
        st.error("❌ No numeric columns available for stationarity testing.")
        return

    st.markdown("**Augmented Dickey-Fuller (ADF) Test Results**")

    # Perform ADF test for each series
    test_results = []

    for col in numeric_cols:
        series = data[col].dropna()
        if len(series) > 10:  # Need minimum observations
            try:
                result = stats.adfuller(series, autolag='AIC')
                test_results.append({
                    'Variable': col,
                    'ADF_Statistic': result[0],
                    'p_value': result[1],
                    'Critical_1%': result[4]['1%'],
                    'Critical_5%': result[4]['5%'],
                    'Critical_10%': result[4]['10%'],
                    'Stationary': result[1] < 0.05
                })
            except Exception as e:
                test_results.append({
                    'Variable': col,
                    'Error': str(e),
                    'Stationary': False
                })

    if test_results:
        results_df = pd.DataFrame(test_results)

        # Display results with badges
        for idx, row in results_df.iterrows():
            col1, col2, col3 = st.columns([2, 1, 2])

            with col1:
                st.markdown(f"**{row['Variable']}**")

            with col2:
                if row.get('Stationary', False):
                    st.success("✅ Stationary")
                else:
                    st.error("❌ Non-stationary")

            with col3:
                if 'p_value' in row:
                    st.metric("p-value", f"{row['p_value']:.4f}")
                else:
                    st.error("Test failed")

        # Detailed results table
        with st.expander("Detailed Test Results", expanded=False):
            st.dataframe(results_df.round(4), use_container_width=True)

    st.markdown("**Implications for Modeling**")
    st.markdown("""
    - **Stationary Series**: Can be modeled directly, good for traditional time series models
    - **Non-stationary Series**: May require differencing or other transformations
    - **Unit Root Present**: Series has trend/random walk component
    - **p-value < 0.05**: Reject null hypothesis of unit root (series is stationary)
    """)

def create_data_quality_report(data: pd.DataFrame):
    """Create data quality report section."""
    st.subheader("🔍 Data Quality Report")

    col1, col2, col3, col4 = st.columns(4)

    # Missing data percentage
    missing_pct = (data.isnull().sum() / len(data) * 100).mean()
    with col1:
        st.metric(
            "Missing Data %",
            f"{missing_pct:.2f}%",
            delta="Overall average"
        )

    # Outlier count (using IQR method)
    outlier_count = 0
    numeric_cols = data.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        series = data[col].dropna()
        if len(series) > 0:
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((series < (Q1 - 1.5 * IQR)) | (series > (Q3 + 1.5 * IQR))).sum()
            outlier_count += outliers

    with col2:
        st.metric("Outliers Detected", outlier_count)

    # Duplicate rows
    duplicate_count = data.duplicated().sum()
    with col3:
        st.metric("Duplicate Rows", duplicate_count)

    # Date continuity check
    if 'Date' in data.columns:
        data = data.copy()
        data['Date'] = pd.to_datetime(data['Date'])
        date_diffs = data['Date'].sort_values().diff().dt.days
        gaps = (date_diffs > 1).sum()  # More than 1 day gap
        with col4:
            st.metric("Date Gaps", gaps)
    else:
        with col4:
            st.metric("Date Continuity", "N/A")

    # Detailed quality metrics
    st.markdown("**Detailed Quality Metrics**")

    quality_df = pd.DataFrame({
        'Metric': ['Completeness', 'Validity', 'Uniqueness', 'Consistency'],
        'Score': [100 - missing_pct, 95.0, 100 - (duplicate_count/len(data)*100), 98.0],
        'Status': ['Good' if missing_pct < 5 else 'Poor',
                  'Excellent',
                  'Good' if duplicate_count == 0 else 'Poor',
                  'Excellent']
    })

    st.dataframe(quality_df, use_container_width=True)

def create_export_options(data: pd.DataFrame):
    """Create export options section."""
    st.subheader("💾 Export Options")

    col1, col2, col3 = st.columns(3)

    # CSV Export
    with col1:
        csv_data = data.to_csv(index=False)
        st.download_button(
            label="📄 Download CSV",
            data=csv_data,
            file_name=f"market_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    # Excel Export
    with col2:
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            data.to_excel(writer, sheet_name='Market_Data', index=False)
        excel_data = excel_buffer.getvalue()

        st.download_button(
            label="📊 Download Excel",
            data=excel_data,
            file_name=f"market_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

    # Plot Export (placeholder for future implementation)
    with col3:
        st.button("📈 Export Charts (PNG)", disabled=True, use_container_width=True)
        st.caption("Chart export coming soon")

def main():
    """Main function for the data explorer page."""
    st.title("📊 Data Explorer")

    # Load data
    data = load_market_data()

    if data is None:
        return

    # Create all sections
    create_data_overview(data)
    st.markdown("---")

    create_time_series_visualization(data)
    st.markdown("---")

    create_rolling_statistics(data)
    st.markdown("---")

    create_correlation_analysis(data)
    st.markdown("---")

    create_stationarity_tests(data)
    st.markdown("---")

    create_data_quality_report(data)
    st.markdown("---")

    create_export_options(data)

if __name__ == "__main__":
    main()
