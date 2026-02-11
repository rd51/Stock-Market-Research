"""
Regime Analysis Page - Market Regime Detection and Analysis
===========================================================

Comprehensive analysis of market regimes, volatility clustering, asymmetric responses,
and regime transition dynamics.

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
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf, pacf
from typing import Dict, List, Any, Optional, Tuple
import io

st.set_page_config(page_title="Regime Analysis - Stock Market AI Analytics", page_icon="🎯")

def load_market_data() -> Optional[pd.DataFrame]:
    """Load market data for regime analysis."""
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
        return None

    return data.copy()

def detect_market_regimes(data: pd.DataFrame) -> pd.DataFrame:
    """Detect market regimes using statistical methods."""
    if data is None or data.empty:
        return pd.DataFrame()

    # Simple regime detection based on volatility and returns
    # In a real implementation, this would use more sophisticated methods like HMM
    data = data.copy()

    # Calculate rolling volatility (20-day window)
    if 'Returns' in data.columns:
        data['Volatility'] = data['Returns'].rolling(20).std() * np.sqrt(252)  # Annualized

        # Calculate rolling returns (20-day window)
        data['Rolling_Returns'] = data['Returns'].rolling(20).mean() * 252  # Annualized

        # Simple regime classification
        vol_median = data['Volatility'].median()
        ret_median = data['Rolling_Returns'].median()

        conditions = [
            (data['Volatility'] > vol_median) & (data['Rolling_Returns'] < ret_median),  # High vol, low returns
            (data['Volatility'] < vol_median) & (data['Rolling_Returns'] > ret_median),  # Low vol, high returns
            (data['Volatility'] > vol_median) & (data['Rolling_Returns'] > ret_median),  # High vol, high returns
        ]
        choices = ['Bear', 'Bull', 'Volatile Bull']
        data['Regime'] = np.select(conditions, choices, default='Sideways')

        # Assign colors to regimes
        regime_colors = {
            'Bull': 'green',
            'Bear': 'red',
            'Sideways': 'yellow',
            'Volatile Bull': 'orange'
        }
        data['Regime_Color'] = data['Regime'].map(regime_colors)

    return data

def create_regime_overview(data: pd.DataFrame):
    """Create regime overview section."""
    st.subheader("🎯 Regime Overview")

    if 'Regime' not in data.columns:
        st.error("❌ Regime data not available. Please run regime detection first.")
        return

    # Current regime
    current_regime = data['Regime'].iloc[-1]
    regime_color = data['Regime_Color'].iloc[-1]

    # Days in current regime
    regime_changes = data['Regime'] != data['Regime'].shift(1)
    regime_groups = regime_changes.cumsum()
    current_regime_start = data[regime_groups == regime_groups.iloc[-1]].index[0]
    days_in_regime = len(data) - data.index.get_loc(current_regime_start)

    # Typical duration by regime type
    regime_durations = []
    for regime in data['Regime'].unique():
        regime_mask = data['Regime'] == regime
        if regime_mask.sum() > 0:
            # Calculate average consecutive days in this regime
            regime_periods = (~regime_mask).cumsum()
            avg_duration = regime_mask.groupby(regime_periods).sum().mean()
            regime_durations.append({'Regime': regime, 'Avg_Duration': avg_duration})

    duration_df = pd.DataFrame(regime_durations)
    current_avg_duration = duration_df[duration_df['Regime'] == current_regime]['Avg_Duration'].iloc[0] if not duration_df.empty else 0

    # Regime distribution
    regime_counts = data['Regime'].value_counts()
    regime_percentages = (regime_counts / len(data) * 100).round(1)

    # Display current regime prominently
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        st.markdown(f"""
        <div style="background-color: {regime_color}; padding: 20px; border-radius: 10px; text-align: center; color: white;">
            <h2>Current Regime: {current_regime}</h2>
            <h3>{days_in_regime} days in regime</h3>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.metric("Days in Current Regime", days_in_regime)
        st.metric("Typical Duration", f"{current_avg_duration:.0f} days")

    with col3:
        st.metric("Total Regimes Detected", len(regime_counts))

    # Regime probability distribution
    st.markdown("**Regime Distribution**")
    fig_pie = px.pie(
        values=regime_percentages.values,
        names=regime_percentages.index,
        title="Market Regime Distribution",
        color=regime_percentages.index,
        color_discrete_map={'Bull': 'green', 'Bear': 'red', 'Sideways': 'yellow', 'Volatile Bull': 'orange'}
    )
    fig_pie.update_layout(height=300)
    st.plotly_chart(fig_pie, use_container_width=True)

def create_regime_timeline(data: pd.DataFrame):
    """Create regime timeline visualization."""
    st.subheader("📅 Regime Timeline")

    if 'Regime' not in data.columns or 'Date' not in data.columns:
        st.error("❌ Required data columns not available.")
        return

    # Create subplot with regime background
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add regime background rectangles
    regime_periods = []
    current_regime = data['Regime'].iloc[0]
    start_idx = 0

    for i in range(1, len(data)):
        if data['Regime'].iloc[i] != current_regime:
            regime_periods.append({
                'regime': current_regime,
                'start': start_idx,
                'end': i-1,
                'color': data['Regime_Color'].iloc[start_idx]
            })
            current_regime = data['Regime'].iloc[i]
            start_idx = i

    # Add the last period
    regime_periods.append({
        'regime': current_regime,
        'start': start_idx,
        'end': len(data)-1,
        'color': data['Regime_Color'].iloc[start_idx]
    })

    # Add background rectangles for regimes
    for period in regime_periods:
        fig.add_shape(
            type="rect",
            x0=data['Date'].iloc[period['start']],
            x1=data['Date'].iloc[period['end']],
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            fillcolor=period['color'],
            opacity=0.1,
            line_width=0,
            layer="below"
        )

    # Add VIX data (primary y-axis)
    if 'VIX' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data['Date'],
                y=data['VIX'],
                name="VIX",
                line=dict(color='purple', width=2),
                hovertemplate="Date: %{x}<br>VIX: %{y:.2f}<extra></extra>"
            ),
            secondary_y=False
        )

    # Add Returns data (secondary y-axis)
    if 'Returns' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data['Date'],
                y=data['Returns'],
                name="Returns",
                line=dict(color='blue', width=1),
                hovertemplate="Date: %{x}<br>Returns: %{y:.4f}<extra></extra>"
            ),
            secondary_y=True
        )

    # Update layout
    fig.update_layout(
        title="Market Regime Timeline with VIX and Returns Overlay",
        xaxis_title="Date",
        height=500,
        hovermode='x unified'
    )

    fig.update_yaxes(title_text="VIX", secondary_y=False)
    fig.update_yaxes(title_text="Returns", secondary_y=True)

    st.plotly_chart(fig, use_container_width=True)

    # Add regime legend
    st.markdown("**Regime Legend**")
    legend_cols = st.columns(4)
    regime_info = [
        ("Bull", "green", "Low volatility, high returns"),
        ("Bear", "red", "High volatility, low returns"),
        ("Sideways", "yellow", "Moderate conditions"),
        ("Volatile Bull", "orange", "High volatility, high returns")
    ]

    for i, (regime, color, desc) in enumerate(regime_info):
        with legend_cols[i]:
            st.markdown(f"""
            <div style="background-color: {color}; padding: 10px; border-radius: 5px; color: white; text-align: center;">
                <strong>{regime}</strong><br>
                <small>{desc}</small>
            </div>
            """, unsafe_allow_html=True)

def create_rolling_correlation_analysis(data: pd.DataFrame):
    """Create rolling correlation analysis section."""
    st.subheader("🔄 Rolling Correlation Analysis")

    if data is None or data.empty:
        st.error("❌ No data available for correlation analysis.")
        return

    # Define correlation pairs
    correlation_pairs = [
        ("VIX", "Returns", "VIX vs Returns"),
        ("VIX", "Unemployment_Rate", "VIX vs Unemployment"),
        ("Returns", "Unemployment_Rate", "Returns vs Unemployment")
    ]

    # Create tabs for each correlation pair
    tab_names = [pair[2] for pair in correlation_pairs]
    tabs = st.tabs(tab_names)

    for i, (tab, (var1, var2, title)) in enumerate(zip(tabs, correlation_pairs)):
        with tab:
            if var1 not in data.columns or var2 not in data.columns:
                st.warning(f"⚠️ Variables {var1} or {var2} not found in data.")
                continue

            # Calculate rolling correlation
            window_size = st.slider(
                f"Rolling window size for {title}",
                min_value=10,
                max_value=100,
                value=30,
                key=f"corr_window_{i}"
            )

            # Remove NaN values for correlation calculation
            clean_data = data[[var1, var2]].dropna()

            if len(clean_data) < window_size:
                st.error(f"❌ Insufficient data for {window_size}-day rolling correlation.")
                continue

            rolling_corr = clean_data[var1].rolling(window_size).corr(clean_data[var2])

            # Current correlation value
            current_corr = rolling_corr.iloc[-1] if not rolling_corr.empty else 0

            col1, col2 = st.columns([1, 3])

            with col1:
                st.metric(
                    f"Current {title} Correlation",
                    f"{current_corr:.3f}",
                    delta=f"{'Positive' if current_corr > 0 else 'Negative'} correlation"
                )

                # Correlation strength indicator
                if abs(current_corr) > 0.7:
                    st.success("🔴 Strong correlation")
                elif abs(current_corr) > 0.3:
                    st.warning("🟡 Moderate correlation")
                else:
                    st.info("🔵 Weak correlation")

            with col2:
                # Create correlation plot
                fig_corr = go.Figure()

                fig_corr.add_trace(go.Scatter(
                    x=data['Date'],
                    y=rolling_corr,
                    name=f'{window_size}-day Rolling Correlation',
                    line=dict(color='darkblue', width=2),
                    hovertemplate="Date: %{x}<br>Correlation: %{y:.3f}<extra></extra>"
                ))

                # Add zero line
                fig_corr.add_hline(y=0, line_dash="dash", line_color="gray")

                # Highlight correlation breakdowns (near zero)
                breakdown_threshold = 0.1
                breakdown_periods = abs(rolling_corr) < breakdown_threshold

                if breakdown_periods.any():
                    fig_corr.add_trace(go.Scatter(
                        x=data['Date'][breakdown_periods],
                        y=rolling_corr[breakdown_periods],
                        mode='markers',
                        name='Correlation Breakdown',
                        marker=dict(color='red', size=6, symbol='circle'),
                        showlegend=True
                    ))

                fig_corr.update_layout(
                    title=f"Rolling Correlation: {title} ({window_size}-day window)",
                    xaxis_title="Date",
                    yaxis_title="Correlation",
                    height=300,
                    yaxis_range=[-1, 1]
                )

                st.plotly_chart(fig_corr, use_container_width=True)

def create_volatility_clustering(data: pd.DataFrame):
    """Create volatility clustering analysis."""
    st.subheader("⚡ Volatility Clustering")

    if 'Returns' not in data.columns:
        st.error("❌ Returns data not available for volatility clustering analysis.")
        return

    # Calculate squared returns (volatility proxy)
    data = data.copy()
    data['Squared_Returns'] = data['Returns'] ** 2

    # Create subplot
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add returns
    fig.add_trace(
        go.Scatter(
            x=data['Date'],
            y=data['Returns'],
            name="Returns",
            line=dict(color='blue', width=1),
            opacity=0.7
        ),
        secondary_y=False
    )

    # Add squared returns
    fig.add_trace(
        go.Scatter(
            x=data['Date'],
            y=data['Squared_Returns'],
            name="Squared Returns (Volatility)",
            line=dict(color='red', width=2),
            fill='tozeroy',
            fillcolor='rgba(255, 0, 0, 0.1)'
        ),
        secondary_y=True
    )

    fig.update_layout(
        title="Volatility Clustering: Returns vs Squared Returns",
        xaxis_title="Date",
        height=400,
        hovermode='x unified'
    )

    fig.update_yaxes(title_text="Returns", secondary_y=False)
    fig.update_yaxes(title_text="Squared Returns", secondary_y=True)

    st.plotly_chart(fig, use_container_width=True)

    # Identify high clustering periods
    st.markdown("**Volatility Clustering Analysis**")

    # Calculate rolling volatility of squared returns
    vol_of_vol = data['Squared_Returns'].rolling(20).std()

    # High clustering threshold (top 20%)
    high_clustering_threshold = vol_of_vol.quantile(0.8)
    high_clustering_periods = vol_of_vol > high_clustering_threshold

    clustering_percentage = high_clustering_periods.sum() / len(high_clustering_periods) * 100

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "High Clustering Periods",
            f"{high_clustering_periods.sum()} days",
            f"{clustering_percentage:.1f}% of time"
        )

    with col2:
        # Simple ARCH test (autocorrelation of squared returns)
        squared_returns = data['Squared_Returns'].dropna()
        if len(squared_returns) > 10:
            arch_test = acf(squared_returns, nlags=5)
            arch_significance = any(abs(arch_test[1:]) > 1.96/np.sqrt(len(squared_returns)))

            if arch_significance:
                st.success("✅ Significant ARCH effects detected")
                st.info("Volatility clustering is statistically significant")
            else:
                st.warning("❌ No significant ARCH effects")
                st.info("Limited evidence of volatility clustering")

    # Interpretation
    st.markdown("""
    **Interpretation:**
    - **Squared Returns**: Proxy for volatility (higher values = more volatility)
    - **Clustering**: Periods where high volatility tends to follow high volatility
    - **ARCH Effects**: Statistical test for volatility persistence
    - **High Clustering Periods**: When volatility of volatility is elevated
    """)

def create_asymmetric_response(data: pd.DataFrame):
    """Create asymmetric response analysis."""
    st.subheader("⚖️ Asymmetric Response Analysis")

    if 'VIX' not in data.columns or 'Returns' not in data.columns:
        st.error("❌ VIX and Returns data required for asymmetric response analysis.")
        return

    # Calculate VIX changes (shocks)
    data = data.copy()
    data['VIX_Change'] = data['VIX'].pct_change()

    # Define positive and negative shocks
    positive_shocks = data['VIX_Change'] > data['VIX_Change'].quantile(0.8)  # Top 20%
    negative_shocks = data['VIX_Change'] < data['VIX_Change'].quantile(0.2)  # Bottom 20%

    # Calculate response in returns for next day
    data['Next_Returns'] = data['Returns'].shift(-1)

    # Response to positive shocks (VIX increases)
    pos_response = data[positive_shocks]['Next_Returns'].mean()
    pos_std = data[positive_shocks]['Next_Returns'].std()

    # Response to negative shocks (VIX decreases)
    neg_response = data[negative_shocks]['Next_Returns'].mean()
    neg_std = data[negative_shocks]['Next_Returns'].std()

    # Create bar chart
    fig_asym = go.Figure()

    fig_asym.add_trace(go.Bar(
        x=['VIX Increase (Fear)', 'VIX Decrease (Confidence)'],
        y=[pos_response, neg_response],
        error_y=dict(
            type='data',
            array=[pos_std, neg_std],
            visible=True
        ),
        marker_color=['red', 'green'],
        name='Average Return Response'
    ))

    fig_asym.update_layout(
        title="Asymmetric Response: Market Reaction to VIX Shocks",
        yaxis_title="Next Day Returns",
        height=400
    )

    st.plotly_chart(fig_asym, use_container_width=True)

    # Analysis by regime
    st.markdown("**By Market Regime**")

    if 'Regime' in data.columns:
        regime_responses = []

        for regime in data['Regime'].unique():
            regime_data = data[data['Regime'] == regime]

            pos_resp_regime = regime_data[positive_shocks]['Next_Returns'].mean()
            neg_resp_regime = regime_data[negative_shocks]['Next_Returns'].mean()

            regime_responses.append({
                'Regime': regime,
                'Positive_Shock_Response': pos_resp_regime,
                'Negative_Shock_Response': neg_resp_regime,
                'Asymmetry': abs(pos_resp_regime) - abs(neg_resp_regime)
            })

        response_df = pd.DataFrame(regime_responses)

        # Display as table
        st.dataframe(
            response_df.round(4),
            column_config={
                'Regime': st.column_config.TextColumn('Regime'),
                'Positive_Shock_Response': st.column_config.NumberColumn('Response to VIX ↑', format='%.4f'),
                'Negative_Shock_Response': st.column_config.NumberColumn('Response to VIX ↓', format='%.4f'),
                'Asymmetry': st.column_config.NumberColumn('Asymmetry', format='%.4f')
            },
            use_container_width=True
        )

    # Interpretation
    asymmetry_ratio = abs(pos_response) / abs(neg_response) if neg_response != 0 else float('inf')

    st.markdown(f"""
    **Key Findings:**
    - **VIX increases** (fear/shocks) lead to average next-day returns of **{pos_response:.4f}**
    - **VIX decreases** (confidence) lead to average next-day returns of **{neg_response:.4f}**
    - **Asymmetric ratio**: VIX shocks have **{asymmetry_ratio:.2f}x stronger effect** than VIX declines
    - **Interpretation**: Markets react more strongly to bad news (VIX increases) than good news
    """)

def create_lag_analysis(data: pd.DataFrame):
    """Create lag analysis section."""
    st.subheader("⏰ Lag Analysis")

    if data is None or data.empty:
        st.error("❌ No data available for lag analysis.")
        return

    # Define variable pairs for lag analysis
    lag_pairs = [
        ("VIX", "Returns", "VIX vs Returns"),
        ("Unemployment_Rate", "Returns", "Unemployment vs Returns"),
        ("Unemployment_Rate", "VIX", "Unemployment vs VIX")
    ]

    selected_pair = st.selectbox(
        "Select variable pair for lag analysis:",
        [pair[2] for pair in lag_pairs]
    )

    # Find selected pair
    var1, var2, title = next(pair for pair in lag_pairs if pair[2] == selected_pair)

    if var1 not in data.columns or var2 not in data.columns:
        st.warning(f"⚠️ Variables {var1} or {var2} not found in data.")
        return

    # Calculate cross-correlation
    max_lags = st.slider("Maximum lag to analyze:", 1, 50, 20, key="max_lags")

    # Remove NaN values
    clean_data = data[[var1, var2]].dropna()

    if len(clean_data) < max_lags * 2:
        st.error(f"❌ Insufficient data for {max_lags}-lag analysis.")
        return

    # Calculate cross-correlation
    cross_corr = []
    for lag in range(-max_lags, max_lags + 1):
        if lag < 0:
            corr = clean_data[var1].shift(lag).corr(clean_data[var2])
        else:
            corr = clean_data[var1].corr(clean_data[var2].shift(lag))
        cross_corr.append({'Lag': lag, 'Correlation': corr})

    corr_df = pd.DataFrame(cross_corr)

    # Create lag plot
    fig_lag = go.Figure()

    # Add correlation line
    fig_lag.add_trace(go.Scatter(
        x=corr_df['Lag'],
        y=corr_df['Correlation'],
        mode='lines+markers',
        name='Cross-Correlation',
        line=dict(color='blue', width=2),
        marker=dict(size=6)
    ))

    # Add zero line
    fig_lag.add_hline(y=0, line_dash="dash", line_color="gray")

    # Mark significant correlations (using approximate 95% confidence interval)
    n = len(clean_data)
    significance_level = 1.96 / np.sqrt(n)  # 95% confidence

    significant_positive = corr_df['Correlation'] > significance_level
    significant_negative = corr_df['Correlation'] < -significance_level

    if significant_positive.any():
        fig_lag.add_trace(go.Scatter(
            x=corr_df[significant_positive]['Lag'],
            y=corr_df[significant_positive]['Correlation'],
            mode='markers',
            name='Significant (+)',
            marker=dict(color='green', size=8, symbol='star'),
            showlegend=True
        ))

    if significant_negative.any():
        fig_lag.add_trace(go.Scatter(
            x=corr_df[significant_negative]['Lag'],
            y=corr_df[significant_negative]['Correlation'],
            mode='markers',
            name='Significant (-)',
            marker=dict(color='red', size=8, symbol='star'),
            showlegend=True
        ))

    fig_lag.update_layout(
        title=f"Cross-Correlation: {title} (lags -{max_lags} to +{max_lags})",
        xaxis_title="Lag (days)",
        yaxis_title="Correlation",
        height=400
    )

    st.plotly_chart(fig_lag, use_container_width=True)

    # Find maximum correlations and their lags
    max_corr_idx = corr_df['Correlation'].abs().idxmax()
    max_corr = corr_df.loc[max_corr_idx]
    max_corr_lag = max_corr['Lag']
    max_corr_value = max_corr['Correlation']

    # Interpretation
    if abs(max_corr_value) > significance_level:
        direction = "leads" if max_corr_lag < 0 else "lags"
        var_leading = var1 if max_corr_lag < 0 else var2
        var_lagging = var2 if max_corr_lag < 0 else var1
        lag_days = abs(max_corr_lag)

        st.markdown(f"""
        **Key Findings:**
        - **Maximum correlation**: {max_corr_value:.3f} at lag {max_corr_lag} days
        - **{var_leading} {direction} {var_lagging}** by {lag_days} days
        - **Statistical significance**: {'✅ Significant' if abs(max_corr_value) > significance_level else '❌ Not significant'}
        - **Interpretation**: Changes in {var_leading} predict changes in {var_lagging} {lag_days} days later
        """)
    else:
        st.info("No statistically significant lags detected at the 95% confidence level.")

def create_transition_matrix(data: pd.DataFrame):
    """Create regime transition matrix."""
    st.subheader("🔄 Transition Matrix")

    if 'Regime' not in data.columns:
        st.error("❌ Regime data not available for transition analysis.")
        return

    # Calculate transition matrix
    regimes = data['Regime'].unique()
    transition_matrix = pd.DataFrame(0, index=regimes, columns=regimes)

    # Count transitions
    for i in range(1, len(data)):
        from_regime = data['Regime'].iloc[i-1]
        to_regime = data['Regime'].iloc[i]
        transition_matrix.loc[from_regime, to_regime] += 1

    # Convert to probabilities
    transition_probabilities = transition_matrix.div(transition_matrix.sum(axis=1), axis=0)

    # Create heatmap
    fig_transition = go.Figure(data=go.Heatmap(
        z=transition_probabilities.values,
        x=transition_probabilities.columns,
        y=transition_probabilities.index,
        colorscale='Blues',
        text=np.round(transition_probabilities.values, 3),
        texttemplate='%{text}',
        textfont={"size": 12},
        hoverongaps=False
    ))

    fig_transition.update_layout(
        title="Regime Transition Probabilities",
        xaxis_title="To Regime",
        yaxis_title="From Regime",
        height=500
    )

    st.plotly_chart(fig_transition, use_container_width=True)

    # Transition statistics
    st.markdown("**Transition Statistics**")

    # Most stable regime (highest self-transition probability)
    diagonal_probs = np.diag(transition_probabilities)
    most_stable_idx = np.argmax(diagonal_probs)
    most_stable_regime = regimes[most_stable_idx]
    stability_prob = diagonal_probs[most_stable_idx]

    # Most volatile regime (lowest self-transition probability)
    most_volatile_idx = np.argmin(diagonal_probs)
    most_volatile_regime = regimes[most_volatile_idx]
    volatility_prob = diagonal_probs[most_volatile_idx]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Most Stable Regime", most_stable_regime, f"{stability_prob:.1%} persistence")

    with col2:
        st.metric("Most Volatile Regime", most_volatile_regime, f"{volatility_prob:.1%} persistence")

    with col3:
        avg_transitions = (1 - diagonal_probs).mean()
        st.metric("Avg Regime Changes", f"{avg_transitions:.1%}", "per day")

    # Interpretation
    st.markdown("""
    **Interpretation:**
    - **Transition Matrix**: Shows probability of moving from one regime to another
    - **Diagonal values**: Probability of staying in the same regime (persistence)
    - **Off-diagonal values**: Probability of regime changes
    - **Higher diagonal values**: More stable regimes
    - **Lower diagonal values**: More volatile/frequent regime changes
    """)

def create_export_options(data: pd.DataFrame):
    """Create export options section."""
    st.subheader("💾 Export Options")

    if data is None or data.empty:
        st.warning("⚠️ No data available for export.")
        return

    col1, col2, col3 = st.columns(3)

    # Export regime statistics
    with col1:
        if 'Regime' in data.columns:
            # Create regime summary statistics
            regime_stats = data.groupby('Regime').agg({
                'Returns': ['mean', 'std', 'count'],
                'VIX': ['mean', 'std'] if 'VIX' in data.columns else 'count'
            }).round(4)

            # Flatten column names
            regime_stats.columns = ['_'.join(col).strip() for col in regime_stats.columns]

            csv_stats = regime_stats.to_csv()
            st.download_button(
                label="📊 Download Regime Stats",
                data=csv_stats,
                file_name=f"regime_statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

    # Export transition matrix
    with col2:
        if 'Regime' in data.columns:
            # Recalculate transition matrix
            regimes = data['Regime'].unique()
            transition_matrix = pd.DataFrame(0, index=regimes, columns=regimes)

            for i in range(1, len(data)):
                from_regime = data['Regime'].iloc[i-1]
                to_regime = data['Regime'].iloc[i]
                transition_matrix.loc[from_regime, to_regime] += 1

            transition_probabilities = transition_matrix.div(transition_matrix.sum(axis=1), axis=0)

            csv_transitions = transition_probabilities.to_csv()
            st.download_button(
                label="🔄 Download Transition Matrix",
                data=csv_transitions,
                file_name=f"regime_transitions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

    # Placeholder for plot exports
    with col3:
        st.button("📈 Export Charts (PNG)", disabled=True, use_container_width=True)
        st.caption("Chart export coming soon")

def main():
    """Main function for the regime analysis page."""
    st.title("🎯 Regime Analysis")

    # Load and process data
    data = load_market_data()

    if data is None:
        return

    # Detect regimes
    data_with_regimes = detect_market_regimes(data)

    if data_with_regimes.empty:
        st.error("❌ Failed to process regime data.")
        return

    # Create all sections
    create_regime_overview(data_with_regimes)
    st.markdown("---")

    create_regime_timeline(data_with_regimes)
    st.markdown("---")

    create_rolling_correlation_analysis(data_with_regimes)
    st.markdown("---")

    create_volatility_clustering(data_with_regimes)
    st.markdown("---")

    create_asymmetric_response(data_with_regimes)
    st.markdown("---")

    create_lag_analysis(data_with_regimes)
    st.markdown("---")

    create_transition_matrix(data_with_regimes)
    st.markdown("---")

    create_export_options(data_with_regimes)

if __name__ == "__main__":
    main()

    # Calculate returns
    df['returns'] = df['NIFTY_Close'].pct_change()
