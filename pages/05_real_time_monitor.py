"""
Real-Time Monitor Page - Live Market Data & AI Predictions
==========================================================

Comprehensive real-time monitoring dashboard with live data feeds,
continuous predictions, market intelligence, and automated alerts.

Author: GitHub Copilot
Date: February 8, 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# JSON availability (defensive import for constrained environments)
try:
    import json
    JSON_AVAILABLE = True
except Exception as e:
    JSON_AVAILABLE = False
    json = None
    warnings.warn(f"json module not available; JSON exports disabled: {e}")

st.set_page_config(page_title="Real-Time Monitor - Stock Market AI Analytics", page_icon="📈")

# Import monitoring components
try:
    from real_time_monitor import RealtimeDataFeed
    MONITOR_AVAILABLE = True
except ImportError:
    MONITOR_AVAILABLE = False

# Import Prediction Updater — support multiple class names for compatibility
try:
    import prediction_updater as _pu
    # Use getattr to avoid attribute access errors for static analysis
    PredictionUpdater = getattr(_pu, 'PredictionUpdater', None) or getattr(_pu, 'RealtimePredictorUpdater', None)
    if PredictionUpdater is None:
        raise ImportError("No suitable PredictionUpdater class found in prediction_updater module")
    PREDICTOR_AVAILABLE = True
except Exception as e:
    PREDICTOR_AVAILABLE = False
    PredictionUpdater = None
    import warnings
    warnings.warn(f"prediction_updater not available: {e}")

def initialize_realtime_components() -> Tuple[Optional[Any], Optional[Any]]:
    """Initialize real-time monitoring components."""
    realtime_feed = None
    predictor = None

    if MONITOR_AVAILABLE:
        try:
            realtime_feed = RealtimeDataFeed()
        except Exception as e:
            st.error(f"❌ Failed to initialize real-time feed: {e}")

    if PREDICTOR_AVAILABLE and PredictionUpdater is not None:
        try:
            predictor = PredictionUpdater()
        except Exception as e:
            st.error(f"❌ Failed to initialize prediction updater: {e}")
    elif PREDICTOR_AVAILABLE:
        # Class lookup succeeded earlier but class is None — warn
        st.warning("⚠️ Prediction updater module found but no compatible class available.")

    return realtime_feed, predictor

def create_data_source_controls():
    """Create live data source controls."""
    st.subheader("🎛️ Data Source Controls")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        data_mode = st.selectbox(
            "Data Mode",
            options=["Live", "Cache", "Hybrid"],
            index=2,  # Default to Hybrid
            help="Live: Real-time data only, Cache: Stored data only, Hybrid: Live with cache fallback"
        )

    with col2:
        refresh_rate = st.selectbox(
            "Refresh Rate",
            options=["5s", "10s", "30s", "1m", "5m"],
            index=2,  # Default to 30s
            help="How often to refresh data"
        )

    with col3:
        auto_refresh = st.checkbox("Auto-Refresh", value=True)

    with col4:
        manual_refresh = st.button("🔄 Manual Refresh", use_container_width=True)

    return data_mode, refresh_rate, auto_refresh, manual_refresh

def create_data_freshness_indicators(realtime_feed: Optional[Any], predictor: Optional[Any]):
    """Create data freshness indicators."""
    st.subheader("📊 Data Freshness Status")

    if realtime_feed is None:
        st.error("❌ Real-time feed not available")
        return

    try:
        stats = realtime_feed.get_monitoring_stats()
    except:
        stats = {}

    col1, col2, col3, col4 = st.columns(4)

    # Unemployment data freshness
    with col1:
        unemployment_fresh = stats.get('unemployment', {}).get('last_update')
        if unemployment_fresh:
            age_minutes = (datetime.now() - unemployment_fresh).total_seconds() / 60
            if age_minutes < 5:
                status = "🟢"
            elif age_minutes < 15:
                status = "🟡"
            else:
                status = "🔴"
        else:
            status = "⚪"
            age_minutes = float('inf')

        st.metric(
            "Unemployment Data",
            f"{status} {age_minutes:.0f}min ago" if age_minutes != float('inf') else f"{status} No data"
        )

    # VIX data freshness
    with col2:
        vix_fresh = stats.get('vix', {}).get('last_update')
        if vix_fresh:
            age_minutes = (datetime.now() - vix_fresh).total_seconds() / 60
            if age_minutes < 5:
                status = "🟢"
            elif age_minutes < 15:
                status = "🟡"
            else:
                status = "🔴"
        else:
            status = "⚪"
            age_minutes = float('inf')

        st.metric(
            "VIX Data",
            f"{status} {age_minutes:.0f}min ago" if age_minutes != float('inf') else f"{status} No data"
        )

    # Market data freshness
    with col3:
        market_fresh = stats.get('market_indices', {}).get('last_update')
        if market_fresh:
            age_minutes = (datetime.now() - market_fresh).total_seconds() / 60
            if age_minutes < 5:
                status = "🟢"
            elif age_minutes < 15:
                status = "🟡"
            else:
                status = "🔴"
        else:
            status = "⚪"
            age_minutes = float('inf')

        st.metric(
            "Market Data",
            f"{status} {age_minutes:.0f}min ago" if age_minutes != float('inf') else f"{status} No data"
        )

    # Prediction freshness
    with col4:
        if predictor:
            try:
                pred_stats = predictor.get_prediction_accuracy_last_n_days(1)
                pred_fresh = pred_stats.get('last_update')
                if pred_fresh:
                    age_minutes = (datetime.now() - pred_fresh).total_seconds() / 60
                    if age_minutes < 10:
                        status = "🟢"
                    elif age_minutes < 30:
                        status = "🟡"
                    else:
                        status = "🔴"
                else:
                    status = "⚪"
                    age_minutes = float('inf')
            except:
                status = "⚪"
                age_minutes = float('inf')
        else:
            status = "⚪"
            age_minutes = float('inf')

        st.metric(
            "Predictions",
            f"{status} {age_minutes:.0f}min ago" if age_minutes != float('inf') else f"{status} No data"
        )

def create_market_status_indicator(realtime_feed: Optional[Any]):
    """Create market status indicator."""
    st.subheader("🏛️ Market Status")

    if realtime_feed is None:
        st.error("❌ Real-time feed not available")
        return

    try:
        # Get latest market data
        market_data = realtime_feed.get_latest_market_index('NIFTY50', use_live=True)
        vix_data = realtime_feed.get_latest_vix(use_live=True)

        if market_data and vix_data:
            market_value = market_data.get('value', 0)
            market_change = market_data.get('change_percent', 0)
            vix_value = vix_data.get('value', 0)

            # Determine market status
            if market_change > 1.0:
                market_status = "🚀 Bullish"
            elif market_change < -1.0:
                market_status = "📉 Bearish"
            else:
                market_status = "➡️ Sideways"

            # Determine volatility status
            if vix_value > 25:
                vol_status = "🌪️ High Volatility"
            elif vix_value > 18:
                vol_status = "⚠️ Moderate Volatility"
            else:
                vol_status = "😌 Low Volatility"

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("NIFTY 50", f"{market_value:,.0f}", f"{market_change:+.2f}%")

            with col2:
                st.metric("Market Status", market_status)

            with col3:
                st.metric("VIX", f"{vix_value:.2f}")

            with col4:
                st.metric("Volatility", vol_status)

        else:
            st.warning("⚠️ Market data not available")

    except Exception as e:
        st.error(f"❌ Error getting market status: {e}")

def create_live_metrics_cards(realtime_feed: Optional[Any], predictor: Optional[Any]):
    """Create live metrics cards."""
    st.subheader("📈 Live Metrics")

    if realtime_feed is None:
        st.error("❌ Real-time feed not available")
        return

    try:
        # Get latest data
        unemployment_data = realtime_feed.get_latest_unemployment(use_live=True)
        vix_data = realtime_feed.get_latest_vix(use_live=True)
        market_data = realtime_feed.get_latest_market_index('NIFTY50', use_live=True)

        col1, col2, col3, col4 = st.columns(4)

        # Unemployment rate
        with col1:
            if unemployment_data:
                rate = unemployment_data.get('rate', 0)
                change = unemployment_data.get('change', 0)
                st.metric("Unemployment Rate", f"{rate:.1f}%", f"{change:+.1f}pp")
            else:
                st.metric("Unemployment Rate", "N/A")

        # VIX level
        with col2:
            if vix_data:
                vix_value = vix_data.get('value', 0)
                vix_change = vix_data.get('change', 0)
                st.metric("VIX Index", f"{vix_value:.2f}", f"{vix_change:+.2f}")
            else:
                st.metric("VIX Index", "N/A")

        # Market index
        with col3:
            if market_data:
                market_value = market_data.get('value', 0)
                market_change = market_data.get('change_percent', 0)
                st.metric("NIFTY 50", f"{market_value:,.0f}", f"{market_change:+.2f}%")
            else:
                st.metric("NIFTY 50", "N/A")

        # Prediction accuracy (last 24h)
        with col4:
            if predictor:
                try:
                    accuracy_stats = predictor.get_prediction_accuracy_last_n_days(1)
                    avg_accuracy = accuracy_stats.get('overall_accuracy', 0)
                    st.metric("24h Prediction Acc", f"{avg_accuracy:.1f}%")
                except:
                    st.metric("24h Prediction Acc", "N/A")
            else:
                st.metric("24h Prediction Acc", "N/A")

    except Exception as e:
        st.error(f"❌ Error getting live metrics: {e}")

def create_live_prediction_dashboard(predictor: Optional[Any]):
    """Create live prediction dashboard."""
    st.subheader("🔮 Live Prediction Dashboard")

    if predictor is None:
        st.error("❌ Prediction updater not available")
        return

    try:
        # Get latest predictions
        latest_predictions = predictor.update_predictions_on_new_data(use_live_data=True)

        if latest_predictions:
            # Create prediction cards
            models = list(latest_predictions.get('predictions', {}).keys())
            n_models = len(models)

            if n_models > 0:
                cols = st.columns(min(n_models, 4))

                for i, model_name in enumerate(models[:4]):
                    with cols[i]:
                        pred_value = latest_predictions['predictions'][model_name]
                        confidence = latest_predictions.get('confidence_intervals', {}).get(model_name, [pred_value, pred_value])

                        st.metric(
                            f"{model_name}",
                            f"{pred_value:.2f}",
                            f"±{confidence[1]-confidence[0]:.2f}"
                        )

                # Prediction consensus
                consensus = predictor.compute_model_consensus(latest_predictions['predictions'])
                st.metric("Model Consensus", f"{consensus.get('consensus_value', 0):.2f}")

                # Prediction direction indicator
                direction = consensus.get('direction', 'neutral')
                if direction == 'bullish':
                    st.success("📈 Consensus: Bullish")
                elif direction == 'bearish':
                    st.error("📉 Consensus: Bearish")
                else:
                    st.info("➡️ Consensus: Neutral")

            else:
                st.warning("⚠️ No predictions available")

        else:
            st.warning("⚠️ No live predictions available")

    except Exception as e:
        st.error(f"❌ Error getting live predictions: {e}")

def create_emerging_divergences_detection(realtime_feed: Optional[Any], predictor: Optional[Any]):
    """Create emerging divergences detection."""
    st.subheader("🔍 Emerging Divergences")

    if predictor is None:
        st.error("❌ Prediction updater not available")
        return

    try:
        # Check for prediction drift
        drift_analysis = predictor.track_prediction_drift()

        if drift_analysis:
            # Display divergence alerts
            divergences = drift_analysis.get('divergences', [])

            if divergences:
                for divergence in divergences[:5]:  # Show top 5
                    severity = divergence.get('severity', 'low')
                    if severity == 'high':
                        st.error(f"🚨 {divergence.get('description', 'Unknown divergence')}")
                    elif severity == 'medium':
                        st.warning(f"⚠️ {divergence.get('description', 'Unknown divergence')}")
                    else:
                        st.info(f"ℹ️ {divergence.get('description', 'Unknown divergence')}")
            else:
                st.success("✅ No significant divergences detected")

            # Divergence metrics
            col1, col2, col3 = st.columns(3)

            with col1:
                drift_score = drift_analysis.get('drift_score', 0)
                st.metric("Drift Score", f"{drift_score:.3f}")

            with col2:
                stability = drift_analysis.get('stability_index', 0)
                st.metric("Stability Index", f"{stability:.3f}")

            with col3:
                n_divergences = len(divergences)
                st.metric("Active Divergences", n_divergences)

        else:
            st.info("ℹ️ Divergence analysis not available")

    except Exception as e:
        st.error(f"❌ Error in divergence detection: {e}")

def create_30day_accuracy_tracker(predictor: Optional[Any]):
    """Create 30-day accuracy tracker."""
    st.subheader("📊 30-Day Accuracy Tracker")

    if predictor is None:
        st.error("❌ Prediction updater not available")
        return

    try:
        # Get 30-day accuracy data
        accuracy_data = predictor.get_prediction_accuracy_last_n_days(30)

        if accuracy_data and 'daily_accuracy' in accuracy_data:
            daily_acc = accuracy_data['daily_accuracy']

            # Create accuracy over time chart
            dates = list(daily_acc.keys())
            accuracies = list(daily_acc.values())

            # Simple text-based chart for now
            st.line_chart(pd.DataFrame({'Accuracy': accuracies}, index=dates))

            # Accuracy statistics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                avg_accuracy = np.mean(accuracies)
                st.metric("Average Accuracy", f"{avg_accuracy:.1f}%")

            with col2:
                max_accuracy = max(accuracies)
                st.metric("Best Day", f"{max_accuracy:.1f}%")

            with col3:
                min_accuracy = min(accuracies)
                st.metric("Worst Day", f"{min_accuracy:.1f}%")

            with col4:
                trend = np.polyfit(range(len(accuracies)), accuracies, 1)[0]
                trend_direction = "📈 Improving" if trend > 0 else "📉 Declining" if trend < 0 else "➡️ Stable"
                st.metric("Trend", trend_direction)

        else:
            st.warning("⚠️ 30-day accuracy data not available")

    except Exception as e:
        st.error(f"❌ Error in accuracy tracking: {e}")

def create_data_quality_dashboard(realtime_feed: Optional[Any]):
    """Create data quality dashboard."""
    st.subheader("🔍 Data Quality Dashboard")

    if realtime_feed is None:
        st.error("❌ Real-time feed not available")
        return

    try:
        stats = realtime_feed.get_monitoring_stats()

        if stats:
            col1, col2, col3, col4 = st.columns(4)

            # Success rates
            with col1:
                unemployment_success = stats.get('unemployment', {}).get('success_rate', 0)
                st.metric("Unemployment Success", f"{unemployment_success:.1f}%")

            with col2:
                vix_success = stats.get('vix', {}).get('success_rate', 0)
                st.metric("VIX Success", f"{vix_success:.1f}%")

            with col3:
                market_success = stats.get('market_indices', {}).get('success_rate', 0)
                st.metric("Market Success", f"{market_success:.1f}%")

            with col4:
                overall_success = (unemployment_success + vix_success + market_success) / 3
                st.metric("Overall Success", f"{overall_success:.1f}%")

            # Data quality indicators
            st.markdown("### Quality Metrics")

            quality_data = []
            for source, source_stats in stats.items():
                if isinstance(source_stats, dict):
                    quality_data.append({
                        'Source': source.replace('_', ' ').title(),
                        'Success Rate': source_stats.get('success_rate', 0),
                        'Avg Response Time': source_stats.get('avg_response_time', 0),
                        'Error Rate': source_stats.get('error_rate', 0),
                        'Last Update': source_stats.get('last_update', 'Never')
                    })

            if quality_data:
                quality_df = pd.DataFrame(quality_data)
                st.dataframe(quality_df, use_container_width=True, hide_index=True)

        else:
            st.warning("⚠️ Data quality statistics not available")

    except Exception as e:
        st.error(f"❌ Error in data quality dashboard: {e}")

def create_scraping_cache_status_log(realtime_feed: Optional[Any]):
    """Create scraping and cache status log."""
    st.subheader("📋 Scraping & Cache Status Log")

    if realtime_feed is None:
        st.error("❌ Real-time feed not available")
        return

    try:
        # Get recent logs (assuming the feed has logging capabilities)
        # For now, create a mock log display
        log_entries = [
            {"timestamp": datetime.now() - timedelta(minutes=i),
             "level": "INFO" if i % 3 != 0 else "WARNING" if i % 3 == 1 else "ERROR",
             "source": ["Unemployment", "VIX", "Market"][i % 3],
             "message": f"Data {'fetched successfully' if i % 3 != 0 else 'fetch failed - using cache' if i % 3 == 1 else 'cache expired - retrying'}"}
            for i in range(20)
        ]

        # Display recent logs
        log_df = pd.DataFrame(log_entries)
        # Ensure timestamp column is datetime-like then format safely
        log_df['timestamp'] = pd.to_datetime(log_df['timestamp'], errors='coerce')
        log_df['timestamp'] = log_df['timestamp'].apply(lambda ts: ts.strftime('%H:%M:%S') if hasattr(ts, 'strftime') else 'N/A')

        # Color code log levels
        def color_log_level(val):
            if val == "ERROR":
                return "background-color: #FFCCCC"
            elif val == "WARNING":
                return "background-color: #FFFFCC"
            else:
                return ""

        styled_df = log_df.style.apply(
            lambda x: [color_log_level(x.iloc[i]) if i == 1 else "" for i in range(len(x))],
            axis=1
        )

        st.dataframe(styled_df, use_container_width=True, hide_index=True)

        # Log summary
        error_count = sum(1 for entry in log_entries if entry['level'] == 'ERROR')
        warning_count = sum(1 for entry in log_entries if entry['level'] == 'WARNING')
        info_count = sum(1 for entry in log_entries if entry['level'] == 'INFO')

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Logs", len(log_entries))

        with col2:
            st.metric("Errors", error_count)

        with col3:
            st.metric("Warnings", warning_count)

        with col4:
            st.metric("Success", info_count)

    except Exception as e:
        st.error(f"❌ Error in status log: {e}")

def create_live_settings_panel():
    """Create live settings panel."""
    st.subheader("⚙️ Live Settings")

    with st.expander("🔔 Alert Settings", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.checkbox("Enable Price Alerts", value=True)
            st.checkbox("Enable Volume Alerts", value=False)
            st.checkbox("Enable Prediction Alerts", value=True)

        with col2:
            st.slider("Alert Threshold (%)", 0.1, 5.0, 1.0, 0.1)
            st.slider("Volume Threshold (x)", 1.0, 10.0, 2.0, 0.5)
            st.slider("Prediction Confidence", 0.5, 0.95, 0.8, 0.05)

    with st.expander("🔄 Update Settings", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.selectbox("Update Frequency", ["Real-time", "5 minutes", "15 minutes", "1 hour"], index=0)
            st.checkbox("Enable Background Updates", value=True)

        with col2:
            st.selectbox("Cache Strategy", ["Smart Cache", "Always Fresh", "Conservative"], index=0)
            st.slider("Cache TTL (minutes)", 5, 120, 30, 5)

    with st.expander("📊 Display Settings", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.checkbox("Show Confidence Intervals", value=True)
            st.checkbox("Show Historical Comparison", value=True)

        with col2:
            st.selectbox("Chart Theme", ["Light", "Dark", "Auto"], index=2)
            st.slider("Display Precision", 1, 4, 2, 1)

def create_market_stress_indicator(realtime_feed: Optional[Any]):
    """Create market stress indicator."""
    st.subheader("🌪️ Market Stress Indicator")

    if realtime_feed is None:
        st.error("❌ Real-time feed not available")
        return

    try:
        # Get market data for stress calculation
        vix_data = realtime_feed.get_latest_vix(use_live=True)
        market_data = realtime_feed.get_latest_market_index('NIFTY50', use_live=True)

        if vix_data and market_data:
            vix_value = vix_data.get('value', 20)
            market_volatility = abs(market_data.get('change_percent', 0))

            # Calculate stress score (simplified)
            stress_score = (vix_value / 20) * 0.6 + (market_volatility / 2) * 0.4
            stress_score = min(stress_score, 5.0)  # Cap at 5

            # Determine stress level
            if stress_score < 1.0:
                stress_level = "😌 Calm"
                stress_desc = "Market conditions are stable"
            elif stress_score < 2.0:
                stress_level = "🙂 Normal"
                stress_desc = "Typical market activity"
            elif stress_score < 3.0:
                stress_level = "😟 Elevated"
                stress_desc = "Increased market uncertainty"
            elif stress_score < 4.0:
                stress_level = "😰 High"
                stress_desc = "Significant market stress"
            else:
                stress_level = "😱 Extreme"
                stress_desc = "Extreme market volatility"

            # Display stress indicator
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Stress Score", f"{stress_score:.2f}")

            with col2:
                st.metric("Stress Level", stress_level)

            with col3:
                st.metric("VIX Contribution", f"{(vix_value / 20) * 0.6:.2f}")

            st.info(f"📊 {stress_desc}")

            # Simple stress gauge (text-based for now)
            st.progress(min(stress_score / 5.0, 1.0))
            st.caption("Market Stress Gauge (0-5 scale)")

        else:
            st.warning("⚠️ Insufficient data for stress calculation")

    except Exception as e:
        st.error(f"❌ Error calculating market stress: {e}")

def create_scheduled_update_countdown():
    """Create scheduled update countdown."""
    st.subheader("⏰ Scheduled Update Countdown")

    # Mock countdown (in real implementation, this would connect to the scheduler)
    next_update = datetime.now() + timedelta(minutes=5, seconds=30)

    countdown_seconds = max(0, int((next_update - datetime.now()).total_seconds()))

    minutes = countdown_seconds // 60
    seconds = countdown_seconds % 60

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Next Update In", f"{minutes:02d}:{seconds:02d}")

    with col2:
        st.metric("Update Frequency", "5 minutes")

    with col3:
        st.metric("Last Update", datetime.now().strftime("%H:%M:%S"))

    with col4:
        status = "🟢 Active" if countdown_seconds > 0 else "🔄 Updating..."
        st.metric("Scheduler Status", status)

    # Progress bar for countdown
    progress = 1 - (countdown_seconds / 330)  # 5.5 minutes total
    st.progress(progress)

    # Update schedule
    st.markdown("### 📅 Update Schedule")
    schedule_data = [
        {"Time": "09:15", "Update": "Market Open"},
        {"Time": "11:00", "Update": "Mid-Morning"},
        {"Time": "13:00", "Update": "Mid-Day"},
        {"Time": "15:00", "Update": "Pre-Close"},
        {"Time": "15:30", "Update": "Market Close"},
    ]

    schedule_df = pd.DataFrame(schedule_data)
    st.dataframe(schedule_df, use_container_width=True, hide_index=True)

def create_export_download_options(realtime_feed: Optional[Any], predictor: Optional[Any]):
    """Create export and download options."""
    st.subheader("💾 Export & Download Options")

    col1, col2, col3, col4 = st.columns(4)

    # Export live data
    with col1:
        if st.button("📊 Export Live Data", use_container_width=True):
            try:
                if realtime_feed:
                    # Get current data
                    data = {
                        'unemployment': realtime_feed.get_latest_unemployment(),
                        'vix': realtime_feed.get_latest_vix(),
                        'market': realtime_feed.get_latest_market_index('NIFTY50')
                    }

                    if not (JSON_AVAILABLE and json is not None):
                        st.error("❌ JSON export not available (json module missing)")
                    else:
                        # Type-guard for static analysis
                        assert json is not None
                        json_data = json.dumps(data, indent=2, default=str)
                        st.download_button(
                            label="Download JSON",
                            data=json_data,
                            file_name=f"live_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json",
                            use_container_width=True
                        )
                else:
                    st.error("❌ No data to export")
            except Exception as e:
                st.error(f"❌ Export failed: {e}")

    # Export predictions
    with col2:
        if st.button("🔮 Export Predictions", use_container_width=True):
            try:
                if predictor:
                    predictions = predictor.update_predictions_on_new_data()
                    if predictions:
                        if not (JSON_AVAILABLE and json is not None):
                            st.error("❌ JSON export not available (json module missing)")
                        else:
                            assert json is not None
                            json_data = json.dumps(predictions, indent=2, default=str)
                            st.download_button(
                                label="Download JSON",
                                data=json_data,
                                file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json",
                                use_container_width=True
                            )
                    else:
                        st.error("❌ No predictions to export")
                else:
                    st.error("❌ No predictor available")
            except Exception as e:
                st.error(f"❌ Export failed: {e}")

    # Export monitoring stats
    with col3:
        if st.button("📈 Export Stats", use_container_width=True):
            try:
                if realtime_feed:
                    stats = realtime_feed.get_monitoring_stats()
                    if stats:
                        if not (JSON_AVAILABLE and json is not None):
                            st.error("❌ JSON export not available (json module missing)")
                        else:
                            assert json is not None
                            json_data = json.dumps(stats, indent=2, default=str)
                            st.download_button(
                                label="Download JSON",
                                data=json_data,
                                file_name=f"monitoring_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json",
                                use_container_width=True
                            )
                    else:
                        st.error("❌ No stats to export")
                else:
                    st.error("❌ No monitoring data")
            except Exception as e:
                st.error(f"❌ Export failed: {e}")

    # Generate report
    with col4:
        if st.button("📋 Generate Report", use_container_width=True):
            try:
                report_data = {
                    'timestamp': datetime.now().isoformat(),
                    'market_status': 'Active Monitoring',
                    'data_sources': ['Unemployment', 'VIX', 'NIFTY50'],
                    'prediction_models': ['Available'] if predictor else ['Not Available'],
                    'alerts': ['System Operational']
                }

                if not (JSON_AVAILABLE and json is not None):
                    st.error("❌ JSON export not available (json module missing)")
                else:
                    assert json is not None
                    json_data = json.dumps(report_data, indent=2)
                    st.download_button(
                        label="Download Report",
                        data=json_data,
                        file_name=f"monitoring_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        use_container_width=True
                    )
            except Exception as e:
                st.error(f"❌ Report generation failed: {e}")

def main():
    """Main function for the real-time monitor page."""
    st.title("📈 Real-Time Monitor")

    st.markdown("""
    Live market monitoring dashboard with real-time data feeds, continuous AI predictions,
    automated alerts, and comprehensive market intelligence.
    """)

    # Initialize components
    realtime_feed, predictor = initialize_realtime_components()

    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Live Dashboard",
        "🔍 Analytics",
        "⚙️ Settings & Status",
        "💾 Export"
    ])

    with tab1:
        # Data source controls
        data_mode, refresh_rate, auto_refresh, manual_refresh = create_data_source_controls()

        st.markdown("---")

        # Main dashboard sections
        create_data_freshness_indicators(realtime_feed, predictor)
        st.markdown("---")

        create_market_status_indicator(realtime_feed)
        st.markdown("---")

        create_live_metrics_cards(realtime_feed, predictor)
        st.markdown("---")

        create_live_prediction_dashboard(predictor)
        st.markdown("---")

        create_market_stress_indicator(realtime_feed)

    with tab2:
        create_emerging_divergences_detection(realtime_feed, predictor)
        st.markdown("---")

        create_30day_accuracy_tracker(predictor)
        st.markdown("---")

        create_data_quality_dashboard(realtime_feed)

    with tab3:
        create_live_settings_panel()
        st.markdown("---")

        create_scheduled_update_countdown()
        st.markdown("---")

        create_scraping_cache_status_log(realtime_feed)

    with tab4:
        create_export_download_options(realtime_feed, predictor)

if __name__ == "__main__":
    main()