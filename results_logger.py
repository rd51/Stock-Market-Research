"""
Results Logger and Analytics System
====================================

Comprehensive logging, aggregation, and reporting system for stock market analysis.
Tracks model performance, predictions, hypothesis validation, and provides automated reporting.

Author: GitHub Copilot
Date: February 8, 2026
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json
import os
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
import warnings
import streamlit as st
import sqlite3
from pathlib import Path
import pytz

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/logs/results_logger.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
IST = pytz.timezone('Asia/Kolkata')
RESULTS_DB = "data/results/results.db"
REPORTS_DIR = "data/reports"
DASHBOARD_DATA_DIR = "data/dashboard"

# Ensure directories exist
for dir_path in [RESULTS_DB, REPORTS_DIR, DASHBOARD_DATA_DIR]:
    Path(dir_path).parent.mkdir(parents=True, exist_ok=True)


@dataclass
class PredictionResult:
    """Data class for storing individual prediction results."""
    timestamp: datetime
    model_name: str
    symbol: str
    prediction_date: str  # Date string in YYYY-MM-DD format
    predicted_direction: str  # 'UP', 'DOWN', 'SIDEWAYS'
    confidence: float
    actual_direction: Optional[str] = None
    accuracy: Optional[float] = None
    features_used: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class PerformanceMetrics:
    """Data class for storing performance metrics."""
    timestamp: datetime
    model_name: str
    period: str  # 'daily', 'weekly', 'monthly'
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    total_predictions: int
    correct_predictions: int
    avg_confidence: float
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None


@dataclass
class HypothesisResult:
    """Data class for hypothesis validation results."""
    hypothesis_id: str
    hypothesis_statement: str
    test_date: datetime
    test_result: bool  # True if hypothesis validated
    confidence_level: float
    p_value: Optional[float] = None
    effect_size: Optional[float] = None
    sample_size: Optional[int] = None
    methodology: Optional[str] = None
    conclusion: Optional[str] = None


@dataclass
class SystemEvent:
    """Data class for system events and alerts."""
    timestamp: datetime
    event_type: str  # 'INFO', 'WARNING', 'ERROR', 'ALERT'
    component: str
    message: str
    severity: str  # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    metadata: Optional[Dict[str, Any]] = None


class ResultsDatabase:
    """SQLite database manager for results storage and retrieval."""

    def __init__(self, db_path: str = RESULTS_DB):
        self.db_path = db_path
        self._init_database()

    def _init_database(self):
        """Initialize database tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Predictions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    prediction_date TEXT NOT NULL,
                    predicted_direction TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    actual_direction TEXT,
                    accuracy REAL,
                    features_used TEXT,
                    metadata TEXT
                )
            ''')

            # Performance metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    period TEXT NOT NULL,
                    accuracy REAL NOT NULL,
                    precision REAL NOT NULL,
                    recall REAL NOT NULL,
                    f1_score REAL NOT NULL,
                    total_predictions INTEGER NOT NULL,
                    correct_predictions INTEGER NOT NULL,
                    avg_confidence REAL NOT NULL,
                    sharpe_ratio REAL,
                    max_drawdown REAL
                )
            ''')

            # Hypothesis results table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS hypothesis_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    hypothesis_id TEXT NOT NULL,
                    hypothesis_statement TEXT NOT NULL,
                    test_date TEXT NOT NULL,
                    test_result BOOLEAN NOT NULL,
                    confidence_level REAL NOT NULL,
                    p_value REAL,
                    effect_size REAL,
                    sample_size INTEGER,
                    methodology TEXT,
                    conclusion TEXT
                )
            ''')

            # System events table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    component TEXT NOT NULL,
                    message TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    metadata TEXT
                )
            ''')

            # Create indexes for better query performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_predictions_model ON predictions(model_name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_performance_timestamp ON performance_metrics(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_performance_model ON performance_metrics(model_name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_hypothesis_id ON hypothesis_results(hypothesis_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_timestamp ON system_events(timestamp)')

            conn.commit()
            logger.info(f"Initialized results database at {self.db_path}")

    def store_prediction(self, prediction: PredictionResult):
        """Store a prediction result."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO predictions (
                    timestamp, model_name, symbol, prediction_date,
                    predicted_direction, confidence, actual_direction,
                    accuracy, features_used, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                prediction.timestamp.isoformat(),
                prediction.model_name,
                prediction.symbol,
                prediction.prediction_date,
                prediction.predicted_direction,
                prediction.confidence,
                prediction.actual_direction,
                prediction.accuracy,
                json.dumps(prediction.features_used) if prediction.features_used else None,
                json.dumps(prediction.metadata) if prediction.metadata else None
            ))
            conn.commit()

    def store_performance_metrics(self, metrics: PerformanceMetrics):
        """Store performance metrics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO performance_metrics (
                    timestamp, model_name, period, accuracy, precision,
                    recall, f1_score, total_predictions, correct_predictions,
                    avg_confidence, sharpe_ratio, max_drawdown
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metrics.timestamp.isoformat(),
                metrics.model_name,
                metrics.period,
                metrics.accuracy,
                metrics.precision,
                metrics.recall,
                metrics.f1_score,
                metrics.total_predictions,
                metrics.correct_predictions,
                metrics.avg_confidence,
                metrics.sharpe_ratio,
                metrics.max_drawdown
            ))
            conn.commit()

    def store_hypothesis_result(self, hypothesis: HypothesisResult):
        """Store hypothesis validation result."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO hypothesis_results (
                    hypothesis_id, hypothesis_statement, test_date,
                    test_result, confidence_level, p_value, effect_size,
                    sample_size, methodology, conclusion
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                hypothesis.hypothesis_id,
                hypothesis.hypothesis_statement,
                hypothesis.test_date.isoformat(),
                hypothesis.test_result,
                hypothesis.confidence_level,
                hypothesis.p_value,
                hypothesis.effect_size,
                hypothesis.sample_size,
                hypothesis.methodology,
                hypothesis.conclusion
            ))
            conn.commit()

    def store_system_event(self, event: SystemEvent):
        """Store system event."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO system_events (
                    timestamp, event_type, component, message, severity, metadata
                ) VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                event.timestamp.isoformat(),
                event.event_type,
                event.component,
                event.message,
                event.severity,
                json.dumps(event.metadata) if event.metadata else None
            ))
            conn.commit()

    def get_predictions(self, model_name: Optional[str] = None,
                       start_date: Optional[datetime] = None,
                       end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Retrieve predictions with optional filtering."""
        query = "SELECT * FROM predictions WHERE 1=1"
        params = []

        if model_name:
            query += " AND model_name = ?"
            params.append(model_name)

        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date.isoformat())

        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date.isoformat())

        query += " ORDER BY timestamp DESC"

        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=params)

        # Parse JSON columns
        if 'features_used' in df.columns:
            df['features_used'] = df['features_used'].apply(
                lambda x: json.loads(x) if x else None
            )
        if 'metadata' in df.columns:
            df['metadata'] = df['metadata'].apply(
                lambda x: json.loads(x) if x else None
            )

        return df

    def get_performance_metrics(self, model_name: Optional[str] = None,
                               period: Optional[str] = None,
                               start_date: Optional[datetime] = None) -> pd.DataFrame:
        """Retrieve performance metrics with optional filtering."""
        query = "SELECT * FROM performance_metrics WHERE 1=1"
        params = []

        if model_name:
            query += " AND model_name = ?"
            params.append(model_name)

        if period:
            query += " AND period = ?"
            params.append(period)

        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date.isoformat())

        query += " ORDER BY timestamp DESC"

        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn, params=params)

    def get_hypothesis_results(self, hypothesis_id: Optional[str] = None) -> pd.DataFrame:
        """Retrieve hypothesis validation results."""
        query = "SELECT * FROM hypothesis_results"
        params = []

        if hypothesis_id:
            query += " WHERE hypothesis_id = ?"
            params.append(hypothesis_id)

        query += " ORDER BY test_date DESC"

        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn, params=params)

    def get_system_events(self, event_type: Optional[str] = None,
                         severity: Optional[str] = None,
                         start_date: Optional[datetime] = None) -> pd.DataFrame:
        """Retrieve system events with optional filtering."""
        query = "SELECT * FROM system_events WHERE 1=1"
        params = []

        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)

        if severity:
            query += " AND severity = ?"
            params.append(severity)

        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date.isoformat())

        query += " ORDER BY timestamp DESC"

        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=params)

        # Parse metadata
        if 'metadata' in df.columns:
            df['metadata'] = df['metadata'].apply(
                lambda x: json.loads(x) if x else None
            )

        return df


class ResultsAggregator:
    """Aggregates and analyzes prediction results across models."""

    def __init__(self, db: ResultsDatabase):
        self.db = db

    def get_ensemble_prediction(self, symbol: str, prediction_date: str) -> Dict[str, Any]:
        """Generate ensemble prediction from multiple models."""
        # Get predictions for the specific date and symbol
        predictions_df = self.db.get_predictions(
            start_date=datetime.now(IST) - timedelta(days=1),
            end_date=datetime.now(IST) + timedelta(days=1)
        )

        # Filter for the specific symbol and date
        relevant_predictions = predictions_df[
            (predictions_df['symbol'] == symbol) &
            (predictions_df['prediction_date'] == prediction_date)
        ]

        if relevant_predictions.empty:
            return {
                'ensemble_direction': 'UNKNOWN',
                'confidence': 0.0,
                'model_count': 0,
                'agreement_score': 0.0
            }

        # Calculate ensemble direction based on weighted confidence
        direction_weights = {'UP': 0, 'DOWN': 0, 'SIDEWAYS': 0}

        for _, pred in relevant_predictions.iterrows():
            direction = pred['predicted_direction']
            confidence = pred['confidence']
            if direction in direction_weights:
                direction_weights[direction] += confidence

        # Find the direction with highest weight
        ensemble_direction = max(direction_weights, key=direction_weights.get)
        total_weight = sum(direction_weights.values())
        ensemble_confidence = direction_weights[ensemble_direction] / total_weight if total_weight > 0 else 0

        # Calculate agreement score (percentage of models agreeing with ensemble)
        agreement_count = sum(1 for _, pred in relevant_predictions.iterrows()
                            if pred['predicted_direction'] == ensemble_direction)
        agreement_score = agreement_count / len(relevant_predictions)

        return {
            'ensemble_direction': ensemble_direction,
            'confidence': ensemble_confidence,
            'model_count': len(relevant_predictions),
            'agreement_score': agreement_score,
            'individual_predictions': relevant_predictions.to_dict('records')
        }

    def calculate_performance_metrics(self, model_name: str, period_days: int = 30) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics for a model."""
        end_date = datetime.now(IST)
        start_date = end_date - timedelta(days=period_days)

        predictions_df = self.db.get_predictions(
            model_name=model_name,
            start_date=start_date,
            end_date=end_date
        )

        # Filter predictions that have actual results
        evaluated_predictions = predictions_df.dropna(subset=['actual_direction'])

        if evaluated_predictions.empty:
            return PerformanceMetrics(
                timestamp=end_date,
                model_name=model_name,
                period=f"{period_days}d",
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                total_predictions=0,
                correct_predictions=0,
                avg_confidence=0.0
            )

        # Calculate accuracy
        correct_predictions = (evaluated_predictions['predicted_direction'] ==
                             evaluated_predictions['actual_direction']).sum()
        total_predictions = len(evaluated_predictions)
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

        # Calculate precision, recall, F1 for UP direction (as example)
        up_predictions = evaluated_predictions[evaluated_predictions['predicted_direction'] == 'UP']
        up_correct = (up_predictions['predicted_direction'] == up_predictions['actual_direction']).sum()

        precision = up_correct / len(up_predictions) if len(up_predictions) > 0 else 0

        up_actual = evaluated_predictions[evaluated_predictions['actual_direction'] == 'UP']
        recall = up_correct / len(up_actual) if len(up_actual) > 0 else 0

        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        avg_confidence = evaluated_predictions['confidence'].mean()

        return PerformanceMetrics(
            timestamp=end_date,
            model_name=model_name,
            period=f"{period_days}d",
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            total_predictions=total_predictions,
            correct_predictions=correct_predictions,
            avg_confidence=avg_confidence
        )

    def get_model_comparison(self, models: List[str], period_days: int = 30) -> pd.DataFrame:
        """Compare performance across multiple models."""
        comparison_data = []

        for model in models:
            metrics = self.calculate_performance_metrics(model, period_days)
            comparison_data.append({
                'model': model,
                'accuracy': metrics.accuracy,
                'precision': metrics.precision,
                'recall': metrics.recall,
                'f1_score': metrics.f1_score,
                'total_predictions': metrics.total_predictions,
                'avg_confidence': metrics.avg_confidence
            })

        return pd.DataFrame(comparison_data)


class ReportGenerator:
    """Generates automated reports for different time periods."""

    def __init__(self, db: ResultsDatabase, aggregator: ResultsAggregator):
        self.db = db
        self.aggregator = aggregator
        self.reports_dir = Path(REPORTS_DIR)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def generate_daily_report(self, report_date: Optional[datetime] = None) -> str:
        """Generate daily performance report."""
        if report_date is None:
            report_date = datetime.now(IST)

        report_path = self.reports_dir / f"daily_report_{report_date.strftime('%Y%m%d')}.md"

        # Get yesterday's predictions and performance
        yesterday = report_date - timedelta(days=1)
        predictions_df = self.db.get_predictions(
            start_date=yesterday.replace(hour=0, minute=0, second=0),
            end_date=yesterday.replace(hour=23, minute=59, second=59)
        )

        # Get system events from yesterday
        events_df = self.db.get_system_events(
            start_date=yesterday.replace(hour=0, minute=0, second=0)
        )

        # Calculate daily metrics
        total_predictions = len(predictions_df)
        avg_confidence = predictions_df['confidence'].mean() if not predictions_df.empty else 0

        # Count predictions by direction
        direction_counts = predictions_df['predicted_direction'].value_counts() if not predictions_df.empty else pd.Series()

        # System health
        error_count = len(events_df[events_df['event_type'] == 'ERROR'])
        warning_count = len(events_df[events_df['event_type'] == 'WARNING'])

        # Generate report
        report_content = f"""# Daily Performance Report - {report_date.strftime('%Y-%m-%d')}

## Executive Summary
- **Total Predictions**: {total_predictions}
- **Average Confidence**: {avg_confidence:.2f}%
- **System Errors**: {error_count}
- **System Warnings**: {warning_count}

## Prediction Distribution
"""

        for direction in ['UP', 'DOWN', 'SIDEWAYS']:
            count = direction_counts.get(direction, 0)
            percentage = (count / total_predictions * 100) if total_predictions > 0 else 0
            report_content += f"- **{direction}**: {count} ({percentage:.1f}%)\n"

        report_content += "\n## Model Performance\n"

        # Get performance by model
        if not predictions_df.empty:
            model_performance = predictions_df.groupby('model_name').agg({
                'confidence': 'mean',
                'symbol': 'count'
            }).rename(columns={'symbol': 'predictions'})

            for model, stats in model_performance.iterrows():
                report_content += f"- **{model}**: {stats['predictions']} predictions, {stats['confidence']:.2f}% avg confidence\n"

        report_content += "\n## System Events\n"

        if not events_df.empty:
            for _, event in events_df.head(10).iterrows():  # Show last 10 events
                report_content += f"- **{event['event_type']}** ({event['component']}): {event['message']}\n"

        report_content += "\n## Recommendations\n"

        if error_count > 5:
            report_content += "- ⚠️ High error rate detected - investigate system issues\n"
        if avg_confidence < 0.5:
            report_content += "- ⚠️ Low prediction confidence - consider model retraining\n"
        if total_predictions < 10:
            report_content += "- ℹ️ Low prediction volume - check data pipeline\n"

        # Save report
        with open(report_path, 'w') as f:
            f.write(report_content)

        logger.info(f"Generated daily report: {report_path}")
        return str(report_path)


class HypothesisTracker:
    """Tracks and validates research hypotheses."""

    def __init__(self, db: ResultsDatabase):
        self.db = db
        self.hypotheses = self._load_hypotheses()

    def _load_hypotheses(self) -> Dict[str, Dict[str, Any]]:
        """Load predefined hypotheses for testing."""
        return {
            "H1": {
                "statement": "VIX is a leading indicator for NIFTY directional changes with >60% accuracy",
                "test_type": "correlation",
                "threshold": 0.6,
                "time_window": 30
            },
            "H2": {
                "statement": "Ensemble models outperform individual models by at least 5% accuracy",
                "test_type": "comparison",
                "threshold": 0.05,
                "time_window": 90
            },
            "H3": {
                "statement": "Unemployment rate changes predict market volatility with 55% confidence",
                "test_type": "predictive_power",
                "threshold": 0.55,
                "time_window": 60
            },
            "H4": {
                "statement": "LSTM models show better performance during high volatility periods",
                "test_type": "conditional_performance",
                "threshold": 0.65,
                "time_window": 120
            }
        }

    def test_hypothesis(self, hypothesis_id: str) -> HypothesisResult:
        """Test a specific hypothesis using available data."""
        if hypothesis_id not in self.hypotheses:
            raise ValueError(f"Unknown hypothesis ID: {hypothesis_id}")

        hypothesis = self.hypotheses[hypothesis_id]
        test_date = datetime.now(IST)

        # Get relevant data based on hypothesis
        if hypothesis_id == "H1":
            result = self._test_vix_leading_indicator(hypothesis)
        elif hypothesis_id == "H2":
            result = self._test_ensemble_advantage(hypothesis)
        elif hypothesis_id == "H3":
            result = self._test_unemployment_volatility(hypothesis)
        elif hypothesis_id == "H4":
            result = self._test_lstm_volatility_performance(hypothesis)
        else:
            result = HypothesisResult(
                hypothesis_id=hypothesis_id,
                hypothesis_statement=hypothesis["statement"],
                test_date=test_date,
                test_result=False,
                confidence_level=0.0,
                conclusion="Test not implemented"
            )

        # Store result
        self.db.store_hypothesis_result(result)

        logger.info(f"Tested hypothesis {hypothesis_id}: {'PASSED' if result.test_result else 'FAILED'}")
        return result

    def _test_vix_leading_indicator(self, hypothesis: Dict[str, Any]) -> HypothesisResult:
        """Test if VIX leads NIFTY directional changes."""
        # Mock result for demonstration
        test_result = np.random.random() > 0.4  # 60% chance of passing
        confidence = np.random.uniform(0.5, 0.8)

        return HypothesisResult(
            hypothesis_id="H1",
            hypothesis_statement=hypothesis["statement"],
            test_date=datetime.now(IST),
            test_result=test_result,
            confidence_level=confidence,
            p_value=0.05 if test_result else 0.15,
            effect_size=0.3 if test_result else 0.1,
            sample_size=100,
            methodology="Correlation analysis between VIX daily changes and NIFTY next-day moves",
            conclusion=f"VIX {'is' if test_result else 'is not'} a reliable leading indicator for NIFTY direction"
        )

    def _test_ensemble_advantage(self, hypothesis: Dict[str, Any]) -> HypothesisResult:
        """Test if ensemble models outperform individual models."""
        # Get performance data for ensemble vs individual models
        performance_df = self.db.get_performance_metrics(
            start_date=datetime.now(IST) - timedelta(days=hypothesis["time_window"])
        )

        if performance_df.empty:
            return HypothesisResult(
                hypothesis_id="H2",
                hypothesis_statement=hypothesis["statement"],
                test_date=datetime.now(IST),
                test_result=False,
                confidence_level=0.0,
                conclusion="Insufficient performance data"
            )

        # Compare ensemble vs individual model performance
        ensemble_perf = performance_df[performance_df['model_name'].str.contains('ensemble', case=False)]
        individual_perf = performance_df[~performance_df['model_name'].str.contains('ensemble', case=False)]

        if ensemble_perf.empty or individual_perf.empty:
            return HypothesisResult(
                hypothesis_id="H2",
                hypothesis_statement=hypothesis["statement"],
                test_date=datetime.now(IST),
                test_result=False,
                confidence_level=0.0,
                conclusion="Missing ensemble or individual model data"
            )

        ensemble_accuracy = ensemble_perf['accuracy'].mean()
        individual_accuracy = individual_perf['accuracy'].mean()
        advantage = ensemble_accuracy - individual_accuracy

        test_result = advantage >= hypothesis["threshold"]
        confidence = min(abs(advantage) / hypothesis["threshold"], 1.0)

        return HypothesisResult(
            hypothesis_id="H2",
            hypothesis_statement=hypothesis["statement"],
            test_date=datetime.now(IST),
            test_result=test_result,
            confidence_level=confidence,
            effect_size=advantage,
            sample_size=len(performance_df),
            methodology="Comparison of mean accuracy between ensemble and individual models",
            conclusion=f"Ensemble models {'outperform' if test_result else 'do not outperform'} individual models by {advantage:.1f}%"
        )

    def _test_unemployment_volatility(self, hypothesis: Dict[str, Any]) -> HypothesisResult:
        """Test if unemployment rate predicts market volatility."""
        # Mock result for demonstration
        test_result = np.random.random() > 0.45
        confidence = np.random.uniform(0.4, 0.7)

        return HypothesisResult(
            hypothesis_id="H3",
            hypothesis_statement=hypothesis["statement"],
            test_date=datetime.now(IST),
            test_result=test_result,
            confidence_level=confidence,
            p_value=0.08 if test_result else 0.2,
            effect_size=0.25 if test_result else 0.05,
            sample_size=80,
            methodology="Regression analysis of unemployment rate changes vs market volatility measures",
            conclusion=f"Unemployment rate {'does' if test_result else 'does not'} predict market volatility"
        )

    def _test_lstm_volatility_performance(self, hypothesis: Dict[str, Any]) -> HypothesisResult:
        """Test if LSTM models perform better during high volatility."""
        # Mock result for demonstration
        test_result = np.random.random() > 0.35
        confidence = np.random.uniform(0.5, 0.8)

        return HypothesisResult(
            hypothesis_id="H4",
            hypothesis_statement=hypothesis["statement"],
            test_date=datetime.now(IST),
            test_result=test_result,
            effect_size=0.4 if test_result else 0.15,
            sample_size=120,
            methodology="Conditional performance analysis during high volatility periods (VIX > 25)",
            conclusion=f"LSTM models {'do' if test_result else 'do not'} show superior performance during high volatility"
        )

    def get_hypothesis_status(self) -> pd.DataFrame:
        """Get current status of all hypotheses."""
        results_df = self.db.get_hypothesis_results()

        if results_df.empty:
            return pd.DataFrame()

        # Get latest result for each hypothesis
        latest_results = results_df.sort_values('test_date').groupby('hypothesis_id').last().reset_index()

        return latest_results


class ResultsDashboard:
    """Streamlit-based dashboard for results visualization."""

    def __init__(self, db: ResultsDatabase, aggregator: ResultsAggregator,
                 hypothesis_tracker: HypothesisTracker):
        self.db = db
        self.aggregator = aggregator
        self.hypothesis_tracker = hypothesis_tracker

    def create_dashboard(self):
        """Create the main dashboard layout."""
        st.set_page_config(
            page_title="Stock Market Analysis - Results Dashboard",
            page_icon="📊",
            layout="wide"
        )

        st.title("📊 Stock Market Analysis - Results Dashboard")
        st.markdown("---")

        # Overview metrics
        self._display_overview_metrics()

        st.markdown("---")

        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Performance Analytics",
            "🎯 Model Comparison",
            "🧪 Hypothesis Testing",
            "⚠️ System Health"
        ])

        with tab1:
            self._performance_analytics_tab()

        with tab2:
            self._model_comparison_tab()

        with tab3:
            self._hypothesis_testing_tab()

        with tab4:
            self._system_health_tab()

    def _display_overview_metrics(self):
        """Display key overview metrics."""
        col1, col2, col3, col4 = st.columns(4)

        # Get recent data
        recent_predictions = self.db.get_predictions(
            start_date=datetime.now(IST) - timedelta(days=7)
        )

        with col1:
            total_predictions = len(recent_predictions)
            st.metric(
                "Total Predictions (7d)",
                f"{total_predictions:,}",
                help="Total predictions made in the last 7 days"
            )

        with col2:
            avg_confidence = recent_predictions['confidence'].mean() if not recent_predictions.empty else 0
            st.metric(
                "Avg Confidence",
                f"{avg_confidence:.1f}%",
                help="Average prediction confidence"
            )

        with col3:
            active_models = recent_predictions['model_name'].nunique() if not recent_predictions.empty else 0
            st.metric(
                "Active Models",
                active_models,
                help="Number of models currently active"
            )

        with col4:
            # System health score (simplified)
            recent_events = self.db.get_system_events(
                start_date=datetime.now(IST) - timedelta(hours=24)
            )
            error_rate = len(recent_events[recent_events['event_type'] == 'ERROR']) / max(len(recent_events), 1)
            health_score = max(0, 100 - (error_rate * 1000))  # Simplified health calculation
            st.metric(
                "System Health",
                f"{health_score:.0f}%",
                help="Overall system health score based on error rates"
            )

    def _performance_analytics_tab(self):
        """Performance analytics tab content."""
        st.header("Performance Analytics")

        # Recent predictions table
        st.subheader("Recent Predictions")
        recent_preds = self.db.get_predictions(
            start_date=datetime.now(IST) - timedelta(hours=1)
        ).head(20)

        if not recent_preds.empty:
            # Format for display
            display_df = recent_preds[[
                'timestamp', 'model_name', 'symbol',
                'predicted_direction', 'confidence', 'actual_direction'
            ]].copy()

            display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%H:%M:%S')
            display_df['confidence'] = display_df['confidence'].round(1)

            st.dataframe(display_df, use_container_width=True)
        else:
            st.info("No recent predictions to display.")

    def _model_comparison_tab(self):
        """Model comparison tab content."""
        st.header("Model Comparison")

        # Get comparison data
        predictions_df = self.db.get_predictions()
        if not predictions_df.empty:
            available_models = predictions_df['model_name'].unique()
            selected_models = st.multiselect(
                "Select Models to Compare",
                available_models,
                default=available_models[:3] if len(available_models) > 3 else available_models
            )

            if selected_models:
                comparison_df = self.aggregator.get_model_comparison(selected_models, period_days=30)

                if not comparison_df.empty:
                    st.subheader("Performance Comparison")
                    st.dataframe(
                        comparison_df.round(3),
                        use_container_width=True
                    )
                else:
                    st.info("No comparison data available.")
            else:
                st.warning("Please select models to compare.")
        else:
            st.info("No model data available yet.")

    def _hypothesis_testing_tab(self):
        """Hypothesis testing tab content."""
        st.header("Hypothesis Validation")

        # Test hypotheses button
        if st.button("🔬 Run Hypothesis Tests"):
            with st.spinner("Testing hypotheses..."):
                progress_bar = st.progress(0)
                hypotheses = list(self.hypothesis_tracker.hypotheses.keys())

                for i, hyp_id in enumerate(hypotheses):
                    try:
                        result = self.hypothesis_tracker.test_hypothesis(hyp_id)
                        st.success(f"✅ Tested {hyp_id}: {'PASSED' if result.test_result else 'FAILED'}")
                    except Exception as e:
                        st.error(f"❌ Failed to test {hyp_id}: {str(e)}")

                    progress_bar.progress((i + 1) / len(hypotheses))

                st.rerun()

        # Display hypothesis results
        hypothesis_df = self.hypothesis_tracker.get_hypothesis_status()

        if not hypothesis_df.empty:
            # Summary stats
            col1, col2, col3 = st.columns(3)
            with col1:
                total_tests = len(hypothesis_df)
                st.metric("Total Hypotheses", total_tests)

            with col2:
                passed = len(hypothesis_df[hypothesis_df['test_result'] == True])
                st.metric("Validated", f"{passed}/{total_tests}")

            with col3:
                avg_confidence = hypothesis_df['confidence_level'].mean()
                st.metric("Avg Confidence", f"{avg_confidence:.1f}%")

            # Results table
            st.subheader("Hypothesis Results")
            display_df = hypothesis_df[[
                'hypothesis_id', 'test_result', 'confidence_level',
                'test_date', 'conclusion'
            ]].copy()

            display_df['test_date'] = pd.to_datetime(display_df['test_date']).dt.strftime('%Y-%m-%d')
            display_df['confidence_level'] = display_df['confidence_level'].round(2)
            display_df['test_result'] = display_df['test_result'].map({True: '✅ Passed', False: '❌ Failed'})

            st.dataframe(display_df, use_container_width=True)
        else:
            st.info("No hypothesis tests have been run yet.")

    def _system_health_tab(self):
        """System health tab content."""
        st.header("System Health Monitor")

        # Recent system events
        st.subheader("Recent System Events")

        events_df = self.db.get_system_events(
            start_date=datetime.now(IST) - timedelta(hours=24)
        ).head(50)

        if not events_df.empty:
            # Event type distribution
            event_counts = events_df['event_type'].value_counts()

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Event Types (24h)")
                st.bar_chart(event_counts)

            with col2:
                st.subheader("Severity Levels")
                severity_counts = events_df['severity'].value_counts()
                st.bar_chart(severity_counts)

            # Events table
            st.subheader("Event Log")
            display_df = events_df[[
                'timestamp', 'event_type', 'component',
                'severity', 'message'
            ]].copy()

            display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%H:%M:%S')

            st.dataframe(display_df, use_container_width=True)
        else:
            st.info("No system events in the last 24 hours.")


class ResultsLogger:
    """Main results logging and analytics system."""

    def __init__(self):
        self.db = ResultsDatabase()
        self.aggregator = ResultsAggregator(self.db)
        self.hypothesis_tracker = HypothesisTracker(self.db)
        self.report_generator = ReportGenerator(self.db, self.aggregator)
        self.dashboard = ResultsDashboard(self.db, self.aggregator, self.hypothesis_tracker)

        logger.info("Results Logger initialized")

    def log_prediction(self, prediction: PredictionResult):
        """Log a prediction result."""
        self.db.store_prediction(prediction)
        logger.debug(f"Logged prediction: {prediction.model_name} -> {prediction.predicted_direction}")

    def log_performance_metrics(self, metrics: PerformanceMetrics):
        """Log performance metrics."""
        self.db.store_performance_metrics(metrics)
        logger.debug(f"Logged performance metrics for {metrics.model_name}")

    def log_system_event(self, event: SystemEvent):
        """Log a system event."""
        self.db.store_system_event(event)
        logger.debug(f"Logged system event: {event.event_type} - {event.component}")

    def get_ensemble_prediction(self, symbol: str, prediction_date: str) -> Dict[str, Any]:
        """Get ensemble prediction for a symbol and date."""
        return self.aggregator.get_ensemble_prediction(symbol, prediction_date)

    def calculate_model_performance(self, model_name: str, period_days: int = 30) -> PerformanceMetrics:
        """Calculate performance metrics for a model."""
        return self.aggregator.calculate_model_performance(model_name, period_days)

    def test_hypothesis(self, hypothesis_id: str) -> HypothesisResult:
        """Test a specific hypothesis."""
        return self.hypothesis_tracker.test_hypothesis(hypothesis_id)

    def generate_report(self, report_type: str, report_date: Optional[datetime] = None) -> str:
        """Generate a report of specified type."""
        if report_type == "daily":
            return self.report_generator.generate_daily_report(report_date)
        elif report_type == "weekly":
            return self.report_generator.generate_weekly_report(report_date)
        elif report_type == "monthly":
            return self.report_generator.generate_monthly_report(report_date)
        else:
            raise ValueError(f"Unknown report type: {report_type}")

    def start_dashboard(self):
        """Start the Streamlit dashboard."""
        self.dashboard.create_dashboard()

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for dashboard display."""
        return {
            'recent_predictions': self.db.get_predictions(
                start_date=datetime.now(IST) - timedelta(hours=24)
            ).to_dict('records'),
            'performance_metrics': self.db.get_performance_metrics(
                start_date=datetime.now(IST) - timedelta(days=7)
            ).to_dict('records'),
            'hypothesis_status': self.hypothesis_tracker.get_hypothesis_status().to_dict('records'),
            'system_events': self.db.get_system_events(
                start_date=datetime.now(IST) - timedelta(hours=24)
            ).to_dict('records')
        }


# Global instance for easy access
results_logger = ResultsLogger()

print("✅ Results Logger module loaded successfully!")
