"""
Model Retraining Scheduler
==========================

Automated model retraining system with performance monitoring and versioning.
Handles scheduled retraining, performance degradation detection, and model updates.

Features:
- Weekly scheduled retraining
- Performance comparison and validation
- Model versioning and rollback
- Automatic retraining triggers on degradation
- Comprehensive logging and monitoring

Author: GitHub Copilot
Date: February 8, 2026
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings
import os
import json
import joblib
from pathlib import Path
import pytz
from dataclasses import dataclass, asdict
import hashlib
warnings.filterwarnings('ignore')

# Scheduling imports
try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.jobstores.memory import MemoryJobStore
    from apscheduler.executors.asyncio import AsyncIOExecutor
    APSCHEDULER_AVAILABLE = True
except ImportError:
    APSCHEDULER_AVAILABLE = False
    BackgroundScheduler = None
    logging.warning("APScheduler not available - scheduling disabled")

# Import existing modules
try:
    from preprocessing_pipeline import DataPreprocessingPipeline
    PREPROCESSING_AVAILABLE = True
except ImportError:
    PREPROCESSING_AVAILABLE = False
    logging.warning("preprocessing_pipeline not available")

try:
    from ensemble_models import EnsembleModelTrainer
    ENSEMBLE_AVAILABLE = True
except ImportError:
    ENSEMBLE_AVAILABLE = False
    logging.warning("ensemble_models not available")

try:
    from model_evaluation import ModelEvaluator
    EVALUATION_AVAILABLE = True
except ImportError:
    EVALUATION_AVAILABLE = False
    logging.warning("model_evaluation not available")

try:
    from prediction_updater import RealtimePredictorUpdater
    PREDICTOR_AVAILABLE = True
except ImportError:
    PREDICTOR_AVAILABLE = False
    logging.warning("prediction_updater not available")

# Constants
IST = pytz.timezone('Asia/Kolkata')
MODEL_DIR = Path("models")
MODEL_VERSIONS_DIR = Path("models/versions")
PERFORMANCE_LOG_FILE = "data/logs/model_performance.json"
RETRAINING_LOG_FILE = "data/logs/retraining_history.json"
CONFIG_FILE = "config/retraining_config.json"

# Default configuration
DEFAULT_CONFIG = {
    "schedule": {
        "weekly_retraining_day": "sunday",  # sunday, monday, etc.
        "weekly_retraining_time": "02:00",  # HH:MM format
        "timezone": "Asia/Kolkata"
    },
    "performance_thresholds": {
        "min_r2_score": 0.6,
        "max_mae_increase_pct": 15.0,
        "max_rmse_increase_pct": 15.0,
        "min_accuracy_decline_pct": -10.0,  # Negative for decline
        "performance_window_days": 30
    },
    "retraining_triggers": {
        "data_age_days": 7,  # Retrain if data is older than this
        "performance_check_interval_hours": 24,
        "auto_retrain_enabled": True,
        "manual_retrain_cooldown_hours": 6
    },
    "model_versioning": {
        "max_versions_to_keep": 10,
        "version_naming_format": "model_{model_name}_{timestamp}_{performance_score:.3f}",
        "backup_before_retrain": True
    },
    "notifications": {
        "email_enabled": False,
        "slack_enabled": False,
        "log_level": "INFO"
    }
}

@dataclass
class ModelVersion:
    """Model version information."""
    model_name: str
    version_id: str
    timestamp: datetime
    performance_metrics: Dict[str, float]
    file_path: str
    is_active: bool = False
    retraining_reason: str = ""

@dataclass
class PerformanceMetrics:
    """Performance metrics for model evaluation."""
    r2_score: float
    mae: float
    rmse: float
    mape: float
    accuracy: Optional[float] = None
    timestamp: Optional[datetime] = None

@dataclass
class RetrainingResult:
    """Result of a retraining operation."""
    success: bool
    model_name: str
    old_version: str
    new_version: str
    performance_improved: bool
    metrics_comparison: Dict[str, Dict[str, float]]
    timestamp: datetime
    error_message: Optional[str] = None

class ModelRetrainingScheduler:
    """
    Automated model retraining system with performance monitoring and versioning.
    """

    def __init__(self, config_file: str = CONFIG_FILE):
        """
        Initialize the retraining scheduler.

        Args:
            config_file: Path to configuration file
        """
        self.config_file = Path(config_file)

        # Setup logging first
        self._setup_logging()

        # Load configuration
        self.config = self._load_config()

        # Update logging level based on config
        self._update_log_level()

        # Setup directories
        MODEL_VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
        Path("data/logs").mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.scheduler = None
        self.model_versions: Dict[str, List[ModelVersion]] = {}
        self.performance_history: Dict[str, List[PerformanceMetrics]] = {}
        self.last_retraining: Dict[str, datetime] = {}

        # Load existing model versions and performance history
        self._load_model_versions()
        self._load_performance_history()

        # Setup scheduler if available
        if APSCHEDULER_AVAILABLE:
            self._setup_scheduler()

        self.logger.info("Model Retraining Scheduler initialized")

    def _setup_logging(self):
        """Setup logging configuration."""
        # Use default log level initially, will be updated after config is loaded
        log_level = logging.INFO

        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('data/logs/retraining_scheduler.log'),
                logging.StreamHandler()
            ]
        )

        self.logger = logging.getLogger(__name__)

    def _update_log_level(self):
        """Update logging level based on configuration."""
        try:
            log_level = getattr(logging, self.config["notifications"]["log_level"])
            logging.getLogger().setLevel(log_level)
            self.logger.setLevel(log_level)
        except (KeyError, AttributeError):
            self.logger.warning("Could not update log level from config, using default")

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                # Log after logger is set up
                print(f"Loaded configuration from {self.config_file}")
                return config
            except Exception as e:
                # Log after logger is set up
                print(f"Failed to load config file: {e}")

        # Save default config
        self._save_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG

    def _save_config(self, config: Dict[str, Any]):
        """Save configuration to file."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2, default=str)
            self.logger.info(f"Saved configuration to {self.config_file}")
        except Exception as e:
            self.logger.error(f"Failed to save config: {e}")

    def _setup_scheduler(self):
        """Setup the background scheduler for automated tasks."""
        if not APSCHEDULER_AVAILABLE:
            return

        self.scheduler = BackgroundScheduler(
            jobstores={'default': MemoryJobStore()},
            executors={'default': AsyncIOExecutor()},
            job_defaults={
                'coalesce': True,
                'max_instances': 1,
                'misfire_grace_time': 30
            },
            timezone=IST
        )

        # Schedule weekly retraining
        self._schedule_weekly_retraining()

        # Schedule performance monitoring
        self._schedule_performance_monitoring()

        self.scheduler.start()
        self.logger.info("Background scheduler started")

    def _schedule_weekly_retraining(self):
        """Schedule weekly retraining job."""
        schedule_config = self.config["schedule"]
        day = schedule_config["weekly_retraining_day"]
        time_str = schedule_config["weekly_retraining_time"]

        # Parse time
        hour, minute = map(int, time_str.split(":"))

        # Map day name to cron day
        day_mapping = {
            'monday': 'mon', 'tuesday': 'tue', 'wednesday': 'wed',
            'thursday': 'thu', 'friday': 'fri', 'saturday': 'sat', 'sunday': 'sun'
        }
        cron_day = day_mapping.get(day.lower(), 'sun')

        trigger = CronTrigger(
            day_of_week=cron_day,
            hour=hour,
            minute=minute,
            timezone=IST
        )

        self.scheduler.add_job(
            self.retrain_all_models,
            trigger=trigger,
            id='weekly_retraining',
            name='Weekly Model Retraining',
            replace_existing=True
        )

        self.logger.info(f"Scheduled weekly retraining for {day} at {time_str}")

    def _schedule_performance_monitoring(self):
        """Schedule performance monitoring job."""
        interval_hours = self.config["retraining_triggers"]["performance_check_interval_hours"]

        self.scheduler.add_job(
            self.check_performance_degradation,
            trigger=f"interval hours={interval_hours}",
            id='performance_monitoring',
            name='Performance Monitoring',
            replace_existing=True
        )

        self.logger.info(f"Scheduled performance monitoring every {interval_hours} hours")

    def _load_model_versions(self):
        """Load existing model versions from disk."""
        if not MODEL_VERSIONS_DIR.exists():
            return

        for model_file in MODEL_VERSIONS_DIR.glob("*.json"):
            try:
                with open(model_file, 'r') as f:
                    version_data = json.load(f)

                model_name = version_data["model_name"]
                if model_name not in self.model_versions:
                    self.model_versions[model_name] = []

                version = ModelVersion(**version_data)
                self.model_versions[model_name].append(version)

            except Exception as e:
                self.logger.error(f"Failed to load model version {model_file}: {e}")

        # Sort versions by timestamp
        for model_name in self.model_versions:
            self.model_versions[model_name].sort(key=lambda x: x.timestamp, reverse=True)

        self.logger.info(f"Loaded {sum(len(v) for v in self.model_versions.values())} model versions")

    def _load_performance_history(self):
        """Load performance history from disk."""
        if not Path(PERFORMANCE_LOG_FILE).exists():
            return

        try:
            with open(PERFORMANCE_LOG_FILE, 'r') as f:
                history_data = json.load(f)

            for model_name, metrics_list in history_data.items():
                self.performance_history[model_name] = [
                    PerformanceMetrics(**metrics) for metrics in metrics_list
                ]

            self.logger.info("Loaded performance history")

        except Exception as e:
            self.logger.error(f"Failed to load performance history: {e}")

    def _save_performance_history(self):
        """Save performance history to disk."""
        try:
            history_data = {}
            for model_name, metrics_list in self.performance_history.items():
                history_data[model_name] = [asdict(metrics) for metrics in metrics_list]

            with open(PERFORMANCE_LOG_FILE, 'w') as f:
                json.dump(history_data, f, indent=2, default=str)

        except Exception as e:
            self.logger.error(f"Failed to save performance history: {e}")

    def _get_current_models(self) -> Dict[str, Any]:
        """Get currently active models."""
        models = {}

        # Load models from the main models directory
        model_files = {
            'random_forest': 'random_forest.joblib',
            'xgboost': 'xgboost.joblib',
            'lightgbm': 'lightgbm.joblib',
            'ensemble': 'ensemble.joblib',
            'ols_static': 'ols_static.joblib'
        }

        for model_name, filename in model_files.items():
            model_path = MODEL_DIR / filename
            if model_path.exists():
                try:
                    models[model_name] = joblib.load(model_path)
                    self.logger.debug(f"Loaded current model: {model_name}")
                except Exception as e:
                    self.logger.error(f"Failed to load model {model_name}: {e}")

        return models

    def _get_current_scalers(self) -> Dict[str, Any]:
        """Get current scalers."""
        scaler_path = MODEL_DIR / "scalers.joblib"
        if scaler_path.exists():
            try:
                return joblib.load(scaler_path)
            except Exception as e:
                self.logger.error(f"Failed to load scalers: {e}")

        return {}

    def evaluate_model_performance(self, model_name: str, model: Any,
                                 X_test: pd.DataFrame, y_test: pd.Series) -> PerformanceMetrics:
        """
        Evaluate model performance on test data.

        Args:
            model_name: Name of the model
            model: Trained model object
            X_test: Test features
            y_test: Test targets

        Returns:
            PerformanceMetrics object
        """
        try:
            # Make predictions
            y_pred = model.predict(X_test)

            # Calculate metrics
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

            metrics = PerformanceMetrics(
                r2_score=r2,
                mae=mae,
                rmse=rmse,
                mape=mape,
                timestamp=datetime.now(IST)
            )

            # Store in history
            if model_name not in self.performance_history:
                self.performance_history[model_name] = []
            self.performance_history[model_name].append(metrics)

            # Keep only recent history
            window_days = self.config["performance_thresholds"]["performance_window_days"]
            cutoff_date = datetime.now(IST) - timedelta(days=window_days)

            self.performance_history[model_name] = [
                m for m in self.performance_history[model_name]
                if m.timestamp and m.timestamp > cutoff_date
            ]

            self._save_performance_history()

            self.logger.info(f"Evaluated {model_name}: R²={r2:.3f}, MAE={mae:.4f}, RMSE={rmse:.4f}")

            return metrics

        except Exception as e:
            self.logger.error(f"Failed to evaluate {model_name}: {e}")
            return PerformanceMetrics(r2_score=0, mae=0, rmse=0, mape=0)

    def check_performance_degradation(self, model_name: Optional[str] = None) -> Dict[str, bool]:
        """
        Check for performance degradation in models.

        Args:
            model_name: Specific model to check, or None for all models

        Returns:
            Dictionary of model_name -> needs_retraining
        """
        self.logger.info("Checking for performance degradation...")

        needs_retraining = {}

        models_to_check = [model_name] if model_name else list(self.performance_history.keys())

        for model in models_to_check:
            if model not in self.performance_history or len(self.performance_history[model]) < 2:
                continue

            metrics_history = self.performance_history[model]
            if len(metrics_history) < 2:
                continue

            # Get recent metrics (last 7 days vs previous period)
            recent_metrics = [m for m in metrics_history if (datetime.now(IST) - m.timestamp).days <= 7]
            older_metrics = [m for m in metrics_history if (datetime.now(IST) - m.timestamp).days > 7]

            if not recent_metrics or not older_metrics:
                continue

            # Calculate average metrics
            recent_r2 = np.mean([m.r2_score for m in recent_metrics])
            older_r2 = np.mean([m.r2_score for m in older_metrics])

            recent_mae = np.mean([m.mae for m in recent_metrics])
            older_mae = np.mean([m.mae for m in older_metrics])

            # Check thresholds
            thresholds = self.config["performance_thresholds"]

            r2_decline = (older_r2 - recent_r2) / abs(older_r2) * 100 if older_r2 != 0 else 0
            mae_increase = (recent_mae - older_mae) / abs(older_mae) * 100 if older_mae != 0 else 0

            needs_retrain = (
                recent_r2 < thresholds["min_r2_score"] or
                r2_decline > abs(thresholds["min_accuracy_decline_pct"]) or
                mae_increase > thresholds["max_mae_increase_pct"]
            )

            needs_retraining[model] = needs_retrain

            if needs_retrain:
                self.logger.warning(
                    f"Performance degradation detected for {model}: "
                    f"R² decline: {r2_decline:.1f}%, MAE increase: {mae_increase:.1f}%"
                )

                # Trigger automatic retraining if enabled
                if self.config["retraining_triggers"]["auto_retrain_enabled"]:
                    self.trigger_retraining(model, f"Performance degradation: R² {r2_decline:.1f}%, MAE {mae_increase:.1f}%")

        return needs_retraining

    def trigger_retraining(self, model_name: str, reason: str):
        """
        Trigger retraining for a specific model.

        Args:
            model_name: Name of the model to retrain
            reason: Reason for retraining
        """
        # Check cooldown period
        cooldown_hours = self.config["retraining_triggers"]["manual_retrain_cooldown_hours"]
        if model_name in self.last_retraining:
            time_since_last = datetime.now(IST) - self.last_retraining[model_name]
            if time_since_last < timedelta(hours=cooldown_hours):
                self.logger.info(f"Retraining cooldown active for {model_name}, skipping")
                return

        self.logger.info(f"Triggering retraining for {model_name}: {reason}")

        # Schedule immediate retraining
        if self.scheduler:
            self.scheduler.add_job(
                self.retrain_model,
                args=[model_name, reason],
                id=f'retrain_{model_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                name=f'Retrain {model_name}',
                replace_existing=False
            )

    def retrain_model(self, model_name: str, reason: str = "Scheduled retraining") -> RetrainingResult:
        """
        Retrain a specific model.

        Args:
            model_name: Name of the model to retrain
            reason: Reason for retraining

        Returns:
            RetrainingResult object
        """
        self.logger.info(f"Starting retraining for {model_name}: {reason}")

        start_time = datetime.now(IST)
        result = RetrainingResult(
            success=False,
            model_name=model_name,
            old_version="",
            new_version="",
            performance_improved=False,
            metrics_comparison={},
            timestamp=start_time
        )

        try:
            # Get current model version
            current_version = None
            if model_name in self.model_versions and self.model_versions[model_name]:
                current_version = self.model_versions[model_name][0]  # Most recent
                result.old_version = current_version.version_id

            # Backup current model if configured
            if self.config["model_versioning"]["backup_before_retrain"] and current_version:
                self._backup_current_model(model_name)

            # Load fresh data
            if not PREPROCESSING_AVAILABLE:
                raise Exception("Preprocessing pipeline not available")

            pipeline = DataPreprocessingPipeline()
            processed_data = pipeline.run_full_pipeline()

            if processed_data is None:
                raise Exception("Failed to process data")

            # Split data for training and testing
            from sklearn.model_selection import train_test_split

            feature_cols = [col for col in processed_data.columns if col not in ['Date', 'Returns']]
            target_col = 'Returns'

            X = processed_data[feature_cols]
            y = processed_data[target_col]

            # Remove any NaN values
            valid_idx = ~(X.isnull().any(axis=1) | y.isnull())
            X = X[valid_idx]
            y = y[valid_idx]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )

            # Retrain model based on type
            new_model = self._train_model_by_type(model_name, X_train, y_train)

            if new_model is None:
                raise Exception(f"Failed to train {model_name}")

            # Evaluate new model
            new_metrics = self.evaluate_model_performance(model_name, new_model, X_test, y_test)

            # Compare with current model
            performance_improved = True
            metrics_comparison = {"new": asdict(new_metrics)}

            if current_version and current_version.performance_metrics:
                old_metrics = current_version.performance_metrics
                metrics_comparison["old"] = old_metrics

                # Check if performance improved
                r2_improved = new_metrics.r2_score > old_metrics.get("r2_score", 0)
                mae_improved = new_metrics.mae < old_metrics.get("mae", float('inf'))

                performance_improved = r2_improved and mae_improved

            # Create new version
            version_id = self._create_version_id(model_name, new_metrics)
            version_file = f"{model_name}_{version_id}.joblib"
            version_path = MODEL_DIR / version_file

            # Save new model
            joblib.dump(new_model, version_path)

            # Create version record
            version = ModelVersion(
                model_name=model_name,
                version_id=version_id,
                timestamp=start_time,
                performance_metrics=asdict(new_metrics),
                file_path=str(version_path),
                is_active=True,
                retraining_reason=reason
            )

            # Update version tracking
            if model_name not in self.model_versions:
                self.model_versions[model_name] = []
            self.model_versions[model_name].insert(0, version)  # Add to front

            # Deactivate old version
            if current_version:
                current_version.is_active = False

            # Save version info
            self._save_version_info(version)

            # Update main model file
            main_model_path = MODEL_DIR / f"{model_name}.joblib"
            joblib.dump(new_model, main_model_path)

            # Cleanup old versions
            self._cleanup_old_versions(model_name)

            # Update result
            result.success = True
            result.new_version = version_id
            result.performance_improved = performance_improved
            result.metrics_comparison = metrics_comparison

            self.last_retraining[model_name] = start_time

            self.logger.info(f"Successfully retrained {model_name}: {version_id}")

        except Exception as e:
            result.error_message = str(e)
            self.logger.error(f"Failed to retrain {model_name}: {e}")

        # Log retraining result
        self._log_retraining_result(result)

        return result

    def _train_model_by_type(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """Train model based on its type."""
        try:
            if not ENSEMBLE_AVAILABLE:
                self.logger.error("Ensemble models not available")
                return None

            trainer = EnsembleModelTrainer()

            if model_name == "random_forest":
                return trainer.train_random_forest(X_train, y_train)
            elif model_name == "xgboost":
                return trainer.train_xgboost(X_train, y_train)
            elif model_name == "lightgbm":
                return trainer.train_lightgbm(X_train, y_train)
            elif model_name == "ensemble":
                # Train individual models first
                rf_model = trainer.train_random_forest(X_train, y_train)
                xgb_model = trainer.train_xgboost(X_train, y_train)
                lgb_model = trainer.train_lightgbm(X_train, y_train)

                return trainer.create_ensemble([rf_model, xgb_model, lgb_model], X_train, y_train)
            elif model_name == "ols_static":
                # For OLS, we can use basic sklearn
                from sklearn.linear_model import LinearRegression
                model = LinearRegression()
                model.fit(X_train, y_train)
                return model
            else:
                self.logger.error(f"Unknown model type: {model_name}")
                return None

        except Exception as e:
            self.logger.error(f"Failed to train {model_name}: {e}")
            return None

    def _create_version_id(self, model_name: str, metrics: PerformanceMetrics) -> str:
        """Create a unique version ID for the model."""
        timestamp = datetime.now(IST).strftime("%Y%m%d_%H%M%S")
        performance_score = metrics.r2_score

        version_format = self.config["model_versioning"]["version_naming_format"]
        version_id = version_format.format(
            model_name=model_name,
            timestamp=timestamp,
            performance_score=performance_score
        )

        return version_id

    def _save_version_info(self, version: ModelVersion):
        """Save version information to disk."""
        try:
            version_file = MODEL_VERSIONS_DIR / f"{version.version_id}.json"
            with open(version_file, 'w') as f:
                json.dump(asdict(version), f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save version info: {e}")

    def _backup_current_model(self, model_name: str):
        """Backup current model before retraining."""
        try:
            main_model_path = MODEL_DIR / f"{model_name}.joblib"
            backup_path = MODEL_DIR / f"{model_name}_backup_{datetime.now(IST).strftime('%Y%m%d_%H%M%S')}.joblib"

            if main_model_path.exists():
                import shutil
                shutil.copy2(main_model_path, backup_path)
                self.logger.info(f"Backed up {model_name} to {backup_path}")

        except Exception as e:
            self.logger.error(f"Failed to backup {model_name}: {e}")

    def _cleanup_old_versions(self, model_name: str):
        """Clean up old model versions."""
        if model_name not in self.model_versions:
            return

        max_versions = self.config["model_versioning"]["max_versions_to_keep"]
        versions = self.model_versions[model_name]

        if len(versions) > max_versions:
            # Keep only the most recent versions
            versions_to_remove = versions[max_versions:]

            for version in versions_to_remove:
                try:
                    # Remove version file
                    if os.path.exists(version.file_path):
                        os.remove(version.file_path)

                    # Remove version info file
                    version_info_file = MODEL_VERSIONS_DIR / f"{version.version_id}.json"
                    if version_info_file.exists():
                        version_info_file.unlink()

                    self.logger.info(f"Removed old version: {version.version_id}")

                except Exception as e:
                    self.logger.error(f"Failed to remove version {version.version_id}: {e}")

            # Update the list
            self.model_versions[model_name] = versions[:max_versions]

    def retrain_all_models(self):
        """Retrain all available models."""
        self.logger.info("Starting weekly retraining of all models")

        current_models = self._get_current_models()
        results = []

        for model_name in current_models.keys():
            result = self.retrain_model(model_name, "Weekly scheduled retraining")
            results.append(result)

        # Summary
        successful = sum(1 for r in results if r.success)
        improved = sum(1 for r in results if r.performance_improved)

        self.logger.info(f"Weekly retraining completed: {successful}/{len(results)} successful, {improved} improved")

        return results

    def _log_retraining_result(self, result: RetrainingResult):
        """Log retraining result to history file."""
        try:
            history_file = Path(RETRAINING_LOG_FILE)

            # Load existing history
            history = []
            if history_file.exists():
                with open(history_file, 'r') as f:
                    history = json.load(f)

            # Add new result
            history.append(asdict(result))

            # Keep only recent history (last 100 entries)
            history = history[-100:]

            # Save
            with open(history_file, 'w') as f:
                json.dump(history, f, indent=2, default=str)

        except Exception as e:
            self.logger.error(f"Failed to log retraining result: {e}")

    def get_model_history(self, model_name: str) -> List[ModelVersion]:
        """Get version history for a model."""
        return self.model_versions.get(model_name, [])

    def get_performance_history(self, model_name: str) -> List[PerformanceMetrics]:
        """Get performance history for a model."""
        return self.performance_history.get(model_name, [])

    def rollback_model(self, model_name: str, version_id: str) -> bool:
        """
        Rollback to a specific model version.

        Args:
            model_name: Name of the model
            version_id: Version ID to rollback to

        Returns:
            True if successful, False otherwise
        """
        try:
            if model_name not in self.model_versions:
                return False

            # Find the version
            target_version = None
            for version in self.model_versions[model_name]:
                if version.version_id == version_id:
                    target_version = version
                    break

            if not target_version:
                return False

            # Load the model
            model = joblib.load(target_version.file_path)

            # Update main model file
            main_model_path = MODEL_DIR / f"{model_name}.joblib"
            joblib.dump(model, main_model_path)

            # Update active status
            for version in self.model_versions[model_name]:
                version.is_active = (version.version_id == version_id)

            self.logger.info(f"Rolled back {model_name} to version {version_id}")

            return True

        except Exception as e:
            self.logger.error(f"Failed to rollback {model_name} to {version_id}: {e}")
            return False

    def get_retraining_status(self) -> Dict[str, Any]:
        """Get current retraining status and statistics."""
        status = {
            "scheduler_active": self.scheduler is not None and self.scheduler.running,
            "models_tracked": list(self.model_versions.keys()),
            "total_versions": sum(len(versions) for versions in self.model_versions.values()),
            "last_retraining": self.last_retraining,
            "performance_checks": {
                model: len(history) for model, history in self.performance_history.items()
            }
        }

        return status

    def manual_retrain(self, model_name: str, reason: str = "Manual retraining") -> RetrainingResult:
        """
        Manually trigger retraining for a specific model.

        Args:
            model_name: Name of the model to retrain
            reason: Reason for retraining

        Returns:
            RetrainingResult object
        """
        self.logger.info(f"Manual retraining requested for {model_name}: {reason}")
        return self.retrain_model(model_name, reason)

    def update_config(self, new_config: Dict[str, Any]):
        """
        Update configuration and restart scheduler if needed.

        Args:
            new_config: New configuration dictionary
        """
        self.config.update(new_config)
        self._save_config(self.config)

        # Restart scheduler with new config
        if self.scheduler:
            self.scheduler.shutdown()
            self._setup_scheduler()

        self.logger.info("Configuration updated")

    def shutdown(self):
        """Shutdown the scheduler and cleanup."""
        if self.scheduler:
            self.scheduler.shutdown()
            self.logger.info("Scheduler shutdown")

        self.logger.info("Model Retraining Scheduler shutdown")


# Global instance for easy access
_scheduler_instance = None

def get_scheduler() -> ModelRetrainingScheduler:
    """Get the global scheduler instance."""
    global _scheduler_instance
    if _scheduler_instance is None:
        _scheduler_instance = ModelRetrainingScheduler()
    return _scheduler_instance

def start_scheduler():
    """Start the retraining scheduler."""
    scheduler = get_scheduler()
    return scheduler

def stop_scheduler():
    """Stop the retraining scheduler."""
    global _scheduler_instance
    if _scheduler_instance:
        _scheduler_instance.shutdown()
        _scheduler_instance = None

if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Model Retraining Scheduler")
    parser.add_argument("--start", action="store_true", help="Start the scheduler")
    parser.add_argument("--stop", action="store_true", help="Stop the scheduler")
    parser.add_argument("--retrain", type=str, help="Manually retrain a specific model")
    parser.add_argument("--retrain-all", action="store_true", help="Retrain all models")
    parser.add_argument("--check-performance", action="store_true", help="Check performance degradation")
    parser.add_argument("--status", action="store_true", help="Show scheduler status")

    args = parser.parse_args()

    if args.start:
        scheduler = start_scheduler()
        print("Scheduler started. Press Ctrl+C to stop.")
        try:
            import time
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            stop_scheduler()
            print("Scheduler stopped.")

    elif args.stop:
        stop_scheduler()
        print("Scheduler stopped.")

    elif args.retrain:
        scheduler = get_scheduler()
        result = scheduler.manual_retrain(args.retrain, "Manual CLI retraining")
        print(f"Retraining result: {result.success}")
        if result.error_message:
            print(f"Error: {result.error_message}")

    elif args.retrain_all:
        scheduler = get_scheduler()
        results = scheduler.retrain_all_models()
        successful = sum(1 for r in results if r.success)
        print(f"Retrained {successful}/{len(results)} models successfully")

    elif args.check_performance:
        scheduler = get_scheduler()
        degradation = scheduler.check_performance_degradation()
        print("Performance degradation check:")
        for model, needs_retrain in degradation.items():
            print(f"  {model}: {'Needs retraining' if needs_retrain else 'OK'}")

    elif args.status:
        scheduler = get_scheduler()
        status = scheduler.get_retraining_status()
        print("Scheduler Status:")
        print(json.dumps(status, indent=2, default=str))

    else:
        parser.print_help()
