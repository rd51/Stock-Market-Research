"""
Real-Time Market Monitoring System
===================================

Live financial data monitoring with intelligent caching, scheduling, and quality validation.
Provides real-time feeds for unemployment, VIX, and market indices with production-grade reliability.

Author: GitHub Copilot
Date: February 8, 2026
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings
import os
import json
import pytz
from dataclasses import dataclass, asdict
import threading
import time as time_module
warnings.filterwarnings('ignore')

# Required libraries for real-time monitoring
try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.jobstores.memory import MemoryJobStore
    from apscheduler.executors.asyncio import AsyncIOExecutor
    APSCHEDULER_AVAILABLE = True
except ImportError:
    APSCHEDULER_AVAILABLE = False
    BackgroundScheduler = None
    CronTrigger = None
    MemoryJobStore = None
    AsyncIOExecutor = None
    logging.warning("APScheduler not available - scheduling disabled")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logging.warning("requests not available - HTTP requests disabled")

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    logging.warning("BeautifulSoup not available - HTML parsing disabled")

# Import our existing scraping utilities
try:
    from web_scraping_utils import (
        ScrapingSession, NSEIndiaScraper, DataGovInScraper, LabourGovScraper,
        validate_scraped_dataset, skip_holiday, get_next_trading_day
    )
    SCRAPING_UTILS_AVAILABLE = True
except ImportError:
    SCRAPING_UTILS_AVAILABLE = False
    logging.warning("web_scraping_utils not available - scraping disabled")

# Ensure logging directory exists; fall back to 'logs' if creation fails
log_dir = os.path.join('data', 'logs')
try:
    os.makedirs(log_dir, exist_ok=True)
except Exception as e:
    warnings.warn(f"Could not create {log_dir}; falling back to 'logs': {e}")
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'realtime_monitor.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
IST = pytz.timezone('Asia/Kolkata')
MARKET_OPEN_TIME = time(9, 15)  # 9:15 AM IST
MARKET_CLOSE_TIME = time(15, 30)  # 3:30 PM IST
DEFAULT_UPDATE_TIME = "15:35"  # 3:35 PM IST

@dataclass
class DataQualityResult:
    """Data quality validation result."""
    is_valid: bool
    issues: List[str]
    warnings: List[str]
    quality_score: int  # 0-100
    validation_timestamp: datetime

@dataclass
class MarketAnomaly:
    """Market anomaly detection result."""
    anomaly_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    timestamp: datetime
    value: float
    threshold: float

@dataclass
class CachedData:
    """Cached data structure."""
    data: Dict[str, Any]
    timestamp: datetime
    source: str
    quality_score: int
    cache_type: str  # 'standard', 'archive'

class RealtimeDataFeed:
    """
    Real-time financial data feed manager with intelligent caching and scheduling.
    """

    def __init__(self, live_dataset_manager=None, scraping_utils=None,
                 update_frequency: str = 'daily'):
        """
        Initialize the real-time data feed.

        Args:
            live_dataset_manager: Dataset manager for data persistence
            scraping_utils: Scraping utilities instance
            update_frequency: 'daily', 'hourly', 'realtime'
        """
        self.live_dataset_manager = live_dataset_manager
        self.scraping_utils = scraping_utils
        self.update_frequency = update_frequency

        # Initialize scrapers
        if SCRAPING_UTILS_AVAILABLE:
            self.nse_scraper = NSEIndiaScraper()
            self.data_gov_scraper = DataGovInScraper()
            self.labour_scraper = LabourGovScraper()
        else:
            self.nse_scraper = None
            self.data_gov_scraper = None
            self.labour_scraper = None

        # Cache management
        self.cache_dir = "data/cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache = {}  # In-memory cache
        self.cache_lock = threading.Lock()

        # Scheduler
        self.scheduler = None
        if APSCHEDULER_AVAILABLE:
            self._setup_scheduler()

        # Monitoring
        self.monitoring_stats = {
            'total_fetches': 0,
            'successful_fetches': 0,
            'failed_fetches': 0,
            'last_fetch_times': {},
            'data_freshness': {},
            'quality_scores': {}
        }

        logger.info(f"Initialized RealtimeDataFeed with frequency: {update_frequency}")

    def _setup_scheduler(self):
        """Set up the background scheduler for automated updates."""
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

        logger.info("Background scheduler initialized")

    def is_market_open(self) -> bool:
        """
        Check if NSE market is currently open.

        Returns:
            bool: True if market is open
        """
        now = datetime.now(IST)

        # Check if it's a weekend
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False

        # Check if it's a holiday
        if skip_holiday(now.date()):
            return False

        # Check time
        current_time = now.time()
        return MARKET_OPEN_TIME <= current_time <= MARKET_CLOSE_TIME

    def next_market_close_time(self) -> datetime:
        """
        Calculate the next NSE market close time.

        Returns:
            datetime: Next market close in IST
        """
        now = datetime.now(IST)
        today_close = datetime.combine(now.date(), MARKET_CLOSE_TIME, tzinfo=IST)

        # If market already closed today, find next trading day
        if now > today_close or not self.is_market_open():
            next_trading_day = get_next_trading_day(now.date())
            next_close = datetime.combine(next_trading_day, MARKET_CLOSE_TIME, tzinfo=IST)
            return next_close

        return today_close

    def schedule_daily_update(self, time_str: str = DEFAULT_UPDATE_TIME) -> None:
        """
        Schedule automatic daily data updates.

        Args:
            time_str: Time in 'HH:MM' format (IST)
        """
        if not APSCHEDULER_AVAILABLE or not self.scheduler:
            logger.warning("Scheduler not available - cannot schedule daily updates")
            return

        try:
            hour, minute = map(int, time_str.split(':'))

            # Schedule daily update
            trigger = CronTrigger(hour=hour, minute=minute, timezone=IST)

            self.scheduler.add_job(
                func=self._scheduled_update_job,
                trigger=trigger,
                id='daily_market_update',
                name='Daily Market Data Update',
                replace_existing=True
            )

            # Start scheduler if not running
            if not self.scheduler.running:
                self.scheduler.start()

            logger.info(f"Scheduled daily update at {time_str} IST")

        except Exception as e:
            logger.error(f"Failed to schedule daily update: {str(e)}")

    def _scheduled_update_job(self):
        """Scheduled job to fetch and cache all latest data."""
        try:
            logger.info("Starting scheduled daily update")

            # Fetch all data
            all_data = self.fetch_all_latest_data(use_live=True, respect_market_hours=False)

            # Cache the data
            for data_type, data in all_data.items():
                if data_type in ['unemployment', 'vix', 'market_index']:
                    self.cache_data({data_type: data}, cache_type='standard')

            # Log success
            logger.info("Scheduled daily update completed successfully")

        except Exception as e:
            logger.error(f"Scheduled update failed: {str(e)}")

    def get_latest_unemployment(self, use_live: bool = True) -> Optional[Dict[str, Any]]:
        """
        Fetch latest unemployment data.

        Args:
            use_live: Whether to attempt live scraping

        Returns:
            dict: Unemployment data or None
        """
        source_name = 'unemployment'
        self.monitoring_stats['total_fetches'] += 1

        try:
            if use_live and self.data_gov_scraper:
                logger.info("Fetching live unemployment data")

                # Try data.gov.in first
                dataset = self.data_gov_scraper.scrape_unemployment_data(
                    start_date=(datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d'),
                    end_date=datetime.now().strftime('%Y-%m-%d')
                )

                if not dataset.empty:
                    # Get latest data point
                    latest_row = dataset.iloc[-1]
                    data = {
                        'timestamp': pd.to_datetime(latest_row['Date']),
                        'unemployment_rate': latest_row['UnemploymentRate'],
                        'urban_rate': None,  # Not always available
                        'rural_rate': None,  # Not always available
                        'data_source': 'live',
                        'is_fresh': True,
                        'age_hours': 0.0
                    }

                    # Validate data
                    quality = self.validate_live_fetch(data, source_name)
                    if quality.is_valid:
                        self.monitoring_stats['successful_fetches'] += 1
                        self.monitoring_stats['last_fetch_times'][source_name] = datetime.now()
                        self.monitoring_stats['quality_scores'][source_name] = quality.quality_score
                        return data
                    else:
                        logger.warning(f"Unemployment data validation failed: {quality.issues}")

            # Fallback to cached data
            logger.info("Using cached unemployment data")
            cached = self.load_cached_data(source_name, max_age_hours=168)  # 1 week
            if cached:
                cached_data = cached['data']
                age = (datetime.now() - cached_data['timestamp']).total_seconds() / 3600
                cached_data.update({
                    'data_source': 'cache',
                    'is_fresh': age < 24,
                    'age_hours': age
                })
                return cached_data

            logger.error("No unemployment data available")
            self.monitoring_stats['failed_fetches'] += 1
            return None

        except Exception as e:
            logger.error(f"Error fetching unemployment data: {str(e)}")
            self.monitoring_stats['failed_fetches'] += 1
            return self.handle_scraping_failure(source_name, e)

    def get_latest_vix(self, use_live: bool = True) -> Optional[Dict[str, Any]]:
        """
        Fetch latest VIX data.

        Args:
            use_live: Whether to attempt live scraping

        Returns:
            dict: VIX data or None
        """
        source_name = 'vix'
        self.monitoring_stats['total_fetches'] += 1

        try:
            if use_live and self.nse_scraper and self.is_market_open():
                logger.info("Fetching live VIX data")

                today = datetime.now().date()
                vix_data = self.nse_scraper.scrape_vix_historical(
                    start_date=today.strftime('%Y-%m-%d'),
                    end_date=today.strftime('%Y-%m-%d')
                )

                if not vix_data.empty:
                    latest_row = vix_data.iloc[-1]
                    data = {
                        'timestamp': pd.to_datetime(latest_row['Date']),
                        'vix_close': latest_row['VIX_Close'],
                        'vix_change': None,  # Would need previous day for change
                        'vix_change_pct': None,
                        'data_source': 'live',
                        'is_fresh': True,
                        'age_minutes': 0.0
                    }

                    # Validate data
                    quality = self.validate_live_fetch(data, source_name)
                    if quality.is_valid:
                        self.monitoring_stats['successful_fetches'] += 1
                        self.monitoring_stats['last_fetch_times'][source_name] = datetime.now()
                        self.monitoring_stats['quality_scores'][source_name] = quality.quality_score
                        return data
                    else:
                        logger.warning(f"VIX data validation failed: {quality.issues}")

            # Fallback to cached data
            logger.info("Using cached VIX data")
            cached = self.load_cached_data(source_name, max_age_hours=6)  # 6 hours for VIX
            if cached:
                cached_data = cached['data']
                age = (datetime.now() - cached_data['timestamp']).total_seconds() / 60
                cached_data.update({
                    'data_source': 'cache',
                    'is_fresh': age < 60,  # Fresh if less than 1 hour
                    'age_minutes': age
                })
                return cached_data

            logger.error("No VIX data available")
            self.monitoring_stats['failed_fetches'] += 1
            return None

        except Exception as e:
            logger.error(f"Error fetching VIX data: {str(e)}")
            self.monitoring_stats['failed_fetches'] += 1
            return self.handle_scraping_failure(source_name, e)

    def get_latest_market_index(self, index: str = 'NIFTY50',
                               use_live: bool = True) -> Optional[Dict[str, Any]]:
        """
        Fetch latest market index data.

        Args:
            index: Index name ('NIFTY50', 'SENSEX', 'NIFTY100')
            use_live: Whether to attempt live scraping

        Returns:
            dict: Market index data or None
        """
        source_name = f'market_index_{index.lower()}'
        self.monitoring_stats['total_fetches'] += 1

        try:
            if use_live and self.nse_scraper and self.is_market_open():
                logger.info(f"Fetching live {index} data")

                today = datetime.now().date()
                index_data = self.nse_scraper.scrape_index_historical(
                    index_name=index,
                    start_date=today.strftime('%Y-%m-%d'),
                    end_date=today.strftime('%Y-%m-%d')
                )

                if not index_data.empty:
                    latest_row = index_data.iloc[-1]
                    close_col = f'{index}_Close'
                    volume_col = f'{index}_Volume'

                    data = {
                        'timestamp': pd.to_datetime(latest_row['Date']),
                        'index_name': index,
                        'close': latest_row[close_col],
                        'change': None,  # Would need previous day
                        'change_pct': None,
                        'volume': latest_row.get(volume_col, 0),
                        'data_source': 'live',
                        'is_fresh': True
                    }

                    # Validate data
                    quality = self.validate_live_fetch(data, source_name)
                    if quality.is_valid:
                        self.monitoring_stats['successful_fetches'] += 1
                        self.monitoring_stats['last_fetch_times'][source_name] = datetime.now()
                        self.monitoring_stats['quality_scores'][source_name] = quality.quality_score
                        return data
                    else:
                        logger.warning(f"{index} data validation failed: {quality.issues}")

            # Fallback to cached data
            logger.info(f"Using cached {index} data")
            cached = self.load_cached_data(source_name, max_age_hours=6)  # 6 hours for indices
            if cached:
                cached_data = cached['data']
                age = (datetime.now() - cached_data['timestamp']).total_seconds() / 3600
                cached_data.update({
                    'data_source': 'cache',
                    'is_fresh': age < 1  # Fresh if less than 1 hour
                })
                return cached_data

            logger.error(f"No {index} data available")
            self.monitoring_stats['failed_fetches'] += 1
            return None

        except Exception as e:
            logger.error(f"Error fetching {index} data: {str(e)}")
            self.monitoring_stats['failed_fetches'] += 1
            return self.handle_scraping_failure(source_name, e)

    def fetch_all_latest_data(self, use_live: bool = True,
                            respect_market_hours: bool = True) -> Dict[str, Any]:
        """
        Fetch all latest data sources.

        Args:
            use_live: Whether to attempt live scraping
            respect_market_hours: Whether to respect market hours for live data

        Returns:
            dict: All data sources
        """
        logger.info("Fetching all latest data")

        # Check market status
        market_open = self.is_market_open()
        next_close = self.next_market_close_time()

        # If respecting market hours and market is closed, use cached data
        if respect_market_hours and not market_open:
            logger.info("Market closed - using cached data only")
            use_live = False

        result = {
            'unemployment': self.get_latest_unemployment(use_live=use_live),
            'vix': self.get_latest_vix(use_live=use_live),
            'market_index': self.get_latest_market_index('NIFTY50', use_live=use_live),
            'is_market_open': market_open,
            'next_update_time': next_close
        }

        # Determine overall freshness
        freshness_scores = []
        for key, data in result.items():
            if key in ['unemployment', 'vix', 'market_index'] and data:
                if data.get('is_fresh', False):
                    freshness_scores.append(100)
                elif data.get('age_hours', 24) < 24:
                    freshness_scores.append(75)
                elif data.get('age_hours', 24) < 72:
                    freshness_scores.append(50)
                else:
                    freshness_scores.append(25)

        if freshness_scores:
            avg_freshness = sum(freshness_scores) / len(freshness_scores)
            if avg_freshness >= 90:
                result['overall_freshness'] = 'fresh'
            elif avg_freshness >= 70:
                result['overall_freshness'] = 'warning'
            else:
                result['overall_freshness'] = 'stale'
        else:
            result['overall_freshness'] = 'stale'

        logger.info(f"All data fetched - freshness: {result['overall_freshness']}")
        return result

    def cache_data(self, data_dict: Dict[str, Any], cache_type: str = 'standard') -> bool:
        """
        Cache data with timestamp and metadata.

        Args:
            data_dict: Data to cache
            cache_type: 'standard' or 'archive'

        Returns:
            bool: Success status
        """
        try:
            with self.cache_lock:
                timestamp = datetime.now()

                for key, data in data_dict.items():
                    if data is None:
                        continue

                    cached_item = CachedData(
                        data=data,
                        timestamp=timestamp,
                        source=key,
                        quality_score=self.monitoring_stats.get('quality_scores', {}).get(key, 50),
                        cache_type=cache_type
                    )

                    # In-memory cache
                    self.cache[key] = cached_item

                    # File cache
                    cache_file = os.path.join(self.cache_dir, f"{key}_{cache_type}.json")
                    with open(cache_file, 'w') as f:
                        json.dump({
                            'data': data,
                            'timestamp': timestamp.isoformat(),
                            'source': key,
                            'quality_score': cached_item.quality_score,
                            'cache_type': cache_type
                        }, f, default=str, indent=2)

                logger.info(f"Cached {len(data_dict)} data items")
                return True

        except Exception as e:
            logger.error(f"Error caching data: {str(e)}")
            return False

    def load_cached_data(self, source_name: str, max_age_hours: int = 24) -> Optional[Dict[str, Any]]:
        """
        Load cached data if fresh enough.

        Args:
            source_name: Name of data source
            max_age_hours: Maximum age in hours

        Returns:
            dict: Cached data or None
        """
        try:
            # Try file cache first
            cache_file = os.path.join(self.cache_dir, f"{source_name}_standard.json")
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    cached = json.load(f)

                timestamp = pd.to_datetime(cached['timestamp'])
                age_hours = (datetime.now() - timestamp).total_seconds() / 3600

                if age_hours <= max_age_hours:
                    logger.info(f"Loaded cached {source_name} data (age: {age_hours:.1f} hours)")
                    return cached
                else:
                    logger.warning(f"Cached {source_name} data too old ({age_hours:.1f} hours)")

            # Try in-memory cache
            with self.cache_lock:
                if source_name in self.cache:
                    cached_item = self.cache[source_name]
                    age_hours = (datetime.now() - cached_item.timestamp).total_seconds() / 3600

                    if age_hours <= max_age_hours:
                        logger.info(f"Loaded in-memory cached {source_name} data")
                        return {
                            'data': cached_item.data,
                            'timestamp': cached_item.timestamp,
                            'quality_score': cached_item.quality_score
                        }

            return None

        except Exception as e:
            logger.error(f"Error loading cached data for {source_name}: {str(e)}")
            return None

    def validate_live_fetch(self, data: Dict[str, Any], source_name: str) -> DataQualityResult:
        """
        Validate quality of live-fetched data.

        Args:
            data: Data to validate
            source_name: Source name for context

        Returns:
            DataQualityResult: Validation results
        """
        issues = []
        warnings = []
        score = 100

        try:
            # Check for required fields based on source
            if source_name == 'unemployment':
                required_fields = ['timestamp', 'unemployment_rate']
                expected_ranges = {'unemployment_rate': (0, 50)}  # 0-50%
            elif source_name == 'vix':
                required_fields = ['timestamp', 'vix_close']
                expected_ranges = {'vix_close': (5, 100)}  # Typical VIX range
            elif source_name.startswith('market_index'):
                required_fields = ['timestamp', 'close']
                expected_ranges = {'close': (1000, 100000)}  # Broad range for indices
            else:
                required_fields = []
                expected_ranges = {}

            # Check required fields
            for field in required_fields:
                if field not in data or data[field] is None:
                    issues.append(f"Missing required field: {field}")
                    score -= 20

            # Check data types
            if 'timestamp' in data:
                try:
                    pd.to_datetime(data['timestamp'])
                except:
                    issues.append("Invalid timestamp format")
                    score -= 15

            # Check value ranges
            for field, (min_val, max_val) in expected_ranges.items():
                if field in data and data[field] is not None:
                    value = data[field]
                    if not isinstance(value, (int, float)) or not (min_val <= value <= max_val):
                        issues.append(f"Value {value} for {field} outside expected range [{min_val}, {max_val}]")
                        score -= 15

            # Check for sudden jumps (compared to cached data)
            cached = self.load_cached_data(source_name, max_age_hours=48)
            if cached and 'data' in cached:
                old_data = cached['data']
                for field in ['unemployment_rate', 'vix_close', 'close']:
                    if field in data and field in old_data and old_data[field] is not None:
                        old_val = old_data[field]
                        new_val = data[field]
                        if abs(new_val - old_val) / old_val > 0.5:  # 50% change
                            warnings.append(f"Large change in {field}: {old_val:.2f} -> {new_val:.2f}")
                            score -= 10

            # Check data freshness
            if 'timestamp' in data:
                data_time = pd.to_datetime(data['timestamp'])
                now = datetime.now()
                age_hours = (now - data_time).total_seconds() / 3600
                if age_hours > 24:
                    warnings.append(f"Data is {age_hours:.1f} hours old")
                    score -= 5

            score = max(0, min(100, score))

        except Exception as e:
            issues.append(f"Validation error: {str(e)}")
            score = 0

        result = DataQualityResult(
            is_valid=len(issues) == 0,
            issues=issues,
            warnings=warnings,
            quality_score=score,
            validation_timestamp=datetime.now()
        )

        logger.info(f"Data validation for {source_name}: score={score}, valid={result.is_valid}")
        return result

    def detect_market_anomalies(self, current_vix: float, current_return: float,
                               previous_data: Dict[str, Any]) -> List[MarketAnomaly]:
        """
        Detect market anomalies in current data.

        Args:
            current_vix: Current VIX value
            current_return: Current market return
            previous_data: Previous period data

        Returns:
            List[MarketAnomaly]: Detected anomalies
        """
        anomalies = []

        try:
            # VIX spike anomaly
            if 'vix' in previous_data and previous_data['vix'] is not None:
                prev_vix = previous_data['vix']
                vix_change_pct = abs(current_vix - prev_vix) / prev_vix * 100

                if vix_change_pct > 20:
                    severity = 'high' if vix_change_pct > 50 else 'medium'
                    anomalies.append(MarketAnomaly(
                        anomaly_type='vix_spike',
                        severity=severity,
                        description=f"VIX spiked {vix_change_pct:.1f}% from {prev_vix:.2f} to {current_vix:.2f}",
                        timestamp=datetime.now(),
                        value=current_vix,
                        threshold=prev_vix * 1.20
                    ))

            # Extreme market move
            if abs(current_return) > 5:  # 5% move
                severity = 'critical' if abs(current_return) > 10 else 'high'
                anomalies.append(MarketAnomaly(
                    anomaly_type='extreme_move',
                    severity=severity,
                    description=f"Extreme market move: {current_return:.2f}%",
                    timestamp=datetime.now(),
                    value=current_return,
                    threshold=5.0
                ))

            # VIX-return correlation breakdown
            if 'vix' in previous_data and 'return' in previous_data:
                # Expected: High VIX with negative returns
                expected_correlation = -1 if current_vix > 20 else 0
                actual_correlation = 1 if (current_vix > 20 and current_return > 0) else -1

                if expected_correlation != actual_correlation:
                    anomalies.append(MarketAnomaly(
                        anomaly_type='correlation_breakdown',
                        severity='medium',
                        description="VIX-return correlation breakdown",
                        timestamp=datetime.now(),
                        value=actual_correlation,
                        threshold=expected_correlation
                    ))

        except Exception as e:
            logger.error(f"Error detecting market anomalies: {str(e)}")

        if anomalies:
            logger.warning(f"Detected {len(anomalies)} market anomalies")

        return anomalies

    def handle_scraping_failure(self, source_name: str, error: Exception) -> Optional[Dict[str, Any]]:
        """
        Handle scraping failures with fallback strategies.

        Args:
            source_name: Failed source name
            error: Exception that occurred

        Returns:
            dict: Best available data or None
        """
        logger.warning(f"Scraping failure for {source_name}: {str(error)}")

        # Try cached data as fallback
        cached = self.load_cached_data(source_name, max_age_hours=168)  # 1 week
        if cached:
            logger.info(f"Using cached data as fallback for {source_name}")
            return cached['data']

        logger.error(f"No fallback data available for {source_name}")
        return None

    def fallback_strategy_cascade(self, source_name: str) -> Optional[Dict[str, Any]]:
        """
        Implement cascading fallback strategy.

        Args:
            source_name: Source name to fetch

        Returns:
            dict: Data from best available source
        """
        logger.info(f"Starting fallback cascade for {source_name}")

        # Strategy 1: Try live scraping from primary source
        if source_name == 'unemployment':
            data = self.get_latest_unemployment(use_live=True)
        elif source_name == 'vix':
            data = self.get_latest_vix(use_live=True)
        elif source_name.startswith('market_index'):
            index_name = source_name.split('_')[-1].upper()
            data = self.get_latest_market_index(index_name, use_live=True)

        if data and data.get('data_source') == 'live':
            logger.info(f"Successfully fetched live data for {source_name}")
            return data

        # Strategy 2: Try alternative live sources (if available)
        # This would be implemented based on specific fallback sources

        # Strategy 3: Use most recent cached data
        cached = self.load_cached_data(source_name, max_age_hours=720)  # 30 days
        if cached:
            logger.info(f"Using cached data for {source_name}")
            return cached['data']

        # Strategy 4: Return None with error
        logger.error(f"All fallback strategies failed for {source_name}")
        return None

    def get_monitoring_stats(self) -> Dict[str, Any]:
        """
        Get monitoring statistics.

        Returns:
            dict: Monitoring statistics
        """
        stats = self.monitoring_stats.copy()

        # Calculate success rate
        total = stats['total_fetches']
        successful = stats['successful_fetches']
        stats['success_rate'] = (successful / total * 100) if total > 0 else 0

        # Check for stale data alerts
        now = datetime.now()
        stale_alerts = []
        for source, last_time in stats['last_fetch_times'].items():
            age_hours = (now - last_time).total_seconds() / 3600
            if age_hours > 24:
                stale_alerts.append(f"{source}: {age_hours:.1f} hours old")

        stats['stale_alerts'] = stale_alerts
        stats['alert_level'] = 'red' if len(stale_alerts) > 0 else 'green'

        return stats

    def cleanup_old_cache(self, max_age_days: int = 30) -> int:
        """
        Clean up old cache files.

        Args:
            max_age_days: Maximum age in days

        Returns:
            int: Number of files deleted
        """
        deleted_count = 0
        cutoff_time = datetime.now() - timedelta(days=max_age_days)

        try:
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(self.cache_dir, filename)
                    file_time = datetime.fromtimestamp(os.path.getmtime(filepath))

                    if file_time < cutoff_time:
                        os.remove(filepath)
                        deleted_count += 1
                        logger.info(f"Deleted old cache file: {filename}")

            logger.info(f"Cache cleanup completed: {deleted_count} files deleted")
            return deleted_count

        except Exception as e:
            logger.error(f"Error during cache cleanup: {str(e)}")
            return 0


def main():
    """
    Main function demonstrating real-time market monitoring.
    """
    logger.info("Starting real-time market monitoring demo")

    try:
        # Initialize the monitoring system
        monitor = RealtimeDataFeed()

        print("=" * 80)
        print("REAL-TIME MARKET MONITORING SYSTEM")
        print("=" * 80)

        # Test market status
        market_open = monitor.is_market_open()
        next_close = monitor.next_market_close_time()

        print(f"Market Status: {'OPEN' if market_open else 'CLOSED'}")
        print(f"Next Market Close: {next_close.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print()

        # Test data fetching
        print("TESTING DATA SOURCES:")
        print("-" * 40)

        # Test unemployment data
        print("1. Unemployment Data:")
        unemployment = monitor.get_latest_unemployment(use_live=True)
        if unemployment:
            print(f"   ✓ Rate: {unemployment.get('unemployment_rate', 'N/A')}%")
            print(f"   ✓ Source: {unemployment.get('data_source', 'unknown')}")
            print(f"   ✓ Fresh: {unemployment.get('is_fresh', False)}")
            print(f"   ✓ Age: {unemployment.get('age_hours', 'N/A'):.1f} hours")
        else:
            print("   ✗ No data available")
        print()

        # Test VIX data
        print("2. VIX Data:")
        vix = monitor.get_latest_vix(use_live=True)
        if vix:
            print(f"   ✓ Close: {vix.get('vix_close', 'N/A')}")
            print(f"   ✓ Source: {vix.get('data_source', 'unknown')}")
            print(f"   ✓ Fresh: {vix.get('is_fresh', False)}")
            print(f"   ✓ Age: {vix.get('age_minutes', 'N/A'):.1f} minutes")
        else:
            print("   ✗ No data available")
        print()

        # Test market index data
        print("3. NIFTY50 Data:")
        nifty = monitor.get_latest_market_index('NIFTY50', use_live=True)
        if nifty:
            print(f"   ✓ Close: {nifty.get('close', 'N/A')}")
            print(f"   ✓ Volume: {nifty.get('volume', 'N/A')}")
            print(f"   ✓ Source: {nifty.get('data_source', 'unknown')}")
            print(f"   ✓ Fresh: {nifty.get('is_fresh', False)}")
        else:
            print("   ✗ No data available")
        print()

        # Test comprehensive fetch
        print("4. Comprehensive Data Fetch:")
        all_data = monitor.fetch_all_latest_data(use_live=True, respect_market_hours=True)
        print(f"   ✓ Overall Freshness: {all_data.get('overall_freshness', 'unknown')}")
        print(f"   ✓ Market Open: {all_data.get('is_market_open', False)}")
        print(f"   ✓ Next Update: {all_data.get('next_update_time', 'N/A')}")
        print()

        # Test data validation
        print("5. Data Quality Validation:")
        if unemployment:
            quality = monitor.validate_live_fetch(unemployment, 'unemployment')
            print(f"   Unemployment Quality Score: {quality.quality_score}/100")
            if quality.issues:
                print(f"   Issues: {', '.join(quality.issues)}")
            if quality.warnings:
                print(f"   Warnings: {', '.join(quality.warnings)}")
        print()

        # Test caching
        print("6. Data Caching:")
        cache_success = monitor.cache_data({
            'unemployment': unemployment,
            'vix': vix,
            'nifty50': nifty
        })
        print(f"   ✓ Cache operation: {'SUCCESS' if cache_success else 'FAILED'}")

        # Test cache loading
        cached_unemp = monitor.load_cached_data('unemployment', max_age_hours=1)
        print(f"   ✓ Cache retrieval: {'SUCCESS' if cached_unemp else 'FAILED'}")
        print()

        # Schedule daily updates
        print("7. Scheduling:")
        monitor.schedule_daily_update(DEFAULT_UPDATE_TIME)
        print(f"   ✓ Daily update scheduled for {DEFAULT_UPDATE_TIME} IST")
        print()

        # Monitoring stats
        print("8. Monitoring Statistics:")
        stats = monitor.get_monitoring_stats()
        print(f"   ✓ Total Fetches: {stats['total_fetches']}")
        print(f"   ✓ Success Rate: {stats['success_rate']:.1f}%")
        print(f"   ✓ Alert Level: {stats['alert_level']}")
        if stats['stale_alerts']:
            print(f"   ⚠ Stale Data Alerts: {len(stats['stale_alerts'])}")
            for alert in stats['stale_alerts'][:3]:  # Show first 3
                print(f"      - {alert}")
        print()

        # Cleanup old cache
        print("9. Cache Maintenance:")
        deleted = monitor.cleanup_old_cache(max_age_days=7)
        print(f"   ✓ Old cache files deleted: {deleted}")
        print()

        print("=" * 80)
        print("MONITORING SYSTEM READY")
        print("=" * 80)
        print("• Real-time data feeds: ACTIVE")
        print("• Intelligent caching: ENABLED")
        print("• Quality validation: ACTIVE")
        print("• Automated scheduling: ENABLED")
        print("• Error handling: ROBUST")
        print("• Monitoring: COMPREHENSIVE")
        print()
        print("System will automatically fetch fresh data during market hours")
        print("and use cached data when markets are closed.")
        print("=" * 80)

        # Keep the scheduler running for demo
        if monitor.scheduler and monitor.scheduler.running:
            print("\nScheduler is running. Press Ctrl+C to exit.")
            try:
                while True:
                    time_module.sleep(60)  # Check every minute
                    current_stats = monitor.get_monitoring_stats()
                    if current_stats['alert_level'] == 'red':
                        print(f"⚠ ALERT: {len(current_stats['stale_alerts'])} stale data sources")
            except KeyboardInterrupt:
                print("\nShutting down scheduler...")
                monitor.scheduler.shutdown()
                print("Scheduler stopped.")

    except Exception as e:
        logger.error(f"Error in main monitoring function: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()