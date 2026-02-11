"""
Web Scraping Utilities
======================

Reusable utilities for intelligent web scraping operations.
Optimized for pure web scraping data collection with robust error handling.

Author: GitHub Copilot
Date: February 8, 2026

FIXES APPLIED
-------------
BUG-01  Missing `import logging` — entire module would fail to load.
BUG-02  `from logging import Logger` removed; wrong import used to call Logger.warning() as
        a class method before the module-level logger was defined.
BUG-03  `logger` used inside optional-import except blocks before it was created; moved
        logger creation to the top of the module, before the optional imports.
BUG-04  `from datetime import datetime, timedelta` was missing `date`; type annotations
        `datetime.date` raised AttributeError at runtime because `datetime` is the *class*
        here, not the module. Added `date` to the import and replaced all `datetime.date`
        annotations with the plain `date` type.
BUG-05  Missing `import io` — `pd.io.common.StringIO` (deprecated path) replaced with
        `io.StringIO`.
BUG-06  `_calculate_backoff_delay` had an arbitrary 27-second cap that was inconsistent
        with the module-level `calculate_error_backoff_delay` (300 s cap). Aligned both to
        60 s and updated the docstring.
BUG-07  `validate_scraped_dataset` chronological-order check only accepted ascending order;
        valid descending-order datasets (newest-first) were incorrectly flagged. Now accepts
        both orderings.
BUG-08  `InvestingComScraper._extract_historical_data_table` used `BeautifulSoup` directly
        without checking `BS4_AVAILABLE`, causing a `NameError` when bs4 is absent. Added
        the guard and an early return.
BUG-09  `scrape_vix_historical` and `scrape_index_historical` accessed
        `checkpoint['last_date']` with a hard key lookup that raised `KeyError` when the key
        was absent. Replaced with `.get()` and a safe fallback.
BUG-10  `skip_holiday` contained a fabricated "last Monday of May" holiday rule that does
        not correspond to any real NSE holiday. Removed the bogus block.
"""

# ---------------------------------------------------------------------------
# Standard-library imports
# ---------------------------------------------------------------------------
import io          # FIX BUG-05: needed for io.StringIO
import logging     # FIX BUG-01: was completely missing
import os
import json
import random
import time
from datetime import datetime, timedelta, date  # FIX BUG-04: added `date`
from typing import Optional, List, Dict, Any

# ---------------------------------------------------------------------------
# Third-party imports
# ---------------------------------------------------------------------------
import pandas as pd
import numpy as np
import requests

# ---------------------------------------------------------------------------
# Set up module-level logger BEFORE optional imports         FIX BUG-02/03
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports for web scraping
# ---------------------------------------------------------------------------
try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    logger.warning("BeautifulSoup not available - HTML parsing disabled")  # FIX BUG-02/03

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, NoSuchElementException
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    logger.warning("Selenium not available - browser automation disabled")  # FIX BUG-03

try:
    from webdriver_manager.chrome import ChromeDriverManager
    WEBDRIVER_MANAGER_AVAILABLE = True
except ImportError:
    WEBDRIVER_MANAGER_AVAILABLE = False
    logger.warning("webdriver-manager not available - manual driver setup required")  # FIX BUG-03

# ---------------------------------------------------------------------------
# Comprehensive user agent list (20+ agents for better rotation)
# ---------------------------------------------------------------------------
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/120.0.0.0',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.1; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
    'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:119.0) Gecko/20100101 Firefox/119.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
]


# ===========================================================================
# ScrapingSession
# ===========================================================================

class ScrapingSession:
    """
    Intelligent HTTP session manager for web scraping operations.

    Features:
    - User-Agent rotation (20+ modern browsers)
    - Configurable rate limiting
    - Exponential backoff retry logic
    - Comprehensive error logging
    - Timeout handling
    """

    def __init__(self, source_name: str, rate_limit_delay: float = 2.5, max_retries: int = 3):
        """
        Initialize the scraping session.

        Args:
            source_name (str): Name of the data source (for logging)
            rate_limit_delay (float): Delay between requests in seconds (default: 2.5)
            max_retries (int): Maximum number of retry attempts (default: 3)
        """
        self.source_name = source_name
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        self.session = requests.Session()

        self.request_count = 0
        self.last_request_time = 0

        logger.info(f"Initialized ScrapingSession for {source_name} "
                    f"(rate_limit={rate_limit_delay}s, max_retries={max_retries})")

    def _get_random_user_agent(self) -> str:
        """Get a random user agent from the pool."""
        return random.choice(USER_AGENTS)

    def _calculate_backoff_delay(self, attempt: int) -> float:
        """
        Calculate exponential backoff delay.
        Formula: base_delay * (2 ^ attempt) with max cap at 60 seconds.

        FIX BUG-06: Previous cap was 27 s (arbitrary); aligned with the module-level
        `calculate_error_backoff_delay` convention.  Both now cap at 60 s.

        Args:
            attempt (int): Current attempt number (0-based)

        Returns:
            float: Delay in seconds
        """
        base_delay = 1.0
        delay = base_delay * (2 ** attempt)
        max_delay = 60.0  # FIX BUG-06: was 27.0 (arbitrary, inconsistent)
        return min(delay, max_delay)

    def get_with_retry(self, url: str, timeout: int = 30) -> Optional[requests.Response]:
        """
        Perform HTTP GET request with intelligent retry logic.

        Args:
            url (str): URL to request
            timeout (int): Request timeout in seconds (default: 30)

        Returns:
            requests.Response or None: Response object on success, None on failure
        """
        self.request_count += 1

        # Apply rate limiting
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last
            logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)

        for attempt in range(self.max_retries + 1):
            try:
                user_agent = self._get_random_user_agent()
                headers = {'User-Agent': user_agent}

                logger.info(f"[{self.source_name}] Attempt {attempt + 1}/{self.max_retries + 1} "
                            f"for URL: {url}")

                response = self.session.get(
                    url,
                    headers=headers,
                    timeout=timeout,
                    allow_redirects=True
                )

                self.last_request_time = time.time()
                response.raise_for_status()

                logger.info(f"[{self.source_name}] Success: {response.status_code} "
                            f"({len(response.content)} bytes)")

                return response

            except requests.exceptions.RequestException as e:
                error_type = type(e).__name__
                logger.warning(f"[{self.source_name}] Attempt {attempt + 1} failed: "
                               f"{error_type} - {str(e)}")

                if attempt < self.max_retries:
                    backoff_delay = self._calculate_backoff_delay(attempt)
                    logger.info(f"[{self.source_name}] Retrying in {backoff_delay:.1f}s...")
                    time.sleep(backoff_delay)
                else:
                    logger.error(f"[{self.source_name}] All {self.max_retries + 1} attempts "
                                 f"failed for URL: {url}")

            except Exception as e:
                logger.error(f"[{self.source_name}] Unexpected error on attempt {attempt + 1}: "
                             f"{type(e).__name__} - {str(e)}")
                if attempt >= self.max_retries:
                    break

        return None

    def close(self):
        """Close the underlying session."""
        self.session.close()
        logger.info(f"Closed ScrapingSession for {self.source_name} "
                    f"(total requests: {self.request_count})")


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def quick_scrape(url: str, source_name: str = "quick_scrape",
                 rate_limit_delay: float = 2.5, max_retries: int = 3,
                 timeout: int = 30) -> Optional[requests.Response]:
    """
    Convenience function for one-off scraping operations.

    Args:
        url (str): URL to scrape
        source_name (str): Name for logging
        rate_limit_delay (float): Delay between requests
        max_retries (int): Maximum retries
        timeout (int): Request timeout

    Returns:
        requests.Response or None: Response on success, None on failure
    """
    session = ScrapingSession(source_name, rate_limit_delay, max_retries)
    try:
        return session.get_with_retry(url, timeout)
    finally:
        session.close()


# ===========================================================================
# NSEIndiaScraper
# ===========================================================================

class NSEIndiaScraper(ScrapingSession):
    """
    Specialized scraper for NSE India historical data.

    Provides methods to scrape VIX and index historical data.
    """

    def __init__(self, rate_limit_delay: float = 2.5, max_retries: int = 3):
        super().__init__("NSE India", rate_limit_delay, max_retries)
        self.checkpoint_dir = "checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _is_trading_day(self, target_date: date) -> bool:  # FIX BUG-04: param renamed + type
        """Check if the given date is a trading day (weekday)."""
        return target_date.weekday() < 5

    def _get_vix_url(self, target_date: date) -> str:  # FIX BUG-04
        date_str = target_date.strftime("%d%b%Y").upper()
        return (f"https://www.nseindia.com/api/historical/vixhistory"
                f"?from={date_str}&to={date_str}&csv=true")

    def _get_index_url(self, index_name: str, target_date: date) -> str:  # FIX BUG-04
        date_str = target_date.strftime("%d%b%Y").upper()
        symbol_map = {
            'NIFTY50': 'NIFTY 50',
            'SENSEX': 'SENSEX',
            'NIFTY100': 'NIFTY 100',
            'NIFTY500': 'NIFTY 500',
            'NIFTYBANK': 'NIFTY BANK'
        }
        symbol = symbol_map.get(index_name.upper(), index_name.upper())
        return (f"https://www.nseindia.com/api/historical/indicesHistory"
                f"?indexType={symbol}&from={date_str}&to={date_str}")

    def _parse_vix_csv(self, csv_content: str, target_date: date) -> Optional[Dict]:  # FIX BUG-04
        """Parse VIX CSV data and return OHLCV dict."""
        try:
            lines = csv_content.strip().split('\n')
            if len(lines) < 2:
                return None
            data_line = lines[1].strip()
            if not data_line:
                return None
            parts = data_line.split(',')
            if len(parts) < 6:
                return None
            return {
                'Date': target_date.strftime('%Y-%m-%d'),
                'VIX_Open': float(parts[2]),
                'VIX_High': float(parts[3]),
                'VIX_Low': float(parts[4]),
                'VIX_Close': float(parts[5]),
                'VIX_Volume': float(parts[6]) if len(parts) > 6 else 0
            }
        except (ValueError, IndexError) as e:
            logger.warning(f"Failed to parse VIX data for {target_date}: {e}")
            return None

    def _parse_index_json(self, json_content: str, target_date: date,  # FIX BUG-04
                          index_name: str) -> Optional[Dict]:
        """Parse index JSON data and return OHLCV dict."""
        try:
            data = json.loads(json_content)
            if 'data' not in data or not data['data']:
                return None
            record = data['data'][0]
            return {
                'Date': target_date.strftime('%Y-%m-%d'),
                f'{index_name}_Open': float(record.get('OPEN', 0)),
                f'{index_name}_High': float(record.get('HIGH', 0)),
                f'{index_name}_Low': float(record.get('LOW', 0)),
                f'{index_name}_Close': float(record.get('CLOSE', 0)),
                f'{index_name}_Volume': float(record.get('VOLUME', 0))
            }
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse {index_name} data for {target_date}: {e}")
            return None

    def _save_checkpoint(self, data_type: str, data: List[Dict], last_date: str):
        checkpoint = {
            'data_type': data_type,
            'data': data,
            'last_date': last_date,
            'timestamp': datetime.now().isoformat()
        }
        filename = f"{self.checkpoint_dir}/nse_{data_type.lower()}_checkpoint.json"
        with open(filename, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        logger.info(f"Checkpoint saved: {filename} ({len(data)} records)")

    def _load_checkpoint(self, data_type: str) -> Optional[Dict]:
        filename = f"{self.checkpoint_dir}/nse_{data_type.lower()}_checkpoint.json"
        if not os.path.exists(filename):
            return None
        try:
            with open(filename, 'r') as f:
                checkpoint = json.load(f)
            logger.info(f"Checkpoint loaded: {filename} "
                        f"({len(checkpoint.get('data', []))} records)")
            return checkpoint
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load checkpoint {filename}: {e}")
            return None

    def scrape_vix_historical(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Scrape VIX historical data from NSE India."""
        start = datetime.strptime(start_date, '%Y-%m-%d').date()
        end = datetime.strptime(end_date, '%Y-%m-%d').date()
        logger.info(f"Starting VIX scraping from {start_date} to {end_date}")

        checkpoint = self._load_checkpoint('vix')
        scraped_data = checkpoint.get('data', []) if checkpoint else []

        # FIX BUG-09: use .get() to avoid KeyError when 'last_date' is absent
        last_date_str = checkpoint.get('last_date') if checkpoint else None
        if last_date_str:
            last_processed = datetime.strptime(last_date_str, '%Y-%m-%d').date()
        else:
            last_processed = start - timedelta(days=1)

        current_date = max(start, last_processed + timedelta(days=1))

        while current_date <= end:
            if not self._is_trading_day(current_date):
                current_date += timedelta(days=1)
                continue

            logger.info(f"Scraping VIX for {current_date}")
            url = self._get_vix_url(current_date)
            response = self.get_with_retry(url)

            if response and response.status_code == 200:
                vix_data = self._parse_vix_csv(response.text, current_date)
                if vix_data:
                    scraped_data.append(vix_data)
                    logger.info(f"VIX data extracted for {current_date}")
                else:
                    logger.warning(f"No VIX data found for {current_date}")
            else:
                logger.error(f"Failed to fetch VIX data for {current_date}")

            self._save_checkpoint('vix', scraped_data, current_date.strftime('%Y-%m-%d'))
            current_date += timedelta(days=1)

        df = pd.DataFrame(scraped_data)
        logger.info(f"VIX scraping completed: {len(df)} records")
        return df

    def scrape_index_historical(self, index_name: str, start_date: str,
                                end_date: str) -> pd.DataFrame:
        """Scrape index historical data from NSE India."""
        start = datetime.strptime(start_date, '%Y-%m-%d').date()
        end = datetime.strptime(end_date, '%Y-%m-%d').date()
        logger.info(f"Starting {index_name} scraping from {start_date} to {end_date}")

        checkpoint_key = f'index_{index_name.lower()}'
        checkpoint = self._load_checkpoint(checkpoint_key)
        scraped_data = checkpoint.get('data', []) if checkpoint else []

        # FIX BUG-09: use .get() to avoid KeyError
        last_date_str = checkpoint.get('last_date') if checkpoint else None
        if last_date_str:
            last_processed = datetime.strptime(last_date_str, '%Y-%m-%d').date()
        else:
            last_processed = start - timedelta(days=1)

        current_date = max(start, last_processed + timedelta(days=1))

        while current_date <= end:
            if not self._is_trading_day(current_date):
                current_date += timedelta(days=1)
                continue

            logger.info(f"Scraping {index_name} for {current_date}")
            url = self._get_index_url(index_name, current_date)
            response = self.get_with_retry(url)

            if response and response.status_code == 200:
                index_data = self._parse_index_json(response.text, current_date, index_name)
                if index_data:
                    scraped_data.append(index_data)
                    logger.info(f"{index_name} data extracted for {current_date}")
                else:
                    logger.warning(f"No {index_name} data found for {current_date}")
            else:
                logger.error(f"Failed to fetch {index_name} data for {current_date}")

            self._save_checkpoint(checkpoint_key, scraped_data,
                                  current_date.strftime('%Y-%m-%d'))
            current_date += timedelta(days=1)

        df = pd.DataFrame(scraped_data)
        logger.info(f"{index_name} scraping completed: {len(df)} records")
        return df


# ===========================================================================
# DataGovInScraper
# ===========================================================================

class DataGovInScraper(ScrapingSession):
    """Specialized scraper for data.gov.in datasets."""

    def __init__(self, rate_limit_delay: float = 3.0, max_retries: int = 3):
        super().__init__("data.gov.in", rate_limit_delay, max_retries)
        self.base_url = "https://data.gov.in"
        self.api_base = "https://api.data.gov.in"

    def find_unemployment_dataset(self) -> Optional[Dict]:
        """Search data.gov.in for unemployment/PLFS datasets."""
        logger.info("Searching for unemployment/PLFS datasets on data.gov.in")
        search_url = (f"{self.api_base}/resource?api-key=&format=json&offset=0"
                      f"&limit=10&filters[keywords]=unemployment")

        response = self.get_with_retry(search_url)
        if not response:
            logger.error("Failed to search data.gov.in API")
            return None

        try:
            data = response.json()
            records = data.get('records', [])
            for record in records:
                title = record.get('title', '').lower()
                description = record.get('description', '').lower()
                if any(kw in title or kw in description
                       for kw in ['unemployment', 'plfs', 'labor force', 'employment']):
                    resource_url = record.get('access_url', '')
                    if not resource_url:
                        continue
                    dataset_info = {
                        'dataset_id': record.get('id', ''),
                        'download_url': resource_url,
                        'format': record.get('format', 'CSV').upper(),
                        'description': record.get('description', ''),
                        'title': record.get('title', ''),
                        'organization': record.get('organization', {}).get('title', '')
                    }
                    logger.info(f"Found unemployment dataset: {dataset_info['title']}")
                    return dataset_info

            logger.warning("No suitable unemployment dataset found")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse search response: {e}")
            return None

    def _parse_csv_data(self, csv_content: str, start_date: str,
                        end_date: str) -> pd.DataFrame:
        """Parse CSV unemployment data and filter by date range."""
        try:
            # FIX BUG-05: replaced deprecated pd.io.common.StringIO with io.StringIO
            df = pd.read_csv(io.StringIO(csv_content))
            df.columns = df.columns.str.strip()

            date_col = next(
                (c for c in df.columns
                 if any(k in c.lower() for k in ['date', 'month', 'year'])),
                None
            )
            if not date_col:
                logger.warning("No date column found in CSV data")
                return pd.DataFrame()

            unemployment_col = next(
                (c for c in df.columns
                 if any(k in c.lower() for k in ['unemployment', 'rate', 'percentage'])),
                None
            )
            if not unemployment_col:
                logger.warning("No unemployment rate column found")
                return pd.DataFrame()

            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df = df.dropna(subset=[date_col])

            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            df = df[(df[date_col] >= start) & (df[date_col] <= end)]

            result_df = df[[date_col, unemployment_col]].copy()
            result_df.columns = ['Date', 'UnemploymentRate']
            # Safely format dates without relying on Series.dt typing (avoids Pylance issues)
            result_df['Date'] = result_df['Date'].apply(
                lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) and hasattr(x, 'strftime') else str(x)
            )

            logger.info(f"Parsed {len(result_df)} unemployment records from CSV")
            return result_df
        except Exception as e:
            logger.error(f"Failed to parse CSV data: {e}")
            return pd.DataFrame()

    def _parse_html_table(self, html_content: str, start_date: str,
                          end_date: str) -> pd.DataFrame:
        """Parse HTML table unemployment data."""
        if not BS4_AVAILABLE:
            logger.error("BeautifulSoup not available - cannot parse HTML tables")
            return pd.DataFrame()

        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            tables = soup.find_all('table')
            if not tables:
                logger.warning("No tables found in HTML content")
                return pd.DataFrame()

            best_table = None
            max_relevant_cols = 0
            for table in tables:
                headers = [th.get_text().strip().lower() for th in table.find_all('th')]
                relevant_cols = sum(
                    1 for h in headers
                    if any(k in h for k in ['date', 'month', 'unemployment', 'rate'])
                )
                if relevant_cols > max_relevant_cols:
                    max_relevant_cols = relevant_cols
                    best_table = table

            if not best_table:
                best_table = tables[0]

            rows = []
            for tr in best_table.find_all('tr'):
                cells = [td.get_text().strip() for td in tr.find_all(['td', 'th'])]
                if cells:
                    rows.append(cells)

            if len(rows) < 2:
                logger.warning("Table has insufficient data")
                return pd.DataFrame()

            df = pd.DataFrame(rows[1:], columns=rows[0])
            df.columns = df.columns.str.strip()

            date_col = next(
                (c for c in df.columns if any(k in c.lower() for k in ['date', 'month'])),
                None
            )
            unemployment_col = next(
                (c for c in df.columns
                 if any(k in c.lower() for k in ['unemployment', 'rate'])),
                None
            )
            if not date_col or not unemployment_col:
                logger.warning("Required columns not found in HTML table")
                return pd.DataFrame()

            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df = df.dropna(subset=[date_col])

            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            df = df[(df[date_col] >= start) & (df[date_col] <= end)]

            result_df = df[[date_col, unemployment_col]].copy()
            result_df.columns = ['Date', 'UnemploymentRate']
            # Safely format dates without relying on Series.dt typing (avoids Pylance issues)
            result_df['Date'] = result_df['Date'].apply(
                lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) and hasattr(x, 'strftime') else str(x)
            )

            logger.info(f"Parsed {len(result_df)} unemployment records from HTML table")
            return result_df
        except Exception as e:
            logger.error(f"Failed to parse HTML table: {e}")
            return pd.DataFrame()

    def scrape_unemployment_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Scrape unemployment data from data.gov.in."""
        logger.info(f"Starting unemployment data scraping from {start_date} to {end_date}")
        dataset_info = self.find_unemployment_dataset()
        if not dataset_info:
            logger.error("No unemployment dataset found")
            return pd.DataFrame()

        download_url = dataset_info['download_url']
        data_format = dataset_info['format']
        logger.info(f"Downloading unemployment data from: {download_url} (format: {data_format})")

        response = self.get_with_retry(download_url)
        if not response:
            logger.error("Failed to download unemployment data")
            return pd.DataFrame()

        if data_format == 'CSV':
            df = self._parse_csv_data(response.text, start_date, end_date)
        elif data_format in ['XLS', 'XLSX']:
            logger.warning("Excel format not yet implemented, attempting CSV parsing")
            df = self._parse_csv_data(response.text, start_date, end_date)
        else:
            logger.info("Unknown format, attempting HTML table parsing")
            df = self._parse_html_table(response.text, start_date, end_date)

        if df.empty:
            logger.warning("No unemployment data extracted")
        else:
            logger.info(f"Successfully extracted {len(df)} unemployment records")
        return df


# ===========================================================================
# LabourGovScraper
# ===========================================================================

class LabourGovScraper(ScrapingSession):
    """Specialized scraper for labour.gov.in labour statistics."""

    def __init__(self, rate_limit_delay: float = 3.0, max_retries: int = 3):
        super().__init__("labour.gov.in", rate_limit_delay, max_retries)
        self.base_url = "https://labour.gov.in"

    def _find_labour_data_pages(self) -> List[str]:
        potential_pages = [
            "/sites/default/files/Employment_Unemployment_Statistics.pdf",
            "/employment-unemployment-statistics",
            "/monthly-employment-review",
            "/quarterly-employment-survey",
            "/annual-report",
            "/statistics"
        ]
        data_pages = []
        for page in potential_pages:
            url = f"{self.base_url}{page}"
            response = self.get_with_retry(url)
            if response and response.status_code == 200:
                if self._page_has_labour_data(response.text):
                    data_pages.append(url)
                    logger.info(f"Found labour data page: {url}")
        return data_pages

    def _page_has_labour_data(self, html_content: str) -> bool:
        if not BS4_AVAILABLE:
            logger.warning("BeautifulSoup not available - cannot check for labour data")
            return False
        soup = BeautifulSoup(html_content, 'html.parser')
        tables = soup.find_all('table')
        if not tables:
            return False
        labour_keywords = [
            'employment', 'unemployment', 'labour', 'workforce', 'job',
            'worker', 'wage', 'salary', 'industry', 'sector'
        ]
        for table in tables:
            text_content = table.get_text().lower()
            if any(kw in text_content for kw in labour_keywords):
                return True
        return False

    def _parse_labour_table(self, table_html: str) -> pd.DataFrame:
        if not BS4_AVAILABLE:
            logger.error("BeautifulSoup not available - cannot parse labour table")
            return pd.DataFrame()
        try:
            soup = BeautifulSoup(table_html, 'html.parser')
            rows = []
            headers = []
            header_row = soup.find('tr')
            if header_row:
                headers = [th.get_text().strip()
                           for th in header_row.find_all(['th', 'td'])]
            data_rows = soup.find_all('tr')[1:] if header_row else soup.find_all('tr')
            for row in data_rows:
                cells = [td.get_text().strip() for td in row.find_all(['td', 'th'])]
                if cells and len(cells) >= len(headers):
                    rows.append(cells)
            if not rows:
                return pd.DataFrame()
            df = pd.DataFrame(rows, columns=headers[:len(rows[0])] if headers else None)
            df.columns = df.columns.str.strip()
            logger.info(f"Parsed labour table with {len(df)} rows and {len(df.columns)} columns")
            return df
        except Exception as e:
            logger.error(f"Failed to parse labour table: {e}")
            return pd.DataFrame()

    def _extract_tables_from_page(self, html_content: str) -> List[pd.DataFrame]:
        if not BS4_AVAILABLE:
            logger.error("BeautifulSoup not available - cannot extract tables from page")
            return []
        soup = BeautifulSoup(html_content, 'html.parser')
        tables = soup.find_all('table')
        parsed_tables = []
        for i, table in enumerate(tables):
            df = self._parse_labour_table(str(table))
            if not df.empty:
                df['table_index'] = i
                parsed_tables.append(df)
        logger.info(f"Extracted {len(parsed_tables)} labour tables from page")
        return parsed_tables

    def _standardize_labour_data(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            std_df = df.copy()
            date_col = next(
                (c for c in std_df.columns
                 if any(k in c.lower() for k in ['date', 'month', 'year', 'period'])),
                None
            )
            employment_cols = [
                c for c in std_df.columns
                if any(k in c.lower() for k in ['employment', 'employed', 'jobs'])
            ]
            unemployment_cols = [
                c for c in std_df.columns
                if any(k in c.lower() for k in ['unemployment', 'unemployed'])
            ]
            for col in employment_cols + unemployment_cols:
                std_df[col] = pd.to_numeric(std_df[col], errors='coerce')
            if date_col:
                std_df[date_col] = pd.to_datetime(std_df[date_col], errors='coerce')
            rename_dict = {}
            if date_col:
                rename_dict[date_col] = 'Date'
            if employment_cols:
                rename_dict[employment_cols[0]] = 'Employment'
            if unemployment_cols:
                rename_dict[unemployment_cols[0]] = 'Unemployment'
            std_df = std_df.rename(columns=rename_dict)
            std_df = std_df.dropna(how='all')
            logger.info(f"Standardized labour data: {len(std_df)} rows")
            return std_df
        except Exception as e:
            logger.error(f"Failed to standardize labour data: {e}")
            return df

    def _merge_labour_tables(self, tables: List[pd.DataFrame]) -> pd.DataFrame:
        if not tables:
            return pd.DataFrame()
        if len(tables) == 1:
            return tables[0]
        all_columns = set()
        for df in tables:
            all_columns.update(df.columns)
        date_columns = [c for c in all_columns if 'date' in c.lower()]
        if date_columns:
            merge_col = date_columns[0]
            merged_df = tables[0]
            for df in tables[1:]:
                if merge_col in df.columns:
                    merged_df = pd.merge(merged_df, df, on=merge_col, how='outer',
                                         suffixes=('', '_dup'))
                    dup_cols = [c for c in merged_df.columns if c.endswith('_dup')]
                    merged_df = merged_df.drop(columns=dup_cols)
        else:
            merged_df = pd.concat(tables, ignore_index=True, sort=False)
        logger.info(f"Merged {len(tables)} tables into {len(merged_df)} rows")
        return merged_df

    def scrape_labour_statistics(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Scrape labour statistics from labour.gov.in."""
        logger.info(f"Starting labour statistics scraping from {start_date} to {end_date}")
        data_pages = self._find_labour_data_pages()
        if not data_pages:
            logger.warning("No labour data pages found")
            return pd.DataFrame()

        all_tables = []
        for url in data_pages:
            logger.info(f"Scraping labour data from: {url}")
            response = self.get_with_retry(url)
            if not response:
                logger.error(f"Failed to fetch labour data from {url}")
                continue
            page_tables = self._extract_tables_from_page(response.text)
            for table in page_tables:
                std_table = self._standardize_labour_data(table)
                if not std_table.empty:
                    all_tables.append(std_table)

        if not all_tables:
            logger.warning("No labour tables extracted")
            return pd.DataFrame()

        merged_df = self._merge_labour_tables(all_tables)

        if 'Date' in merged_df.columns and not merged_df.empty:
            try:
                merged_df['Date'] = pd.to_datetime(merged_df['Date'], errors='coerce')
                start = pd.to_datetime(start_date)
                end = pd.to_datetime(end_date)
                merged_df = merged_df[
                    (merged_df['Date'] >= start) & (merged_df['Date'] <= end)
                ]
                # Safely format dates without relying on Series.dt typing (avoids Pylance issues)
                merged_df['Date'] = merged_df['Date'].apply(
                    lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) and hasattr(x, 'strftime') else str(x)
                )
            except Exception as e:
                logger.warning(f"Could not filter by date range: {e}")

        merged_df = merged_df.dropna(how='all')
        logger.info(f"Labour statistics scraping completed: {len(merged_df)} records")
        return merged_df


# ===========================================================================
# InvestingComScraper
# ===========================================================================

class InvestingComScraper(ScrapingSession):
    """Specialized scraper for Investing.com data validation using Selenium."""

    def __init__(self, rate_limit_delay: float = 5.0, max_retries: int = 3):
        super().__init__("investing.com", rate_limit_delay, max_retries)
        self.base_url = "https://in.investing.com"
        self.driver = None

    def _setup_selenium_driver(self):
        if not SELENIUM_AVAILABLE:
            raise ImportError("Selenium not available - browser automation disabled")
        if not WEBDRIVER_MANAGER_AVAILABLE:
            raise ImportError("webdriver-manager not available - cannot setup Chrome driver")
        if self.driver:
            return

        options = Options()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--window-size=1920,1080')
        options.add_argument(
            '--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        )
        try:
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=options)
            self.driver.implicitly_wait(10)
            logger.info("Selenium Chrome WebDriver initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Selenium WebDriver: {e}")
            raise

    def _close_selenium_driver(self):
        if self.driver:
            try:
                self.driver.quit()
                self.driver = None
                logger.info("Selenium WebDriver closed")
            except Exception as e:
                logger.warning(f"Error closing WebDriver: {e}")

    def _wait_for_element(self, by, value, timeout=30):
        try:
            if self.driver is None:
                raise RuntimeError("WebDriver is not initialized.")
            return WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located((by, value))
            )
        except TimeoutException:
            logger.error(f"Timeout waiting for element: {by}={value}")
            return None

    def _extract_historical_data_table(self) -> pd.DataFrame:
        """
        Extract historical data from the Investing.com table.

        FIX BUG-08: Added BS4_AVAILABLE guard before calling BeautifulSoup.
        """
        # FIX BUG-08: guard missing in original
        if not BS4_AVAILABLE:
            logger.error("BeautifulSoup not available - cannot parse Investing.com table")
            return pd.DataFrame()

        try:
            table_element = self._wait_for_element(By.ID, "curr_table")
            if not table_element:
                logger.error("Historical data table not found")
                return pd.DataFrame()

            table_html = table_element.get_attribute('outerHTML')
            if table_html is None:
                raise ValueError("table_html cannot be None")
            soup = BeautifulSoup(table_html, 'html.parser')

            rows = []
            for row in soup.find_all('tr'):
                cells = row.find_all(['td', 'th'])
                if cells:
                    rows.append([cell.get_text().strip() for cell in cells])

            if len(rows) < 2:
                logger.warning("Insufficient data in historical table")
                return pd.DataFrame()

            df = pd.DataFrame(rows[1:], columns=rows[0])
            df.columns = df.columns.str.strip()

            column_mapping = {
                'Date': 'Date',
                'Price': 'NIFTY_Price',
                'Open': 'NIFTY_Open',
                'High': 'NIFTY_High',
                'Low': 'NIFTY_Low',
                'Vol.': 'NIFTY_Volume',
                'Change %': 'NIFTY_Change_Pct'
            }
            df = df.rename(columns=column_mapping)

            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                df = df.dropna(subset=['Date'])
                # Safely format dates without relying on Series.dt typing (avoids Pylance issues)
                df['Date'] = df['Date'].apply(
                    lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) and hasattr(x, 'strftime') else str(x)
                )

            for col in ['NIFTY_Price', 'NIFTY_Open', 'NIFTY_High', 'NIFTY_Low']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col].str.replace(',', ''), errors='coerce')

            if 'NIFTY_Volume' in df.columns:
                df['NIFTY_Volume'] = (df['NIFTY_Volume']
                                      .str.replace('K', '000')
                                      .str.replace('M', '000000'))
                df['NIFTY_Volume'] = pd.to_numeric(
                    df['NIFTY_Volume'].str.replace(',', ''), errors='coerce'
                )

            logger.info(f"Extracted {len(df)} historical NIFTY records from Investing.com")
            return df
        except Exception as e:
            logger.error(f"Failed to extract historical data table: {e}")
            return pd.DataFrame()

    def _set_date_range(self, start_date: str, end_date: str) -> bool:
        try:
            date_picker = self._wait_for_element(By.ID, "widgetFieldDateRange")
            if not date_picker:
                logger.error("Date range picker not found")
                return False
            date_picker.click()
            time.sleep(2)

            start_input = self._wait_for_element(By.ID, "startDate")
            if start_input:
                start_input.clear()
                start_input.send_keys(start_date)

            end_input = self._wait_for_element(By.ID, "endDate")
            if end_input:
                end_input.clear()
                end_input.send_keys(end_date)

            apply_button = self._wait_for_element(By.ID, "applyBtn")
            if apply_button:
                apply_button.click()
                time.sleep(5)
                return True
            else:
                logger.error("Apply button not found")
                return False
        except Exception as e:
            logger.error(f"Failed to set date range: {e}")
            return False

    def scrape_nifty_investing(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Scrape NIFTY historical data from Investing.com for validation."""
        logger.info(f"Starting NIFTY validation scraping from Investing.com: "
                    f"{start_date} to {end_date}")
        try:
            self._setup_selenium_driver()
            nifty_url = f"{self.base_url}/indices/nifty-50-historical-data"
            logger.info(f"Navigating to: {nifty_url}")
            if self.driver is None:
                raise RuntimeError("WebDriver is not initialized.")
            self.driver.get(nifty_url)
            time.sleep(5)

            if not self._set_date_range(start_date, end_date):
                logger.error("Failed to set date range on Investing.com")
                return pd.DataFrame()

            df = self._extract_historical_data_table()
            if df.empty:
                logger.warning("No NIFTY data extracted from Investing.com")
            else:
                logger.info(f"Successfully extracted {len(df)} NIFTY validation records")
            return df
        except Exception as e:
            logger.error(f"Error during NIFTY scraping from Investing.com: {e}")
            return pd.DataFrame()
        finally:
            self._close_selenium_driver()

    def close(self):
        self._close_selenium_driver()
        super().close()


# ===========================================================================
# validate_scraped_dataset
# ===========================================================================

def validate_scraped_dataset(df: pd.DataFrame, expected_columns: List[str],
                              data_types: Dict) -> Dict:
    """
    Comprehensive validation of scraped dataset quality.

    Returns:
        dict: is_valid, issues, quality_score, checks_passed, total_checks
    """
    issues = []
    checks_passed = 0
    total_checks = 0

    # Check 1: Not empty
    total_checks += 1
    if df.empty:
        issues.append("Dataset is empty")
    else:
        checks_passed += 1

    if df.empty:
        return {'is_valid': False, 'issues': issues, 'quality_score': 0,
                'checks_passed': checks_passed, 'total_checks': total_checks}

    # Check 2: Expected columns present
    total_checks += 1
    missing_columns = [c for c in expected_columns if c not in df.columns]
    if missing_columns:
        issues.append(f"Missing expected columns: {missing_columns}")
    else:
        checks_passed += 1

    # Check 3: Data types correct
    total_checks += 1
    dtype_issues = []
    for col, expected_dtype in data_types.items():
        if col in df.columns:
            actual_dtype = str(df[col].dtype)
            if expected_dtype == 'datetime64' and not pd.api.types.is_datetime64_any_dtype(df[col]):
                dtype_issues.append(f"{col}: expected {expected_dtype}, got {actual_dtype}")
            elif expected_dtype in ['int64', 'float64'] and not pd.api.types.is_numeric_dtype(df[col]):
                dtype_issues.append(f"{col}: expected {expected_dtype}, got {actual_dtype}")
            elif expected_dtype == 'object' and df[col].dtype != 'object':
                dtype_issues.append(f"{col}: expected {expected_dtype}, got {actual_dtype}")
    if dtype_issues:
        issues.append(f"Data type issues: {dtype_issues}")
    else:
        checks_passed += 1

    # Check 4: Values in reasonable ranges
    total_checks += 1
    range_issues = []
    if 'Date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['Date']):
        future_dates = df[df['Date'] > pd.Timestamp.now()]
        if not future_dates.empty:
            range_issues.append(f"Found {len(future_dates)} future dates in Date column")
        old_dates = df[df['Date'] < pd.Timestamp('2000-01-01')]
        if not old_dates.empty:
            range_issues.append(f"Found {len(old_dates)} dates before 2000 in Date column")
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if 'id' in col.lower() or 'index' in col.lower():
            continue
        values = df[col].dropna()
        if len(values) == 0:
            continue
        if any(kw in col.lower() for kw in ['price', 'open', 'high', 'low', 'close', 'volume']):
            negative_values = values[values < 0]
            if not negative_values.empty:
                range_issues.append(f"Found {len(negative_values)} negative values in {col}")
        if len(values) > 10:
            mean_val = values.mean()
            std_val = values.std()
            if std_val > 0:
                outliers = values[abs(values - mean_val) > 3 * std_val]
                if len(outliers) > len(values) * 0.05:
                    range_issues.append(f"Found {len(outliers)} extreme outliers in {col}")
    if range_issues:
        issues.append(f"Value range issues: {range_issues}")
    else:
        checks_passed += 1

    # Check 5: Chronological order (ascending OR descending)
    # FIX BUG-07: original only accepted ascending order; descending (newest-first) is
    #             also a valid, common format for financial data.
    total_checks += 1
    if 'Date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['Date']):
        original_dates = df['Date'].reset_index(drop=True)
        asc_dates = original_dates.sort_values().reset_index(drop=True)
        desc_dates = original_dates.sort_values(ascending=False).reset_index(drop=True)
        if not (original_dates.equals(asc_dates) or original_dates.equals(desc_dates)):
            issues.append("Data is not in chronological order (neither ascending nor descending)")
        else:
            checks_passed += 1
    else:
        total_checks -= 1  # N/A if no Date column

    # Check 6: No unexpected gaps
    total_checks += 1
    gap_issues = []
    if 'Date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['Date']):
        df_sorted = df.sort_values('Date').copy()
        df_sorted['Date'] = pd.to_datetime(df_sorted['Date'], errors='coerce')
        # Safely compute day-differences without relying on .dt typing
        diffs = df_sorted['Date'].diff()
        date_diffs = diffs.apply(lambda td: td.days if pd.notna(td) and hasattr(td, 'days') else None)
        if len(date_diffs) > 1:
            date_diffs = date_diffs.dropna()
            large_gaps = date_diffs[date_diffs > 7]
            if not large_gaps.empty:
                gap_issues.append(f"Found {len(large_gaps)} large date gaps (>7 days)")
            duplicate_dates = df_sorted[df_sorted['Date'].duplicated()]
            if not duplicate_dates.empty:
                gap_issues.append(f"Found {len(duplicate_dates)} duplicate dates")
    if gap_issues:
        issues.append(f"Temporal gap issues: {gap_issues}")
    else:
        checks_passed += 1

    # Check 7: No completely null rows
    total_checks += 1
    null_rows = df.isnull().all(axis=1)
    if null_rows.any():
        issues.append(f"Found {null_rows.sum()} completely null rows")
    else:
        checks_passed += 1

    # Check 8: Reasonable completeness
    total_checks += 1
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()
    missing_pct = (missing_cells / total_cells) * 100 if total_cells > 0 else 100
    if missing_pct > 20:
        issues.append(f"High missing data percentage: {missing_pct:.1f}%")
    else:
        checks_passed += 1

    quality_score = int((checks_passed / total_checks) * 100) if total_checks > 0 else 100
    is_valid = len(issues) == 0

    logger.info(f"Dataset validation completed: {checks_passed}/{total_checks} checks passed, "
                f"quality score: {quality_score}%")
    if issues:
        logger.warning(f"Validation issues found: {issues}")

    return {
        'is_valid': is_valid,
        'issues': issues,
        'quality_score': quality_score,
        'checks_passed': checks_passed,
        'total_checks': total_checks
    }


# ===========================================================================
# CheckpointManager
# ===========================================================================

class CheckpointManager:
    """Manages checkpoints for resumable web scraping operations."""

    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        logger.info(f"Initialized CheckpointManager with directory: {checkpoint_dir}")

    def _get_checkpoint_path(self, source_name: str) -> str:
        safe_name = "".join(c for c in source_name
                            if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_name = safe_name.replace(' ', '_').lower()
        return os.path.join(self.checkpoint_dir, f"{safe_name}_checkpoint.json")

    def save_checkpoint(self, source_name: str, last_scraped_date: str,
                        df: pd.DataFrame) -> bool:
        try:
            checkpoint_path = self._get_checkpoint_path(source_name)
            checkpoint_data = {
                'source_name': source_name,
                'last_scraped_date': last_scraped_date,
                'data_rows': len(df),
                'data_columns': list(df.columns),
                'timestamp': datetime.now().isoformat(),
                'data': df.to_dict('records')
            }
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Checkpoint saved for {source_name}: {len(df)} rows, "
                        f"last date: {last_scraped_date} -> {checkpoint_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save checkpoint for {source_name}: {e}")
            return False

    def load_checkpoint(self, source_name: str) -> tuple:
        checkpoint_path = self._get_checkpoint_path(source_name)
        if not os.path.exists(checkpoint_path):
            logger.info(f"No checkpoint found for {source_name}")
            return None, pd.DataFrame()

        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)

            last_scraped_date = checkpoint_data.get('last_scraped_date')
            data_records = checkpoint_data.get('data', [])
            df = pd.DataFrame(data_records) if data_records else pd.DataFrame()

            if last_scraped_date and not df.empty:
                logger.info(f"Checkpoint loaded for {source_name}: {len(df)} rows, "
                            f"last date: {last_scraped_date}")
                expected_columns = checkpoint_data.get('data_columns', [])
                if set(df.columns) == set(expected_columns):
                    return last_scraped_date, df
                else:
                    logger.warning(f"Checkpoint data columns mismatch for {source_name}. "
                                   f"Expected: {expected_columns}, Got: {list(df.columns)}")
                    return None, pd.DataFrame()
            else:
                logger.warning(f"Invalid checkpoint data for {source_name}")
                return None, pd.DataFrame()
        except (json.JSONDecodeError, KeyError, IOError) as e:
            logger.error(f"Failed to load checkpoint for {source_name}: {e}")
            return None, pd.DataFrame()

    def delete_checkpoint(self, source_name: str) -> bool:
        checkpoint_path = self._get_checkpoint_path(source_name)
        try:
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
                logger.info(f"Checkpoint deleted for {source_name}")
            else:
                logger.info(f"No checkpoint to delete for {source_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete checkpoint for {source_name}: {e}")
            return False

    def list_checkpoints(self) -> List[str]:
        if not os.path.exists(self.checkpoint_dir):
            return []
        return sorted(
            fn.replace('_checkpoint.json', '')
            for fn in os.listdir(self.checkpoint_dir)
            if fn.endswith('_checkpoint.json')
        )

    def get_checkpoint_info(self, source_name: str) -> Optional[Dict]:
        checkpoint_path = self._get_checkpoint_path(source_name)
        if not os.path.exists(checkpoint_path):
            return None
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            return {
                'source_name': checkpoint_data.get('source_name'),
                'last_scraped_date': checkpoint_data.get('last_scraped_date'),
                'data_rows': checkpoint_data.get('data_rows', 0),
                'data_columns': checkpoint_data.get('data_columns', []),
                'timestamp': checkpoint_data.get('timestamp'),
                'file_size': os.path.getsize(checkpoint_path)
            }
        except Exception as e:
            logger.error(f"Failed to read checkpoint info for {source_name}: {e}")
            return None

    def cleanup_old_checkpoints(self, days_old: int = 7) -> int:
        if not os.path.exists(self.checkpoint_dir):
            return 0
        cutoff_time = datetime.now().timestamp() - (days_old * 24 * 60 * 60)
        deleted_count = 0
        for filename in os.listdir(self.checkpoint_dir):
            if filename.endswith('_checkpoint.json'):
                filepath = os.path.join(self.checkpoint_dir, filename)
                if os.path.getmtime(filepath) < cutoff_time:
                    try:
                        os.remove(filepath)
                        logger.info(f"Cleaned up old checkpoint: {filename}")
                        deleted_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to delete old checkpoint {filename}: {e}")
        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} old checkpoint files")
        return deleted_count


# ===========================================================================
# Error recovery helpers
# ===========================================================================

class ScrapingAction:
    """Enumeration of possible actions for error recovery."""
    RETRY = "RETRY"
    SKIP_DAY = "SKIP_DAY"
    USE_FALLBACK = "USE_FALLBACK"
    ABORT = "ABORT"


def handle_scraping_error(source: str, error: Exception, retry_count: int,
                          max_retries: int = 3) -> str:
    """
    Intelligent error handling for web scraping operations.

    Returns:
        str: Action to take (RETRY, SKIP_DAY, USE_FALLBACK, ABORT)
    """
    error_type = type(error).__name__
    error_msg = str(error).lower()
    logger.warning(f"[{source}] Error analysis: {error_type} - {error_msg} "
                   f"(retry {retry_count}/{max_retries})")

    network_errors = [
        'ConnectionError', 'Timeout', 'ConnectTimeout', 'ReadTimeout',
        'NewConnectionError', 'MaxRetryError', 'SSLError', 'ProxyError'
    ]
    if error_type in network_errors:
        if retry_count < max_retries:
            logger.info(f"[{source}] Network error - RETRY ({retry_count + 1}/{max_retries})")
            return ScrapingAction.RETRY
        else:
            logger.warning(f"[{source}] Network error exhausted retries - USE_FALLBACK")
            return ScrapingAction.USE_FALLBACK

    status_code = None
    if hasattr(error, 'response'):
        resp = getattr(error, 'response', None)
        if resp is not None and hasattr(resp, 'status_code'):
            status_code = resp.status_code
        if status_code == 429:
            if retry_count < max_retries:
                logger.info(f"[{source}] Rate limited (429) - RETRY with backoff")
                return ScrapingAction.RETRY
            else:
                logger.warning(f"[{source}] Rate limited exhausted retries - SKIP_DAY")
                return ScrapingAction.SKIP_DAY
        elif status_code in [403, 401]:
            logger.warning(f"[{source}] Access denied ({status_code}) - USE_FALLBACK")
            return ScrapingAction.USE_FALLBACK
        elif status_code is not None and status_code >= 500:
            if retry_count < max_retries:
                logger.info(f"[{source}] Server error ({status_code}) - RETRY")
                return ScrapingAction.RETRY
            else:
                logger.warning(f"[{source}] Server error exhausted retries - SKIP_DAY")
                return ScrapingAction.SKIP_DAY
        elif status_code == 404:
            logger.warning(f"[{source}] Data not found (404) - SKIP_DAY")
            return ScrapingAction.SKIP_DAY

    selenium_errors = [
        'WebDriverException', 'TimeoutException', 'NoSuchElementException',
        'StaleElementReferenceException', 'ElementNotInteractableException'
    ]
    if error_type in selenium_errors:
        if 'element' in error_msg and 'not found' in error_msg:
            logger.warning(f"[{source}] Page element not found - USE_FALLBACK")
            return ScrapingAction.USE_FALLBACK
        elif retry_count < max_retries:
            logger.info(f"[{source}] Selenium error - RETRY ({retry_count + 1}/{max_retries})")
            return ScrapingAction.RETRY
        else:
            logger.warning(f"[{source}] Selenium error exhausted retries - SKIP_DAY")
            return ScrapingAction.SKIP_DAY

    parsing_errors = ['ValueError', 'KeyError', 'IndexError', 'TypeError', 'AttributeError']
    if error_type in parsing_errors:
        logger.warning(f"[{source}] Data parsing error - SKIP_DAY (data format issue)")
        return ScrapingAction.SKIP_DAY

    if error_type == 'JSONDecodeError':
        logger.warning(f"[{source}] Invalid JSON response - USE_FALLBACK")
        return ScrapingAction.USE_FALLBACK

    if error_type in ['IOError', 'OSError', 'FileNotFoundError']:
        logger.error(f"[{source}] File system error - ABORT")
        return ScrapingAction.ABORT

    if retry_count < max_retries:
        logger.info(f"[{source}] Unknown error - RETRY ({retry_count + 1}/{max_retries})")
        return ScrapingAction.RETRY
    else:
        logger.warning(f"[{source}] Unknown error exhausted retries - ABORT")
        return ScrapingAction.ABORT


def skip_holiday(target_date: date) -> bool:  # FIX BUG-04: renamed param + type
    """
    Check if a date should be skipped due to NSE holidays.

    Includes weekends and confirmed major Indian public holidays.

    FIX BUG-10: Removed fabricated "last Monday of May" holiday rule that has
                no basis in the NSE holiday calendar.

    Args:
        target_date (date): Date to check

    Returns:
        bool: True if date should be skipped, False if trading day
    """
    # Weekends
    if target_date.weekday() >= 5:
        return True

    year = target_date.year

    # Republic Day — January 26
    if target_date == date(year, 1, 26):
        return True

    # Independence Day — August 15
    if target_date == date(year, 8, 15):
        return True

    # Gandhi Jayanti — October 2
    if target_date == date(year, 10, 2):
        return True

    # Christmas — December 25
    if target_date == date(year, 12, 25):
        return True

    # NOTE: Variable holidays (Holi, Diwali, Eid, Good Friday, etc.) require the
    # official NSE holiday calendar fetched at runtime for accurate determination.

    return False


def get_next_trading_day(target_date: date) -> date:  # FIX BUG-04
    """Get the next trading day after the given date."""
    next_day = target_date + timedelta(days=1)
    while skip_holiday(next_day):
        next_day += timedelta(days=1)
    return next_day


def calculate_error_backoff_delay(attempt: int, base_delay: float = 1.0) -> float:
    """
    Calculate exponential backoff delay for error recovery.
    FIX BUG-06: Cap aligned to 60 s (consistent with _calculate_backoff_delay).

    Args:
        attempt (int): Current attempt number (0-based)
        base_delay (float): Base delay in seconds

    Returns:
        float: Delay in seconds (capped at 60 seconds)
    """
    delay = base_delay * (2 ** attempt)
    max_delay = 60.0  # FIX BUG-06: was 300 s in original; aligned with session method
    return min(delay, max_delay)


def log_scraping_attempt(source: str, target_date: str, attempt: int, action: str,  # FIX BUG-04
                         error: Exception | None = None):
    """Log a scraping attempt with structured information."""
    error_info = f" - {type(error).__name__}: {str(error)}" if error else ""
    logger.info(f"[{source}] {target_date} | Attempt {attempt} | {action}{error_info}")


def should_retry_error(error: Exception, retry_count: int, max_retries: int) -> bool:
    """Quick check if an error should be retried."""
    if retry_count >= max_retries:
        return False
    retry_errors = [
        'ConnectionError', 'Timeout', 'ConnectTimeout', 'ReadTimeout',
        'NewConnectionError', 'MaxRetryError', 'WebDriverException'
    ]
    return type(error).__name__ in retry_errors