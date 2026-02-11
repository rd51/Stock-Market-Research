"""
Data Preprocessing Pipeline
===========================

Orchestrates the complete data preparation workflow for freshly scraped financial data.
Chains validation, cleaning, feature engineering, normalization, and data splitting.

Author: GitHub Copilot
Date: February 8, 2026
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import os
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from web_scraping_utils import ScrapingSession, NSEIndiaScraper, DataGovInScraper, LabourGovScraper, InvestingComScraper
from feature_engineering import FeatureEngineer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataPreprocessingPipeline:
    """
    Complete data preprocessing pipeline for freshly scraped financial data.
    
    Orchestrates the entire workflow from raw scraped data to ML-ready datasets.
    Designed specifically for freshly scraped datasets with no historical dependencies.
    """

    def __init__(self, output_dir: str = "processed_data"):
        """
        Initialize the preprocessing pipeline.

        Args:
            output_dir (str): Directory to save processed data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Initialize components
        self.scraping_session = ScrapingSession("preprocessing_pipeline")
        self.feature_engineer = FeatureEngineer()

        # Pipeline state
        self.raw_data = {}
        self.processed_data = {}
        self.validation_results = {}
        self.pipeline_metadata = {}

        logger.info(f"Initialized DataPreprocessingPipeline with output directory: {self.output_dir}")

    def load_scraped_data(self, data_sources: Dict[str, Union[str, pd.DataFrame]]) -> Dict[str, pd.DataFrame]:
        """
        Load freshly scraped data from various sources.

        Args:
            data_sources (Dict[str, Union[str, pd.DataFrame]]): Dictionary mapping source names to file paths or DataFrames

        Returns:
            Dict[str, pd.DataFrame]: Loaded dataframes
        """
        logger.info(f"Loading scraped data from {len(data_sources)} sources")

        loaded_data = {}

        for source_name, data_source in data_sources.items():
            try:
                if isinstance(data_source, pd.DataFrame):
                    # Direct DataFrame input
                    df = data_source.copy()
                    logger.info(f"Loaded {source_name} from DataFrame: {df.shape}")
                elif isinstance(data_source, str):
                    # File path
                    if data_source.endswith('.csv'):
                        df = pd.read_csv(data_source)
                    elif data_source.endswith('.json'):
                        df = pd.read_json(data_source)
                    elif data_source.endswith('.xlsx') or data_source.endswith('.xls'):
                        df = pd.read_excel(data_source)
                    else:
                        logger.warning(f"Unsupported file format for {source_name}: {data_source}")
                        continue

                    logger.info(f"Loaded {source_name} from file: {df.shape}")
                else:
                    logger.warning(f"Unsupported data source type for {source_name}: {type(data_source)}")
                    continue

                # Basic data cleaning
                df = self._basic_data_cleaning(df, source_name)
                loaded_data[source_name] = df

            except Exception as e:
                logger.error(f"Error loading {source_name}: {str(e)}")
                continue

        self.raw_data = loaded_data
        logger.info(f"Successfully loaded {len(loaded_data)} datasets")
        return loaded_data

    def _basic_data_cleaning(self, df: pd.DataFrame, source_name: str) -> pd.DataFrame:
        """
        Perform basic data cleaning operations.

        Args:
            df (pd.DataFrame): Input dataframe
            source_name (str): Name of the data source

        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        df_copy = df.copy()

        # Convert date columns
        date_columns = [col for col in df_copy.columns if 'date' in col.lower() or 'time' in col.lower()]
        for col in date_columns:
            try:
                df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')
            except:
                pass

        # Remove completely empty rows
        initial_rows = len(df_copy)
        df_copy = df_copy.dropna(how='all')
        final_rows = len(df_copy)

        if initial_rows != final_rows:
            logger.info(f"Removed {initial_rows - final_rows} completely empty rows from {source_name}")

        return df_copy

    def validate_data(self, data_dict: Optional[Dict[str, pd.DataFrame]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Validate data quality and structure for each dataset.

        Args:
            data_dict (Optional[Dict[str, pd.DataFrame]]): Data to validate (uses self.raw_data if None)

        Returns:
            Dict[str, Dict[str, Any]]: Validation results for each dataset
        """
        if data_dict is None:
            data_dict = self.raw_data

        logger.info(f"Validating {len(data_dict)} datasets")

        validation_results = {}

        for source_name, df in data_dict.items():
            validation_results[source_name] = self._validate_single_dataset(df, source_name)

        self.validation_results = validation_results

        # Log summary
        total_issues = sum([result['total_issues'] for result in validation_results.values()])
        logger.info(f"Data validation complete. Total issues found: {total_issues}")

        return validation_results

    def _validate_single_dataset(self, df: pd.DataFrame, source_name: str) -> Dict[str, Any]:
        """
        Validate a single dataset.

        Args:
            df (pd.DataFrame): Dataset to validate
            source_name (str): Name of the dataset

        Returns:
            Dict[str, Any]: Validation results
        """
        results = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'issues': [],
            'total_issues': 0
        }

        # Check for required columns based on source type
        if 'nse' in source_name.lower():
            required_cols = ['Date', 'VIX']
            expected_numeric = ['VIX', 'Close', 'Open', 'High', 'Low']
        elif 'datagov' in source_name.lower():
            required_cols = ['Date']
            expected_numeric = ['unemployment_rate', 'labour_participation']
        elif 'labour' in source_name.lower():
            required_cols = ['Date']
            expected_numeric = ['employment_rate', 'wage_index']
        elif 'investing' in source_name.lower():
            required_cols = ['Date']
            expected_numeric = ['price', 'volume']
        else:
            required_cols = ['Date']
            expected_numeric = []

        # Check required columns
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            results['issues'].append(f"Missing required columns: {missing_cols}")

        # Check data types
        for col in expected_numeric:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                results['issues'].append(f"Column {col} should be numeric but is {df[col].dtype}")

        # Check for date column
        date_cols = [col for col in df.columns if 'date' in col.lower()]
        if not date_cols:
            results['issues'].append("No date column found")
        else:
            date_col = date_cols[0]
            if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
                results['issues'].append(f"Date column {date_col} is not datetime type")

        # Check for missing values
        missing_summary = df.isnull().sum()
        total_missing = missing_summary.sum()
        if total_missing > 0:
            cols_with_missing = missing_summary[missing_summary > 0]
            results['issues'].append(f"Missing values in columns: {dict(cols_with_missing)}")

        # Check for duplicate rows
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            results['issues'].append(f"Found {duplicates} duplicate rows")

        # Check data range validity
        if 'VIX' in df.columns:
            vix_values = df['VIX'].dropna()
            if len(vix_values) > 0:
                invalid_vix = ((vix_values < 0) | (vix_values > 100)).sum()
                if invalid_vix > 0:
                    results['issues'].append(f"Found {invalid_vix} invalid VIX values (outside 0-100 range)")

        results['total_issues'] = len(results['issues'])
        return results

    def handle_missing_values(self, data_dict: Optional[Dict[str, pd.DataFrame]] = None,
                            strategy: str = 'interpolate') -> Dict[str, pd.DataFrame]:
        """
        Handle missing values in datasets.

        Args:
            data_dict (Optional[Dict[str, pd.DataFrame]]): Data to process
            strategy (str): Missing value handling strategy ('drop', 'fill', 'interpolate')

        Returns:
            Dict[str, pd.DataFrame]: Data with missing values handled
        """
        if data_dict is None:
            data_dict = self.raw_data.copy()

        logger.info(f"Handling missing values using strategy: {strategy}")

        processed_data = {}

        for source_name, df in data_dict.items():
            df_processed = df.copy()

            # Strategy-specific handling
            if strategy == 'drop':
                initial_rows = len(df_processed)
                df_processed = df_processed.dropna()
                final_rows = len(df_processed)
                logger.info(f"Dropped {initial_rows - final_rows} rows with missing values from {source_name}")

            elif strategy == 'fill':
                # Fill numeric columns with median, categorical with mode
                for col in df_processed.columns:
                    if pd.api.types.is_numeric_dtype(df_processed[col]):
                        df_processed[col] = df_processed[col].fillna(df_processed[col].median())
                    else:
                        mode_val = df_processed[col].mode()
                        if not mode_val.empty:
                            df_processed[col] = df_processed[col].fillna(mode_val.iloc[0])

                logger.info(f"Filled missing values in {source_name}")

            elif strategy == 'interpolate':
                # Time series interpolation for numeric columns
                date_cols = [col for col in df_processed.columns if 'date' in col.lower()]
                if date_cols:
                    df_processed = df_processed.sort_values(date_cols[0])

                # Interpolate numeric columns
                numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
                df_processed[numeric_cols] = df_processed[numeric_cols].interpolate(method='linear')

                # Fill remaining NaNs with forward/backward fill
                df_processed = df_processed.ffill().bfill()

                logger.info(f"Interpolated missing values in {source_name}")

            processed_data[source_name] = df_processed

        self.processed_data = processed_data
        logger.info(f"Missing value handling complete for {len(processed_data)} datasets")
        return processed_data

    def remove_outliers(self, data_dict: Optional[Dict[str, pd.DataFrame]] = None,
                       method: str = 'iqr', threshold: float = 1.5) -> Dict[str, pd.DataFrame]:
        """
        Remove outliers from datasets.

        Args:
            data_dict (Optional[Dict[str, pd.DataFrame]]): Data to process
            method (str): Outlier detection method ('iqr', 'zscore', 'isolation_forest')
            threshold (float): Threshold for outlier detection

        Returns:
            Dict[str, pd.DataFrame]: Data with outliers removed
        """
        if data_dict is None:
            data_dict = self.processed_data.copy()

        logger.info(f"Removing outliers using method: {method}")

        processed_data = {}

        for source_name, df in data_dict.items():
            df_clean = df.copy()

            if method == 'iqr':
                # IQR method for numeric columns
                numeric_cols = df_clean.select_dtypes(include=[np.number]).columns

                for col in numeric_cols:
                    if col.lower() in ['year', 'month', 'day', 'day_of_week', 'quarter']:
                        continue  # Skip date-related columns

                    Q1 = df_clean[col].quantile(0.25)
                    Q3 = df_clean[col].quantile(0.75)
                    IQR = Q3 - Q1

                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR

                    outliers = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound))
                    df_clean.loc[outliers, col] = np.nan

                # Interpolate outliers
                df_clean = df_clean.interpolate(method='linear').ffill().bfill()

            elif method == 'zscore':
                # Z-score method
                numeric_cols = df_clean.select_dtypes(include=[np.number]).columns

                for col in numeric_cols:
                    if col.lower() in ['year', 'month', 'day', 'day_of_week', 'quarter']:
                        continue

                    z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
                    outliers = z_scores > threshold
                    df_clean.loc[outliers, col] = np.nan

                df_clean = df_clean.interpolate(method='linear').ffill().bfill()

            # Log outlier removal
            logger.info(f"Processed outliers in {source_name}: {df.shape[0] - df_clean.shape[0]} rows affected")

            processed_data[source_name] = df_clean

        self.processed_data = processed_data
        logger.info(f"Outlier removal complete for {len(processed_data)} datasets")
        return processed_data

    def create_features(self, data_dict: Optional[Dict[str, pd.DataFrame]] = None,
                       feature_config: Optional[Dict[str, Any]] = None) -> Dict[str, pd.DataFrame]:
        """
        Create features using the FeatureEngineer.

        Args:
            data_dict (Optional[Dict[str, pd.DataFrame]]): Data to process
            feature_config (Optional[Dict[str, Any]]): Feature engineering configuration

        Returns:
            Dict[str, pd.DataFrame]: Data with features created
        """
        if data_dict is None:
            data_dict = self.processed_data.copy()

        if feature_config is None:
            feature_config = {
                'vix_col': 'VIX',
                'price_col': 'Close',
                'create_comprehensive': True
            }

        logger.info("Creating features for all datasets")

        featured_data = {}

        for source_name, df in data_dict.items():
            try:
                if feature_config.get('create_comprehensive', True):
                    # Use comprehensive feature creation
                    df_featured = self.feature_engineer.create_comprehensive_features(
                        df,
                        vix_col=feature_config.get('vix_col', 'VIX'),
                        price_col=feature_config.get('price_col')
                    )
                else:
                    # Apply individual feature engineering steps
                    df_featured = df.copy()

                    # Add basic features based on available columns
                    if 'VIX' in df_featured.columns:
                        df_featured = self.feature_engineer.create_rolling_statistics(df_featured, 'VIX', [7, 14, 30])
                        df_featured = self.feature_engineer.create_regime_indicators(df_featured, 'VIX')

                    if 'Date' in df_featured.columns:
                        df_featured = self.feature_engineer.create_seasonality_features(df_featured, 'Date')

                featured_data[source_name] = df_featured
                logger.info(f"Created features for {source_name}: {df_featured.shape[1]} features")

            except Exception as e:
                logger.error(f"Error creating features for {source_name}: {str(e)}")
                featured_data[source_name] = df  # Keep original data if feature creation fails

        self.processed_data = featured_data
        logger.info(f"Feature creation complete for {len(featured_data)} datasets")
        return featured_data

    def normalize_data(self, data_dict: Optional[Dict[str, pd.DataFrame]] = None,
                      method: str = 'standard', columns_to_normalize: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Normalize data for machine learning.

        Args:
            data_dict (Optional[Dict[str, pd.DataFrame]]): Data to normalize
            method (str): Normalization method ('standard', 'minmax', 'robust')
            columns_to_normalize (Optional[List[str]]): Specific columns to normalize

        Returns:
            Dict[str, pd.DataFrame]: Normalized data
        """
        if data_dict is None:
            data_dict = self.processed_data.copy()

        logger.info(f"Normalizing data using method: {method}")

        normalized_data = {}

        for source_name, df in data_dict.items():
            df_normalized = df.copy()

            # Determine columns to normalize
            if columns_to_normalize is None:
                # Auto-select numeric columns (exclude date-related and categorical)
                exclude_patterns = ['year', 'month', 'day', 'week', 'quarter', 'regime', 'shock', 'is_', 'crash']
                numeric_cols = []

                for col in df_normalized.select_dtypes(include=[np.number]).columns:
                    if not any(pattern in col.lower() for pattern in exclude_patterns):
                        numeric_cols.append(col)

                columns_to_normalize = numeric_cols[:10]  # Limit to first 10 to avoid over-normalization

            if columns_to_normalize:
                df_normalized = self.feature_engineer.normalize_features(
                    df_normalized, columns_to_normalize, method=method
                )

            normalized_data[source_name] = df_normalized
            logger.info(f"Normalized {len(columns_to_normalize)} columns in {source_name}")

        self.processed_data = normalized_data
        logger.info(f"Data normalization complete for {len(normalized_data)} datasets")
        return normalized_data

    def split_data(self, data_dict: Optional[Dict[str, pd.DataFrame]] = None,
                  train_ratio: float = 0.7, val_ratio: float = 0.15,
                  time_based_split: bool = True) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Split data into train/validation/test sets.

        Args:
            data_dict (Optional[Dict[str, pd.DataFrame]]): Data to split
            train_ratio (float): Proportion for training
            val_ratio (float): Proportion for validation
            time_based_split (bool): Whether to use time-based splitting

        Returns:
            Dict[str, Dict[str, pd.DataFrame]]: Split data for each source
        """
        if data_dict is None:
            data_dict = self.processed_data.copy()

        logger.info(f"Splitting data with train_ratio={train_ratio}, val_ratio={val_ratio}")

        split_data = {}

        for source_name, df in data_dict.items():
            if len(df) < 10:
                logger.warning(f"Dataset {source_name} too small for splitting: {len(df)} rows")
                split_data[source_name] = {'train': df, 'val': pd.DataFrame(), 'test': pd.DataFrame()}
                continue

            if time_based_split:
                # Time-based split (chronological order)
                n_total = len(df)
                n_train = int(n_total * train_ratio)
                n_val = int(n_total * val_ratio)

                train_data = df.iloc[:n_train]
                val_data = df.iloc[n_train:n_train + n_val]
                test_data = df.iloc[n_train + n_val:]
            else:
                # Random split
                df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

                n_total = len(df_shuffled)
                n_train = int(n_total * train_ratio)
                n_val = int(n_total * val_ratio)

                train_data = df_shuffled.iloc[:n_train]
                val_data = df_shuffled.iloc[n_train:n_train + n_val]
                test_data = df_shuffled.iloc[n_train + n_val:]

            split_data[source_name] = {
                'train': train_data,
                'val': val_data,
                'test': test_data
            }

            logger.info(f"Split {source_name}: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

        self.split_data_result = split_data
        logger.info(f"Data splitting complete for {len(split_data)} datasets")
        return split_data

    def save_processed_data(self, split_data: Optional[Dict[str, Dict[str, pd.DataFrame]]] = None,
                           save_format: str = 'csv') -> Dict[str, Dict[str, str]]:
        """
        Save processed data to files.

        Args:
            split_data (Optional[Dict[str, Dict[str, pd.DataFrame]]]): Data to save
            save_format (str): File format ('csv', 'json', 'parquet')

        Returns:
            Dict[str, Dict[str, str]]: File paths for saved data
        """
        if split_data is None:
            split_data = getattr(self, 'split_data_result', {})

        logger.info(f"Saving processed data in {save_format} format")

        saved_files = {}

        for source_name, splits in split_data.items():
            saved_files[source_name] = {}

            for split_name, df in splits.items():
                if df.empty:
                    continue

                # Create filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{source_name}_{split_name}_{timestamp}.{save_format}"
                filepath = self.output_dir / filename

                try:
                    if save_format == 'csv':
                        df.to_csv(filepath, index=False)
                    elif save_format == 'json':
                        df.to_json(filepath, orient='records', date_format='iso')
                    elif save_format == 'parquet':
                        df.to_parquet(filepath, index=False)
                    else:
                        logger.warning(f"Unsupported save format: {save_format}")
                        continue

                    saved_files[source_name][split_name] = str(filepath)
                    logger.info(f"Saved {source_name}_{split_name} to {filepath}")

                except Exception as e:
                    logger.error(f"Error saving {source_name}_{split_name}: {str(e)}")

        # Save pipeline metadata
        metadata_file = self.output_dir / f"pipeline_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        metadata = {
            'pipeline_run_date': datetime.now().isoformat(),
            'raw_data_shapes': {k: v.shape for k, v in self.raw_data.items()},
            'processed_data_shapes': {k: v.shape for k, v in self.processed_data.items()},
            'validation_results': self.validation_results,
            'saved_files': saved_files
        }

        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"Pipeline metadata saved to {metadata_file}")
        return saved_files

    def run_complete_pipeline(self, data_sources: Dict[str, Union[str, pd.DataFrame]],
                            config: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, str]]:
        """
        Run the complete preprocessing pipeline.

        Args:
            data_sources (Dict[str, Union[str, pd.DataFrame]]): Input data sources
            config (Optional[Dict[str, Any]]): Pipeline configuration

        Returns:
            Dict[str, Dict[str, str]]: Paths to saved processed data
        """
        if config is None:
            config = {
                'missing_value_strategy': 'interpolate',
                'outlier_method': 'iqr',
                'normalization_method': 'standard',
                'train_ratio': 0.7,
                'val_ratio': 0.15,
                'time_based_split': True,
                'save_format': 'csv'
            }

        logger.info("Starting complete preprocessing pipeline")

        try:
            # Step 1: Load data
            self.load_scraped_data(data_sources)

            # Step 2: Validate data
            validation_results = self.validate_data()
            total_issues = sum([result['total_issues'] for result in validation_results.values()])
            if total_issues > 0:
                logger.warning(f"Data validation found {total_issues} issues. Continuing with pipeline...")

            # Step 3: Handle missing values
            self.handle_missing_values(strategy=config['missing_value_strategy'])

            # Step 4: Remove outliers
            self.remove_outliers(method=config['outlier_method'])

            # Step 5: Create features
            self.create_features()

            # Step 6: Normalize data
            self.normalize_data(method=config['normalization_method'])

            # Step 7: Split data
            split_data = self.split_data(
                train_ratio=config['train_ratio'],
                val_ratio=config['val_ratio'],
                time_based_split=config['time_based_split']
            )

            # Step 8: Save processed data
            saved_files = self.save_processed_data(split_data, save_format=config['save_format'])

            logger.info("Complete preprocessing pipeline finished successfully")
            return saved_files

        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise

    def get_pipeline_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the pipeline execution.

        Returns:
            Dict[str, Any]: Pipeline summary
        """
        summary = {
            'raw_data_sources': list(self.raw_data.keys()),
            'raw_data_shapes': {k: v.shape for k, v in self.raw_data.items()},
            'processed_data_shapes': {k: v.shape for k, v in self.processed_data.items()},
            'validation_issues': {k: v['total_issues'] for k, v in self.validation_results.items()},
            'pipeline_metadata': self.pipeline_metadata
        }

        return summary