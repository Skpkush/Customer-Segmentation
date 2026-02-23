"""
data_preprocessing.py
---------------------
Functions for loading, cleaning, and preprocessing the Online Retail dataset.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_data(filepath: str, encoding: str = 'ISO-8859-1') -> pd.DataFrame:
    """
    Load the Online Retail dataset from a CSV or Excel file.

    Args:
        filepath: Path to the data file (.csv or .xlsx)
        encoding: File encoding (ISO-8859-1 works for most Online Retail files)

    Returns:
        Raw DataFrame
    """
    logger.info(f"Loading data from: {filepath}")
    if filepath.endswith('.xlsx'):
        df = pd.read_excel(filepath, engine='openpyxl')
    else:
        df = pd.read_csv(filepath, encoding=encoding)

    logger.info(f"Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def initial_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a data quality report showing missing values, dtypes, and unique counts.

    Args:
        df: Raw DataFrame

    Returns:
        Quality report DataFrame
    """
    report = pd.DataFrame({
        'dtype': df.dtypes,
        'null_count': df.isnull().sum(),
        'null_pct': (df.isnull().sum() / len(df) * 100).round(2),
        'unique_count': df.nunique(),
        'sample_value': df.iloc[0]
    })
    return report


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all cleaning steps to the Online Retail dataset:
      - Remove rows with missing CustomerID
      - Remove cancelled transactions (InvoiceNo starting with 'C')
      - Remove non-positive Quantity and UnitPrice
      - Parse InvoiceDate to datetime
      - Cast CustomerID to string

    Args:
        df: Raw DataFrame

    Returns:
        Cleaned DataFrame
    """
    logger.info("Starting data cleaning...")
    original_rows = len(df)

    # Drop rows with missing CustomerID — we can't segment without it
    df = df.dropna(subset=['CustomerID'])
    logger.info(f"After dropping missing CustomerID: {len(df):,} rows (removed {original_rows - len(df):,})")

    # Remove cancellations (InvoiceNo starts with 'C')
    before = len(df)
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
    logger.info(f"After removing cancellations: {len(df):,} rows (removed {before - len(df):,})")

    # Remove non-positive Quantity and UnitPrice
    before = len(df)
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    logger.info(f"After removing invalid Quantity/UnitPrice: {len(df):,} rows (removed {before - len(df):,})")

    # Parse InvoiceDate
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    # CustomerID as string (avoids float formatting issues)
    df['CustomerID'] = df['CustomerID'].astype(int).astype(str)

    # Create TotalAmount feature
    df['TotalAmount'] = df['Quantity'] * df['UnitPrice']

    logger.info(f"Cleaning complete. Final shape: {df.shape}")
    return df.reset_index(drop=True)


def remove_outliers_iqr(df: pd.DataFrame, column: str, multiplier: float = 3.0) -> pd.DataFrame:
    """
    Remove outliers from a column using IQR method.

    Args:
        df: DataFrame
        column: Column name to clean
        multiplier: IQR multiplier (3.0 = very conservative, keeps most data)

    Returns:
        DataFrame with outliers removed
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - multiplier * IQR
    upper = Q3 + multiplier * IQR
    before = len(df)
    df = df[(df[column] >= lower) & (df[column] <= upper)]
    logger.info(f"Outlier removal on '{column}': removed {before - len(df):,} rows")
    return df


def get_snapshot_date(df: pd.DataFrame, offset_days: int = 1) -> pd.Timestamp:
    """
    Get the analysis snapshot date (1 day after last transaction).
    This is used as the 'today' reference for Recency calculation.

    Args:
        df: Cleaned DataFrame with InvoiceDate column
        offset_days: Days to add after the last date

    Returns:
        Snapshot date as Timestamp
    """
    snapshot = df['InvoiceDate'].max() + pd.Timedelta(days=offset_days)
    logger.info(f"Snapshot date (analysis reference): {snapshot.date()}")
    return snapshot