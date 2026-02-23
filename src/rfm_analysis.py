"""
rfm_analysis.py
---------------
RFM (Recency, Frequency, Monetary) scoring and business-rule-based segmentation.
"""

import pandas as pd
import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


def calculate_rfm(df: pd.DataFrame, snapshot_date: pd.Timestamp) -> pd.DataFrame:
    """
    Calculate raw RFM metrics per customer.

    Args:
        df: Cleaned transaction DataFrame
        snapshot_date: Reference date for recency calculation

    Returns:
        DataFrame with columns: CustomerID, Recency, Frequency, Monetary
    """
    rfm = df.groupby('CustomerID').agg(
        Recency=('InvoiceDate', lambda x: (snapshot_date - x.max()).days),
        Frequency=('InvoiceNo', 'nunique'),
        Monetary=('TotalAmount', 'sum')
    ).reset_index()

    logger.info(f"RFM calculated for {len(rfm):,} customers")
    logger.info(f"Recency  — mean: {rfm['Recency'].mean():.1f} days, median: {rfm['Recency'].median():.1f}")
    logger.info(f"Frequency — mean: {rfm['Frequency'].mean():.1f}, median: {rfm['Frequency'].median():.1f}")
    logger.info(f"Monetary  — mean: £{rfm['Monetary'].mean():.2f}, median: £{rfm['Monetary'].median():.2f}")

    return rfm


def score_rfm(rfm: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
    """
    Score RFM metrics on a 1-5 scale using quantile binning.

    Note: Recency is scored INVERSELY (lower recency = higher score = more recent = better).

    Args:
        rfm: DataFrame with Recency, Frequency, Monetary columns
        n_bins: Number of scoring bins (default: 5)

    Returns:
        DataFrame with added R_Score, F_Score, M_Score, RFM_Score, RFM_Total columns
    """
    rfm = rfm.copy()

    # Recency: lower is better → reverse scoring
    rfm['R_Score'] = pd.qcut(rfm['Recency'], q=n_bins, labels=range(n_bins, 0, -1), duplicates='drop').astype(int)

    # Frequency: higher is better
    rfm['F_Score'] = pd.qcut(rfm['Frequency'], q=n_bins, labels=range(1, n_bins + 1), duplicates='drop').astype(int)

    # Monetary: higher is better
    rfm['M_Score'] = pd.qcut(rfm['Monetary'], q=n_bins, labels=range(1, n_bins + 1), duplicates='drop').astype(int)

    # Composite RFM string (e.g., "555") and total score
    rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)
    rfm['RFM_Total'] = rfm['R_Score'] + rfm['F_Score'] + rfm['M_Score']

    return rfm


def assign_rfm_segment(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Assign human-readable business segments based on RFM scores.
    Uses a rule-based approach with 8 meaningful segments.

    Args:
        rfm: Scored RFM DataFrame

    Returns:
        DataFrame with added 'RFM_Segment' column
    """
    rfm = rfm.copy()

    def segment_rule(row):
        r, f, m = row['R_Score'], row['F_Score'], row['M_Score']

        if r >= 4 and f >= 4 and m >= 4:
            return 'Champions'
        elif r >= 3 and f >= 3:
            return 'Loyal Customers'
        elif r >= 4 and f <= 2:
            return 'Potential Loyalists'
        elif r == 5 and f == 1:
            return 'New Customers'
        elif r >= 3 and m >= 4:
            return 'Big Spenders'
        elif r <= 2 and f >= 3 and m >= 3:
            return 'At Risk'
        elif r <= 2 and f >= 4:
            return 'Cannot Lose Them'
        else:
            return 'Lost Customers'

    rfm['RFM_Segment'] = rfm.apply(segment_rule, axis=1)

    dist = rfm['RFM_Segment'].value_counts()
    logger.info("RFM Segment distribution:\n" + dist.to_string())

    return rfm


def calculate_clv(rfm: pd.DataFrame, avg_margin: float = 0.20, periods: int = 12) -> pd.DataFrame:
    """
    Calculate simplified Customer Lifetime Value (CLV) per customer.

    Formula: CLV = (Monetary / Frequency) * (Frequency / Recency_months) * avg_margin * periods

    Args:
        rfm: RFM DataFrame
        avg_margin: Average profit margin (default 20%)
        periods: Projection period in months (default 12)

    Returns:
        DataFrame with added 'CLV' column
    """
    rfm = rfm.copy()
    # Avoid division by zero
    rfm['Recency_months'] = rfm['Recency'].clip(lower=1) / 30
    rfm['Avg_Order_Value'] = rfm['Monetary'] / rfm['Frequency'].clip(lower=1)
    rfm['Purchase_Rate'] = rfm['Frequency'] / rfm['Recency_months'].clip(lower=1)
    rfm['CLV'] = (rfm['Avg_Order_Value'] * rfm['Purchase_Rate'] * avg_margin * periods).round(2)

    logger.info(f"CLV calculated — mean: £{rfm['CLV'].mean():.2f}, top 10%: £{rfm['CLV'].quantile(0.9):.2f}")
    return rfm