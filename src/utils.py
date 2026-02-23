"""
utils.py
--------
Utility functions: automated segmentation pipeline, business recommendations,
marketing strategy generator, and model persistence.
"""

import pandas as pd
import numpy as np
import pickle
import os
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


# ── Segment name mapping for ML clusters ──────────────────────────────────────

SEGMENT_DESCRIPTIONS = {
    'Champions': {
        'description': 'Bought recently, buy often, spend the most',
        'strategy': 'Reward them. Can be early adopters for new products. Promote them.',
        'channel': 'Email, Loyalty Program, VIP Access',
        'offer': 'Exclusive previews, loyalty points, premium membership',
        'frequency': 'Monthly engagement + event-based',
        'retention_risk': 'Low',
        'budget_priority': '⭐⭐⭐⭐⭐ Highest'
    },
    'Loyal Customers': {
        'description': 'Spend good money, respond to promotions',
        'strategy': 'Upsell higher-value products. Ask for reviews.',
        'channel': 'Email, SMS',
        'offer': 'Volume discounts, category expansion offers',
        'frequency': 'Bi-weekly',
        'retention_risk': 'Low-Medium',
        'budget_priority': '⭐⭐⭐⭐ High'
    },
    'Potential Loyalists': {
        'description': 'Recent customers, bought more than once',
        'strategy': 'Offer membership / loyalty program. Recommend other products.',
        'channel': 'Email, Push Notifications',
        'offer': '10-15% discount on second/third purchase, free shipping',
        'frequency': 'Weekly for first month',
        'retention_risk': 'Medium',
        'budget_priority': '⭐⭐⭐ Medium-High'
    },
    'New Customers': {
        'description': 'Recently bought for the first time',
        'strategy': 'Provide onboarding support, introduce to products.',
        'channel': 'Email, Welcome Series',
        'offer': 'Welcome discount (10%), how-to guides',
        'frequency': 'Welcome series — 3 emails in 2 weeks',
        'retention_risk': 'High (first-time)',
        'budget_priority': '⭐⭐⭐ Medium'
    },
    'Big Spenders': {
        'description': 'High monetary value, moderate frequency',
        'strategy': 'Focus on frequency increase. Premium product lines.',
        'channel': 'Email, Phone (top tier)',
        'offer': 'Premium bundles, limited editions, concierge service',
        'frequency': 'Monthly',
        'retention_risk': 'Medium',
        'budget_priority': '⭐⭐⭐⭐ High'
    },
    'At Risk': {
        'description': 'Spent big money, bought often, but long time ago',
        'strategy': 'Send personalised reactivation campaign. Offer to renew.',
        'channel': 'Email, SMS, Retargeting Ads',
        'offer': '20% win-back discount, "We miss you" campaign',
        'frequency': 'Immediate: 3-touch sequence over 2 weeks',
        'retention_risk': 'Very High',
        'budget_priority': '⭐⭐⭐⭐ High (win-back ROI)'
    },
    'Cannot Lose Them': {
        'description': 'Used to buy frequently but haven't returned',
        'strategy': 'Win them back via renewals or newer products.',
        'channel': 'Email, Direct Mail, Phone',
        'offer': '25-30% reactivation discount, personal account manager',
        'frequency': 'Urgent: immediate outreach',
        'retention_risk': 'Critical',
        'budget_priority': '⭐⭐⭐⭐⭐ Urgent Investment'
    },
    'Lost Customers': {
        'description': 'Lowest recency, frequency, and monetary scores',
        'strategy': 'Reactivate with compelling offer; if no response, let go.',
        'channel': 'Retargeting Ads, Email',
        'offer': 'Deep discount (30%), clearance items',
        'frequency': 'One final campaign; low spend',
        'retention_risk': 'Extreme',
        'budget_priority': '⭐ Minimal (low ROI)'
    }
}


def generate_business_report(rfm: pd.DataFrame, segment_col: str = 'RFM_Segment') -> pd.DataFrame:
    """
    Generate a business-level summary table per segment.

    Returns a DataFrame with revenue, customer count, avg CLV, and recommendations.
    """
    summary = rfm.groupby(segment_col).agg(
        Customer_Count=('CustomerID', 'count'),
        Total_Revenue=('Monetary', 'sum'),
        Avg_Revenue=('Monetary', 'mean'),
        Avg_Frequency=('Frequency', 'mean'),
        Avg_Recency=('Recency', 'mean'),
        Avg_CLV=('CLV', 'mean') if 'CLV' in rfm.columns else ('Monetary', 'mean'),
    ).round(2).reset_index()

    summary['Revenue_Share_Pct'] = (summary['Total_Revenue'] / summary['Total_Revenue'].sum() * 100).round(1)
    summary['Customer_Share_Pct'] = (summary['Customer_Count'] / summary['Customer_Count'].sum() * 100).round(1)

    # Merge segment descriptions
    if segment_col == 'RFM_Segment':
        summary['Marketing_Channel'] = summary[segment_col].map(
            lambda s: SEGMENT_DESCRIPTIONS.get(s, {}).get('channel', 'N/A')
        )
        summary['Offer_Strategy'] = summary[segment_col].map(
            lambda s: SEGMENT_DESCRIPTIONS.get(s, {}).get('offer', 'N/A')
        )
        summary['Budget_Priority'] = summary[segment_col].map(
            lambda s: SEGMENT_DESCRIPTIONS.get(s, {}).get('budget_priority', 'N/A')
        )

    summary = summary.sort_values('Total_Revenue', ascending=False)
    return summary


def name_ml_clusters(rfm: pd.DataFrame, cluster_col: str = 'KMeans_Cluster') -> pd.DataFrame:
    """
    Map numeric ML cluster IDs to meaningful business names based on RFM centroid characteristics.

    Args:
        rfm: DataFrame with cluster labels and R/F/M scores
        cluster_col: Column with cluster integer labels

    Returns:
        rfm DataFrame with added '{cluster_col}_Name' column
    """
    # Profile clusters by mean R, F, M scores
    profile = rfm.groupby(cluster_col)[['R_Score', 'F_Score', 'M_Score']].mean()

    name_map = {}
    for cluster_id, row in profile.iterrows():
        r, f, m = row['R_Score'], row['F_Score'], row['M_Score']
        total = r + f + m

        if total >= 12:
            name_map[cluster_id] = 'Champions'
        elif r >= 4 and f >= 3:
            name_map[cluster_id] = 'Loyal Customers'
        elif m >= 4 and f <= 2:
            name_map[cluster_id] = 'Big Spenders'
        elif r <= 2 and f >= 3:
            name_map[cluster_id] = 'At Risk'
        elif r >= 4 and f <= 2:
            name_map[cluster_id] = 'Potential Loyalists'
        else:
            name_map[cluster_id] = 'Lost Customers'

    rfm[f'{cluster_col}_Name'] = rfm[cluster_col].map(name_map)
    logger.info(f"Cluster name mapping:\n{name_map}")
    return rfm


def save_model(model, path: str):
    """Save a trained model to disk using pickle."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to {path}")


def load_model(path: str):
    """Load a model from disk."""
    with open(path, 'rb') as f:
        model = pickle.load(f)
    logger.info(f"Model loaded from {path}")
    return model


def automated_segmentation_pipeline(df_raw: pd.DataFrame,
                                     n_clusters: int = 4,
                                     features: list = None) -> pd.DataFrame:
    """
    End-to-end automated segmentation pipeline.
    Takes raw transaction data → returns customer segments.

    This function is PRODUCTION-READY — just pass in new data.

    Args:
        df_raw: Raw transactions DataFrame
        n_clusters: Number of K-Means clusters
        features: List of features to use (default: ['Recency', 'Frequency', 'Monetary'])

    Returns:
        DataFrame with CustomerID, RFM metrics, scores, segments, and ML cluster labels
    """
    from src.data_preprocessing import clean_data, get_snapshot_date
    from src.rfm_analysis import calculate_rfm, score_rfm, assign_rfm_segment, calculate_clv
    from src.clustering_models import scale_features, fit_kmeans
    from src.utils import name_ml_clusters

    if features is None:
        features = ['Recency', 'Frequency', 'Monetary']

    logger.info("=== AUTOMATED SEGMENTATION PIPELINE STARTED ===")

    # Step 1: Clean
    df_clean = clean_data(df_raw)

    # Step 2: RFM
    snapshot = get_snapshot_date(df_clean)
    rfm = calculate_rfm(df_clean, snapshot)
    rfm = score_rfm(rfm)
    rfm = assign_rfm_segment(rfm)
    rfm = calculate_clv(rfm)

    # Step 3: ML Clustering
    X_scaled, scaler = scale_features(rfm, features)
    km_model, labels = fit_kmeans(X_scaled, n_clusters=n_clusters)
    rfm['KMeans_Cluster'] = labels
    rfm = name_ml_clusters(rfm, cluster_col='KMeans_Cluster')

    logger.info("=== PIPELINE COMPLETE ===")
    return rfm