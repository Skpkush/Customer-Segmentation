"""
visualization.py
----------------
All plotting functions for the customer segmentation project.
Covers RFM distributions, cluster plots, segment profiling, and business insights.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from typing import Optional, List
import warnings
warnings.filterwarnings('ignore')

# ── Global style ──────────────────────────────────────────────────────────────
PALETTE = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B',
           '#44BBA4', '#E94F37', '#393E41']
plt.rcParams.update({'figure.dpi': 120, 'font.family': 'DejaVu Sans'})


# ── 1. RFM Distribution Plots ──────────────────────────────────────────────────

def plot_rfm_distributions(rfm: pd.DataFrame, save_path: Optional[str] = None):
    """Histograms and KDE plots for raw R, F, M values."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = ['Recency', 'Frequency', 'Monetary']
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    units = ['Days since last purchase', 'Number of orders', 'Total spend (£)']

    for ax, col, color, unit in zip(axes, metrics, colors, units):
        # Use log scale for skewed data
        data = np.log1p(rfm[col]) if col in ['Frequency', 'Monetary'] else rfm[col]
        ax.hist(data, bins=50, color=color, alpha=0.75, edgecolor='white', linewidth=0.5)
        ax.set_title(f'{col} Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel(f'log({unit})' if col in ['Frequency', 'Monetary'] else unit, fontsize=11)
        ax.set_ylabel('Customer Count', fontsize=11)
        ax.axvline(data.median(), color='red', linestyle='--', linewidth=1.5,
                   label=f'Median: {rfm[col].median():.0f}')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.suptitle('RFM Metric Distributions', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_rfm_boxplots(rfm: pd.DataFrame, save_path: Optional[str] = None):
    """Box plots for outlier detection in RFM metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = ['Recency', 'Frequency', 'Monetary']

    for ax, col in zip(axes, metrics):
        ax.boxplot(rfm[col], patch_artist=True,
                   boxprops=dict(facecolor='#2E86AB', alpha=0.6),
                   medianprops=dict(color='red', linewidth=2))
        ax.set_title(f'{col} — Outlier Check', fontsize=13, fontweight='bold')
        ax.set_ylabel(col, fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')

        # Annotate quartiles
        q1, median, q3 = rfm[col].quantile([0.25, 0.5, 0.75])
        ax.text(1.12, median, f'Median: {median:.0f}', va='center', fontsize=9, color='red')

    plt.suptitle('RFM Outlier Detection (Box Plots)', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_correlation_heatmap(rfm: pd.DataFrame, features: List[str],
                              save_path: Optional[str] = None):
    """Correlation heatmap for RFM features."""
    corr = rfm[features].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.3f',
                cmap='RdYlGn', center=0, vmin=-1, vmax=1,
                ax=ax, linewidths=0.5, annot_kws={'size': 12})
    ax.set_title('RFM Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# ── 2. Segment Distribution Plots ─────────────────────────────────────────────

def plot_segment_treemap(rfm: pd.DataFrame, segment_col: str = 'RFM_Segment',
                         save_path: Optional[str] = None):
    """Treemap showing proportion of customers per segment (using matplotlib patches)."""
    counts = rfm[segment_col].value_counts()
    total = counts.sum()
    labels = [f"{s}\n{c:,}\n({c/total*100:.1f}%)" for s, c in counts.items()]

    # Simple rectangle treemap using matplotlib
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    colors = PALETTE[:len(counts)]
    x, y, row_h = 0, 0, 0
    total_w = 100

    for i, (label, count) in enumerate(zip(labels, counts.values)):
        w = (count / total) * total_w
        if x + w > total_w:
            y += row_h
            x, row_h = 0, 0
        h = 35 if y < 50 else 35
        rect = FancyBboxPatch((x + 0.5, y + 0.5), w - 1, h - 1,
                              boxstyle="round,pad=0.3", facecolor=colors[i % len(colors)],
                              alpha=0.85, edgecolor='white', linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha='center', va='center',
                fontsize=9, fontweight='bold', color='white', wrap=True)
        x += w
        row_h = max(row_h, h)

    ax.set_title(f'Customer Segment Treemap — {segment_col}', fontsize=15, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_segment_bar(rfm: pd.DataFrame, segment_col: str = 'RFM_Segment',
                     metric: str = 'Monetary', agg: str = 'sum',
                     save_path: Optional[str] = None):
    """Bar chart of a metric aggregated by segment."""
    grouped = rfm.groupby(segment_col)[metric].agg(agg).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(grouped.index, grouped.values,
                  color=PALETTE[:len(grouped)], edgecolor='black', linewidth=0.5)

    ax.set_title(f'{agg.capitalize()} {metric} by {segment_col}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Segment', fontsize=12)
    ax.set_ylabel(f'{agg.capitalize()} {metric}', fontsize=12)
    ax.tick_params(axis='x', rotation=30)
    ax.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, grouped.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.02,
                f'£{val:,.0f}' if metric == 'Monetary' else f'{val:,.0f}',
                ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# ── 3. Cluster Scatter Plots ───────────────────────────────────────────────────

def plot_2d_clusters(rfm: pd.DataFrame, x_col: str, y_col: str,
                     cluster_col: str, save_path: Optional[str] = None):
    """2D scatter plot colored by cluster."""
    fig, ax = plt.subplots(figsize=(10, 7))
    clusters = sorted(rfm[cluster_col].unique())

    for i, cluster in enumerate(clusters):
        mask = rfm[cluster_col] == cluster
        ax.scatter(rfm.loc[mask, x_col], rfm.loc[mask, y_col],
                   c=PALETTE[i % len(PALETTE)], label=f'Cluster {cluster}',
                   alpha=0.6, s=30, edgecolors='none')

    ax.set_xlabel(x_col, fontsize=12)
    ax.set_ylabel(y_col, fontsize=12)
    ax.set_title(f'Clusters: {x_col} vs {y_col}', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_3d_clusters_matplotlib(X_pca: np.ndarray, labels: np.ndarray,
                                 title: str = '3D PCA Cluster Plot',
                                 save_path: Optional[str] = None):
    """3D scatter plot of PCA-reduced clusters using Matplotlib."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    unique_labels = np.unique(labels)

    for i, label in enumerate(unique_labels):
        if label == -1:
            ax.scatter(X_pca[labels == label, 0], X_pca[labels == label, 1],
                       X_pca[labels == label, 2], c='grey', alpha=0.3, s=10, label='Noise')
        else:
            ax.scatter(X_pca[labels == label, 0], X_pca[labels == label, 1],
                       X_pca[labels == label, 2], c=PALETTE[i % len(PALETTE)],
                       alpha=0.6, s=30, label=f'Cluster {label}')

    ax.set_xlabel('PC1', fontsize=11)
    ax.set_ylabel('PC2', fontsize=11)
    ax.set_zlabel('PC3', fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# ── 4. Radar Chart for Segment Profiling ──────────────────────────────────────

def plot_radar_chart(cluster_profiles: pd.DataFrame, metrics: List[str],
                     save_path: Optional[str] = None):
    """
    Radar chart showing normalized metric profiles for each cluster.

    Args:
        cluster_profiles: DataFrame indexed by cluster name, columns = metrics
        metrics: List of metric column names to include
        save_path: Optional save path
    """
    # Normalize 0-1
    normalized = cluster_profiles[metrics].copy()
    for col in metrics:
        col_min, col_max = normalized[col].min(), normalized[col].max()
        if col_max > col_min:
            normalized[col] = (normalized[col] - col_min) / (col_max - col_min)

    n_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))

    for i, (segment, row) in enumerate(normalized.iterrows()):
        values = row[metrics].tolist()
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2,
                color=PALETTE[i % len(PALETTE)], label=str(segment))
        ax.fill(angles, values, alpha=0.15, color=PALETTE[i % len(PALETTE)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title('Segment Profile Radar Chart\n(Normalized Metrics)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', bbox_to_anchor=(1.3, -0.1), fontsize=10)
    ax.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# ── 5. Segment × Metrics Heatmap ──────────────────────────────────────────────

def plot_segment_heatmap(cluster_profiles: pd.DataFrame, metrics: List[str],
                         save_path: Optional[str] = None):
    """Heatmap of cluster profiles — segments as rows, metrics as columns."""
    normalized = cluster_profiles[metrics].copy()
    for col in metrics:
        col_min, col_max = normalized[col].min(), normalized[col].max()
        if col_max > col_min:
            normalized[col] = (normalized[col] - col_min) / (col_max - col_min)

    fig, ax = plt.subplots(figsize=(max(10, len(metrics) * 1.5), len(cluster_profiles) * 0.8 + 2))
    sns.heatmap(normalized, annot=True, fmt='.2f', cmap='YlOrRd',
                ax=ax, linewidths=0.5, cbar_kws={'label': 'Normalized Value'},
                annot_kws={'size': 10})
    ax.set_title('Cluster Profile Heatmap (Normalized)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Segment', fontsize=12)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# ── 6. PCA Explained Variance ─────────────────────────────────────────────────

def plot_pca_variance(pca, save_path: Optional[str] = None):
    """Scree plot for PCA explained variance."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Individual variance
    axes[0].bar(range(1, len(pca.explained_variance_ratio_) + 1),
                pca.explained_variance_ratio_ * 100,
                color='steelblue', alpha=0.8, edgecolor='black')
    axes[0].set_title('PCA — Individual Explained Variance', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Principal Component', fontsize=11)
    axes[0].set_ylabel('Explained Variance (%)', fontsize=11)
    axes[0].grid(True, alpha=0.3, axis='y')

    # Cumulative variance
    cumulative = np.cumsum(pca.explained_variance_ratio_) * 100
    axes[1].plot(range(1, len(cumulative) + 1), cumulative, 'bo-', linewidth=2, markersize=8)
    axes[1].axhline(y=80, color='red', linestyle='--', label='80% threshold')
    axes[1].set_title('PCA — Cumulative Explained Variance', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Principal Components', fontsize=11)
    axes[1].set_ylabel('Cumulative Variance (%)', fontsize=11)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# ── 7. Cohort Analysis ─────────────────────────────────────────────────────────

def plot_cohort_heatmap(df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Build and visualize a cohort retention heatmap.

    Args:
        df: Cleaned transactions DataFrame with CustomerID and InvoiceDate
        save_path: Optional save path
    """
    df = df.copy()
    df['CohortMonth'] = df.groupby('CustomerID')['InvoiceDate'].transform('min').dt.to_period('M')
    df['InvoiceMonth'] = df['InvoiceDate'].dt.to_period('M')
    df['CohortIndex'] = (df['InvoiceMonth'] - df['CohortMonth']).apply(lambda x: x.n)

    cohort_data = df.groupby(['CohortMonth', 'CohortIndex'])['CustomerID'].nunique().reset_index()
    cohort_pivot = cohort_data.pivot_table(index='CohortMonth', columns='CohortIndex', values='CustomerID')

    # Retention rate
    cohort_size = cohort_pivot.iloc[:, 0]
    retention = cohort_pivot.divide(cohort_size, axis=0) * 100

    # Plot (limit to first 12 months for readability)
    retention_display = retention.iloc[:, :13]

    fig, ax = plt.subplots(figsize=(16, 8))
    sns.heatmap(retention_display, annot=True, fmt='.0f', cmap='YlOrRd_r',
                ax=ax, linewidths=0.3, cbar_kws={'label': 'Retention Rate (%)'},
                annot_kws={'size': 8})
    ax.set_title('Customer Cohort Retention Heatmap (%)', fontsize=15, fontweight='bold')
    ax.set_xlabel('Months Since First Purchase', fontsize=12)
    ax.set_ylabel('Cohort (First Purchase Month)', fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    return retention