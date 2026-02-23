"""
clustering_models.py
--------------------
K-Means, Hierarchical, and DBSCAN clustering with evaluation metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage
from typing import Tuple, Dict, Optional
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


def scale_features(df: pd.DataFrame, features: list) -> Tuple[np.ndarray, StandardScaler]:
    """
    Normalize features using StandardScaler.

    Args:
        df: DataFrame containing feature columns
        features: List of column names to scale

    Returns:
        Tuple of (scaled_array, fitted_scaler)
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    logger.info(f"Scaled {len(features)} features: {features}")
    return X_scaled, scaler


def elbow_method(X: np.ndarray, k_range: range = range(2, 12),
                 save_path: Optional[str] = None) -> Dict[int, float]:
    """
    Run Elbow Method to find optimal K for K-Means.

    Args:
        X: Scaled feature array
        k_range: Range of K values to test
        save_path: Optional path to save the plot

    Returns:
        Dict mapping k → inertia
    """
    inertias = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        inertias[k] = km.inertia_

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(list(inertias.keys()), list(inertias.values()), 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Clusters (K)', fontsize=13)
    ax.set_ylabel('Inertia (WCSS)', fontsize=13)
    ax.set_title('Elbow Method — Optimal K Selection', fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Annotate the "elbow" region
    ax.axvline(x=4, color='red', linestyle='--', alpha=0.7, label='Suggested elbow (K=4)')
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Elbow plot saved to {save_path}")
    plt.show()

    return inertias


def silhouette_analysis(X: np.ndarray, k_range: range = range(2, 8),
                        save_path: Optional[str] = None) -> Dict[int, float]:
    """
    Run silhouette analysis across multiple K values.

    Args:
        X: Scaled feature array
        k_range: Range of K values to test
        save_path: Optional path to save the plot

    Returns:
        Dict mapping k → silhouette score
    """
    scores = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        score = silhouette_score(X, labels)
        scores[k] = round(score, 4)
        logger.info(f"K={k} — Silhouette Score: {score:.4f}")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(list(scores.keys()), list(scores.values()), color='steelblue', alpha=0.8, edgecolor='black')
    ax.set_xlabel('Number of Clusters (K)', fontsize=13)
    ax.set_ylabel('Silhouette Score', fontsize=13)
    ax.set_title('Silhouette Analysis — K Selection', fontsize=15, fontweight='bold')
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Score = 0.5 threshold')
    for k, s in scores.items():
        ax.text(k, s + 0.005, f'{s:.3f}', ha='center', fontsize=10)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    return scores


def fit_kmeans(X: np.ndarray, n_clusters: int, random_state: int = 42) -> Tuple[KMeans, np.ndarray]:
    """
    Fit K-Means with the optimal number of clusters.

    Args:
        X: Scaled feature array
        n_clusters: Number of clusters
        random_state: Random seed

    Returns:
        Tuple of (fitted KMeans model, cluster labels)
    """
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10, max_iter=300)
    labels = km.fit_predict(X)
    score = silhouette_score(X, labels)
    db = davies_bouldin_score(X, labels)
    ch = calinski_harabasz_score(X, labels)

    logger.info(f"K-Means (K={n_clusters}) Results:")
    logger.info(f"  Silhouette Score:      {score:.4f}  (higher is better, >0.5 = good)")
    logger.info(f"  Davies-Bouldin Index:  {db:.4f}  (lower is better)")
    logger.info(f"  Calinski-Harabasz:     {ch:.2f}  (higher is better)")

    return km, labels


def fit_hierarchical(X: np.ndarray, n_clusters: int,
                     linkage_method: str = 'ward') -> Tuple[AgglomerativeClustering, np.ndarray]:
    """
    Fit Agglomerative Hierarchical Clustering.

    Args:
        X: Scaled feature array
        n_clusters: Number of clusters
        linkage_method: Linkage criterion ('ward', 'complete', 'average', 'single')

    Returns:
        Tuple of (fitted model, cluster labels)
    """
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
    labels = model.fit_predict(X)
    score = silhouette_score(X, labels)
    db = davies_bouldin_score(X, labels)

    logger.info(f"Hierarchical Clustering (linkage={linkage_method}, K={n_clusters}):")
    logger.info(f"  Silhouette Score:     {score:.4f}")
    logger.info(f"  Davies-Bouldin Index: {db:.4f}")

    return model, labels


def fit_dbscan(X: np.ndarray, eps: float = 0.5,
               min_samples: int = 5) -> Tuple[DBSCAN, np.ndarray]:
    """
    Fit DBSCAN clustering. Noise points get label -1.

    Args:
        X: Scaled feature array
        eps: Neighborhood radius
        min_samples: Min points to form a core point

    Returns:
        Tuple of (fitted DBSCAN model, cluster labels)
    """
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()

    logger.info(f"DBSCAN (eps={eps}, min_samples={min_samples}):")
    logger.info(f"  Clusters found:  {n_clusters}")
    logger.info(f"  Noise points:    {n_noise} ({n_noise / len(labels) * 100:.1f}%)")

    if n_clusters > 1:
        # Exclude noise for silhouette score
        mask = labels != -1
        score = silhouette_score(X[mask], labels[mask])
        logger.info(f"  Silhouette Score (excl. noise): {score:.4f}")

    return model, labels


def plot_dendrogram(X: np.ndarray, method: str = 'ward',
                    n_sample: int = 500, save_path: Optional[str] = None):
    """
    Plot hierarchical clustering dendrogram on a random sample.

    Args:
        X: Scaled feature array
        method: Linkage method
        n_sample: Number of samples to use (full dataset is slow)
        save_path: Optional path to save plot
    """
    np.random.seed(42)
    idx = np.random.choice(len(X), min(n_sample, len(X)), replace=False)
    X_sample = X[idx]

    Z = linkage(X_sample, method=method)

    fig, ax = plt.subplots(figsize=(14, 6))
    dendrogram(Z, ax=ax, truncate_mode='lastp', p=30,
               leaf_rotation=45, leaf_font_size=10,
               show_contracted=True, color_threshold=0.7 * max(Z[:, 2]))

    ax.set_title(f'Hierarchical Clustering Dendrogram (method={method}, n={n_sample})',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Customer Cluster Index', fontsize=12)
    ax.set_ylabel('Distance', fontsize=12)
    ax.axhline(y=8, color='red', linestyle='--', linewidth=1.5, label='Cut line (K≈4)')
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def apply_pca(X: np.ndarray, n_components: int = 3) -> Tuple[np.ndarray, PCA]:
    """
    Apply PCA for dimensionality reduction and visualization.

    Args:
        X: Scaled feature array
        n_components: Number of principal components

    Returns:
        Tuple of (transformed array, fitted PCA model)
    """
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)

    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)
    logger.info(f"PCA ({n_components} components):")
    for i, (exp, cum) in enumerate(zip(explained, cumulative)):
        logger.info(f"  PC{i+1}: {exp*100:.2f}% variance (cumulative: {cum*100:.2f}%)")

    return X_pca, pca


def evaluate_all_models(X: np.ndarray, kmeans_labels: np.ndarray,
                        hierarchical_labels: np.ndarray,
                        dbscan_labels: Optional[np.ndarray] = None) -> pd.DataFrame:
    """
    Create a comparison table of all clustering models.

    Args:
        X: Scaled feature array
        kmeans_labels: K-Means cluster labels
        hierarchical_labels: Hierarchical cluster labels
        dbscan_labels: DBSCAN cluster labels (optional)

    Returns:
        Comparison DataFrame
    """
    results = []

    for name, labels in [('K-Means', kmeans_labels), ('Hierarchical', hierarchical_labels)]:
        results.append({
            'Model': name,
            'N_Clusters': len(np.unique(labels)),
            'Silhouette_Score': round(silhouette_score(X, labels), 4),
            'Davies_Bouldin': round(davies_bouldin_score(X, labels), 4),
            'Calinski_Harabasz': round(calinski_harabasz_score(X, labels), 2)
        })

    if dbscan_labels is not None:
        mask = dbscan_labels != -1
        n_clusters = len(set(dbscan_labels[mask]))
        if n_clusters > 1:
            results.append({
                'Model': 'DBSCAN',
                'N_Clusters': n_clusters,
                'Silhouette_Score': round(silhouette_score(X[mask], dbscan_labels[mask]), 4),
                'Davies_Bouldin': round(davies_bouldin_score(X[mask], dbscan_labels[mask]), 4),
                'Calinski_Harabasz': round(calinski_harabasz_score(X[mask], dbscan_labels[mask]), 2)
            })

    df_results = pd.DataFrame(results)
    logger.info("\n=== Model Comparison ===\n" + df_results.to_string(index=False))
    return df_results