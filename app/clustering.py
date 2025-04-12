import logging
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import (
    DBSCAN,
    HDBSCAN,
    AgglomerativeClustering,
    MiniBatchKMeans,
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import normalize

try:
    import umap
except ImportError:
    umap = None
    logging.warning(
        "umap-learn library not found. UMAP functionality will be unavailable. Install with 'pip install umap-learn'"
    )


from app.exceptions import DatabaseError, PipelineError
from core.vectorizer import Vectorizer
from data_access.vector_db_interface import VectorDBInterface

logger = logging.getLogger(__name__)


def run_minibatch_kmeans(
    embeddings: np.ndarray,
    n_clusters: int,
    batch_size: int = 1024,
    n_init: int = 3,
    **kwargs,
) -> Tuple[np.ndarray, float]:

    if not isinstance(embeddings, np.ndarray) or embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D NumPy array.")
    if embeddings.shape[0] < n_clusters:
        raise ValueError(
            f"Number of samples ({embeddings.shape[0]}) must be >= number of clusters ({n_clusters})"
        )

    logger.info(
        f"Running MiniBatchKMeans (k={n_clusters}, batch_size={batch_size}, n_init={n_init})..."
    )
    start_time = time.time()
    embeddings_normalized = normalize(embeddings, norm="l2", axis=1)
    logger.debug("Embeddings normalized (L2) for MiniBatchKMeans.")
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=42,
        batch_size=batch_size,
        n_init=n_init,
        max_iter=300,
        verbose=0,
        **kwargs,
    )
    labels = kmeans.fit_predict(embeddings_normalized)
    inertia = kmeans.inertia_
    end_time = time.time()
    logger.info(
        f"MiniBatchKMeans completed in {end_time - start_time:.2f}s. Inertia: {inertia:.4f}"
    )
    return labels, inertia


def run_hdbscan(
    embeddings: np.ndarray,
    min_cluster_size: int = 5,
    min_samples: Optional[int] = None,
    metric: str = "euclidean",
    **kwargs,
) -> Tuple[np.ndarray, Dict[str, Any]]:

    if not isinstance(embeddings, np.ndarray) or embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D NumPy array.")
    logger.info(
        f"Running HDBSCAN (min_cluster_size={min_cluster_size}, min_samples={min_samples}, metric='{metric}')..."
    )
    start_time = time.time()
    hdb = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        **kwargs,
    )
    labels = hdb.fit_predict(embeddings)
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    num_noise = np.sum(labels == -1)
    stats = {"num_clusters": num_clusters, "num_noise_points": num_noise}
    end_time = time.time()
    logger.info(
        f"HDBSCAN completed in {end_time - start_time:.2f}s. Clusters found: {num_clusters}, Noise points: {num_noise}"
    )
    return labels, stats


def apply_pca(embeddings: np.ndarray, n_components: int = 2) -> np.ndarray:

    if not isinstance(embeddings, np.ndarray) or embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D NumPy array.")
    if n_components >= embeddings.shape[1]:
        logger.warning(
            f"PCA n_components ({n_components}) >= original dim ({embeddings.shape[1]}). Returning original embeddings."
        )
        return embeddings
    if n_components <= 0:
        raise ValueError("PCA n_components must be positive.")
    logger.info(
        f"Applying PCA to reduce from {embeddings.shape[1]} to {n_components} dimensions..."
    )
    start_time = time.time()
    pca = PCA(n_components=n_components, random_state=42)
    reduced_embeddings = pca.fit_transform(embeddings)
    explained_variance = np.sum(pca.explained_variance_ratio_)
    end_time = time.time()
    logger.info(
        f"PCA completed in {end_time - start_time:.2f}s. Explained variance: {explained_variance:.4f}"
    )
    return reduced_embeddings


def apply_umap(
    embeddings: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    **kwargs,
) -> np.ndarray:

    if umap is None:
        raise ImportError("umap-learn library is not installed.")
    if not isinstance(embeddings, np.ndarray) or embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D NumPy array.")
    if n_components >= embeddings.shape[1]:
        logger.warning(
            f"UMAP n_components ({n_components}) >= original dim ({embeddings.shape[1]}). Returning original embeddings."
        )
        return embeddings
    if n_components <= 0:
        raise ValueError("UMAP n_components must be positive.")
    logger.info(
        f"Applying UMAP (n_components={n_components}, n_neighbors={n_neighbors}, min_dist={min_dist}, metric='{metric}')..."
    )
    start_time = time.time()
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=42,
        verbose=False,
        **kwargs,
    )
    reduced_embeddings = reducer.fit_transform(embeddings)
    end_time = time.time()
    logger.info(f"UMAP completed in {end_time - start_time:.2f}s.")
    return reduced_embeddings


def apply_tsne(
    embeddings: np.ndarray,
    n_components: int = 2,
    perplexity: float = 30.0,
    n_iter: int = 1000,
    metric: str = "cosine",
    **kwargs,
) -> np.ndarray:

    if not isinstance(embeddings, np.ndarray) or embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D NumPy array.")
    if n_components >= embeddings.shape[1]:
        logger.warning(
            f"t-SNE n_components ({n_components}) >= original dim ({embeddings.shape[1]}). Returning original embeddings."
        )
        return embeddings
    if n_components <= 0:
        raise ValueError("t-SNE n_components must be positive.")
    if embeddings.shape[0] > 5000:
        logger.warning(
            f"t-SNE with {embeddings.shape[0]} data points can be very slow."
        )
    logger.info(
        f"Applying t-SNE (n_components={n_components}, perplexity={perplexity}, n_iter={n_iter}, metric='{metric}')..."
    )
    start_time = time.time()
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        n_iter=n_iter,
        metric=metric,
        random_state=42,
        verbose=1,
        init="pca",
        learning_rate="auto",
        **kwargs,
    )
    reduced_embeddings = tsne.fit_transform(embeddings)
    end_time = time.time()
    logger.info(f"t-SNE completed in {end_time - start_time:.2f}s.")
    return reduced_embeddings


def calculate_internal_metrics(
    embeddings: np.ndarray, labels: np.ndarray, metric: str = "cosine"
) -> Dict[str, Optional[float]]:

    metrics = {
        "silhouette": None,
        "davies_bouldin": None,
        "calinski_harabasz": None,
    }
    if not isinstance(embeddings, np.ndarray) or embeddings.ndim != 2:
        logger.error("Invalid embeddings format.")
        return metrics
    if not isinstance(labels, np.ndarray) or labels.ndim != 1:
        logger.error("Invalid labels format.")
        return metrics
    if embeddings.shape[0] != labels.shape[0]:
        logger.error("Embeddings and labels count mismatch.")
        return metrics
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    if n_clusters < 2:
        logger.warning(
            f"Need >= 2 clusters (found {n_clusters}) for internal metrics."
        )
        return metrics
    non_noise_mask = labels != -1
    if not np.any(non_noise_mask):
        logger.warning("No non-noise points for metrics.")
        return metrics
    embeddings_filtered = embeddings[non_noise_mask]
    labels_filtered = labels[non_noise_mask]
    n_clusters_filtered = len(np.unique(labels_filtered))
    if embeddings_filtered.shape[0] < 2 or n_clusters_filtered < 2:
        logger.warning("Insufficient filtered data for metrics.")
        return metrics
    logger.info(
        f"Calculating internal metrics for {n_clusters_filtered} clusters ({embeddings_filtered.shape[0]} points)..."
    )
    try:
        embeddings_for_silhouette = embeddings_filtered
        if metric == "cosine":
            norms = np.linalg.norm(embeddings_filtered, axis=1)
            if not np.allclose(norms, 1.0):
                embeddings_for_silhouette = normalize(
                    embeddings_filtered, norm="l2", axis=1
                )
        sample_size = min(5000, embeddings_for_silhouette.shape[0])
        metrics["silhouette"] = silhouette_score(
            embeddings_for_silhouette,
            labels_filtered,
            metric=metric,
            sample_size=sample_size,
            random_state=42,
        )
    except Exception as e:
        logger.warning(f"Silhouette calc failed: {e}")
    try:
        metrics["davies_bouldin"] = davies_bouldin_score(
            embeddings_filtered, labels_filtered
        )
    except Exception as e:
        logger.warning(f"Davies-Bouldin calc failed: {e}")
    try:
        metrics["calinski_harabasz"] = calinski_harabasz_score(
            embeddings_filtered, labels_filtered
        )
    except Exception as e:
        logger.warning(f"Calinski-Harabasz calc failed: {e}")
    logger.info(f"Internal metrics calculated: {metrics}")
    return metrics
