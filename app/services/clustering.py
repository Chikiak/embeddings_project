# --- app/services/clustering.py ---
import logging
import time
from typing import Any, Dict, Optional, Tuple, List # Añadir List
import numpy as np
import pandas as pd
from sklearn.cluster import (
    # DBSCAN, # No usado actualmente
    # AgglomerativeClustering, # No usado actualmente
    HDBSCAN,
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

# --- NUEVAS IMPORTACIONES ---
import config # Para acceder a las etiquetas predefinidas y nombres base
from app.exceptions import DatabaseError, PipelineError, VectorizerError
from core.factory import get_or_create_text_db # Importar la nueva función del factory
# Importar la lista de etiquetas predefinidas desde el nuevo archivo
from data_access.text_data.text_labels import PREDEFINED_LABELS
from core.vectorizer import Vectorizer
from data_access.vector_db_interface import VectorDBInterface
# --- FIN NUEVAS IMPORTACIONES ---

logger = logging.getLogger(__name__)


def run_minibatch_kmeans(
    embeddings: np.ndarray,
    n_clusters: int,
    batch_size: int = 1024,
    n_init: int = 3,
    **kwargs,
) -> Tuple[np.ndarray, float]:
    """
    Ejecuta MiniBatchKMeans sobre los embeddings proporcionados.

    Normaliza los embeddings (L2) antes de ejecutar, asumiendo que son para similitud coseno.

    Args:
        embeddings: Array NumPy 2D de embeddings.
        n_clusters: Número deseado de clusters (k).
        batch_size: Tamaño del lote para MiniBatchKMeans.
        n_init: Número de inicializaciones a probar.
        **kwargs: Argumentos adicionales para MiniBatchKMeans.

    Returns:
        Una tupla (labels, inertia), donde labels es un array NumPy de etiquetas de cluster
        e inertia es la inercia final del modelo.

    Raises:
        ValueError: Si los embeddings son inválidos o n_clusters es mayor que el número de muestras.
    """
    if not isinstance(embeddings, np.ndarray) or embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D NumPy array.")
    if embeddings.shape[0] == 0:
         raise ValueError("Cannot run KMeans on empty embeddings array.")
    if n_clusters <= 0:
         raise ValueError("Number of clusters (k) must be positive.")
    if embeddings.shape[0] < n_clusters:
        # Adjust k if necessary instead of raising an error, or log a warning
        logger.warning(
            f"Number of samples ({embeddings.shape[0]}) is less than number of clusters ({n_clusters}). "
            f"Adjusting k to {embeddings.shape[0]}."
        )
        n_clusters = embeddings.shape[0]
        if n_clusters <= 0 : # If after adjustment k is 0 or less (edge case)
             raise ValueError("Adjusted number of clusters is non-positive.")


    logger.info(
        f"Running MiniBatchKMeans (k={n_clusters}, batch_size={batch_size}, n_init={n_init})..."
    )
    start_time = time.time()
    # Normalize embeddings so Euclidean distance in KMeans approximates Cosine similarity
    embeddings_normalized = normalize(embeddings, norm="l2", axis=1)
    logger.debug("Embeddings normalized (L2) for MiniBatchKMeans.")

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=42, # For reproducibility
        batch_size=batch_size,
        n_init=n_init,
        max_iter=300, # Iteration limit per run
        verbose=0, # Do not show scikit-learn logs
        # compute_labels=True, # Default True
        # init='k-means++', # Default
        # max_no_improvement=10, # Default
        **kwargs, # Pass other arguments if any
    )
    labels = kmeans.fit_predict(embeddings_normalized)
    # Inertia is calculated on the normalized data
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
    """
    Ejecuta HDBSCAN sobre los embeddings proporcionados.

    Args:
        embeddings: Array NumPy 2D de embeddings.
        min_cluster_size: Tamaño mínimo para considerar un grupo como cluster.
        min_samples: Número de muestras en una vecindad para que un punto sea considerado core point.
                     Si es None, se usa min_cluster_size.
        metric: Métrica de distancia a usar ('cosine', 'euclidean', etc.).
        **kwargs: Argumentos adicionales para HDBSCAN.

    Returns:
        Una tupla (labels, stats), donde labels es un array NumPy de etiquetas de cluster (-1 para ruido)
        y stats es un diccionario con 'num_clusters' y 'num_noise_points'.

    Raises:
        ValueError: Si los embeddings son inválidos.
        ImportError: Si HDBSCAN no está instalado.
    """
    try:
        from sklearn.cluster import HDBSCAN # Import here to check installation
    except ImportError:
        logger.error("HDBSCAN requires scikit-learn >= 0.24 (approx). Please check your installation.")
        raise ImportError("HDBSCAN not available. Check scikit-learn version.")

    if not isinstance(embeddings, np.ndarray) or embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D NumPy array.")
    if embeddings.shape[0] == 0:
         raise ValueError("Cannot run HDBSCAN on empty embeddings array.")

    logger.info(
        f"Running HDBSCAN (min_cluster_size={min_cluster_size}, min_samples={min_samples or min_cluster_size}, metric='{metric}')..."
    )
    start_time = time.time()

    # HDBSCAN can be sensitive to scale depending on the metric,
    # but is generally applied to the data as is.
    # If the metric is 'cosine', it internally handles normalization or angular distances.
    hdb = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples, # If None, HDBSCAN defaults to min_cluster_size
        metric=metric,
        # core_dist_n_jobs=-1, # Use all cores if possible (may vary)
        # cluster_selection_epsilon=0.0, # Default
        # allow_single_cluster=False, # Default False
        **kwargs,
    )
    labels = hdb.fit_predict(embeddings)

    # Calculate statistics
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    num_noise = np.sum(labels == -1)
    stats = {"num_clusters": num_clusters, "num_noise_points": num_noise}

    end_time = time.time()
    logger.info(
        f"HDBSCAN completed in {end_time - start_time:.2f}s. Clusters found: {num_clusters}, Noise points: {num_noise}"
    )
    return labels, stats

# --- Dimensionality Reduction Functions (PCA, UMAP, t-SNE) ---

def apply_pca(embeddings: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Aplica PCA para reducir la dimensionalidad."""
    if not isinstance(embeddings, np.ndarray) or embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D NumPy array.")
    if embeddings.shape[0] < n_components:
         raise ValueError(f"Number of samples ({embeddings.shape[0]}) must be >= n_components ({n_components}) for PCA.")
    if n_components <= 0:
        raise ValueError("PCA n_components must be positive.")
    # If n_components is >= original dimension, PCA is meaningless
    if n_components >= embeddings.shape[1]:
        logger.warning(
            f"PCA n_components ({n_components}) >= original dim ({embeddings.shape[1]}). Returning original embeddings."
        )
        return embeddings

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
    """Aplica UMAP para reducir la dimensionalidad."""
    if umap is None:
        raise ImportError("umap-learn library is not installed. Cannot use apply_umap.")
    if not isinstance(embeddings, np.ndarray) or embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D NumPy array.")
    if embeddings.shape[0] < n_components:
         raise ValueError(f"Number of samples ({embeddings.shape[0]}) must be >= n_components ({n_components}) for UMAP.")
    if n_components <= 0:
        raise ValueError("UMAP n_components must be positive.")
    # If n_components is >= original dimension, UMAP is meaningless
    if n_components >= embeddings.shape[1]:
        logger.warning(
            f"UMAP n_components ({n_components}) >= original dim ({embeddings.shape[1]}). Returning original embeddings."
        )
        return embeddings
    # n_neighbors must be less than the number of samples
    if embeddings.shape[0] <= n_neighbors:
        logger.warning(f"n_neighbors ({n_neighbors}) >= number of samples ({embeddings.shape[0]}). Reducing n_neighbors to {max(1, embeddings.shape[0] - 1)}.")
        n_neighbors = max(1, embeddings.shape[0] - 1)


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
        verbose=False, # Controls UMAP logs
        # low_memory=True, # Consider if memory issues arise
        n_jobs=-1, # Use all available cores
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
    """Aplica t-SNE para reducir la dimensionalidad."""
    if not isinstance(embeddings, np.ndarray) or embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D NumPy array.")
    if embeddings.shape[0] < n_components:
         raise ValueError(f"Number of samples ({embeddings.shape[0]}) must be >= n_components ({n_components}) for t-SNE.")
    if n_components <= 0:
        raise ValueError("t-SNE n_components must be positive.")
    # If n_components is >= original dimension
    if n_components >= embeddings.shape[1]:
        logger.warning(
            f"t-SNE n_components ({n_components}) >= original dim ({embeddings.shape[1]}). Returning original embeddings."
        )
        return embeddings
    # Perplexity must be less than the number of samples
    if embeddings.shape[0] <= perplexity:
        logger.warning(f"Perplexity ({perplexity}) >= number of samples ({embeddings.shape[0]}). Reducing perplexity to {max(1.0, embeddings.shape[0] - 1.0)}.")
        perplexity = max(1.0, embeddings.shape[0] - 1.0)

    if embeddings.shape[0] > 5000:
        logger.warning(
            f"t-SNE with {embeddings.shape[0]} data points can be very slow and memory intensive."
        )

    logger.info(
        f"Applying t-SNE (n_components={n_components}, perplexity={perplexity}, n_iter={n_iter}, metric='{metric}')..."
    )
    start_time = time.time()
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        n_iter=n_iter,
        metric=metric, # 'cosine' or 'euclidean' are common
        random_state=42,
        verbose=1, # Show t-SNE progress
        init='pca', # Use PCA for initialization (more stable)
        learning_rate='auto', # Recommended in recent versions
        n_jobs=-1, # Use all cores
        **kwargs,
    )
    reduced_embeddings = tsne.fit_transform(embeddings)
    end_time = time.time()
    logger.info(f"t-SNE completed in {end_time - start_time:.2f}s.")
    return reduced_embeddings


# --- Internal Evaluation Function ---

def calculate_internal_metrics(
    embeddings: np.ndarray, labels: np.ndarray, metric: str = "cosine"
) -> Dict[str, Optional[float]]:
    """
    Calcula métricas internas de clustering (Silhouette, Davies-Bouldin, Calinski-Harabasz).

    Excluye puntos etiquetados como ruido (-1). Requiere al menos 2 clusters válidos.

    Args:
        embeddings: Array NumPy 2D de los embeddings usados para el clustering.
        labels: Array NumPy 1D con las etiquetas de cluster asignadas.
        metric: Métrica de distancia usada para Silhouette ('cosine', 'euclidean').

    Returns:
        Un diccionario con los scores calculados. Los valores son None si no se pueden calcular.
    """
    metrics = {
        "silhouette": None,
        "davies_bouldin": None,
        "calinski_harabasz": None,
    }
    # Basic validations
    if not isinstance(embeddings, np.ndarray) or embeddings.ndim != 2:
        logger.error("Invalid embeddings format for metrics calculation.")
        return metrics
    if not isinstance(labels, np.ndarray) or labels.ndim != 1:
        logger.error("Invalid labels format for metrics calculation.")
        return metrics
    if embeddings.shape[0] != labels.shape[0]:
        logger.error("Embeddings and labels count mismatch for metrics calculation.")
        return metrics

    # Filter noise (-1)
    non_noise_mask = labels != -1
    if not np.any(non_noise_mask):
        logger.warning("No non-noise points found. Cannot calculate internal metrics.")
        return metrics

    embeddings_filtered = embeddings[non_noise_mask]
    labels_filtered = labels[non_noise_mask]

    # Check number of clusters after filtering noise
    unique_labels_filtered = np.unique(labels_filtered)
    n_clusters_filtered = len(unique_labels_filtered)

    if n_clusters_filtered < 2:
        logger.warning(
            f"Need >= 2 non-noise clusters for internal metrics (found {n_clusters_filtered}). Cannot calculate."
        )
        return metrics
    if embeddings_filtered.shape[0] < 2:
         logger.warning("Need >= 2 non-noise samples for internal metrics. Cannot calculate.")
         return metrics


    logger.info(
        f"Calculating internal metrics for {n_clusters_filtered} clusters ({embeddings_filtered.shape[0]} non-noise points)..."
    )
    calc_start_time = time.time()

    # Calculate Silhouette Score
    try:
        # Silhouette can be expensive, use sample_size for large datasets
        sample_size = min(5000, embeddings_filtered.shape[0])
        # Note: silhouette_score uses the metric directly
        metrics["silhouette"] = silhouette_score(
            embeddings_filtered,
            labels_filtered,
            metric=metric,
            sample_size=sample_size,
            random_state=42,
        )
    except ValueError as e:
         # Can fail if only 1 cluster or insufficient data after sampling
         logger.warning(f"Could not calculate Silhouette Score: {e}")
    except Exception as e:
        logger.warning(f"Silhouette Score calculation failed unexpectedly: {e}", exc_info=True)

    # Calculate Davies-Bouldin Index (uses Euclidean distance implicitly in scikit-learn)
    # To make it more comparable with Silhouette(cosine), we could use normalized embeddings if metric='cosine'
    try:
        embeddings_for_db_ch = embeddings_filtered
        if metric == 'cosine':
             # Normalize if the main metric is cosine, so DB/CH (based on variance/distance) are more comparable
             embeddings_for_db_ch = normalize(embeddings_filtered, norm='l2', axis=1)

        metrics["davies_bouldin"] = davies_bouldin_score(
            embeddings_for_db_ch, labels_filtered
        )
    except ValueError as e:
         logger.warning(f"Could not calculate Davies-Bouldin Score: {e}")
    except Exception as e:
        logger.warning(f"Davies-Bouldin Score calculation failed unexpectedly: {e}", exc_info=True)

    # Calculate Calinski-Harabasz Index (uses Euclidean distance implicitly)
    try:
        # Use the same embeddings as for Davies-Bouldin
        metrics["calinski_harabasz"] = calinski_harabasz_score(
            embeddings_for_db_ch, labels_filtered
        )
    except ValueError as e:
         logger.warning(f"Could not calculate Calinski-Harabasz Score: {e}")
    except Exception as e:
        logger.warning(f"Calinski-Harabasz Score calculation failed unexpectedly: {e}", exc_info=True)

    calc_end_time = time.time()
    logger.info(f"Internal metrics calculated in {calc_end_time - calc_start_time:.2f}s: {metrics}")
    return metrics


# --- FUNCTION FOR LABELING ---
# ** MODIFIED: Changed parameter names to match the calling context **
def label_clusters_with_text(
    clustering_embeddings: np.ndarray, # Embeddings used for clustering (N-dim or M-dim)
    cluster_labels: np.ndarray,       # Cluster labels assigned
    cluster_member_ids: List[str],    # IDs corresponding to clustering_embeddings
    vectorizer: Vectorizer,
    image_db_target_dim: VectorDBInterface, # Image DB instance corresponding to target_dimension (M-dim)
    target_dimension: Optional[int], # The target semantic dimension (M-dim, None for native)
    k_text_neighbors: int = 1,       # How many text labels to search per cluster centroid
) -> Dict[int, str]:
    """
    Genera etiquetas textuales para clusters de imágenes buscando en una BD de texto.
    Crea y puebla la BD de texto bajo demanda si no existe para la dimensión actual.

    Args:
        clustering_embeddings: The embeddings used for clustering (can be original or reduced).
        cluster_labels: The cluster labels assigned to each image embedding.
        cluster_member_ids: The IDs corresponding to each image embedding.
        vectorizer: The Vectorizer instance.
        image_db_target_dim: The VectorDBInterface instance of the image database
                             corresponding to the target_dimension (M-dim).
        target_dimension: The target semantic dimension (M-dim, None for native).
        k_text_neighbors: Number of text neighbors to search for each centroid.

    Returns:
        Un diccionario mapeando ID de cluster (int) a etiqueta de texto generada (str).
    """
    logger.info("--- Starting Cluster Labeling Process ---")
    labeling_start_time = time.time()
    generated_cluster_labels: Dict[int, str] = {}
    # Get unique valid cluster labels (excluding noise -1)
    unique_image_labels = sorted([lbl for lbl in np.unique(cluster_labels) if lbl != -1])

    if not unique_image_labels:
        logger.warning("No valid clusters found (only noise or empty). Skipping labeling.")
        return generated_cluster_labels

    # 1. Get/Create and Populate Text Database if necessary
    text_db: Optional[VectorDBInterface] = None
    text_collection_name: str = "N/A"
    try:
        logger.info("Getting or creating text database for labeling...")
        # Use the factory to get the correct instance (with cosine metric)
        text_db = get_or_create_text_db(vectorizer, target_dimension) # Use target_dimension (M-dim)
        if not text_db or not text_db.is_initialized:
            raise PipelineError("Failed to initialize text database via factory.")

        text_collection_name = getattr(text_db, 'collection_name', 'N/A')
        logger.info(f"Using text collection: '{text_collection_name}' (Metric: Cosine expected)")

        # Check if the text DB needs populating (first time for this dimension)
        if text_db.count() <= 0:
            logger.info(f"Text collection '{text_collection_name}' is empty. Populating with predefined labels...")
            # Use the imported list from app.text_labels
            predefined_labels = PREDEFINED_LABELS
            if not predefined_labels:
                 logger.warning("No predefined text labels found in app.text_labels. Cannot label clusters.")
                 return generated_cluster_labels # Cannot label without texts

            logger.info(f"Vectorizing {len(predefined_labels)} predefined text labels using '{vectorizer.model_name}' (Dim: {target_dimension or 'Full'})...")
            vectorization_start_time = time.time()
            # Vectorize using the same target dimension as the images for semantic comparison
            text_embeddings = vectorizer.vectorize_texts(
                texts=predefined_labels,
                batch_size=config.BATCH_SIZE_TEXT, # Use config
                truncate_dim=target_dimension # IMPORTANT! Use M-dim
            )
            logger.info(f"Text vectorization took {time.time() - vectorization_start_time:.2f}s.")

            # Filter failed results and prepare for DB
            valid_texts = []
            valid_embeddings_for_db = []
            valid_ids_for_db = []
            valid_metadatas_for_db = []

            for i, (text, emb) in enumerate(zip(predefined_labels, text_embeddings)):
                if emb: # If vectorization was successful for this text
                    valid_texts.append(text)
                    valid_embeddings_for_db.append(emb)
                    # Create a unique and descriptive ID
                    valid_ids_for_db.append(f"label_{i}_{text.replace(' ','_').lower()[:30]}") # Limit ID length
                    # Store the original text in metadata for retrieval
                    valid_metadatas_for_db.append({"source_text": text})
                else:
                    logger.warning(f"Failed to vectorize predefined label: '{text}'")

            # Add to the database if there are valid vectors
            if valid_embeddings_for_db:
                logger.info(f"Adding {len(valid_embeddings_for_db)} text embeddings to '{text_collection_name}'...")
                add_start_time = time.time()
                success = text_db.add_embeddings(
                    ids=valid_ids_for_db,
                    embeddings=valid_embeddings_for_db,
                    metadatas=valid_metadatas_for_db
                )
                logger.info(f"Adding text embeddings took {time.time() - add_start_time:.2f}s.")
                if not success:
                    # If adding fails, it's a serious issue
                    raise PipelineError(f"Failed to add text embeddings to '{text_collection_name}'.")
                logger.info("Predefined text labels added successfully to the database.")
            else:
                 # If no labels could be vectorized, cannot proceed
                 logger.error("Failed to vectorize any predefined labels. Cannot proceed with labeling.")
                 return generated_cluster_labels
        else:
             # If the collection already has data, assume it's correctly populated
             logger.info(f"Text collection '{text_collection_name}' already populated ({text_db.count()} items). Skipping population.")

    except (DatabaseError, PipelineError, VectorizerError, ValueError) as e:
        logger.error(f"Error preparing text database for labeling: {e}", exc_info=True)
        # Don't fail the entire clustering, just return empty labels and log the error
        return generated_cluster_labels
    except Exception as e:
         # Unexpected error during text DB preparation
         logger.error(f"Unexpected error preparing text database: {e}", exc_info=True)
         return generated_cluster_labels

    # 2. Calculate Centroids using the TARGET SEMANTIC DIMENSION (M-dim)
    centroids: Dict[int, List[float]] = {}
    try:
        logger.info(f"Calculating centroids for {len(unique_image_labels)} clusters...")
        centroid_calc_start_time = time.time()

        # Determine the dimension of the embeddings used for clustering (N-dim)
        clustering_dim = clustering_embeddings.shape[1]
        native_dim = vectorizer.native_dimension
        target_dim_actual = target_dimension if target_dimension is not None else native_dim

        # Check if the clustering dimension matches the target semantic dimension (M-dim)
        needs_original_embeddings = clustering_dim != target_dim_actual

        if needs_original_embeddings:
            logger.info(f"Clustering dimension ({clustering_dim}) differs from target semantic dimension ({target_dim_actual}). Fetching original {target_dim_actual}-dim embeddings for centroid calculation.")
        else:
            logger.info(f"Clustering dimension ({clustering_dim}) matches target semantic dimension. Using provided embeddings for centroids.")

        # Group embeddings by cluster label
        # Store either the N-dim embeddings directly or the IDs to fetch M-dim embeddings later
        cluster_members_data: Dict[int, Union[List[np.ndarray], List[str]]] = {}
        for i, label in enumerate(cluster_labels):
            if label != -1: # Exclude noise points
                if label not in cluster_members_data:
                    cluster_members_data[label] = []
                if needs_original_embeddings:
                    # Store the ID to fetch the M-dim embedding later
                    cluster_members_data[label].append(cluster_member_ids[i])
                else:
                    # Store the M-dim embedding directly (since N==M)
                    cluster_members_data[label].append(clustering_embeddings[i])

        # Calculate centroids
        for label in unique_image_labels:
            member_data = cluster_members_data.get(label)
            if not member_data:
                logger.warning(f"Cluster {label} has no members. Skipping centroid calculation.")
                continue

            cluster_member_embeddings_m_dim: Optional[np.ndarray] = None

            if needs_original_embeddings:
                # Fetch the M-dim embeddings using the stored IDs
                ids_to_fetch = member_data # These are IDs
                logger.debug(f"Fetching {len(ids_to_fetch)} M-dim embeddings for cluster {label} from '{getattr(image_db_target_dim, 'collection_name', 'N/A')}'...")
                fetched_embeddings_dict = image_db_target_dim.get_embeddings_by_ids(ids=ids_to_fetch)

                if not fetched_embeddings_dict:
                    logger.error(f"Failed to fetch M-dim embeddings for cluster {label}. Skipping centroid.")
                    continue

                valid_fetched_embeddings = [emb for emb in fetched_embeddings_dict.values() if emb is not None and len(emb) == target_dim_actual]
                if not valid_fetched_embeddings:
                    logger.warning(f"No valid M-dim embeddings found for cluster {label} after fetching. Skipping centroid.")
                    continue

                cluster_member_embeddings_m_dim = np.array(valid_fetched_embeddings, dtype=np.float32)
                logger.debug(f"Successfully fetched and validated {cluster_member_embeddings_m_dim.shape[0]} M-dim embeddings for cluster {label}.")

            else:
                # We already have the M-dim embeddings (since N==M)
                cluster_member_embeddings_m_dim = np.array(member_data, dtype=np.float32) # These are embeddings

            if cluster_member_embeddings_m_dim is None or cluster_member_embeddings_m_dim.shape[0] == 0:
                logger.warning(f"No valid M-dim embeddings available for cluster {label}. Skipping centroid calculation.")
                continue

            # Calculate the mean vector (centroid) in M-dim space
            centroid_vector = np.mean(cluster_member_embeddings_m_dim, axis=0)
            # Normalize the centroid (important for cosine similarity search)
            norm = np.linalg.norm(centroid_vector)
            if norm > 1e-6:
                centroid_vector /= norm
            else:
                # Handle zero-norm centroid (rare case)
                logger.warning(f"Centroid for cluster {label} has near-zero norm. Skipping normalization.")
            centroids[label] = centroid_vector.tolist() # Store as list of floats

        logger.info(f"Centroid calculation took {time.time() - centroid_calc_start_time:.2f}s. Calculated {len(centroids)} centroids using {target_dim_actual}-dim embeddings.")

    except (DatabaseError, ValueError) as e:
         logger.error(f"Error during centroid calculation or M-dim embedding retrieval: {e}", exc_info=True)
         return generated_cluster_labels # Cannot proceed without centroids
    except Exception as e:
        logger.error(f"Unexpected error calculating centroids: {e}", exc_info=True)
        return generated_cluster_labels # Cannot proceed without centroids

    # 3. Search for Similar Text for Each Centroid and Assign Label (Top-1)
    if not centroids:
         logger.warning("No centroids were calculated. Cannot perform text search for labels.")
         return generated_cluster_labels
    if not text_db: # Should not happen if initialization worked, but safety check
        logger.error("Text database instance is not available for querying labels.")
        return generated_cluster_labels

    logger.info(f"Querying text database '{text_collection_name}' for labels using {len(centroids)} centroids...")
    query_start_time = time.time()
    num_labels_assigned = 0
    for label, centroid in centroids.items():
        try:
            # Query the text DB using the normalized M-dim centroid
            search_results = text_db.query_similar(
                query_embedding=centroid,
                n_results=k_text_neighbors # Usually 1 for the closest label
            )

            # Extract the label from the closest result
            if search_results and not search_results.is_empty:
                top_item = search_results.items[0]
                # Retrieve the original text from metadata
                label_text = top_item.metadata.get("source_text", f"Cluster_{label}_?") # Fallback
                generated_cluster_labels[label] = label_text
                num_labels_assigned += 1
                logger.debug(f"  Cluster {label} -> Label '{label_text}' (Dist: {top_item.distance:.4f})")
            else:
                # If no text neighbors are found
                generated_cluster_labels[label] = f"[Cluster {label} - Sin etiqueta]"
                logger.warning(f"No similar text found in '{text_collection_name}' for cluster {label}.")

        except (DatabaseError, ValueError) as e:
            # Error during query for this specific centroid
            generated_cluster_labels[label] = f"[Cluster {label} - Error Búsqueda]"
            logger.error(f"Error querying text DB for cluster {label}: {e}", exc_info=False)
        except Exception as e:
             # Unexpected error during query for this centroid
             generated_cluster_labels[label] = f"[Cluster {label} - Error Inesperado]"
             logger.error(f"Unexpected error querying text DB for cluster {label}: {e}", exc_info=True)

    logger.info(f"Text query phase took {time.time() - query_start_time:.2f}s.")
    total_labeling_duration = time.time() - labeling_start_time
    logger.info(f"--- Cluster Labeling Process Finished in {total_labeling_duration:.2f}s. Generated {num_labels_assigned}/{len(unique_image_labels)} labels. ---")
    return generated_cluster_labels


