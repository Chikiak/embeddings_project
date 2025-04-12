# app/pipeline.py
import math
import os
import time
import warnings
import logging
from typing import Dict, Optional, Any, List, Tuple, Callable

import numpy as np
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning

# Importar componentes del proyecto y modelos de datos
from core.image_processor import find_image_files, batch_load_images, load_image
from core.vectorizer import Vectorizer
from data_access.vector_db_interface import VectorDBInterface  # Usar la interfaz
from app.models import SearchResultItem, SearchResults  # Usar modelos de datos
from config import (
    BATCH_SIZE_IMAGES,
    VECTOR_DIMENSION,
    DEFAULT_N_CLUSTERS,
    DEFAULT_N_RESULTS,
)

logger = logging.getLogger(__name__)


# --- Helper interno para procesar un lote ---
def _process_batch_for_indexing(
        batch_paths: List[str],
        vectorizer: Vectorizer,
        db: VectorDBInterface,
        truncate_dim: Optional[int],
        batch_num_info: str  # e.g., "Batch 1/10"
) -> Tuple[int, int, int, int]:
    """
    Processes a single batch of image paths for indexing.
    Loads, vectorizes, and adds valid embeddings to the database.

    Returns:
        Tuple: (num_loaded, num_vectorized, num_added_to_db, num_failed_db_add)
    """
    batch_start_time = time.time()
    num_in_batch = len(batch_paths)
    logger.info(f"--- Processing {batch_num_info} ({num_in_batch} files) ---")

    # 1. Load images (handles individual load errors)
    logger.info(f"{batch_num_info}: Loading images...")
    loaded_images, loaded_paths = batch_load_images(batch_paths)
    num_loaded = len(loaded_images)
    if num_loaded == 0:
        logger.warning(f"{batch_num_info}: No images successfully loaded. Skipping batch.")
        return 0, 0, 0, 0

    logger.info(f"{batch_num_info}: Successfully loaded {num_loaded}/{num_in_batch} images.")

    # 2. Vectorize loaded images
    logger.info(f"{batch_num_info}: Vectorizing {num_loaded} loaded images...")
    embedding_results = vectorizer.vectorize_images(
        loaded_images,
        batch_size=num_loaded,  # Vectorize all loaded images in one go
        truncate_dim=truncate_dim,
    )

    # 3. Filter successful results and prepare for DB
    valid_embeddings = []
    valid_ids = []
    valid_metadatas = []
    num_failed_vec_in_batch = 0

    if len(embedding_results) != len(loaded_paths):
        logger.error(
            f"CRITICAL WARNING! {batch_num_info}: Mismatch between embedding results ({len(embedding_results)}) and loaded paths ({len(loaded_paths)}). Skipping DB insertion for this batch.")
        # Consider all loaded as failed vectorization in this critical case
        return num_loaded, 0, 0, 0

    for idx, embedding in enumerate(embedding_results):
        original_path = loaded_paths[idx]
        if embedding is not None and isinstance(embedding, list):
            valid_embeddings.append(embedding)
            valid_ids.append(original_path)
            valid_metadatas.append({"source_path": original_path})
        else:
            num_failed_vec_in_batch += 1
            logger.warning(f"{batch_num_info}: Failed to vectorize image (result was None/invalid): {original_path}")

    num_vectorized = len(valid_ids)
    logger.info(
        f"{batch_num_info}: Vectorization resulted in {num_vectorized} valid embeddings out of {num_loaded} loaded images.")

    # 4. Add valid embeddings to DB
    num_added_to_db = 0
    num_failed_db_add = 0
    if valid_ids:
        logger.info(f"{batch_num_info}: Adding {num_vectorized} valid embeddings to the database...")
        try:
            success = db.add_embeddings(
                ids=valid_ids,
                embeddings=valid_embeddings,
                metadatas=valid_metadatas,
            )
            if success:
                num_added_to_db = num_vectorized
                logger.info(f"{batch_num_info}: Successfully added/updated {num_added_to_db} embeddings in DB.")
            else:
                # add_embeddings failed, assume all failed for this batch add attempt
                num_failed_db_add = num_vectorized
                logger.error(f"{batch_num_info}: Failed to add batch to database (check VectorDatabase logs).")
        except Exception as e:
            num_failed_db_add = num_vectorized  # Mark all as failed DB add on exception
            logger.error(f"{batch_num_info}: Error adding embeddings batch to database: {e}", exc_info=True)
    else:
        logger.info(f"{batch_num_info}: No valid embeddings generated to add to the database.")

    batch_end_time = time.time()
    logger.info(f"--- {batch_num_info} finished in {batch_end_time - batch_start_time:.2f} seconds ---")

    return num_loaded, num_vectorized, num_added_to_db, num_failed_db_add


# --- Public Pipeline Functions ---

def process_directory(
        directory_path: str,
        vectorizer: Vectorizer,
        db: VectorDBInterface,  # Use interface type hint
        batch_size: int = BATCH_SIZE_IMAGES,
        truncate_dim: Optional[int] = VECTOR_DIMENSION,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        status_callback: Optional[Callable[[str], None]] = None,
) -> int:
    """
    Processes images in a directory: finds, vectorizes (handling partial failures),
    and stores successful embeddings in the vector database.

    Uses the VectorDBInterface for database operations.
    Notifies progress and status via optional callbacks.

    Args:
        directory_path: Path to the directory containing images.
        vectorizer: Initialized Vectorizer instance.
        db: Initialized instance conforming to VectorDBInterface.
        batch_size: Number of image files to process in each batch.
        truncate_dim: Dimension to truncate embeddings to (passed to vectorizer).
        progress_callback: Optional function called with (processed_files, total_files).
        status_callback: Optional function called with (status_message).

    Returns:
        The total number of images successfully processed and stored/updated in the DB.
    """
    pipeline_start_time = time.time()
    if status_callback: status_callback(f"Iniciando pipeline para: {directory_path}")
    logger.info(f"--- Starting Image Processing Pipeline for Directory: {directory_path} ---")

    # --- Initialization and Validation ---
    if not db.is_initialized:
        logger.error("Database is not initialized. Aborting pipeline.")
        if status_callback: status_callback("Error: La base de datos no está inicializada.")
        return 0

    if not os.path.isdir(directory_path):
        logger.error(f"Input directory not found: {directory_path}")
        if status_callback: status_callback(f"Error: Directorio no encontrado en {directory_path}")
        return 0

    # --- Find Image Files ---
    if status_callback: status_callback("Buscando archivos de imagen...")
    image_paths = find_image_files(directory_path)  # find_image_files already logs if none found

    if not image_paths:
        logger.warning("No image files found in the specified directory or subdirectories.")
        if status_callback: status_callback("Advertencia: No se encontraron archivos de imagen.")
        if progress_callback: progress_callback(0, 0)  # Indicate 0/0
        return 0

    total_files_found = len(image_paths)
    logger.info(f"Found {total_files_found} potential image files to process.")
    if status_callback: status_callback(f"Encontrados {total_files_found} archivos.")
    if progress_callback: progress_callback(0, total_files_found)  # Initial progress 0%

    # --- Batch Processing ---
    num_batches = math.ceil(total_files_found / batch_size)
    logger.info(f"Processing in approximately {num_batches} batches of size {batch_size}.")

    # --- Aggregate Counters ---
    total_successfully_stored = 0
    total_files_processed_for_progress = 0
    total_failed_loading = 0
    total_failed_vectorizing = 0
    total_failed_db_add = 0

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch_paths = image_paths[start_idx:end_idx]
        current_batch_num = i + 1
        num_in_this_batch = len(batch_paths)
        batch_num_info = f"Lote {current_batch_num}/{num_batches}"

        if status_callback: status_callback(f"Procesando {batch_num_info} ({num_in_this_batch} archivos)")

        # Call the internal batch processing helper
        num_loaded, num_vectorized, num_added, num_failed_db = _process_batch_for_indexing(
            batch_paths, vectorizer, db, truncate_dim, batch_num_info
        )

        # Update aggregate counters based on helper results
        total_failed_loading += (num_in_this_batch - num_loaded)
        total_failed_vectorizing += (num_loaded - num_vectorized)  # Failures after successful load
        total_successfully_stored += num_added
        total_failed_db_add += num_failed_db

        # Update progress based on the number of *files attempted* in the batch
        total_files_processed_for_progress += num_in_this_batch
        if progress_callback:
            progress_callback(total_files_processed_for_progress, total_files_found)

        # Update status callback with batch summary
        if status_callback:
            status_parts = [f"{batch_num_info}: Completado."]
            if num_in_this_batch - num_loaded > 0: status_parts.append(
                f"Fallo carga: {num_in_this_batch - num_loaded}.")
            if num_loaded - num_vectorized > 0: status_parts.append(f"Fallo vectoriz.: {num_loaded - num_vectorized}.")
            if num_failed_db > 0: status_parts.append(f"Fallo BD: {num_failed_db}.")
            if num_added > 0: status_parts.append(f"Guardados: {num_added}.")
            status_callback(" ".join(status_parts))

    # --- Final Summary ---
    pipeline_end_time = time.time()
    total_time = pipeline_end_time - pipeline_start_time
    logger.info(f"--- Image Processing Pipeline Finished in {total_time:.2f} seconds ---")
    logger.info(f"Summary for directory: {directory_path}")
    logger.info(f"  Total files found: {total_files_found}")
    logger.info(f"  Failed to load: {total_failed_loading}")
    logger.info(f"  Failed to vectorize (after loading): {total_failed_vectorizing}")
    logger.info(f"  Failed during DB add (after vectorizing): {total_failed_db_add}")
    logger.info(f"  Successfully processed and stored/updated in DB: {total_successfully_stored}")

    if status_callback:
        status_callback(
            f"Pipeline finalizado en {total_time:.2f}s. Imágenes guardadas/actualizadas: {total_successfully_stored}/{total_files_found}.")

    return total_successfully_stored


def search_by_text(
        query_text: str,
        vectorizer: Vectorizer,
        db: VectorDBInterface,  # Use interface type hint
        n_results: int = DEFAULT_N_RESULTS,
        truncate_dim: Optional[int] = VECTOR_DIMENSION,
) -> Optional[SearchResults]:  # Return SearchResults object or None on failure
    """
    Searches for images similar to a given text query using the VectorDBInterface.

    Args:
        query_text: The text description to search for.
        vectorizer: Initialized Vectorizer instance.
        db: Initialized instance conforming to VectorDBInterface.
        n_results: The maximum number of results to return.
        truncate_dim: Dimension used for vectorization (must match indexed vectors).

    Returns:
        A SearchResults object containing the results, or None if the search failed
        (e.g., failed to vectorize query, database error). Returns an empty
        SearchResults object if the search was successful but found no matches.
    """
    logger.info(f"--- Performing Text-to-Image Search for query: '{query_text}' ---")
    if not query_text or not isinstance(query_text, str):
        logger.error("Invalid query text provided (empty or not a string).")
        return None
    if not db.is_initialized or db.count() == 0:
        logger.warning("Cannot perform search: The database is empty or not initialized.")
        # Return empty results object, as the "search" technically succeeded on an empty DB
        return SearchResults(items=[])

    # 1. Vectorize query text
    logger.info("Vectorizing query text...")
    query_embedding_list = vectorizer.vectorize_texts(
        [query_text],
        batch_size=1,
        truncate_dim=truncate_dim,
        is_query=True,  # Indicate it's a query embedding if model supports it
    )

    if not query_embedding_list or query_embedding_list[0] is None:
        logger.error("Failed to vectorize the query text. Cannot perform search.")
        return None  # Cannot search without a query vector

    query_embedding = query_embedding_list[0]
    logger.info(f"Query text vectorized successfully (dim: {len(query_embedding)}).")

    # 2. Query the database using the interface
    logger.info(f"Querying database for top {n_results} similar images...")
    try:
        # The query_similar method of the interface handles DB interaction
        results: Optional[SearchResults] = db.query_similar(
            query_embedding=query_embedding, n_results=n_results
        )

        if results is None:
            # This indicates a failure within the db.query_similar method
            logger.error("Database query failed. Check VectorDatabase logs.")
            return None
        elif results.is_empty:
            logger.info("Search successful, but no similar images found in the database.")
            # Return the empty SearchResults object provided by db.query_similar
        else:
            logger.info(f"Search successful. Found {results.count} results.")

        logger.info("--- Text-to-Image Search Finished ---")
        return results  # Return the SearchResults object (could be empty)

    except Exception as e:
        # Catch unexpected errors during the pipeline's orchestration of the query
        logger.error(f"Unexpected error during text search pipeline: {e}", exc_info=True)
        return None


def search_by_image(
        query_image_path: str,
        vectorizer: Vectorizer,
        db: VectorDBInterface,  # Use interface type hint
        n_results: int = DEFAULT_N_RESULTS,
        truncate_dim: Optional[int] = VECTOR_DIMENSION,
) -> Optional[SearchResults]:  # Return SearchResults object or None on failure
    """
    Searches for images similar to a given query image using the VectorDBInterface.
    Filters the query image itself from the results if found.

    Args:
        query_image_path: Path to the query image file.
        vectorizer: Initialized Vectorizer instance.
        db: Initialized instance conforming to VectorDBInterface.
        n_results: The maximum number of results to return (excluding the query image itself).
        truncate_dim: Dimension used for vectorization.

    Returns:
        A SearchResults object containing the results, or None if the search failed
        (e.g., failed to load/vectorize query, database error). Returns an empty
        SearchResults object if the search was successful but found no matches.
    """
    logger.info(f"--- Performing Image-to-Image Search using query image: '{query_image_path}' ---")

    # --- Input Validation ---
    if not query_image_path or not isinstance(query_image_path, str):
        logger.error("Invalid query image path provided.")
        return None
    if not os.path.isfile(query_image_path):
        logger.error(f"Query image file not found at path: {query_image_path}")
        return None
    if not db.is_initialized or db.count() == 0:
        logger.warning("Cannot perform search: The database is empty or not initialized.")
        return SearchResults(items=[])  # Return empty results

    # 1. Load query image
    logger.info("Loading query image...")
    query_image = load_image(query_image_path)  # load_image handles errors
    if not query_image:
        logger.error(f"Failed to load query image at {query_image_path}. Cannot perform search.")
        return None

    # 2. Vectorize query image
    logger.info("Vectorizing query image...")
    query_embedding_list = vectorizer.vectorize_images(
        [query_image], batch_size=1, truncate_dim=truncate_dim
    )
    if not query_embedding_list or query_embedding_list[0] is None:
        logger.error("Failed to vectorize the query image. Cannot perform search.")
        return None
    query_embedding = query_embedding_list[0]
    logger.info(f"Query image vectorized successfully (dim: {len(query_embedding)}).")

    # 3. Query the database (request n+1 results to allow filtering self)
    effective_n_results = n_results + 1
    logger.info(f"Querying database for top {effective_n_results} similar images (will filter query image)...")

    try:
        initial_results: Optional[SearchResults] = db.query_similar(
            query_embedding=query_embedding, n_results=effective_n_results
        )

        if initial_results is None:
            logger.error("Database query failed. Check VectorDatabase logs.")
            return None

        # 4. Filter the query image itself from the results
        filtered_items = []
        if not initial_results.is_empty:
            logger.info(f"Search successful. Found {initial_results.count} potential results (before filtering).")
            normalized_query_path = os.path.abspath(query_image_path)

            for item in initial_results.items:
                normalized_result_path = os.path.abspath(item.id)
                # Check for path match and very small distance (potential exact match)
                is_exact_match = (normalized_result_path == normalized_query_path) and \
                                 (item.distance is not None and item.distance < 1e-6)

                if not is_exact_match:
                    filtered_items.append(item)
                else:
                    logger.info(f"  (Excluding query image '{item.id}' itself from results)")

            # Truncate to the originally requested number of results *after* filtering
            final_items = filtered_items[:n_results]
            final_results = SearchResults(items=final_items, query_vector=query_embedding)

            if final_results.is_empty:
                logger.info("No similar images found (after filtering the exact query match).")
            else:
                logger.info(f"Returning {final_results.count} results after filtering.")

            results_to_return = final_results
        else:
            logger.info("Search successful, but no similar images found in the database initially.")
            results_to_return = initial_results  # Return the empty results object

    except Exception as e:
        logger.error(f"Unexpected error during image search pipeline or filtering: {e}", exc_info=True)
        return None

    logger.info("--- Image-to-Image Search Finished ---")
    return results_to_return


def perform_clustering(
        db: VectorDBInterface,  # Use interface type hint
        n_clusters: int = DEFAULT_N_CLUSTERS
) -> Optional[Dict[str, int]]:
    """
    Performs KMeans clustering on all embeddings stored in the database via the interface.

    Args:
        db: Initialized instance conforming to VectorDBInterface.
        n_clusters: The desired number of clusters.

    Returns:
        A dictionary mapping image IDs (paths) to cluster IDs (integers),
        or None if clustering fails or there's insufficient data.
        Returns an empty dict {} if the database is empty.
    """
    logger.info(f"--- Performing Clustering ---")
    if n_clusters <= 0:
        logger.error(f"Invalid number of clusters requested: {n_clusters}. Must be > 0.")
        return None
    if not db.is_initialized:
        logger.error("Cannot perform clustering: Database collection not initialized.")
        return None

    logger.info(f"Target number of clusters: {n_clusters}")

    # 1. Get all embeddings and IDs using the interface
    logger.info("Retrieving all embeddings from the database...")
    retrieved_data = db.get_all_embeddings_with_ids()  # Interface method handles pagination

    if retrieved_data is None:
        logger.error("Failed to retrieve embeddings from the database. Cannot perform clustering.")
        return None  # Critical failure during data retrieval

    ids, embeddings_array = retrieved_data
    num_samples = embeddings_array.shape[0] if isinstance(embeddings_array, np.ndarray) else 0

    if num_samples == 0:
        logger.warning("No embeddings found in the database to cluster.")
        return {}  # Return empty dict for empty DB

    logger.info(f"Retrieved {num_samples} embeddings for clustering.")

    # Adjust n_clusters if fewer samples than requested clusters
    effective_n_clusters = n_clusters
    if num_samples < n_clusters:
        logger.warning(
            f"Number of samples ({num_samples}) is less than desired clusters ({n_clusters}). Adjusting n_clusters to {num_samples}.")
        effective_n_clusters = num_samples

    # Check again if effective_n_clusters became zero (shouldn't if num_samples > 0)
    if effective_n_clusters <= 0:
        logger.error(f"Effective number of clusters is {effective_n_clusters}. Cannot perform clustering.")
        return None

    # 2. Perform KMeans clustering
    logger.info(f"Running KMeans clustering with {effective_n_clusters} clusters...")
    kmeans = KMeans(
        n_clusters=effective_n_clusters,
        random_state=42,  # For reproducibility
        n_init='auto',  # Recommended setting for n_init
        # algorithm='lloyd' # Default, explicitly stated
    )

    # Suppress ConvergenceWarning (clustering might still be useful)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn.cluster._kmeans")
        try:
            start_time_kmeans = time.time()
            # Ensure embeddings_array is valid 2D array
            if not isinstance(embeddings_array, np.ndarray) or embeddings_array.ndim != 2:
                logger.error(
                    f"Embeddings array has incorrect type/dimensions ({embeddings_array.shape if isinstance(embeddings_array, np.ndarray) else type(embeddings_array)}) for KMeans. Expected 2D numpy array.")
                return None

            cluster_labels = kmeans.fit_predict(embeddings_array)
            end_time_kmeans = time.time()
            logger.info(f"KMeans fitting complete in {end_time_kmeans - start_time_kmeans:.2f}s.")

        except ValueError as ve:
            logger.error(
                f"ValueError during KMeans fitting: {ve}. Check data shape ({embeddings_array.shape}) and n_clusters ({effective_n_clusters}).",
                exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Unexpected error during KMeans fitting: {e}", exc_info=True)
            return None

    # 3. Create ID-to-cluster mapping
    if len(ids) != len(cluster_labels):
        logger.error(
            f"CRITICAL: Mismatch between number of IDs ({len(ids)}) and cluster labels ({len(cluster_labels)}) after KMeans. Result invalid.")
        return None

    # Create the final dictionary
    id_to_cluster_map = dict(zip(ids, cluster_labels.tolist()))  # Convert numpy int to standard int

    logger.info(f"Clustering successful. Assigned {len(id_to_cluster_map)} items to {effective_n_clusters} clusters.")
    # logger.debug(f"Cluster assignments (sample): {list(id_to_cluster_map.items())[:5]}") # Uncomment for debug

    logger.info("--- Clustering Finished ---")
    return id_to_cluster_map
