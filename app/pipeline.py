import math
import os
from typing import Dict, Optional, Any, List, Tuple
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
import warnings
import logging
import time # Para medir tiempos
from core.image_processor import find_image_files, batch_load_images, load_image
from core.vectorizer import Vectorizer
from data_access.vector_db import VectorDatabase
from config import (
    BATCH_SIZE_IMAGES,
    VECTOR_DIMENSION,
    DEFAULT_N_CLUSTERS,
    DEFAULT_N_RESULTS,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_directory(
    directory_path: str,
    vectorizer: Vectorizer,
    db: VectorDatabase,
    batch_size: int = BATCH_SIZE_IMAGES,
    truncate_dim: Optional[int] = VECTOR_DIMENSION,
) -> int:
    """
    Processes images in a directory: finds, vectorizes (handling partial failures),
    and stores successful embeddings in the vector database.

    Args:
        directory_path: Path to the directory containing images.
        vectorizer: An initialized Vectorizer instance.
        db: An initialized VectorDatabase instance.
        batch_size: Number of images to process in each vectorization batch.
        truncate_dim: Dimension to truncate embeddings to (passed to vectorizer).

    Returns:
        The total number of images successfully processed and stored in the DB.
    """
    pipeline_start_time = time.time()
    logging.info(f"--- Starting Image Processing Pipeline for Directory: {directory_path} ---")
    successfully_stored_count = 0 # Contador de los realmente guardados
    total_files_found = 0
    total_files_failed_loading = 0
    total_files_failed_vectorizing = 0
    total_files_failed_db_add = 0 # Podría ser por lote

    # 1. Find all image files
    if not os.path.isdir(directory_path):
         logging.error(f"Input directory not found: {directory_path}")
         return 0
    image_paths = find_image_files(directory_path) # find_image_files ya loguea si no encuentra
    if not image_paths:
        logging.warning("No image files found in the specified directory or subdirectories.")
        return 0

    total_files_found = len(image_paths)
    logging.info(f"Found {total_files_found} potential image files to process.")

    num_batches = math.ceil(total_files_found / batch_size)
    logging.info(f"Processing in approximately {num_batches} batches of size {batch_size}.")

    # 2. Process in batches
    for i in range(num_batches):
        batch_start_time = time.time()
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch_paths = image_paths[start_idx:end_idx]
        current_batch_num = i + 1

        logging.info(f"--- Processing Batch {current_batch_num}/{num_batches} ({len(batch_paths)} files) ---")

        # 3. Load images in the batch (batch_load_images maneja errores individuales)
        logging.info("Loading images...")
        loaded_images, loaded_paths = batch_load_images(batch_paths)
        num_loaded = len(loaded_images)
        num_failed_load_in_batch = len(batch_paths) - num_loaded
        total_files_failed_loading += num_failed_load_in_batch

        if not loaded_images:
            logging.warning(f"Batch {current_batch_num}: No images successfully loaded. Skipping.")
            continue
        logging.info(f"Batch {current_batch_num}: Successfully loaded {num_loaded}/{len(batch_paths)} images.")
        if num_failed_load_in_batch > 0:
             logging.warning(f"Batch {current_batch_num}: Failed to load {num_failed_load_in_batch} files (check logs from image_processor).")

        # 4. Vectorize loaded images (vectorizer ahora devuelve List[Optional[List[float]]])
        logging.info(f"Batch {current_batch_num}: Vectorizing {num_loaded} loaded images...")
        # Usar un tamaño de lote para la inferencia que podría ser diferente al tamaño de lote de archivos
        # Aquí usamos len(loaded_images) para procesar todo lo cargado en una llamada al vectorizador
        embedding_results = vectorizer.vectorize_images(
            loaded_images, batch_size=len(loaded_images), truncate_dim=truncate_dim
        )

        # 5. Filtrar resultados exitosos y preparar para BD
        valid_embeddings = []
        valid_ids = []
        valid_metadatas = []
        num_failed_vec_in_batch = 0

        # Verificar consistencia grave (longitud de resultados vs. imágenes cargadas)
        if len(embedding_results) != len(loaded_paths):
            logging.error(f"CRITICAL WARNING! Batch {current_batch_num}: Mismatch between embedding results ({len(embedding_results)}) and loaded paths ({len(loaded_paths)}). Skipping DB insertion for this batch.")
            total_files_failed_vectorizing += len(loaded_paths) # Asumimos que todos fallaron en este caso raro
            continue

        # Iterar sobre los resultados de embeddings
        for idx, embedding in enumerate(embedding_results):
            original_path = loaded_paths[idx] # Ruta correspondiente a este embedding/None
            if embedding is not None and isinstance(embedding, list):
                # Éxito en la vectorización
                valid_embeddings.append(embedding)
                valid_ids.append(original_path) # Usar ruta como ID
                valid_metadatas.append({"source_path": original_path})
            else:
                # Fallo en la vectorización para esta imagen específica
                num_failed_vec_in_batch += 1
                logging.warning(f"Batch {current_batch_num}: Failed to vectorize image: {original_path} (Result was None or invalid)")

        total_files_failed_vectorizing += num_failed_vec_in_batch
        num_valid_embeddings = len(valid_ids)
        logging.info(f"Batch {current_batch_num}: Vectorization resulted in {num_valid_embeddings} valid embeddings out of {num_loaded} loaded images.")

        # 6. Add valid embeddings to database
        if valid_ids:
            logging.info(f"Batch {current_batch_num}: Adding {num_valid_embeddings} valid embeddings to the database...")
            try:
                db.add_embeddings(ids=valid_ids, embeddings=valid_embeddings, metadatas=valid_metadatas)
                successfully_stored_count += num_valid_embeddings
                logging.info(f"Batch {current_batch_num}: Successfully added to DB.")
            except Exception as e:
                # Error al añadir el lote completo a la BD
                total_files_failed_db_add += len(valid_ids) # Marcar todos como fallidos en este intento
                logging.error(f"Batch {current_batch_num}: Error adding embeddings to database: {e}", exc_info=True)
                # Considerar estrategias de reintento para el lote si es crítico
        else:
            logging.info(f"Batch {current_batch_num}: No valid embeddings generated to add to the database.")

        batch_end_time = time.time()
        logging.info(f"--- Batch {current_batch_num} finished in {batch_end_time - batch_start_time:.2f} seconds ---")


    pipeline_end_time = time.time()
    logging.info(f"--- Image Processing Pipeline Finished in {pipeline_end_time - pipeline_start_time:.2f} seconds ---")
    logging.info(f"Summary:")
    logging.info(f"  Total files found: {total_files_found}")
    logging.info(f"  Failed to load: {total_files_failed_loading}")
    logging.info(f"  Failed to vectorize (after loading): {total_files_failed_vectorizing}")
    logging.info(f"  Failed during DB add (after vectorizing): {total_files_failed_db_add}")
    logging.info(f"  Successfully processed and stored in DB: {successfully_stored_count}")

    return successfully_stored_count


def search_by_text(
    query_text: str,
    vectorizer: Vectorizer,
    db: VectorDatabase,
    n_results: int = DEFAULT_N_RESULTS,
    truncate_dim: Optional[int] = VECTOR_DIMENSION,
) -> Optional[Dict[str, Any]]:
    """
    Searches for images similar to a given text query. Handles vectorization errors.
    """
    logging.info(f"--- Performing Text-to-Image Search ---")
    if not query_text or not isinstance(query_text, str):
         logging.error("Invalid query text provided.")
         return None
    logging.info(f"Query: '{query_text}'")

    # 1. Vectorize the query text
    logging.info("Vectorizing query text...")
    # Vectorizer ahora devuelve List[Optional[List[float]]]
    query_embedding_list = vectorizer.vectorize_texts(
        [query_text], batch_size=1, truncate_dim=truncate_dim, is_query=True # Marcar como query
    )

    # Verificar si la vectorización tuvo éxito (primer elemento de la lista)
    if not query_embedding_list or query_embedding_list[0] is None:
        logging.error("Failed to vectorize query text.")
        return None # No se puede buscar sin vector de consulta

    query_embedding = query_embedding_list[0]
    logging.info(f"Query vector generated (dim: {len(query_embedding)}).")

    # 2. Query the database
    logging.info(f"Querying database for top {n_results} similar images...")
    try:
        results = db.query_similar(query_embedding=query_embedding, n_results=n_results)
        if results is None: # query_similar devuelve None en error
             logging.error("Database query failed.")
             return None
        elif not results.get("ids"):
             logging.info("Search successful, but no similar images found in the database.")
        else:
             logging.info(f"Search successful. Found {len(results['ids'])} results.")

    except Exception as e:
         logging.error(f"Unexpected error during database query: {e}", exc_info=True)
         return None

    logging.info("--- Text-to-Image Search Finished ---")
    return results


def search_by_image(
    query_image_path: str,
    vectorizer: Vectorizer,
    db: VectorDatabase,
    n_results: int = DEFAULT_N_RESULTS,
    truncate_dim: Optional[int] = VECTOR_DIMENSION,
) -> Optional[Dict[str, Any]]:
    """
    Searches for images similar to a given query image. Handles load/vectorization errors.
    """
    logging.info(f"--- Performing Image-to-Image Search ---")
    if not query_image_path or not isinstance(query_image_path, str):
         logging.error("Invalid query image path provided.")
         return None
    logging.info(f"Query Image Path: '{query_image_path}'")

    # 1. Load the query image (load_image maneja errores)
    logging.info("Loading query image...")
    query_image = load_image(query_image_path)
    if not query_image:
        # load_image ya habrá logueado el error específico
        logging.error(f"Failed to load query image at {query_image_path}. Cannot perform search.")
        return None
    logging.info("Query image loaded successfully.")

    # 2. Vectorize the query image
    logging.info("Vectorizing query image...")
    # Vectorizer devuelve List[Optional[List[float]]]
    query_embedding_list = vectorizer.vectorize_images(
        [query_image], batch_size=1, truncate_dim=truncate_dim
    )

    if not query_embedding_list or query_embedding_list[0] is None:
        logging.error("Failed to vectorize query image.")
        return None
    query_embedding = query_embedding_list[0]
    logging.info(f"Query vector generated (dim: {len(query_embedding)}).")

    # 3. Query the database
    logging.info(f"Querying database for top {n_results} similar images...")
    # Nota: Podríamos querer n_results+1 para filtrar la imagen exacta si está en la BD
    effective_n_results = n_results + 1 # Pedir uno extra para posible filtrado

    try:
        results = db.query_similar(query_embedding=query_embedding, n_results=effective_n_results)
        if results is None:
             logging.error("Database query failed.")
             return None

        # Filtrar la imagen de consulta si aparece como el resultado más cercano
        if results.get("ids"):
            logging.info(f"Search successful. Found {len(results['ids'])} potential results (before filtering).")
            ids_list = results["ids"]
            distances_list = results["distances"]
            metadatas_list = results["metadatas"]
            filtered_ids = []
            filtered_distances = []
            filtered_metadatas = []
            query_basename = os.path.basename(query_image_path) # Para comparación simple

            for i in range(len(ids_list)):
                # Comprobar si el ID coincide (o basename si IDs son rutas) Y la distancia es casi cero
                is_exact_match = (ids_list[i] == query_image_path or os.path.basename(ids_list[i]) == query_basename) and distances_list[i] < 1e-6

                if not is_exact_match:
                    filtered_ids.append(ids_list[i])
                    filtered_distances.append(distances_list[i])
                    filtered_metadatas.append(metadatas_list[i])
                else:
                     logging.info(f"  (Excluding query image '{ids_list[i]}' itself from results)")

            # Truncar a los n_results solicitados originalmente DESPUÉS de filtrar
            final_results = {
                "ids": filtered_ids[:n_results],
                "distances": filtered_distances[:n_results],
                "metadatas": filtered_metadatas[:n_results],
            }
            if not final_results.get("ids"):
                 logging.info("No similar images found (after filtering exact match).")

            results = final_results # Usar los resultados filtrados y truncados

        else:
             logging.info("Search successful, but no similar images found in the database.")

    except Exception as e:
         logging.error(f"Unexpected error during database query or filtering: {e}", exc_info=True)
         return None


    logging.info("--- Image-to-Image Search Finished ---")
    return results


def perform_clustering(
    db: VectorDatabase, n_clusters: int = DEFAULT_N_CLUSTERS
) -> Optional[Dict[str, int]]:
    """
    Performs KMeans clustering on all embeddings stored in the database.

    Args:
        db: An initialized VectorDatabase instance.
        n_clusters: The desired number of clusters.

    Returns:
        A dictionary mapping image IDs (file paths) to cluster IDs (integers),
        or None if clustering fails or there are not enough data points.
        Returns empty dict {} if DB is empty.
    """
    logging.info(f"--- Performing Clustering ---")
    if n_clusters <= 0:
         logging.error(f"Invalid number of clusters requested: {n_clusters}. Must be > 0.")
         return None
    logging.info(f"Target number of clusters: {n_clusters}")

    # 1. Get all embeddings and IDs from the database
    logging.info("Retrieving all embeddings from the database...")
    retrieved_data = db.get_all_embeddings_with_ids() # Ya maneja errores y caso vacío

    if retrieved_data is None:
        logging.error("Failed to retrieve embeddings from the database. Cannot perform clustering.")
        return None # Error crítico al obtener datos

    ids, embeddings_array = retrieved_data
    num_samples = embeddings_array.shape[0]

    if num_samples == 0:
        logging.warning("No embeddings found in the database to cluster.")
        return {} # No hay datos, devuelve diccionario vacío

    logging.info(f"Retrieved {num_samples} embeddings for clustering.")

    # Ajustar n_clusters si hay menos muestras que clústeres deseados
    effective_n_clusters = n_clusters
    if num_samples < n_clusters:
        logging.warning(f"Number of samples ({num_samples}) is less than the desired number of clusters ({n_clusters}). Adjusting n_clusters to {num_samples}.")
        effective_n_clusters = num_samples

    if effective_n_clusters <= 0: # Doble chequeo por si num_samples era 0
         logging.error("Effective number of clusters is <= 0. Cannot perform clustering.")
         return None

    # 2. Perform KMeans clustering
    logging.info(f"Running KMeans clustering with {effective_n_clusters} clusters...")
    kmeans = KMeans(
        n_clusters=effective_n_clusters,
        random_state=42, # Para reproducibilidad
        n_init='auto' # 'auto' es generalmente recomendado en sklearn reciente
    )

    # Suprimir ConvergenceWarning si KMeans no converge completamente
    # (puede pasar con pocos datos o distribuciones difíciles)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn.cluster._kmeans")
        try:
            start_time_kmeans = time.time()
            cluster_labels = kmeans.fit_predict(embeddings_array)
            end_time_kmeans = time.time()
            logging.info(f"KMeans fitting complete in {end_time_kmeans - start_time_kmeans:.2f}s.")
        except Exception as e:
            logging.error(f"Error during KMeans fitting: {e}", exc_info=True)
            return None

    # 3. Create mapping from ID to cluster label
    # Asegurar que las longitudes coincidan
    if len(ids) != len(cluster_labels):
         logging.error(f"Mismatch between number of IDs ({len(ids)}) and cluster labels ({len(cluster_labels)}). Clustering result invalid.")
         return None

    id_to_cluster_map = dict(zip(ids, cluster_labels.tolist())) # Convertir labels numpy a int

    logging.info(f"Clustering successful. Assigned {len(id_to_cluster_map)} items to {effective_n_clusters} clusters.")
    # logging.debug(f"Cluster assignments (sample): {list(id_to_cluster_map.items())[:5]}")

    logging.info("--- Clustering Finished ---")
    return id_to_cluster_map