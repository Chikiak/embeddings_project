import math
import os
import time
import warnings
import logging
from typing import Dict, Optional, Any, List, Tuple, Callable  # Importar Callable

import numpy as np
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning

# Importar componentes del proyecto
from core.image_processor import find_image_files, batch_load_images, load_image
from core.vectorizer import Vectorizer
from data_access.vector_db import VectorDatabase
from config import (
    BATCH_SIZE_IMAGES,
    VECTOR_DIMENSION,
    DEFAULT_N_CLUSTERS,
    DEFAULT_N_RESULTS,
    IMAGE_EXTENSIONS,  # Importar extensiones para validación
)

# Obtener un logger para este módulo (la configuración se hace en main.py/streamlit_app.py)
logger = logging.getLogger(__name__)


def process_directory(
    directory_path: str,
    vectorizer: Vectorizer,
    db: VectorDatabase,
    batch_size: int = BATCH_SIZE_IMAGES,
    truncate_dim: Optional[int] = VECTOR_DIMENSION,
    # --- NUEVOS ARGUMENTOS CALLBACK ---
    progress_callback: Optional[Callable[[int, int], None]] = None,
    status_callback: Optional[Callable[[str], None]] = None,
) -> int:
    """
    Procesa imágenes en un directorio: encuentra, vectoriza (manejando fallos parciales),
    y almacena embeddings exitosos en la base de datos vectorial.

    Notifica el progreso y estado a través de callbacks opcionales.

    Args:
        directory_path: Ruta al directorio que contiene imágenes.
        vectorizer: Instancia inicializada de Vectorizer.
        db: Instancia inicializada de VectorDatabase.
        batch_size: Número de imágenes a procesar en cada lote de vectorización.
        truncate_dim: Dimensión a la que truncar los embeddings (pasado al vectorizer).
        progress_callback: Función opcional llamada con (procesados, total).
        status_callback: Función opcional llamada con (mensaje_de_estado).

    Returns:
        El número total de imágenes procesadas y almacenadas exitosamente en la BD.
    """
    pipeline_start_time = time.time()
    if status_callback:
        status_callback(f"Iniciando pipeline para: {directory_path}")
    logger.info(
        f"--- Starting Image Processing Pipeline for Directory: {directory_path} ---"
    )

    successfully_stored_count = 0
    total_files_found = 0
    total_files_failed_loading = 0
    total_files_failed_vectorizing = 0
    total_files_failed_db_add = 0
    processed_count_for_progress = 0  # Contador para el callback de progreso

    # 1. Encontrar todos los archivos de imagen
    if not os.path.isdir(directory_path):
        logger.error(f"Input directory not found: {directory_path}")
        if status_callback:
            status_callback(f"Error: Directorio no encontrado en {directory_path}")
        return 0

    if status_callback:
        status_callback("Buscando archivos de imagen...")
    image_paths = find_image_files(
        directory_path
    )  # find_image_files ya loguea si no encuentra

    if not image_paths:
        logger.warning(
            "No image files found in the specified directory or subdirectories."
        )
        if status_callback:
            status_callback(
                "Advertencia: No se encontraron archivos de imagen en el directorio."
            )
        if progress_callback:
            progress_callback(0, 0)  # Indicar 0/0
        return 0

    total_files_found = len(image_paths)
    logger.info(f"Found {total_files_found} potential image files to process.")
    if status_callback:
        status_callback(
            f"Encontrados {total_files_found} archivos de imagen potenciales."
        )
    if progress_callback:
        progress_callback(0, total_files_found)  # Progreso inicial 0%

    num_batches = math.ceil(total_files_found / batch_size)
    logger.info(
        f"Processing in approximately {num_batches} batches of size {batch_size}."
    )

    # 2. Procesar en lotes
    for i in range(num_batches):
        batch_start_time = time.time()
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch_paths = image_paths[start_idx:end_idx]
        current_batch_num = i + 1
        num_in_batch = len(batch_paths)

        logger.info(
            f"--- Processing Batch {current_batch_num}/{num_batches} ({num_in_batch} files) ---"
        )
        if status_callback:
            status_callback(
                f"Procesando Lote {current_batch_num}/{num_batches} ({num_in_batch} archivos)"
            )

        # 3. Cargar imágenes en el lote (batch_load_images maneja errores individuales)
        logger.info("Loading images...")
        if status_callback:
            status_callback(f" Lote {current_batch_num}: Cargando imágenes...")
        # batch_load_images ya loguea advertencias sobre archivos no cargados
        loaded_images, loaded_paths = batch_load_images(batch_paths)
        num_loaded = len(loaded_images)
        num_failed_load_in_batch = num_in_batch - num_loaded
        total_files_failed_loading += num_failed_load_in_batch

        if not loaded_images:
            logger.warning(
                f"Batch {current_batch_num}: No images successfully loaded. Skipping vectorization and DB add for this batch."
            )
            if status_callback:
                status_callback(
                    f" Lote {current_batch_num}: Ninguna imagen válida cargada. Saltando."
                )
            # Actualizar progreso aunque el lote falle
            processed_count_for_progress += num_in_batch
            if progress_callback:
                progress_callback(processed_count_for_progress, total_files_found)
            continue  # Saltar al siguiente lote

        logger.info(
            f"Batch {current_batch_num}: Successfully loaded {num_loaded}/{num_in_batch} images."
        )
        if num_failed_load_in_batch > 0:
            logger.warning(
                f"Batch {current_batch_num}: Failed to load {num_failed_load_in_batch} files (check logs from image_processor)."
            )
            if status_callback:
                status_callback(
                    f" Lote {current_batch_num}: {num_failed_load_in_batch} archivos no pudieron cargarse."
                )

        # 4. Vectorizar imágenes cargadas (vectorizer devuelve List[Optional[List[float]]])
        logger.info(
            f"Batch {current_batch_num}: Vectorizing {num_loaded} loaded images..."
        )
        if status_callback:
            status_callback(
                f" Lote {current_batch_num}: Vectorizando {num_loaded} imágenes..."
            )
        # Usar un tamaño de lote para la inferencia que podría ser diferente al tamaño de lote de archivos
        # Aquí usamos len(loaded_images) para procesar todo lo cargado en una llamada al vectorizador
        embedding_results = vectorizer.vectorize_images(
            loaded_images,
            batch_size=num_loaded,
            truncate_dim=truncate_dim,  # Usar num_loaded como batch_size aquí
        )

        # 5. Filtrar resultados exitosos y preparar para BD
        valid_embeddings = []
        valid_ids = []
        valid_metadatas = []
        num_failed_vec_in_batch = 0

        # Verificar consistencia grave (longitud de resultados vs. imágenes cargadas)
        if len(embedding_results) != len(loaded_paths):
            logger.error(
                f"CRITICAL WARNING! Batch {current_batch_num}: Mismatch between embedding results ({len(embedding_results)}) and loaded paths ({len(loaded_paths)}). Skipping DB insertion for this batch."
            )
            if status_callback:
                status_callback(
                    f" Lote {current_batch_num}: ¡Error Crítico! Discrepancia en vectorización. Saltando guardado."
                )
            # Asumir que todos fallaron en este caso raro para el conteo
            total_files_failed_vectorizing += len(loaded_paths)
            # Actualizar progreso
            processed_count_for_progress += (
                num_in_batch  # Contar los archivos originales del lote
            )
            if progress_callback:
                progress_callback(processed_count_for_progress, total_files_found)
            continue  # Saltar al siguiente lote

        # Iterar sobre los resultados de embeddings (que ahora pueden ser None)
        for idx, embedding in enumerate(embedding_results):
            original_path = loaded_paths[
                idx
            ]  # Ruta correspondiente a este embedding/None
            if embedding is not None and isinstance(embedding, list):
                # Éxito en la vectorización
                valid_embeddings.append(embedding)
                valid_ids.append(original_path)  # Usar ruta como ID único
                valid_metadatas.append(
                    {"source_path": original_path}
                )  # Guardar ruta original en metadatos
            else:
                # Fallo en la vectorización para esta imagen específica
                num_failed_vec_in_batch += 1
                # Loguear advertencia (vectorizer ya debería haber logueado el error específico)
                logger.warning(
                    f"Batch {current_batch_num}: Failed to vectorize image (result was None or invalid): {original_path}"
                )

        total_files_failed_vectorizing += num_failed_vec_in_batch
        num_valid_embeddings = len(valid_ids)
        logger.info(
            f"Batch {current_batch_num}: Vectorization resulted in {num_valid_embeddings} valid embeddings out of {num_loaded} loaded images."
        )
        if num_failed_vec_in_batch > 0 and status_callback:
            status_callback(
                f" Lote {current_batch_num}: {num_failed_vec_in_batch} imágenes no pudieron vectorizarse."
            )

        # 6. Añadir embeddings válidos a la base de datos
        if valid_ids:
            logger.info(
                f"Batch {current_batch_num}: Adding {num_valid_embeddings} valid embeddings to the database..."
            )
            if status_callback:
                status_callback(
                    f" Lote {current_batch_num}: Guardando {num_valid_embeddings} embeddings en la BD..."
                )
            try:
                # Usar upsert para añadir o actualizar
                db.add_embeddings(
                    ids=valid_ids,
                    embeddings=valid_embeddings,
                    metadatas=valid_metadatas,
                )
                successfully_stored_count += num_valid_embeddings
                logger.info(
                    f"Batch {current_batch_num}: Successfully added/updated {num_valid_embeddings} embeddings in DB."
                )
                if status_callback:
                    status_callback(
                        f" Lote {current_batch_num}: {num_valid_embeddings} guardados/actualizados en BD."
                    )
            except Exception as e:
                # Error al añadir el lote completo a la BD
                total_files_failed_db_add += len(
                    valid_ids
                )  # Marcar todos como fallidos en este intento
                logger.error(
                    f"Batch {current_batch_num}: Error adding embeddings batch to database: {e}",
                    exc_info=True,
                )
                if status_callback:
                    status_callback(
                        f" Lote {current_batch_num}: ¡Error al guardar en BD! {e}"
                    )
                # Considerar estrategias de reintento para el lote si es crítico
        else:
            logger.info(
                f"Batch {current_batch_num}: No valid embeddings generated in this batch to add to the database."
            )
            if status_callback:
                status_callback(
                    f" Lote {current_batch_num}: No hay embeddings válidos para guardar."
                )

        batch_end_time = time.time()
        logger.info(
            f"--- Batch {current_batch_num} finished in {batch_end_time - batch_start_time:.2f} seconds ---"
        )

        # Actualizar progreso al final del lote
        processed_count_for_progress += (
            num_in_batch  # Contar archivos originales del lote
        )
        if progress_callback:
            progress_callback(processed_count_for_progress, total_files_found)

    # --- Finalización del Pipeline ---
    pipeline_end_time = time.time()
    total_time = pipeline_end_time - pipeline_start_time
    logger.info(
        f"--- Image Processing Pipeline Finished in {total_time:.2f} seconds ---"
    )
    logger.info(f"Summary for directory: {directory_path}")
    logger.info(f"  Total files found: {total_files_found}")
    logger.info(f"  Failed to load: {total_files_failed_loading}")
    logger.info(
        f"  Failed to vectorize (after loading): {total_files_failed_vectorizing}"
    )
    logger.info(
        f"  Failed during DB add (after vectorizing): {total_files_failed_db_add}"
    )
    logger.info(
        f"  Successfully processed and stored/updated in DB: {successfully_stored_count}"
    )

    if status_callback:
        status_callback(
            f"Pipeline finalizado en {total_time:.2f}s. Imágenes guardadas/actualizadas: {successfully_stored_count}/{total_files_found}."
        )

    return successfully_stored_count


def search_by_text(
    query_text: str,
    vectorizer: Vectorizer,
    db: VectorDatabase,
    n_results: int = DEFAULT_N_RESULTS,
    truncate_dim: Optional[int] = VECTOR_DIMENSION,
) -> Optional[Dict[str, Any]]:
    """
    Busca imágenes similares a una consulta de texto dada. Maneja errores de vectorización.
    """
    logger.info(f"--- Performing Text-to-Image Search for query: '{query_text}' ---")
    if not query_text or not isinstance(query_text, str):
        logger.error("Invalid query text provided (empty or not a string).")
        return None  # No se puede buscar sin texto válido
    if db.collection is None or db.collection.count() == 0:
        logger.warning(
            "Cannot perform search: The database collection is empty or not initialized."
        )
        return {
            "ids": [],
            "distances": [],
            "metadatas": [],
        }  # Devolver vacío si no hay nada que buscar

    # 1. Vectorizar el texto de consulta
    logger.info("Vectorizing query text...")
    # Vectorizer ahora devuelve List[Optional[List[float]]]
    query_embedding_list = vectorizer.vectorize_texts(
        [query_text],
        batch_size=1,
        truncate_dim=truncate_dim,
        is_query=True,  # Marcar como query si el modelo lo soporta
    )

    # Verificar si la vectorización tuvo éxito (primer y único elemento de la lista)
    if not query_embedding_list or query_embedding_list[0] is None:
        logger.error(
            "Failed to vectorize the provided query text. Cannot perform search."
        )
        # El vectorizer ya debería haber logueado el error específico
        return None  # No se puede buscar sin vector de consulta

    query_embedding = query_embedding_list[0]
    logger.info(f"Query text vectorized successfully (dim: {len(query_embedding)}).")

    # 2. Consultar la base de datos
    logger.info(f"Querying database for top {n_results} similar images...")
    try:
        # La función query_similar de la BD ya maneja el caso de colección vacía internamente
        results = db.query_similar(query_embedding=query_embedding, n_results=n_results)

        if results is None:  # Indica un fallo en la consulta de la BD
            logger.error("Database query failed. Check VectorDatabase logs.")
            return None
        elif not results.get("ids"):  # Consulta exitosa, pero sin resultados
            logger.info(
                "Search successful, but no similar images found in the database for this query."
            )
            # Devolver la estructura vacía que ya devuelve query_similar en este caso
        else:  # Consulta exitosa con resultados
            logger.info(f"Search successful. Found {len(results['ids'])} results.")

    except Exception as e:
        # Capturar cualquier otro error inesperado durante la consulta
        logger.error(f"Unexpected error during database query: {e}", exc_info=True)
        return None

    logger.info("--- Text-to-Image Search Finished ---")
    return results  # Devuelve el diccionario de resultados o None si falló


def search_by_image(
    query_image_path: str,
    vectorizer: Vectorizer,
    db: VectorDatabase,
    n_results: int = DEFAULT_N_RESULTS,
    truncate_dim: Optional[int] = VECTOR_DIMENSION,
) -> Optional[Dict[str, Any]]:
    """
    Busca imágenes similares a una imagen de consulta dada. Maneja errores de carga/vectorización
    y filtra la propia imagen de consulta de los resultados si se encuentra.
    """
    logger.info(
        f"--- Performing Image-to-Image Search using query image: '{query_image_path}' ---"
    )
    if not query_image_path or not isinstance(query_image_path, str):
        logger.error("Invalid query image path provided (empty or not a string).")
        return None
    if not os.path.isfile(query_image_path):
        logger.error(f"Query image file not found at path: {query_image_path}")
        return None
    if db.collection is None or db.collection.count() == 0:
        logger.warning(
            "Cannot perform search: The database collection is empty or not initialized."
        )
        return {
            "ids": [],
            "distances": [],
            "metadatas": [],
        }  # Devolver vacío si no hay nada que buscar

    # 1. Cargar la imagen de consulta (load_image maneja errores y loguea)
    logger.info("Loading query image...")
    query_image = load_image(query_image_path)
    if not query_image:
        # load_image ya habrá logueado el error específico
        logger.error(
            f"Failed to load query image at {query_image_path}. Cannot perform search."
        )
        return None  # No se puede buscar sin la imagen cargada
    logger.info("Query image loaded successfully.")

    # 2. Vectorizar la imagen de consulta
    logger.info("Vectorizing query image...")
    # Vectorizer devuelve List[Optional[List[float]]]
    query_embedding_list = vectorizer.vectorize_images(
        [query_image], batch_size=1, truncate_dim=truncate_dim
    )

    if not query_embedding_list or query_embedding_list[0] is None:
        logger.error("Failed to vectorize the query image. Cannot perform search.")
        # Vectorizer ya logueó el error
        return None  # No se puede buscar sin el vector
    query_embedding = query_embedding_list[0]
    logger.info(f"Query image vectorized successfully (dim: {len(query_embedding)}).")

    # 3. Consultar la base de datos
    # Pedir n_results + 1 para tener margen para filtrar la imagen exacta si aparece
    effective_n_results = n_results + 1
    logger.info(
        f"Querying database for top {effective_n_results} similar images (will filter query image)..."
    )

    try:
        results = db.query_similar(
            query_embedding=query_embedding, n_results=effective_n_results
        )

        if results is None:
            logger.error("Database query failed. Check VectorDatabase logs.")
            return None

        # Filtrar la imagen de consulta si aparece como el resultado más cercano
        # Comparar por ID (ruta completa) y distancia muy baja
        if results.get("ids"):
            logger.info(
                f"Search successful. Found {len(results['ids'])} potential results (before filtering)."
            )
            ids_list = results["ids"]
            distances_list = results["distances"]
            metadatas_list = (
                results["metadatas"]
                if results.get("metadatas")
                else [{}] * len(ids_list)
            )

            # Crear listas filtradas
            filtered_ids = []
            filtered_distances = []
            filtered_metadatas = []

            # Normalizar la ruta de consulta para comparación robusta
            normalized_query_path = os.path.abspath(query_image_path)

            for i in range(len(ids_list)):
                # Normalizar la ruta del resultado
                normalized_result_path = os.path.abspath(ids_list[i])
                # Comprobar si las rutas normalizadas coinciden Y la distancia es casi cero
                is_exact_match = (normalized_result_path == normalized_query_path) and (
                    distances_list[i] < 1e-6
                )

                if not is_exact_match:
                    filtered_ids.append(ids_list[i])
                    filtered_distances.append(distances_list[i])
                    # Asegurar que el índice es válido para metadatas
                    if i < len(metadatas_list):
                        filtered_metadatas.append(metadatas_list[i])
                    else:  # Añadir dict vacío si falta metadata por alguna razón
                        filtered_metadatas.append({})
                else:
                    logger.info(
                        f"  (Excluding query image '{ids_list[i]}' itself from results)"
                    )

            # Truncar a los n_results solicitados originalmente DESPUÉS de filtrar
            final_results = {
                "ids": filtered_ids[:n_results],
                "distances": filtered_distances[:n_results],
                "metadatas": filtered_metadatas[:n_results],
            }

            if not final_results.get("ids"):
                logger.info(
                    "No similar images found (after filtering the exact query match)."
                )
            else:
                logger.info(
                    f"Returning {len(final_results['ids'])} results after filtering."
                )

            results = final_results  # Usar los resultados filtrados y truncados

        else:  # Consulta exitosa, pero sin resultados iniciales
            logger.info(
                "Search successful, but no similar images found in the database."
            )
            # results ya es {"ids": [], ...} en este caso

    except Exception as e:
        logger.error(
            f"Unexpected error during database query or result filtering: {e}",
            exc_info=True,
        )
        return None

    logger.info("--- Image-to-Image Search Finished ---")
    return results


def perform_clustering(
    db: VectorDatabase, n_clusters: int = DEFAULT_N_CLUSTERS
) -> Optional[Dict[str, int]]:
    """
    Realiza clustering KMeans sobre todos los embeddings almacenados en la base de datos.

    Args:
        db: Instancia inicializada de VectorDatabase.
        n_clusters: El número deseado de clústeres.

    Returns:
        Un diccionario mapeando IDs de imagen (rutas) a IDs de clúster (enteros),
        o None si falla el clustering o no hay suficientes puntos de datos.
        Devuelve un diccionario vacío {} si la BD está vacía.
    """
    logger.info(f"--- Performing Clustering ---")
    if n_clusters <= 0:
        logger.error(
            f"Invalid number of clusters requested: {n_clusters}. Must be > 0."
        )
        return None
    if db.collection is None:
        logger.error("Cannot perform clustering: Database collection not initialized.")
        return None

    logger.info(f"Target number of clusters: {n_clusters}")

    # 1. Obtener todos los embeddings e IDs de la base de datos
    logger.info("Retrieving all embeddings from the database...")
    # get_all_embeddings_with_ids ya maneja errores y caso vacío internamente
    retrieved_data = db.get_all_embeddings_with_ids()

    if retrieved_data is None:
        logger.error(
            "Failed to retrieve embeddings from the database. Cannot perform clustering."
        )
        return None  # Error crítico al obtener datos

    ids, embeddings_array = retrieved_data
    num_samples = (
        embeddings_array.shape[0] if isinstance(embeddings_array, np.ndarray) else 0
    )

    if num_samples == 0:
        logger.warning("No embeddings found in the database to cluster.")
        return {}  # No hay datos, devuelve diccionario vacío

    logger.info(f"Retrieved {num_samples} embeddings for clustering.")

    # Ajustar n_clusters si hay menos muestras que clústeres deseados
    effective_n_clusters = n_clusters
    if num_samples < n_clusters:
        logger.warning(
            f"Number of samples ({num_samples}) is less than the desired number of clusters ({n_clusters}). Adjusting n_clusters to {num_samples}."
        )
        effective_n_clusters = num_samples

    # Doble chequeo por si num_samples era 0 y n_clusters también
    if effective_n_clusters <= 0:
        logger.error(
            f"Effective number of clusters is {effective_n_clusters}. Cannot perform clustering."
        )
        return None

    # 2. Realizar clustering KMeans
    logger.info(f"Running KMeans clustering with {effective_n_clusters} clusters...")
    kmeans = KMeans(
        n_clusters=effective_n_clusters,
        random_state=42,  # Para reproducibilidad
        n_init="auto",  # Usar 'auto' para la inicialización (recomendado)
    )

    # Suprimir ConvergenceWarning si KMeans no converge completamente
    # (puede pasar con pocos datos o distribuciones difíciles, el resultado puede ser útil igualmente)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=ConvergenceWarning, module="sklearn.cluster._kmeans"
        )
        try:
            start_time_kmeans = time.time()
            # Asegurarse de que embeddings_array es un array 2D válido
            if embeddings_array.ndim != 2:
                logger.error(
                    f"Embeddings array has incorrect dimensions ({embeddings_array.ndim}) for KMeans. Expected 2D."
                )
                return None
            cluster_labels = kmeans.fit_predict(embeddings_array)
            end_time_kmeans = time.time()
            logger.info(
                f"KMeans fitting complete in {end_time_kmeans - start_time_kmeans:.2f}s."
            )
        except ValueError as ve:
            logger.error(
                f"ValueError during KMeans fitting: {ve}. Check data shape and n_clusters.",
                exc_info=True,
            )
            return None
        except Exception as e:
            logger.error(f"Unexpected error during KMeans fitting: {e}", exc_info=True)
            return None

    # 3. Crear mapeo de ID a etiqueta de clúster
    # Asegurar que las longitudes coincidan (deberían si KMeans no falló)
    if len(ids) != len(cluster_labels):
        logger.error(
            f"CRITICAL: Mismatch between number of IDs ({len(ids)}) and cluster labels ({len(cluster_labels)}) after KMeans. Clustering result invalid."
        )
        return None

    # Crear el diccionario final
    id_to_cluster_map = dict(
        zip(ids, cluster_labels.tolist())
    )  # Convertir labels numpy a int nativo

    logger.info(
        f"Clustering successful. Assigned {len(id_to_cluster_map)} items to {effective_n_clusters} clusters."
    )
    # logger.debug(f"Cluster assignments (sample): {list(id_to_cluster_map.items())[:5]}") # Descomentar para depuración

    logger.info("--- Clustering Finished ---")
    return id_to_cluster_map
