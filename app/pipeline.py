# app/pipeline.py
import math
import os
import time
import logging
from typing import Optional, List, Tuple, Callable

from core.image_processor import find_image_files, batch_load_images, load_image
from core.vectorizer import Vectorizer
from data_access.vector_db_interface import VectorDBInterface
from app.models import SearchResults
from app.exceptions import (
    PipelineError,
    DatabaseError,
    VectorizerError,
    ImageProcessingError,
)
from config import (
    BATCH_SIZE_IMAGES,
    VECTOR_DIMENSION,
    DEFAULT_N_RESULTS,
)

logger = logging.getLogger(__name__)


def _process_batch_for_indexing(
    batch_paths: List[str],
    vectorizer: Vectorizer,
    db: VectorDBInterface,
    truncate_dim: Optional[int],
    batch_num_info: str,
) -> Tuple[int, int, int, int]:
    """
    Procesa un único lote de rutas de imágenes para la indexación.

    Carga, vectoriza y añade embeddings válidos a la base de datos.

    Args:
        batch_paths: Lista de rutas de imágenes en el lote.
        vectorizer: Instancia de Vectorizer inicializada.
        db: Instancia que cumple con VectorDBInterface.
        truncate_dim: Dimensión a la que truncar los embeddings.
        batch_num_info: Información del número de lote (ej: "Lote 1/10").

    Returns:
        Tupla: (num_cargadas, num_vectorizadas, num_añadidas_a_bd, num_fallo_bd)

    Raises:
        PipelineError: Si ocurre un error crítico durante el procesamiento del lote.
    """
    batch_start_time = time.time()
    num_in_batch = len(batch_paths)
    logger.info(f"--- Processing {batch_num_info} ({num_in_batch} files) ---")

    num_loaded = 0
    num_vectorized = 0
    num_added_to_db = 0
    num_failed_db_add = 0

    try:
        logger.info(f"{batch_num_info}: Loading images...")
        loaded_images, loaded_paths = batch_load_images(batch_paths)
        num_loaded = len(loaded_images)
        if num_loaded == 0:
            logger.warning(
                f"{batch_num_info}: No images successfully loaded. Skipping batch."
            )
            return 0, 0, 0, 0
        logger.info(
            f"{batch_num_info}: Successfully loaded {num_loaded}/{num_in_batch} images."
        )

        logger.info(f"{batch_num_info}: Vectorizing {num_loaded} loaded images...")
        embedding_results = vectorizer.vectorize_images(
            loaded_images,
            batch_size=num_loaded,
            truncate_dim=truncate_dim,
        )

        valid_embeddings = []
        valid_ids = []
        valid_metadatas = []
        num_failed_vec_in_batch = 0

        if len(embedding_results) != len(loaded_paths):
            logger.error(
                f"CRITICAL WARNING! {batch_num_info}: Mismatch between embedding results ({len(embedding_results)}) and loaded paths ({len(loaded_paths)}). Skipping DB insertion for this batch."
            )
            return num_loaded, 0, 0, 0

        for idx, embedding in enumerate(embedding_results):
            original_path = loaded_paths[idx]
            if embedding is not None and isinstance(embedding, list):
                valid_embeddings.append(embedding)
                valid_ids.append(original_path)
                valid_metadatas.append({"source_path": original_path})
            else:
                num_failed_vec_in_batch += 1
                logger.warning(
                    f"{batch_num_info}: Failed to vectorize image (result was None/invalid): {original_path}"
                )

        num_vectorized = len(valid_ids)
        logger.info(
            f"{batch_num_info}: Vectorization resulted in {num_vectorized} valid embeddings out of {num_loaded} loaded images."
        )

        if valid_ids:
            logger.info(
                f"{batch_num_info}: Adding {num_vectorized} valid embeddings to the database..."
            )
            success = db.add_embeddings(
                ids=valid_ids,
                embeddings=valid_embeddings,
                metadatas=valid_metadatas,
            )
            if success:
                num_added_to_db = num_vectorized
                logger.info(
                    f"{batch_num_info}: Successfully added/updated {num_added_to_db} embeddings in DB."
                )
            else:
                num_failed_db_add = num_vectorized
                logger.error(
                    f"{batch_num_info}: Failed to add batch to database (check VectorDatabase logs)."
                )
        else:
            logger.info(
                f"{batch_num_info}: No valid embeddings generated to add to the database."
            )

    except (VectorizerError, DatabaseError, ImageProcessingError) as e:
        logger.error(f"{batch_num_info}: Error processing batch: {e}", exc_info=True)
        raise PipelineError(f"Failed processing batch {batch_num_info}: {e}") from e
    except Exception as e:
        logger.error(
            f"{batch_num_info}: Unexpected error processing batch: {e}", exc_info=True
        )
        raise PipelineError(f"Unexpected failure in batch {batch_num_info}: {e}") from e
    finally:
        batch_end_time = time.time()
        logger.info(
            f"--- {batch_num_info} finished in {batch_end_time - batch_start_time:.2f} seconds ---"
        )

    return num_loaded, num_vectorized, num_added_to_db, num_failed_db_add


def process_directory(
    directory_path: str,
    vectorizer: Vectorizer,
    db: VectorDBInterface,
    batch_size: int = BATCH_SIZE_IMAGES,
    truncate_dim: Optional[int] = VECTOR_DIMENSION,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    status_callback: Optional[Callable[[str], None]] = None,
) -> int:
    """
    Procesa imágenes en un directorio: encuentra, vectoriza y almacena embeddings.

    Utiliza VectorDBInterface para operaciones de base de datos y notifica
    el progreso y estado a través de callbacks opcionales.

    Args:
        directory_path: Ruta al directorio que contiene las imágenes.
        vectorizer: Instancia de Vectorizer inicializada.
        db: Instancia inicializada que cumple con VectorDBInterface.
        batch_size: Número de archivos de imagen a procesar en cada lote.
        truncate_dim: Dimensión a la que truncar los embeddings.
        progress_callback: Función opcional llamada con (archivos_procesados, archivos_totales).
        status_callback: Función opcional llamada con (mensaje_estado).

    Returns:
        El número total de imágenes procesadas y almacenadas/actualizadas con éxito en la BD.

    Raises:
        PipelineError: Si ocurre un error irrecuperable durante el pipeline.
    """
    pipeline_start_time = time.time()
    if status_callback:
        status_callback(f"Iniciando pipeline para: {directory_path}")
    logger.info(
        f"--- Starting Image Processing Pipeline for Directory: {directory_path} ---"
    )

    if not db.is_initialized:
        msg = "Database is not initialized. Aborting pipeline."
        logger.error(msg)
        if status_callback:
            status_callback(f"Error: {msg}")
        raise PipelineError(msg)

    if not os.path.isdir(directory_path):
        msg = f"Input directory not found: {directory_path}"
        logger.error(msg)
        if status_callback:
            status_callback(f"Error: {msg}")
        raise PipelineError(msg)

    if status_callback:
        status_callback("Buscando archivos de imagen...")
    try:
        image_paths = find_image_files(directory_path)
    except Exception as e:
        msg = f"Failed to find image files in {directory_path}: {e}"
        logger.error(msg, exc_info=True)
        if status_callback:
            status_callback(f"Error: {msg}")
        raise PipelineError(msg) from e

    if not image_paths:
        logger.warning(
            "No image files found in the specified directory or subdirectories."
        )
        if status_callback:
            status_callback("Advertencia: No se encontraron archivos de imagen.")
        if progress_callback:
            progress_callback(0, 0)
        return 0

    total_files_found = len(image_paths)
    logger.info(f"Found {total_files_found} potential image files to process.")
    if status_callback:
        status_callback(f"Encontrados {total_files_found} archivos.")
    if progress_callback:
        progress_callback(0, total_files_found)

    num_batches = math.ceil(total_files_found / batch_size)
    logger.info(
        f"Processing in approximately {num_batches} batches of size {batch_size}."
    )

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

        if status_callback:
            status_callback(
                f"Procesando {batch_num_info} ({num_in_this_batch} archivos)"
            )

        try:
            num_loaded, num_vectorized, num_added, num_failed_db = (
                _process_batch_for_indexing(
                    batch_paths, vectorizer, db, truncate_dim, batch_num_info
                )
            )
        except PipelineError as e:
            logger.error(
                f"Pipeline stopped due to error in batch {batch_num_info}: {e}",
                exc_info=True,
            )
            if status_callback:
                status_callback(
                    f"Error crítico en {batch_num_info}, pipeline detenido."
                )
            # Re-raise or handle depending on desired behavior (stop vs continue)
            raise e  # Stop the pipeline on batch failure

        total_failed_loading += num_in_this_batch - num_loaded
        total_failed_vectorizing += num_loaded - num_vectorized
        total_successfully_stored += num_added
        total_failed_db_add += num_failed_db

        total_files_processed_for_progress += num_in_this_batch
        if progress_callback:
            progress_callback(total_files_processed_for_progress, total_files_found)

        if status_callback:
            status_parts = [f"{batch_num_info}: Completado."]
            if num_in_this_batch - num_loaded > 0:
                status_parts.append(f"Fallo carga: {num_in_this_batch - num_loaded}.")
            if num_loaded - num_vectorized > 0:
                status_parts.append(f"Fallo vectoriz.: {num_loaded - num_vectorized}.")
            if num_failed_db > 0:
                status_parts.append(f"Fallo BD: {num_failed_db}.")
            if num_added > 0:
                status_parts.append(f"Guardados: {num_added}.")
            status_callback(" ".join(status_parts))

    pipeline_end_time = time.time()
    total_time = pipeline_end_time - pipeline_start_time
    logger.info(
        f"--- Image Processing Pipeline Finished in {total_time:.2f} seconds ---"
    )
    logger.info(f"Summary for directory: {directory_path}")
    logger.info(f"  Total files found: {total_files_found}")
    logger.info(f"  Failed to load: {total_failed_loading}")
    logger.info(f"  Failed to vectorize (after loading): {total_failed_vectorizing}")
    logger.info(f"  Failed during DB add (after vectorizing): {total_failed_db_add}")
    logger.info(
        f"  Successfully processed and stored/updated in DB: {total_successfully_stored}"
    )

    if status_callback:
        status_callback(
            f"Pipeline finalizado en {total_time:.2f}s. Imágenes guardadas/actualizadas: {total_successfully_stored}/{total_files_found}."
        )

    return total_successfully_stored


def search_by_text(
    query_text: str,
    vectorizer: Vectorizer,
    db: VectorDBInterface,
    n_results: int = DEFAULT_N_RESULTS,
    truncate_dim: Optional[int] = VECTOR_DIMENSION,
) -> SearchResults:
    """
    Busca imágenes similares a una consulta de texto dada usando VectorDBInterface.

    Args:
        query_text: La descripción de texto a buscar.
        vectorizer: Instancia de Vectorizer inicializada.
        db: Instancia inicializada que cumple con VectorDBInterface.
        n_results: El número máximo de resultados a devolver.
        truncate_dim: Dimensión utilizada para la vectorización.

    Returns:
        Un objeto SearchResults que contiene los resultados. Devuelve un
        SearchResults vacío si la búsqueda fue exitosa pero no encontró coincidencias.

    Raises:
        PipelineError: Si la búsqueda falla (ej: fallo al vectorizar consulta, error de BD).
        ValueError: Si la entrada query_text es inválida.
    """
    logger.info(f"--- Performing Text-to-Image Search for query: '{query_text}' ---")
    if not query_text or not isinstance(query_text, str):
        msg = "Invalid query text provided (empty or not a string)."
        logger.error(msg)
        raise ValueError(msg)
    if not db.is_initialized or db.count() == 0:
        logger.warning(
            "Cannot perform search: The database is empty or not initialized."
        )
        return SearchResults(items=[])

    try:
        logger.info("Vectorizing query text...")
        query_embedding_list = vectorizer.vectorize_texts(
            [query_text],
            batch_size=1,
            truncate_dim=truncate_dim,
            is_query=True,
        )

        if not query_embedding_list or query_embedding_list[0] is None:
            msg = "Failed to vectorize the query text. Cannot perform search."
            logger.error(msg)
            raise PipelineError(msg)

        query_embedding = query_embedding_list[0]
        logger.info(
            f"Query text vectorized successfully (dim: {len(query_embedding)})."
        )

        logger.info(f"Querying database for top {n_results} similar images...")
        results: Optional[SearchResults] = db.query_similar(
            query_embedding=query_embedding, n_results=n_results
        )

        if results is None:
            msg = "Database query failed. Check VectorDatabase logs."
            logger.error(msg)
            raise PipelineError(msg)
        elif results.is_empty:
            logger.info(
                "Search successful, but no similar images found in the database."
            )
        else:
            logger.info(f"Search successful. Found {results.count} results.")

        logger.info("--- Text-to-Image Search Finished ---")
        return results

    except (VectorizerError, DatabaseError) as e:
        logger.error(f"Error during text search pipeline: {e}", exc_info=True)
        raise PipelineError(f"Text search failed: {e}") from e
    except Exception as e:
        logger.error(
            f"Unexpected error during text search pipeline: {e}", exc_info=True
        )
        raise PipelineError(f"Unexpected text search failure: {e}") from e


def search_by_image(
    query_image_path: str,
    vectorizer: Vectorizer,
    db: VectorDBInterface,
    n_results: int = DEFAULT_N_RESULTS,
    truncate_dim: Optional[int] = VECTOR_DIMENSION,
) -> SearchResults:
    """
    Busca imágenes similares a una imagen de consulta dada usando VectorDBInterface.

    Filtra la propia imagen de consulta de los resultados si se encuentra.

    Args:
        query_image_path: Ruta al archivo de imagen de consulta.
        vectorizer: Instancia de Vectorizer inicializada.
        db: Instancia inicializada que cumple con VectorDBInterface.
        n_results: El número máximo de resultados a devolver (excluyendo la imagen de consulta).
        truncate_dim: Dimensión utilizada para la vectorización.

    Returns:
        Un objeto SearchResults que contiene los resultados. Devuelve un
        SearchResults vacío si la búsqueda fue exitosa pero no encontró coincidencias.

    Raises:
        PipelineError: Si la búsqueda falla (ej: fallo al cargar/vectorizar consulta, error de BD).
        FileNotFoundError: Si el archivo query_image_path no existe.
        ValueError: Si la ruta query_image_path es inválida.
    """
    logger.info(
        f"--- Performing Image-to-Image Search using query image: '{query_image_path}' ---"
    )

    if not query_image_path or not isinstance(query_image_path, str):
        msg = "Invalid query image path provided."
        logger.error(msg)
        raise ValueError(msg)
    if not os.path.isfile(query_image_path):
        msg = f"Query image file not found at path: {query_image_path}"
        logger.error(msg)
        raise FileNotFoundError(msg)
    if not db.is_initialized or db.count() == 0:
        logger.warning(
            "Cannot perform search: The database is empty or not initialized."
        )
        return SearchResults(items=[])

    try:
        logger.info("Loading query image...")
        query_image = load_image(query_image_path)
        if not query_image:
            # load_image already logged the error
            raise PipelineError(f"Failed to load query image at {query_image_path}.")

        logger.info("Vectorizing query image...")
        query_embedding_list = vectorizer.vectorize_images(
            [query_image], batch_size=1, truncate_dim=truncate_dim
        )
        if not query_embedding_list or query_embedding_list[0] is None:
            raise PipelineError("Failed to vectorize the query image.")

        query_embedding = query_embedding_list[0]
        logger.info(
            f"Query image vectorized successfully (dim: {len(query_embedding)})."
        )

        effective_n_results = n_results + 1
        logger.info(
            f"Querying database for top {effective_n_results} similar images (will filter query image)..."
        )

        initial_results: Optional[SearchResults] = db.query_similar(
            query_embedding=query_embedding, n_results=effective_n_results
        )

        if initial_results is None:
            raise PipelineError("Database query failed. Check VectorDatabase logs.")

        filtered_items = []
        if not initial_results.is_empty:
            logger.info(
                f"Search successful. Found {initial_results.count} potential results (before filtering)."
            )
            normalized_query_path = os.path.abspath(query_image_path)

            for item in initial_results.items:
                normalized_result_path = os.path.abspath(item.id)
                is_exact_match = (normalized_result_path == normalized_query_path) and (
                    item.distance is not None and item.distance < 1e-6
                )

                if not is_exact_match:
                    filtered_items.append(item)
                else:
                    logger.info(
                        f"  (Excluding query image '{item.id}' itself from results)"
                    )

            final_items = filtered_items[:n_results]
            final_results = SearchResults(
                items=final_items, query_vector=query_embedding
            )

            if final_results.is_empty:
                logger.info(
                    "No similar images found (after filtering the exact query match)."
                )
            else:
                logger.info(f"Returning {final_results.count} results after filtering.")

            results_to_return = final_results
        else:
            logger.info(
                "Search successful, but no similar images found in the database initially."
            )
            results_to_return = initial_results

    except (VectorizerError, DatabaseError, ImageProcessingError) as e:
        logger.error(f"Error during image search pipeline: {e}", exc_info=True)
        raise PipelineError(f"Image search failed: {e}") from e
    except FileNotFoundError as e:  # Re-raise specific error
        raise e
    except Exception as e:
        logger.error(
            f"Unexpected error during image search pipeline or filtering: {e}",
            exc_info=True,
        )
        raise PipelineError(f"Unexpected image search failure: {e}") from e

    logger.info("--- Image-to-Image Search Finished ---")
    return results_to_return
