# app/indexing.py
import math
import os
import time
import logging
from typing import Optional, List, Tuple, Callable, Any, Dict # Added Any, Dict
import numpy as np # Added numpy

# Imports internos del proyecto
from core.image_processor import find_image_files, batch_load_images
from core.vectorizer import Vectorizer
from data_access.vector_db_interface import VectorDBInterface
# from app.models import SearchResults, SearchResultItem # No se usan directamente aquí
from app.exceptions import (
    PipelineError,
    DatabaseError,
    VectorizerError,
    ImageProcessingError,
)
# from config import DEFAULT_N_RESULTS # No se usa directamente aquí

logger = logging.getLogger(__name__)


def _process_batch_for_indexing(
    batch_paths: List[str],
    vectorizer: Vectorizer, # Passed as argument
    db: VectorDBInterface,  # Passed as argument
    truncate_dim: Optional[int], # Passed as argument
    batch_num_info: str,
) -> Tuple[int, int, int, int]:
    """
    Procesa un único lote de rutas de imágenes para la indexación.

    Carga, vectoriza y añade embeddings válidos a la base de datos.
    Utiliza rutas absolutas como IDs.

    Args:
        batch_paths: Lista de rutas de imágenes en el lote.
        vectorizer: Instancia de Vectorizer inicializada (para el modelo correcto).
        db: Instancia que cumple con VectorDBInterface.
        truncate_dim: Dimensión a la que truncar los embeddings (puede ser None).
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
        # Usa la función de core.image_processor
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
        # Usa la instancia de vectorizer proporcionada y el truncate_dim
        embedding_results = vectorizer.vectorize_images(
            loaded_images,
            batch_size=num_loaded, # Vectorize all loaded images in one go within this function
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
            # Consider how to handle this - maybe return partial success? For now, failing the batch.
            return num_loaded, 0, 0, num_in_batch # Count all as failed DB add for this batch

        for idx, embedding in enumerate(embedding_results):
            original_path = loaded_paths[idx]
            if embedding is not None and isinstance(embedding, list):
                # Convierte la ruta original a absoluta ANTES de usarla como ID
                absolute_path_id = os.path.abspath(original_path)
                logger.debug(f"Using absolute path as ID: '{absolute_path_id}' (Original: '{original_path}')")

                valid_embeddings.append(embedding)
                valid_ids.append(absolute_path_id) # Almacena la ruta absoluta como ID
                # Guarda ambas rutas en metadatos para referencia (opcional pero útil)
                valid_metadatas.append({
                    "source_path": original_path,     # La ruta original tal como se encontró
                    "absolute_path": absolute_path_id # La ruta absoluta usada como ID
                })
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
                f"{batch_num_info}: Adding {num_vectorized} valid embeddings (using absolute paths as IDs) to the database..."
            )
            # Usa la instancia de db proporcionada
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
        # Propagate error to stop the main loop or handle as needed
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
    vectorizer: Vectorizer, # Passed as argument
    db: VectorDBInterface,  # Passed as argument
    batch_size: int,        # Passed as argument
    truncate_dim: Optional[int], # Passed as argument
    progress_callback: Optional[Callable[[int, int], None]] = None,
    status_callback: Optional[Callable[[str], None]] = None,
) -> int:
    """
    Procesa imágenes en un directorio: encuentra, vectoriza y almacena embeddings.

    Utiliza instancias específicas de Vectorizer y VectorDBInterface.
    Almacena rutas absolutas como IDs.

    Args:
        directory_path: Ruta al directorio que contiene las imágenes.
        vectorizer: Instancia de Vectorizer inicializada (para el modelo y dimensión correctos).
        db: Instancia inicializada que cumple con VectorDBInterface.
        batch_size: Número de archivos de imagen a procesar en cada lote.
        truncate_dim: Dimensión a la que truncar los embeddings (puede ser None).
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
    logger.info(f"  Using Vectorizer Model: {vectorizer.model_name}")
    logger.info(f"  Target Embedding Dimension: {truncate_dim or 'Full'}")
    logger.info(f"  Processing Batch Size: {batch_size}")
    logger.info(f"  Storing Absolute Paths as IDs: True") # Log the change


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
        # Usa la función de core.image_processor
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
            # Pasa los argumentos correctos a la función interna _process_batch_for_indexing
            num_loaded, num_vectorized, num_added, num_failed_db = (
                _process_batch_for_indexing(
                    batch_paths=batch_paths,
                    vectorizer=vectorizer,
                    db=db,
                    truncate_dim=truncate_dim,
                    batch_num_info=batch_num_info
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
        # Correct calculation for vectorization failures within the batch
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
            # Correct status reporting for vectorization failures
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
