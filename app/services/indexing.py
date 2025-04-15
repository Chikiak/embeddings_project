import logging
import math
import os
import time
from typing import Callable, List, Optional, Tuple

import numpy as np

from app.exceptions import (
    DatabaseError,
    ImageProcessingError,
    PipelineError,
    VectorizerError,
)
from core.image_processor import batch_load_images, find_image_files
from core.vectorizer import Vectorizer
from data_access.chroma_db import ChromaVectorDB
from data_access.vector_db_interface import VectorDBInterface

logger = logging.getLogger(__name__)


def _process_batch_for_indexing(
    batch_paths: List[str],
    vectorizer: Vectorizer,
    db: VectorDBInterface,
    truncate_dim: Optional[int],
    batch_num_info: str,
) -> Tuple[int, int, int, int, int]:
    """
    Procesa un único lote de rutas de imágenes para la indexación,
    verificando primero si las imágenes ya existen en la BD.

    Args:
        batch_paths: Lista de rutas de archivo en este lote.
        vectorizer: Instancia del Vectorizer.
        db: Instancia de la interfaz de BD.
        truncate_dim: Dimensión de truncamiento.
        batch_num_info: String informativo sobre el número de lote.

    Returns:
        Tuple[int, int, int, int, int]: num_loaded, num_vectorized, num_added_to_db, num_failed_db_add, num_skipped
                                         (Contadores para imágenes *nuevas* procesadas en este lote, excepto skipped)
    """
    batch_start_time = time.time()
    num_in_batch = len(batch_paths)

    current_collection_name = getattr(db, "collection_name", "N/A")

    if num_in_batch == 0:
        logger.debug(f"{batch_num_info}: Received empty batch. Skipping.")

        return 0, 0, 0, 0, 0

    logger.info(
        f"--- Processing {batch_num_info} ({num_in_batch} potential files) for Collection '{current_collection_name}' ---"
    )

    num_loaded = 0
    num_vectorized = 0
    num_added_to_db = 0
    num_failed_db_add = 0
    num_skipped = 0
    paths_to_process = []

    try:

        if isinstance(db, ChromaVectorDB) and db.collection is not None:
            absolute_batch_paths = [os.path.abspath(p) for p in batch_paths]
            logger.debug(
                f"{batch_num_info}: Checking existence for {len(absolute_batch_paths)} potential IDs in DB..."
            )

            existing_data = db.collection.get(
                ids=absolute_batch_paths, include=[]
            )
            existing_ids_set = (
                set(existing_data["ids"])
                if existing_data and existing_data.get("ids")
                else set()
            )

            if existing_ids_set:
                logger.info(
                    f"{batch_num_info}: Found {len(existing_ids_set)} images already indexed in this batch."
                )

            paths_to_process = [
                p
                for p, abs_p in zip(batch_paths, absolute_batch_paths)
                if abs_p not in existing_ids_set
            ]
            num_skipped = num_in_batch - len(paths_to_process)

            if num_skipped > 0:
                logger.info(
                    f"{batch_num_info}: Skipping {num_skipped} images already present in the database."
                )

            if not paths_to_process:
                logger.info(
                    f"{batch_num_info}: No new images to process in this batch."
                )

                batch_end_time = time.time()
                logger.info(
                    f"--- {batch_num_info} finished (only checks) in {batch_end_time - batch_start_time:.2f} seconds ---"
                )

                return 0, 0, 0, 0, num_skipped
        else:
            logger.warning(
                f"{batch_num_info}: DB instance is not ChromaDB or collection not available. Cannot check existing IDs. Processing all potential files."
            )
            paths_to_process = batch_paths
            num_skipped = 0

    except (DatabaseError, Exception) as e_get:
        logger.error(
            f"{batch_num_info}: Error checking existing IDs in DB: {e_get}. Processing all potential files as a fallback.",
            exc_info=True,
        )

        paths_to_process = batch_paths
        num_skipped = 0

    try:

        logger.info(
            f"{batch_num_info}: Loading {len(paths_to_process)} new/updated images..."
        )
        loaded_images, loaded_paths = batch_load_images(paths_to_process)
        num_loaded = len(loaded_images)

        if num_loaded == 0:

            logger.warning(
                f"{batch_num_info}: No new images successfully loaded from the filtered list. Skipping rest of batch processing."
            )

            return 0, 0, 0, 0, num_skipped

        logger.info(
            f"{batch_num_info}: Successfully loaded {num_loaded}/{len(paths_to_process)} new images."
        )

        logger.info(
            f"{batch_num_info}: Vectorizing {num_loaded} loaded images (truncate: {truncate_dim or 'Full'})..."
        )
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
                f"CRITICAL WARNING! {batch_num_info}: Mismatch embeddings ({len(embedding_results)}) vs paths ({len(loaded_paths)}). Skipping DB insert for this batch."
            )

            num_failed_db_add = num_loaded
            return num_loaded, 0, 0, num_failed_db_add, num_skipped

        for idx, embedding in enumerate(embedding_results):
            original_path = loaded_paths[idx]
            if embedding is not None and isinstance(embedding, list):
                absolute_path_id = os.path.abspath(original_path)
                valid_embeddings.append(embedding)
                valid_ids.append(absolute_path_id)
                valid_metadatas.append(
                    {
                        "source_path": original_path,
                        "absolute_path": absolute_path_id,
                    }
                )
            else:
                num_failed_vec_in_batch += 1
                logger.warning(
                    f"{batch_num_info}: Failed to vectorize image: {original_path}"
                )

        num_vectorized = len(valid_ids)
        logger.info(
            f"{batch_num_info}: Vectorization resulted in {num_vectorized} valid embeddings for new images."
        )

        if valid_ids:
            logger.info(
                f"{batch_num_info}: Adding/updating {num_vectorized} embeddings to DB '{current_collection_name}'..."
            )

            success = db.add_embeddings(
                ids=valid_ids,
                embeddings=valid_embeddings,
                metadatas=valid_metadatas,
            )
            if success:
                num_added_to_db = num_vectorized
                logger.info(
                    f"{batch_num_info}: Successfully added/updated {num_added_to_db} embeddings."
                )
            else:
                num_failed_db_add = num_vectorized
                logger.error(
                    f"{batch_num_info}: Failed to add batch to database."
                )
        else:
            logger.info(
                f"{batch_num_info}: No valid new embeddings generated to add."
            )

    except (VectorizerError, DatabaseError, ImageProcessingError) as e:
        logger.error(
            f"{batch_num_info}: Error processing batch of new images: {e}",
            exc_info=True,
        )

        num_failed_db_add = num_loaded - num_added_to_db

        raise PipelineError(
            f"Failed processing batch {batch_num_info}: {e}"
        ) from e
    except Exception as e:
        logger.error(
            f"{batch_num_info}: Unexpected error processing batch of new images: {e}",
            exc_info=True,
        )
        num_failed_db_add = num_loaded - num_added_to_db
        raise PipelineError(
            f"Unexpected failure in batch {batch_num_info}: {e}"
        ) from e
    finally:
        batch_end_time = time.time()
        logger.info(
            f"--- {batch_num_info} finished (processing new images) in {batch_end_time - batch_start_time:.2f} seconds ---"
        )

    return (
        num_loaded,
        num_vectorized,
        num_added_to_db,
        num_failed_db_add,
        num_skipped,
    )


def process_directory(
    directory_path: str,
    vectorizer: Vectorizer,
    db: VectorDBInterface,
    batch_size: int,
    truncate_dim: Optional[int],
    progress_callback: Optional[Callable[[int, int], None]] = None,
    status_callback: Optional[Callable[[str], None]] = None,
) -> int:
    """
    Procesa imágenes en un directorio: encuentra, verifica existencia, vectoriza
    y almacena embeddings de imágenes *nuevas* en la instancia de base de datos (`db`) proporcionada.

    Args:
        directory_path: Ruta al directorio que contiene las imágenes.
        vectorizer: Instancia de Vectorizer inicializada.
        db: Instancia de VectorDBInterface inicializada y apuntando a la
            colección CORRECTA para la dimensión `truncate_dim`.
        batch_size: Número de archivos de imagen a procesar en cada lote.
        truncate_dim: Dimensión a la que truncar los embeddings (puede ser None).
        progress_callback: Función opcional llamada con (archivos_procesados, archivos_totales).
        status_callback: Función opcional llamada con (mensaje_estado).

    Returns:
        El número total de imágenes *nuevas* procesadas y almacenadas/actualizadas con éxito en la BD.

    Raises:
        PipelineError: Si ocurre un error irrecuperable durante el pipeline.
    """
    pipeline_start_time = time.time()

    current_collection_name = getattr(db, "collection_name", "N/A")
    effective_dim_str = str(truncate_dim) if truncate_dim else "Full"

    if status_callback:
        status_callback(
            f"Iniciando indexación para: {directory_path} en Colección '{current_collection_name}' (Dim: {effective_dim_str})"
        )
    logger.info(
        f"--- Starting Image Processing Pipeline for Directory: {directory_path} ---"
    )
    logger.info(f"  Using DB Collection: '{current_collection_name}'")
    logger.info(f"  Using Vectorizer Model: {vectorizer.model_name}")
    logger.info(f"  Target Embedding Dimension: {effective_dim_str}")
    logger.info(f"  Processing Batch Size: {batch_size}")

    if not db.is_initialized:
        msg = f"Database (Collection: '{current_collection_name}') is not initialized. Aborting pipeline."
        logger.error(msg)
        if status_callback:
            status_callback(f"Error: {msg}")
        raise PipelineError(msg)

    if not os.path.isdir(directory_path):
        msg = f"Input directory not found or is not a directory: {directory_path}"
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
            status_callback(f"Error buscando archivos: {e}")
        raise PipelineError(msg) from e

    if not image_paths:
        msg = f"No compatible image files found in {directory_path}."
        logger.warning(msg)
        if status_callback:
            status_callback(msg)
        return 0

    total_files_found = len(image_paths)
    logger.info(f"Found {total_files_found} potential image files to process.")
    if status_callback:
        status_callback(
            f"Encontrados {total_files_found} archivos potenciales."
        )
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
    total_skipped = 0

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch_paths = image_paths[start_idx:end_idx]
        current_batch_num = i + 1
        num_in_this_batch = len(batch_paths)

        batch_num_info = f"Lote {current_batch_num}/{num_batches} (Col: {current_collection_name})"

        if status_callback:
            status_callback(
                f"Procesando {batch_num_info} ({num_in_this_batch} archivos potenciales)"
            )

        try:

            (
                num_loaded,
                num_vectorized,
                num_added,
                num_failed_db,
                num_skipped,
            ) = _process_batch_for_indexing(
                batch_paths=batch_paths,
                vectorizer=vectorizer,
                db=db,
                truncate_dim=truncate_dim,
                batch_num_info=batch_num_info,
            )

            total_skipped += num_skipped

        except PipelineError as e:

            logger.error(
                f"Pipeline stopped due to critical error in {batch_num_info}.",
                exc_info=False,
            )
            if status_callback:
                status_callback(
                    f"Error crítico en {batch_num_info}, pipeline detenido."
                )

            raise e

        num_potentially_new = num_in_this_batch - num_skipped
        total_failed_loading += num_potentially_new - num_loaded
        total_failed_vectorizing += num_loaded - num_vectorized
        total_successfully_stored += num_added
        total_failed_db_add += num_failed_db

        total_files_processed_for_progress += num_in_this_batch
        if progress_callback:
            progress_callback(
                total_files_processed_for_progress, total_files_found
            )

        if status_callback:
            status_parts = [f"{batch_num_info}: Completado."]
            if num_skipped > 0:
                status_parts.append(f"Saltados(existentes): {num_skipped}.")

            if num_potentially_new - num_loaded > 0:
                status_parts.append(
                    f"Fallo carga nuevos: {num_potentially_new - num_loaded}."
                )
            if num_loaded - num_vectorized > 0:
                status_parts.append(
                    f"Fallo vectoriz. nuevos: {num_loaded - num_vectorized}."
                )
            if num_failed_db > 0:
                status_parts.append(f"Fallo BD nuevos: {num_failed_db}.")
            if num_added > 0:
                status_parts.append(f"Guardados nuevos: {num_added}.")
            status_callback(" ".join(status_parts))

    pipeline_end_time = time.time()
    total_time = pipeline_end_time - pipeline_start_time
    logger.info(
        f"--- Image Processing Pipeline Finished in {total_time:.2f} seconds ---"
    )
    logger.info(
        f"Summary for directory {directory_path} on collection '{current_collection_name}':"
    )
    logger.info(f"  Total potential files found: {total_files_found}")
    logger.info(f"  Skipped (already indexed): {total_skipped}")
    logger.info(
        f"  New files successfully stored/updated in DB: {total_successfully_stored}"
    )
    logger.info(f"  Failed to load (new files): {total_failed_loading}")
    logger.info(
        f"  Failed to vectorize (new files): {total_failed_vectorizing}"
    )
    logger.info(f"  Failed during DB add (new files): {total_failed_db_add}")

    if status_callback:
        status_callback(
            f"Pipeline finalizado en {total_time:.2f}s. "
            f"Imágenes nuevas guardadas/actualizadas: {total_successfully_stored}/{total_files_found - total_skipped}. "
            f"Imágenes saltadas (ya existían): {total_skipped}."
        )

    return total_successfully_stored
