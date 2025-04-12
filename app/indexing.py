# app/indexing.py
import math
import os
import time
import logging
from typing import Optional, List, Tuple, Callable, Any, Dict
import numpy as np

# Imports internos del proyecto
from core.image_processor import find_image_files, batch_load_images
from core.vectorizer import Vectorizer
from data_access.vector_db_interface import VectorDBInterface # Mantenido
# Importar ChromaVectorDB específicamente para acceder a 'collection.get'
# Nota: Idealmente, la interfaz tendría un método 'check_ids_exist'
from data_access.chroma_db import ChromaVectorDB
from app.exceptions import (
    PipelineError,
    DatabaseError,
    VectorizerError,
    ImageProcessingError,
)

logger = logging.getLogger(__name__)


def _process_batch_for_indexing(
    batch_paths: List[str],
    vectorizer: Vectorizer,
    db: VectorDBInterface,  # Recibe la instancia de BD correcta
    truncate_dim: Optional[int],
    batch_num_info: str,
) -> Tuple[int, int, int, int, int]: # <-- Añadido un valor de retorno para 'skipped'
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
    # Usa getattr para obtener el nombre de forma segura
    current_collection_name = getattr(db, 'collection_name', 'N/A')

    if num_in_batch == 0:
        logger.debug(f"{batch_num_info}: Received empty batch. Skipping.")
        # loaded, vectorized, added_db, failed_db, skipped
        return 0, 0, 0, 0, 0

    logger.info(f"--- Processing {batch_num_info} ({num_in_batch} potential files) for Collection '{current_collection_name}' ---")

    num_loaded = 0
    num_vectorized = 0
    num_added_to_db = 0
    num_failed_db_add = 0
    num_skipped = 0
    paths_to_process = [] # Lista de rutas que realmente necesitan procesamiento

    # --- INICIO: Verificación de IDs Existentes ---
    try:
        # Comprueba si es una instancia de ChromaDB para usar .collection.get
        # Idealmente, esto estaría en la interfaz VectorDBInterface
        if isinstance(db, ChromaVectorDB) and db.collection is not None:
            absolute_batch_paths = [os.path.abspath(p) for p in batch_paths]
            logger.debug(f"{batch_num_info}: Checking existence for {len(absolute_batch_paths)} potential IDs in DB...")

            # Consulta ChromaDB para ver qué IDs de este lote ya existen
            # include=[] hace que sea más rápido ya que solo necesitamos saber si existen los IDs
            existing_data = db.collection.get(ids=absolute_batch_paths, include=[])
            existing_ids_set = set(existing_data['ids']) if existing_data and existing_data.get('ids') else set()

            if existing_ids_set:
                logger.info(f"{batch_num_info}: Found {len(existing_ids_set)} images already indexed in this batch.")

            # Filtra batch_paths para incluir solo aquellos cuya ruta absoluta NO está en el conjunto existente
            paths_to_process = [
                p for p, abs_p in zip(batch_paths, absolute_batch_paths)
                if abs_p not in existing_ids_set
            ]
            num_skipped = num_in_batch - len(paths_to_process)

            if num_skipped > 0:
                logger.info(f"{batch_num_info}: Skipping {num_skipped} images already present in the database.")

            if not paths_to_process:
                logger.info(f"{batch_num_info}: No new images to process in this batch.")
                # No hay nada que cargar/vectorizar/añadir, así que podemos retornar temprano
                batch_end_time = time.time()
                logger.info(f"--- {batch_num_info} finished (only checks) in {batch_end_time - batch_start_time:.2f} seconds ---")
                # loaded, vectorized, added_db, failed_db, skipped
                return 0, 0, 0, 0, num_skipped
        else:
             logger.warning(f"{batch_num_info}: DB instance is not ChromaDB or collection not available. Cannot check existing IDs. Processing all potential files.")
             paths_to_process = batch_paths # Procesar todo como fallback si no es ChromaDB o no está lista
             num_skipped = 0

    except (DatabaseError, Exception) as e_get:
        logger.error(f"{batch_num_info}: Error checking existing IDs in DB: {e_get}. Processing all potential files as a fallback.", exc_info=True)
        # Fallback: Si la verificación falla, procesa todas las rutas para evitar perder datos nuevos
        paths_to_process = batch_paths
        num_skipped = 0 # No podemos confirmar que se saltó algo
    # --- FIN: Verificación de IDs Existentes ---


    # --- Procesamiento Normal (solo con paths_to_process) ---
    try:
        # Usa la lista filtrada 'paths_to_process' aquí
        logger.info(f"{batch_num_info}: Loading {len(paths_to_process)} new/updated images...")
        loaded_images, loaded_paths = batch_load_images(paths_to_process) # Usa la lista filtrada
        num_loaded = len(loaded_images)

        if num_loaded == 0:
            # Esto puede ocurrir si los archivos filtrados fallan al cargar
            logger.warning(f"{batch_num_info}: No new images successfully loaded from the filtered list. Skipping rest of batch processing.")
            # Los 'skipped' ya se contaron arriba. No se cargó, vectorizó ni añadió nada nuevo.
            return 0, 0, 0, 0, num_skipped

        logger.info(f"{batch_num_info}: Successfully loaded {num_loaded}/{len(paths_to_process)} new images.")

        logger.info(f"{batch_num_info}: Vectorizing {num_loaded} loaded images (truncate: {truncate_dim or 'Full'})...")
        embedding_results = vectorizer.vectorize_images(
            loaded_images,
            batch_size=num_loaded, # Procesar todo el sublote cargado
            truncate_dim=truncate_dim,
        )

        valid_embeddings = []
        valid_ids = []
        valid_metadatas = []
        num_failed_vec_in_batch = 0

        if len(embedding_results) != len(loaded_paths):
            logger.error(f"CRITICAL WARNING! {batch_num_info}: Mismatch embeddings ({len(embedding_results)}) vs paths ({len(loaded_paths)}). Skipping DB insert for this batch.")
            # Contar todos los cargados como fallos de DB porque no podemos asegurar la correspondencia
            num_failed_db_add = num_loaded
            return num_loaded, 0, 0, num_failed_db_add, num_skipped

        for idx, embedding in enumerate(embedding_results):
            original_path = loaded_paths[idx]
            if embedding is not None and isinstance(embedding, list):
                absolute_path_id = os.path.abspath(original_path) # ID usado en la BD
                valid_embeddings.append(embedding)
                valid_ids.append(absolute_path_id)
                valid_metadatas.append({
                    "source_path": original_path,
                    "absolute_path": absolute_path_id
                    # Puedes añadir más metadatos aquí si es necesario (ej: timestamp)
                })
            else:
                num_failed_vec_in_batch += 1
                logger.warning(f"{batch_num_info}: Failed to vectorize image: {original_path}")

        num_vectorized = len(valid_ids)
        logger.info(f"{batch_num_info}: Vectorization resulted in {num_vectorized} valid embeddings for new images.")

        if valid_ids:
            logger.info(f"{batch_num_info}: Adding/updating {num_vectorized} embeddings to DB '{current_collection_name}'...")
            # Usa la instancia de db proporcionada
            success = db.add_embeddings(
                ids=valid_ids,
                embeddings=valid_embeddings,
                metadatas=valid_metadatas,
            )
            if success:
                num_added_to_db = num_vectorized
                logger.info(f"{batch_num_info}: Successfully added/updated {num_added_to_db} embeddings.")
            else:
                num_failed_db_add = num_vectorized
                logger.error(f"{batch_num_info}: Failed to add batch to database.")
        else:
            logger.info(f"{batch_num_info}: No valid new embeddings generated to add.")

    except (VectorizerError, DatabaseError, ImageProcessingError) as e:
        logger.error(f"{batch_num_info}: Error processing batch of new images: {e}", exc_info=True)
        # Los 'skipped' se mantienen, el resto falló.
        # Asumimos que todos los 'num_loaded' que no se añadieron, fallaron en algún punto.
        num_failed_db_add = num_loaded - num_added_to_db
        # Propagar el error detendrá el pipeline general en process_directory
        raise PipelineError(f"Failed processing batch {batch_num_info}: {e}") from e
    except Exception as e:
        logger.error(f"{batch_num_info}: Unexpected error processing batch of new images: {e}", exc_info=True)
        num_failed_db_add = num_loaded - num_added_to_db
        raise PipelineError(f"Unexpected failure in batch {batch_num_info}: {e}") from e
    finally:
        batch_end_time = time.time()
        logger.info(f"--- {batch_num_info} finished (processing new images) in {batch_end_time - batch_start_time:.2f} seconds ---")

    # Devuelve los contadores del procesamiento de imágenes *nuevas* en este lote
    return num_loaded, num_vectorized, num_added_to_db, num_failed_db_add, num_skipped
# --- FIN _process_batch_for_indexing ---


# --- MODIFICADO process_directory ---
def process_directory(
    directory_path: str,
    vectorizer: Vectorizer,
    db: VectorDBInterface, # <-- AHORA RECIBE LA INSTANCIA DE BD CORRECTA (determinada por el llamador)
    batch_size: int,
    truncate_dim: Optional[int], # Dimensión seleccionada por el usuario
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
    # Obtener nombre de la colección de forma segura
    current_collection_name = getattr(db, 'collection_name', 'N/A')
    effective_dim_str = str(truncate_dim) if truncate_dim else 'Full'

    if status_callback: status_callback(f"Iniciando indexación para: {directory_path} en Colección '{current_collection_name}' (Dim: {effective_dim_str})")
    logger.info(f"--- Starting Image Processing Pipeline for Directory: {directory_path} ---")
    logger.info(f"  Using DB Collection: '{current_collection_name}'") # Loguear la colección que se está usando
    logger.info(f"  Using Vectorizer Model: {vectorizer.model_name}")
    logger.info(f"  Target Embedding Dimension: {effective_dim_str}")
    logger.info(f"  Processing Batch Size: {batch_size}")

    if not db.is_initialized: # Comprobar la BD recibida
        msg = f"Database (Collection: '{current_collection_name}') is not initialized. Aborting pipeline."
        logger.error(msg)
        if status_callback: status_callback(f"Error: {msg}")
        raise PipelineError(msg)

    if not os.path.isdir(directory_path):
        msg = f"Input directory not found or is not a directory: {directory_path}"
        logger.error(msg)
        if status_callback: status_callback(f"Error: {msg}")
        raise PipelineError(msg)

    if status_callback: status_callback("Buscando archivos de imagen...")
    try:
        image_paths = find_image_files(directory_path)
    except Exception as e:
        msg = f"Failed to find image files in {directory_path}: {e}"
        logger.error(msg, exc_info=True)
        if status_callback: status_callback(f"Error buscando archivos: {e}")
        raise PipelineError(msg) from e

    if not image_paths:
        msg = f"No compatible image files found in {directory_path}."
        logger.warning(msg)
        if status_callback: status_callback(msg)
        return 0 # No hay nada que procesar

    total_files_found = len(image_paths)
    logger.info(f"Found {total_files_found} potential image files to process.")
    if status_callback: status_callback(f"Encontrados {total_files_found} archivos potenciales.")
    if progress_callback: progress_callback(0, total_files_found)

    num_batches = math.ceil(total_files_found / batch_size)
    logger.info(f"Processing in approximately {num_batches} batches of size {batch_size}.")

    # Contadores totales para el resumen final
    total_successfully_stored = 0 # Nuevas imágenes añadidas/actualizadas
    total_files_processed_for_progress = 0 # Archivos totales considerados (incluye saltados)
    total_failed_loading = 0 # Fallos al cargar imágenes *nuevas*
    total_failed_vectorizing = 0 # Fallos al vectorizar imágenes *nuevas*
    total_failed_db_add = 0 # Fallos al añadir imágenes *nuevas* a la BD
    total_skipped = 0 # Imágenes saltadas porque ya existían

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch_paths = image_paths[start_idx:end_idx]
        current_batch_num = i + 1
        num_in_this_batch = len(batch_paths) # Número potencial en el lote
        # Añadir nombre de colección al info del lote
        batch_num_info = f"Lote {current_batch_num}/{num_batches} (Col: {current_collection_name})"

        if status_callback: status_callback(f"Procesando {batch_num_info} ({num_in_this_batch} archivos potenciales)")

        try:
            # Llama a la función modificada y recibe 5 valores
            num_loaded, num_vectorized, num_added, num_failed_db, num_skipped = ( # <-- Recibe num_skipped
                _process_batch_for_indexing(
                    batch_paths=batch_paths,
                    vectorizer=vectorizer,
                    db=db, # Pasa la instancia de db correcta
                    truncate_dim=truncate_dim, # Pasa la dimensión de truncamiento
                    batch_num_info=batch_num_info
                )
            )
            # Actualiza el contador total de saltados
            total_skipped += num_skipped # <-- ACTUALIZA TOTAL SKIPPED

        except PipelineError as e:
            # El error ya se logueó en _process_batch_for_indexing
            logger.error(f"Pipeline stopped due to critical error in {batch_num_info}.", exc_info=False)
            if status_callback: status_callback(f"Error crítico en {batch_num_info}, pipeline detenido.")
            # Propaga el error para detener todo el pipeline
            raise e

        # Actualizar contadores totales (basados en los *nuevos* procesados)
        num_potentially_new = num_in_this_batch - num_skipped
        total_failed_loading += (num_potentially_new - num_loaded) # Fallos al cargar de los potencialmente nuevos
        total_failed_vectorizing += (num_loaded - num_vectorized) # Fallos al vectoriz. de los nuevos cargados
        total_successfully_stored += num_added
        total_failed_db_add += num_failed_db

        # El progreso se basa en los archivos *intentados* (incluyendo los saltados)
        total_files_processed_for_progress += num_in_this_batch
        if progress_callback: progress_callback(total_files_processed_for_progress, total_files_found)

        # Mensaje de estado del lote
        if status_callback:
            status_parts = [f"{batch_num_info}: Completado."]
            if num_skipped > 0: status_parts.append(f"Saltados(existentes): {num_skipped}.")
            # Los siguientes contadores se refieren a imágenes NUEVAS procesadas en este lote
            if num_potentially_new - num_loaded > 0: status_parts.append(f"Fallo carga nuevos: {num_potentially_new - num_loaded}.")
            if num_loaded - num_vectorized > 0: status_parts.append(f"Fallo vectoriz. nuevos: {num_loaded - num_vectorized}.")
            if num_failed_db > 0: status_parts.append(f"Fallo BD nuevos: {num_failed_db}.")
            if num_added > 0: status_parts.append(f"Guardados nuevos: {num_added}.")
            status_callback(" ".join(status_parts))


    pipeline_end_time = time.time()
    total_time = pipeline_end_time - pipeline_start_time
    logger.info(f"--- Image Processing Pipeline Finished in {total_time:.2f} seconds ---")
    logger.info(f"Summary for directory {directory_path} on collection '{current_collection_name}':")
    logger.info(f"  Total potential files found: {total_files_found}")
    logger.info(f"  Skipped (already indexed): {total_skipped}") # <-- NUEVO LOG
    logger.info(f"  New files successfully stored/updated in DB: {total_successfully_stored}")
    logger.info(f"  Failed to load (new files): {total_failed_loading}")
    logger.info(f"  Failed to vectorize (new files): {total_failed_vectorizing}")
    logger.info(f"  Failed during DB add (new files): {total_failed_db_add}")


    if status_callback:
        status_callback(
            f"Pipeline finalizado en {total_time:.2f}s. "
            f"Imágenes nuevas guardadas/actualizadas: {total_successfully_stored}/{total_files_found - total_skipped}. " # Muestra sobre las potencialmente nuevas
            f"Imágenes saltadas (ya existían): {total_skipped}." # <-- NUEVO MENSAJE STATUS
        )

    # Devuelve solo las procesadas con éxito *nuevas*
    return total_successfully_stored
