# --- app/cli_handlers.py ---
import logging
import os
import time
from argparse import Namespace
from typing import Optional

import config # Importa config actualizado
from app import indexing, searching
from app.exceptions import DatabaseError, InitializationError, PipelineError
# Importa la función refactorizada para obtener la BD correcta
from app.factory import get_or_create_db_for_model_and_dim
from core.vectorizer import Vectorizer
from data_access.vector_db_interface import VectorDBInterface

logger = logging.getLogger(__name__)

# Deprecated: _get_target_db_for_cli ya no es necesario, usamos get_or_create_db_for_model_and_dim
# def _get_target_db_for_cli(...) -> VectorDBInterface: ...

def handle_indexing(args: Namespace, vectorizer: Vectorizer):
    """Maneja la acción --index."""
    logger.info(f"--- ACTION: Indexing Directory: {args.image_dir} ---")
    if not os.path.isdir(args.image_dir):
        logger.error(f"Image directory not found: '{args.image_dir}'")
        return

    try:
        # Obtiene la instancia de BD correcta basada en el vectorizador y truncate_dim
        db_to_use = get_or_create_db_for_model_and_dim(vectorizer, args.truncate_dim)
        collection_name = getattr(db_to_use, 'collection_name', 'N/A')

        logger.info(f"Starting indexing process on collection '{collection_name}'...")
        start_index_time = time.time()
        processed_count = indexing.process_directory(
            directory_path=args.image_dir,
            vectorizer=vectorizer,
            db=db_to_use,
            batch_size=config.BATCH_SIZE_IMAGES, # Usa config para batch size
            truncate_dim=args.truncate_dim,
            # No hay callbacks en CLI
        )
        end_index_time = time.time()
        logger.info(f"Finished indexing in {end_index_time - start_index_time:.2f}s. Stored/updated {processed_count} embeddings in '{collection_name}'.")

    except (PipelineError, InitializationError, ValueError, DatabaseError) as e:
        logger.error(f"Indexing pipeline failed: {e}", exc_info=False) # No mostrar traceback para errores esperados
    except Exception as e:
        logger.error(f"Unexpected error during indexing: {e}", exc_info=True) # Mostrar traceback para inesperados

def handle_text_search(args: Namespace, vectorizer: Vectorizer):
    """Maneja --search-text."""
    logger.info(f"--- ACTION: Text Search for query: '{args.search_text}' ---")
    if not args.search_text:
        logger.error("No text query provided for --search-text.")
        return

    try:
        db_to_use = get_or_create_db_for_model_and_dim(vectorizer, args.truncate_dim)
        collection_name = getattr(db_to_use, 'collection_name', 'N/A')

        if not db_to_use.is_initialized or db_to_use.count() <= 0:
            logger.warning(f"Cannot search: Collection '{collection_name}' is empty or not initialized.")
            return

        logger.info(f"Performing search on collection '{collection_name}'...")
        start_search_time = time.time()
        search_results = searching.search_by_text(
            query_text=args.search_text,
            vectorizer=vectorizer,
            db=db_to_use,
            n_results=args.n_results,
            truncate_dim=args.truncate_dim,
        )
        end_search_time = time.time()
        logger.info(f"Search completed in {end_search_time - start_search_time:.2f}s.")

        # Muestra resultados (simplificado)
        if search_results.is_empty:
            logger.info("Search successful, but no similar images found.")
        else:
            logger.info(f"\nTop {search_results.count} matching images for query:")
            for i, item in enumerate(search_results.items):
                dist_str = f"{item.distance:.4f}" if item.distance is not None else "N/A"
                # Simplificado: Muestra solo ID y Distancia
                logger.info(f"  {i + 1}. ID: {item.id} (Dist: {dist_str})")

    except (PipelineError, InitializationError, ValueError, DatabaseError) as e:
        logger.error(f"Text search failed: {e}", exc_info=False)
    except Exception as e:
        logger.error(f"Unexpected error during text search: {e}", exc_info=True)

def handle_image_search(args: Namespace, vectorizer: Vectorizer):
    """Maneja --search-image."""
    logger.info(f"--- ACTION: Image Search using image: '{args.search_image}' ---")
    if not args.search_image:
         logger.error("No image path provided for --search-image.")
         return
    if not os.path.isfile(args.search_image):
        logger.error(f"Query image file not found: '{args.search_image}'.")
        return

    try:
        db_to_use = get_or_create_db_for_model_and_dim(vectorizer, args.truncate_dim)
        collection_name = getattr(db_to_use, 'collection_name', 'N/A')

        if not db_to_use.is_initialized or db_to_use.count() <= 0:
            logger.warning(f"Cannot search: Collection '{collection_name}' is empty or not initialized.")
            return

        logger.info(f"Performing search on collection '{collection_name}'...")
        start_search_time = time.time()
        search_results = searching.search_by_image(
            query_image_path=args.search_image,
            vectorizer=vectorizer,
            db=db_to_use,
            n_results=args.n_results,
            truncate_dim=args.truncate_dim,
        )
        end_search_time = time.time()
        logger.info(f"Search completed in {end_search_time - start_search_time:.2f}s.")

        # Muestra resultados (simplificado)
        if search_results.is_empty:
            logger.info("Search successful, but no similar images found (after filtering).")
        else:
            logger.info(f"\nTop {search_results.count} similar images found:")
            for i, item in enumerate(search_results.items):
                 dist_str = f"{item.distance:.4f}" if item.distance is not None else "N/A"
                 logger.info(f"  {i + 1}. ID: {item.id} (Dist: {dist_str})")

    except FileNotFoundError:
         # Ya manejado al inicio
         pass
    except (PipelineError, InitializationError, ValueError, DatabaseError) as e:
        logger.error(f"Image search failed: {e}", exc_info=False)
    except Exception as e:
        logger.error(f"Unexpected error during image search: {e}", exc_info=True)

def _confirm_action(prompt_message: str, confirmation_keyword: Optional[str] = None) -> bool:
    """Función auxiliar para obtener confirmación del usuario (sin cambios)."""
    try:
        confirm = input(f"{prompt_message} [yes/NO]: ")
        if confirm.lower() != "yes":
            return False
        if confirmation_keyword:
            confirm2 = input(f"Please type '{confirmation_keyword}' to confirm: ")
            if confirm2 != confirmation_keyword:
                return False
        return True
    except EOFError:
         logger.warning("EOFError during confirmation prompt (running non-interactively?). Assuming NO.")
         return False # Asume NO si no es interactivo

def handle_clear_collection(args: Namespace, vectorizer: Vectorizer):
    """Maneja --clear-collection."""
    try:
        db_to_use = get_or_create_db_for_model_and_dim(vectorizer, args.truncate_dim)
        collection_name = getattr(db_to_use, 'collection_name', 'N/A')

        logger.warning(f"ACTION: Clearing all items from collection '{collection_name}'...")
        prompt = f"Are you sure you want to clear ALL items from '{collection_name}'?"
        if _confirm_action(prompt):
            if not db_to_use.is_initialized:
                logger.error(f"Cannot clear: Collection '{collection_name}' is not initialized.")
                return
            success = db_to_use.clear_collection()
            if success:
                logger.info(f"Collection '{collection_name}' cleared successfully.")
            else:
                logger.error(f"Failed to clear collection '{collection_name}'.")
        else:
            logger.info("Clear operation cancelled by user.")
    except (DatabaseError, PipelineError, InitializationError, ValueError) as e:
        logger.error(f"Error during clear_collection: {e}", exc_info=False)
    except Exception as e:
        logger.error(f"Unexpected error during clear_collection: {e}", exc_info=True)

def handle_delete_collection(args: Namespace, vectorizer: Vectorizer):
    """Maneja --delete-collection."""
    try:
        db_to_use = get_or_create_db_for_model_and_dim(vectorizer, args.truncate_dim)
        collection_name = getattr(db_to_use, 'collection_name', 'N/A') # Obtiene el nombre antes de intentar borrar

        logger.warning(f"ACTION: Deleting the ENTIRE collection '{collection_name}'...")
        prompt = f"Are you absolutely sure you want to DELETE the collection '{collection_name}'? This is IRREVERSIBLE."
        if _confirm_action(prompt, confirmation_keyword="DELETE"):
            # No necesitamos verificar is_initialized aquí, delete_collection lo maneja
            success = db_to_use.delete_collection() # delete_collection ahora está en la interfaz
            if success:
                logger.info(f"Collection '{collection_name}' deleted successfully.")
            else:
                # El error ya debería haber sido logueado por delete_collection
                logger.error(f"Failed to delete collection '{collection_name}'. Check previous logs.")
        else:
            logger.info("Delete operation cancelled by user.")
    except (DatabaseError, PipelineError, InitializationError, ValueError) as e:
        logger.error(f"Error during delete_collection: {e}", exc_info=False)
    except Exception as e:
        logger.error(f"Unexpected error during delete_collection: {e}", exc_info=True)
