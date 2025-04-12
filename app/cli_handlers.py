# app/cli_handlers.py
import os
import logging
import time
from argparse import Namespace

# Imports internos del proyecto
from core.vectorizer import Vectorizer
from data_access.vector_db_interface import VectorDBInterface
# --- Importar desde los nuevos módulos ---
from app import indexing # Importa el módulo de indexación
from app import searching # Importa el módulo de búsqueda
# --------------------------------------
from app.exceptions import PipelineError, DatabaseError
import config

logger = logging.getLogger(__name__)


def handle_indexing(args: Namespace, vectorizer: Vectorizer, db: VectorDBInterface):
    """
    Maneja la acción --index del CLI.

    Procesa el directorio de imágenes especificado, vectoriza las imágenes
    y las almacena en la base de datos vectorial.

    Args:
        args: Objeto Namespace de argparse con los argumentos del CLI.
        vectorizer: Instancia de Vectorizer inicializada.
        db: Instancia de VectorDBInterface inicializada.
    """
    logger.info(f"--- ACTION: Processing Image Directory: {args.image_dir} ---")
    if not os.path.isdir(args.image_dir):
        logger.error(f"Image directory not found: '{args.image_dir}'.")
        try:
            logger.info(f"Attempting to create directory: '{args.image_dir}'")
            os.makedirs(args.image_dir)
            logger.info(
                f"Directory '{args.image_dir}' created. Please add images and re-run --index."
            )
        except OSError as e:
            logger.error(f"Could not create directory '{args.image_dir}': {e}")
        return

    try:
        # --- Llamar a la función desde el módulo indexing ---
        processed_count = indexing.process_directory(
            directory_path=args.image_dir,
            vectorizer=vectorizer,
            db=db,
            batch_size=config.BATCH_SIZE_IMAGES,
            truncate_dim=args.truncate_dim,
        )
        # ---------------------------------------------------
        logger.info(
            f"Finished processing directory. Stored/updated {processed_count} embeddings."
        )
    except PipelineError as e:
        logger.error(f"Indexing pipeline failed: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected error during indexing: {e}", exc_info=True)


def handle_text_search(args: Namespace, vectorizer: Vectorizer, db: VectorDBInterface):
    """
    Maneja la acción --search-text del CLI.

    Realiza una búsqueda de imágenes basada en una consulta de texto.

    Args:
        args: Objeto Namespace de argparse con los argumentos del CLI.
        vectorizer: Instancia de Vectorizer inicializada.
        db: Instancia de VectorDBInterface inicializada.
    """
    logger.info(f"--- ACTION: Text-to-Image Search for query: '{args.search_text}' ---")
    if not db.is_initialized or db.count() == 0:
        logger.warning(
            "Cannot perform search: The database is empty or not initialized."
        )
        return

    start_search_time = time.time()
    try:
        # --- Llamar a la función desde el módulo searching ---
        search_results = searching.search_by_text(
            query_text=args.search_text,
            vectorizer=vectorizer,
            db=db,
            n_results=args.n_results,
            truncate_dim=args.truncate_dim,
        )
        # ----------------------------------------------------
        end_search_time = time.time()
        logger.info(f"Search completed in {end_search_time - start_search_time:.2f}s.")

        if search_results.is_empty:
            logger.info(
                "Search successful, but no similar images found in the database."
            )
        else:
            logger.info(f"\nTop {search_results.count} matching images for query:")
            for i, item in enumerate(search_results.items):
                similarity_str = (
                    f"{item.similarity:.4f}" if item.similarity is not None else "N/A"
                )
                logger.info(
                    f"  {i + 1}. ID: {item.id} (Similarity: {similarity_str}) Metadata: {item.metadata}"
                )

    except PipelineError as e:
        logger.error(f"Text search failed: {e}", exc_info=True)
    except ValueError as e:
        logger.error(f"Invalid input for text search: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during text search: {e}", exc_info=True)


def handle_image_search(args: Namespace, vectorizer: Vectorizer, db: VectorDBInterface):
    """
    Maneja la acción --search-image del CLI.

    Realiza una búsqueda de imágenes basada en una imagen de consulta.

    Args:
        args: Objeto Namespace de argparse con los argumentos del CLI.
        vectorizer: Instancia de Vectorizer inicializada.
        db: Instancia de VectorDBInterface inicializada.
    """
    logger.info(
        f"--- ACTION: Image-to-Image Search using image: '{args.search_image}' ---"
    )
    if not os.path.isfile(args.search_image):
        logger.error(
            f"Query image file not found: '{args.search_image}'. Please check the path."
        )
        return
    if not db.is_initialized or db.count() == 0:
        logger.warning(
            "Cannot perform search: The database is empty or not initialized."
        )
        return

    start_search_time = time.time()
    try:
        # --- Llamar a la función desde el módulo searching ---
        search_results = searching.search_by_image(
            query_image_path=args.search_image,
            vectorizer=vectorizer,
            db=db,
            n_results=args.n_results,
            truncate_dim=args.truncate_dim,
        )
        # ----------------------------------------------------
        end_search_time = time.time()
        logger.info(f"Search completed in {end_search_time - start_search_time:.2f}s.")

        if search_results.is_empty:
            logger.info(
                "Search successful, but no similar images found (after potential self-filtering)."
            )
        else:
            logger.info(f"\nTop {search_results.count} similar images found:")
            for i, item in enumerate(search_results.items):
                similarity_str = (
                    f"{item.similarity:.4f}" if item.similarity is not None else "N/A"
                )
                logger.info(
                    f"  {i + 1}. ID: {item.id} (Similarity: {similarity_str}) Metadata: {item.metadata}"
                )

    except FileNotFoundError:
        # Error already logged by pipeline or initial check
        pass
    except PipelineError as e:
        logger.error(f"Image search failed: {e}", exc_info=True)
    except ValueError as e:
        logger.error(f"Invalid input for image search: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during image search: {e}", exc_info=True)


def _confirm_action(
    prompt_message: str, confirmation_keyword: str | None = None
) -> bool:
    """
    Función auxiliar para obtener confirmación del usuario en la consola.

    Args:
        prompt_message: Mensaje a mostrar al usuario.
        confirmation_keyword: Palabra clave opcional que el usuario debe escribir para confirmar.

    Returns:
        True si el usuario confirma, False en caso contrario.
    """
    confirm = input(f"{prompt_message} [yes/NO]: ")
    if confirm.lower() != "yes":
        return False
    if confirmation_keyword:
        confirm2 = input(f"Please type '{confirmation_keyword}' to confirm: ")
        if confirm2 != confirmation_keyword:
            return False
    return True


def handle_clear_collection(args: Namespace, db: VectorDBInterface):
    """
    Maneja la acción --clear-collection del CLI.

    Elimina todos los elementos de la colección especificada, pidiendo confirmación.

    Args:
        args: Objeto Namespace de argparse con los argumentos del CLI.
        db: Instancia de VectorDBInterface inicializada.
    """
    # Esta función interactúa directamente con la BD, no necesita los módulos indexing/searching
    logger.warning(
        f"ACTION: Clearing all items from collection '{args.collection_name}' at path '{args.db_path}'..."
    )
    prompt = f"Are you absolutely sure you want to clear ALL items from '{args.collection_name}'?"
    if _confirm_action(prompt):
        try:
            success = db.clear_collection()
            if success:
                logger.info(
                    f"Collection '{args.collection_name}' cleared successfully."
                )
            else:
                logger.error("Failed to clear collection (check previous logs).")
        except DatabaseError as e:
            logger.error(f"Database error during clear_collection: {e}", exc_info=True)
        except Exception as e:
            logger.error(
                f"Unexpected error during clear_collection: {e}", exc_info=True
            )
    else:
        logger.info("Clear operation cancelled by user.")


def handle_delete_collection(args: Namespace, db: VectorDBInterface):
    """
    Maneja la acción --delete-collection del CLI.

    Elimina toda la colección especificada, pidiendo doble confirmación.

    Args:
        args: Objeto Namespace de argparse con los argumentos del CLI.
        db: Instancia de VectorDBInterface inicializada.
    """
    # Esta función interactúa directamente con la BD, no necesita los módulos indexing/searching
    logger.warning(
        f"ACTION: Deleting the ENTIRE collection '{args.collection_name}' from path '{args.db_path}'..."
    )
    prompt = f"Are you absolutely sure you want to DELETE the collection '{args.collection_name}'? This is IRREVERSIBLE."
    if _confirm_action(prompt, confirmation_keyword="DELETE"):
        try:
            success = db.delete_collection()
            if success:
                logger.info(
                    f"Collection '{args.collection_name}' deleted successfully."
                )
            else:
                logger.error("Failed to delete collection (check previous logs).")
        except DatabaseError as e:
            logger.error(f"Database error during delete_collection: {e}", exc_info=True)
        except Exception as e:
            logger.error(
                f"Unexpected error during delete_collection: {e}", exc_info=True
            )
    else:
        logger.info("Delete operation cancelled by user.")
