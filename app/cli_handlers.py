import logging
import os
import time
from argparse import Namespace
from typing import Optional

import config
from app import indexing, searching
from app.exceptions import DatabaseError, InitializationError, PipelineError
from app.factory import create_vector_database
from core.vectorizer import Vectorizer
from data_access.vector_db_interface import VectorDBInterface

logger = logging.getLogger(__name__)


def _get_target_db_for_cli(
    args: Namespace, vectorizer: Vectorizer, initial_db: VectorDBInterface
) -> VectorDBInterface:
    """Determina la instancia de BD correcta basada en args.truncate_dim."""
    base_collection_name = args.collection_name.split("_dim")[0].split(
        "_full"
    )[0]
    selected_dimension = args.truncate_dim

    native_dim = vectorizer.native_dimension
    if not native_dim:
        logger.warning(
            "Cannot determine vectorizer's native dimension. Using default DB collection."
        )
        return initial_db

    is_truncated = (
        selected_dimension is not None
        and selected_dimension > 0
        and selected_dimension < native_dim
    )
    if is_truncated:
        effective_target_dimension = selected_dimension
        target_collection_suffix = f"_dim{selected_dimension}"
    else:
        effective_target_dimension = "full"
        target_collection_suffix = "_full"

    target_collection_name = (
        f"{base_collection_name}{target_collection_suffix}"
    )
    logger.info(
        f"Target dimension for CLI operation: {effective_target_dimension} -> Collection: '{target_collection_name}'"
    )

    current_collection_name = getattr(initial_db, "collection_name", None)

    if current_collection_name == target_collection_name:
        logger.info(
            f"Target collection '{target_collection_name}' matches initial DB. Using it."
        )

        meta_dim = initial_db.get_dimension_from_metadata()
        if meta_dim is not None and meta_dim != effective_target_dimension:
            logger.warning(
                f"Metadata mismatch! Collection '{target_collection_name}' has metadata '{meta_dim}' but expected '{effective_target_dimension}'."
            )
        return initial_db
    else:
        logger.info(
            f"Target collection '{target_collection_name}' differs from initial '{current_collection_name}'. Getting/Creating target DB instance..."
        )
        try:

            target_db = create_vector_database(
                collection_name=target_collection_name,
                expected_dimension_metadata=effective_target_dimension,
            )
            if not target_db.is_initialized:
                raise InitializationError(
                    f"Failed to initialize DB for target collection '{target_collection_name}'"
                )

            target_meta_dim = target_db.get_dimension_from_metadata()
            if (
                target_meta_dim is not None
                and target_meta_dim != effective_target_dimension
            ):
                logger.error(
                    f"CRITICAL METADATA MISMATCH! Target collection '{target_collection_name}' exists with dimension metadata '{target_meta_dim}', expected '{effective_target_dimension}'. Aborting."
                )
                raise PipelineError(
                    f"Dimension metadata mismatch in target collection '{target_collection_name}'."
                )

            logger.info(
                f"Successfully obtained DB instance for collection '{target_collection_name}'."
            )
            return target_db
        except (InitializationError, PipelineError, Exception) as e:
            logger.error(
                f"Failed to get/create DB for target collection '{target_collection_name}': {e}",
                exc_info=True,
            )
            raise


def handle_indexing(
    args: Namespace, vectorizer: Vectorizer, initial_db: VectorDBInterface
):
    """Maneja la acción --index, usando la colección de BD correcta."""
    logger.info(f"--- ACTION: Indexing Directory: {args.image_dir} ---")
    if not os.path.isdir(args.image_dir):

        return

    try:

        db_to_use = _get_target_db_for_cli(args, vectorizer, initial_db)

        logger.info(
            f"Starting indexing process on collection '{getattr(db_to_use, 'collection_name', 'N/A')}'..."
        )
        processed_count = indexing.process_directory(
            directory_path=args.image_dir,
            vectorizer=vectorizer,
            db=db_to_use,
            batch_size=config.BATCH_SIZE_IMAGES,
            truncate_dim=args.truncate_dim,
        )
        logger.info(
            f"Finished indexing. Stored/updated {processed_count} embeddings in '{getattr(db_to_use, 'collection_name', 'N/A')}'."
        )
    except (PipelineError, InitializationError) as e:
        logger.error(f"Indexing pipeline failed: {e}", exc_info=False)
    except Exception as e:
        logger.error(f"Unexpected error during indexing: {e}", exc_info=True)


def handle_text_search(
    args: Namespace, vectorizer: Vectorizer, initial_db: VectorDBInterface
):
    """Maneja --search-text, usando la colección de BD correcta."""
    logger.info(f"--- ACTION: Text Search for query: '{args.search_text}' ---")

    try:

        db_to_use = _get_target_db_for_cli(args, vectorizer, initial_db)

        collection_name = getattr(db_to_use, "collection_name", "N/A")
        if not db_to_use.is_initialized or db_to_use.count() <= 0:
            logger.warning(
                f"Cannot search: Collection '{collection_name}' is empty or not initialized."
            )
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
        logger.info(
            f"Search completed in {end_search_time - start_search_time:.2f}s."
        )

        if search_results.is_empty:
            logger.info("Search successful, but no similar images found.")
        else:
            logger.info(
                f"\nTop {search_results.count} matching images for query:"
            )
            for i, item in enumerate(search_results.items):
                similarity_str = (
                    f"{item.similarity:.4f}"
                    if item.similarity is not None
                    else "N/A"
                )
                dist_str = (
                    f"{item.distance:.4f}"
                    if item.distance is not None
                    else "N/A"
                )
                logger.info(
                    f"  {i + 1}. ID: {item.id} (Dist: {dist_str}, Sim: {similarity_str}) Meta: {item.metadata}"
                )

    except (PipelineError, InitializationError, ValueError) as e:
        logger.error(f"Text search failed: {e}", exc_info=False)
    except Exception as e:
        logger.error(
            f"Unexpected error during text search: {e}", exc_info=True
        )


def handle_image_search(
    args: Namespace, vectorizer: Vectorizer, initial_db: VectorDBInterface
):
    """Maneja --search-image, usando la colección de BD correcta."""
    logger.info(
        f"--- ACTION: Image Search using image: '{args.search_image}' ---"
    )
    if not os.path.isfile(args.search_image):
        logger.error(f"Query image file not found: '{args.search_image}'.")
        return

    try:

        db_to_use = _get_target_db_for_cli(args, vectorizer, initial_db)

        collection_name = getattr(db_to_use, "collection_name", "N/A")
        if not db_to_use.is_initialized or db_to_use.count() <= 0:
            logger.warning(
                f"Cannot search: Collection '{collection_name}' is empty or not initialized."
            )
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
        logger.info(
            f"Search completed in {end_search_time - start_search_time:.2f}s."
        )

        if search_results.is_empty:
            logger.info(
                "Search successful, but no similar images found (after filtering)."
            )
        else:
            logger.info(f"\nTop {search_results.count} similar images found:")
            for i, item in enumerate(search_results.items):
                similarity_str = (
                    f"{item.similarity:.4f}"
                    if item.similarity is not None
                    else "N/A"
                )
                dist_str = (
                    f"{item.distance:.4f}"
                    if item.distance is not None
                    else "N/A"
                )
                logger.info(
                    f"  {i + 1}. ID: {item.id} (Dist: {dist_str}, Sim: {similarity_str}) Meta: {item.metadata}"
                )

    except FileNotFoundError:
        pass
    except (PipelineError, InitializationError, ValueError) as e:
        logger.error(f"Image search failed: {e}", exc_info=False)
    except Exception as e:
        logger.error(
            f"Unexpected error during image search: {e}", exc_info=True
        )


def _confirm_action(
    prompt_message: str, confirmation_keyword: Optional[str] = None
) -> bool:
    """Función auxiliar para obtener confirmación del usuario."""

    confirm = input(f"{prompt_message} [yes/NO]: ")
    if confirm.lower() != "yes":
        return False
    if confirmation_keyword:
        confirm2 = input(f"Please type '{confirmation_keyword}' to confirm: ")
        if confirm2 != confirmation_keyword:
            return False
    return True


def handle_clear_collection(
    args: Namespace, vectorizer: Vectorizer, initial_db: VectorDBInterface
):
    """Maneja --clear-collection, operando sobre la colección determinada por los args."""
    try:

        db_to_use = _get_target_db_for_cli(args, vectorizer, initial_db)
        collection_name = getattr(db_to_use, "collection_name", "N/A")

        logger.warning(
            f"ACTION: Clearing all items from collection '{collection_name}'..."
        )
        prompt = f"Are you absolutely sure you want to clear ALL items from '{collection_name}'?"
        if _confirm_action(prompt):
            if not db_to_use.is_initialized:
                logger.error(
                    f"Cannot clear: Collection '{collection_name}' is not initialized."
                )
                return
            success = db_to_use.clear_collection()
            if success:
                logger.info(
                    f"Collection '{collection_name}' cleared successfully."
                )
            else:
                logger.error(
                    f"Failed to clear collection '{collection_name}'."
                )
        else:
            logger.info("Clear operation cancelled by user.")
    except (DatabaseError, PipelineError, InitializationError) as e:
        logger.error(f"Error during clear_collection: {e}", exc_info=False)
    except Exception as e:
        logger.error(
            f"Unexpected error during clear_collection: {e}", exc_info=True
        )


def handle_delete_collection(
    args: Namespace, vectorizer: Vectorizer, initial_db: VectorDBInterface
):
    """Maneja --delete-collection, operando sobre la colección determinada por los args."""
    try:

        db_to_use = _get_target_db_for_cli(args, vectorizer, initial_db)
        collection_name = getattr(db_to_use, "collection_name", "N/A")

        logger.warning(
            f"ACTION: Deleting the ENTIRE collection '{collection_name}'..."
        )
        prompt = f"Are you absolutely sure you want to DELETE the collection '{collection_name}'? This is IRREVERSIBLE."
        if _confirm_action(prompt, confirmation_keyword="DELETE"):

            if not db_to_use.is_initialized:
                logger.error(
                    f"Cannot delete: Instance for collection '{collection_name}' seems uninitialized."
                )
                return
            success = db_to_use.delete_collection()
            if success:
                logger.info(
                    f"Collection '{collection_name}' deleted successfully."
                )
            else:
                logger.error(
                    f"Failed to delete collection '{collection_name}'."
                )
        else:
            logger.info("Delete operation cancelled by user.")
    except (DatabaseError, PipelineError, InitializationError) as e:
        logger.error(f"Error during delete_collection: {e}", exc_info=False)
    except Exception as e:
        logger.error(
            f"Unexpected error during delete_collection: {e}", exc_info=True
        )
