# app/factory.py
import logging
import config
from core.vectorizer import Vectorizer
# Import the concrete implementation and the interface
from data_access.chroma_db import ChromaVectorDB
from data_access.vector_db_interface import VectorDBInterface

logger = logging.getLogger(__name__)

# Cache instances to avoid reloading models/reconnecting DB unnecessarily within the same run
_vectorizer_instance: Vectorizer | None = None
_db_instance: VectorDBInterface | None = None


def create_vectorizer(force_reload: bool = False) -> Vectorizer:
    """
    Creates or retrieves a cached Vectorizer instance.

    Args:
        force_reload: If True, ignores cache and creates a new instance.

    Returns:
        An initialized Vectorizer instance.

    Raises:
        RuntimeError: If initialization fails.
    """
    global _vectorizer_instance
    if _vectorizer_instance is None or force_reload:
        logger.info(f"Creating Vectorizer instance (Model: {config.MODEL_NAME}, Device: {config.DEVICE})...")
        try:
            _vectorizer_instance = Vectorizer(
                model_name=config.MODEL_NAME,
                device=config.DEVICE,
                trust_remote_code=config.TRUST_REMOTE_CODE,
            )
            logger.info("Vectorizer instance created successfully.")
        except Exception as e:
            logger.error(f"Failed to create Vectorizer instance: {e}", exc_info=True)
            raise RuntimeError(f"Vectorizer initialization failed: {e}") from e
    else:
        logger.debug("Returning cached Vectorizer instance.")
    return _vectorizer_instance


def create_vector_database(force_reload: bool = False) -> VectorDBInterface:
    """
    Creates or retrieves a cached VectorDatabase instance (currently ChromaDB).

    Args:
        force_reload: If True, ignores cache and creates a new instance.

    Returns:
        An initialized instance conforming to VectorDBInterface.

    Raises:
        RuntimeError: If initialization fails.
    """
    global _db_instance
    if _db_instance is None or force_reload:
        logger.info(
            f"Creating VectorDatabase instance (Type: ChromaDB, Path: {config.CHROMA_DB_PATH}, Collection: {config.CHROMA_COLLECTION_NAME})...")
        try:
            # Instantiate the concrete ChromaDB implementation
            _db_instance = ChromaVectorDB(
                path=config.CHROMA_DB_PATH,
                collection_name=config.CHROMA_COLLECTION_NAME,
            )
            logger.info("VectorDatabase instance (ChromaDB) created successfully.")
        except Exception as e:
            logger.error(f"Failed to create VectorDatabase instance: {e}", exc_info=True)
            raise RuntimeError(f"VectorDatabase initialization failed: {e}") from e
    else:
        logger.debug("Returning cached VectorDatabase instance.")
    return _db_instance


def get_initialized_components(force_reload: bool = False) -> tuple[Vectorizer, VectorDBInterface]:
    """
    Convenience function to get both initialized components.

    Args:
        force_reload: If True, forces reloading of both components.

    Returns:
        A tuple containing the initialized (vectorizer, database).

    Raises:
        RuntimeError: If initialization of either component fails.
    """
    logger.info("Initializing application components...")
    start_time = time.time()
    try:
        vectorizer = create_vectorizer(force_reload=force_reload)
        db = create_vector_database(force_reload=force_reload)
        end_time = time.time()
        logger.info(f"Components initialized successfully in {end_time - start_time:.2f} seconds.")
        return vectorizer, db
    except RuntimeError as e:
        logger.critical(f"Fatal error during component initialization: {e}", exc_info=True)
        raise  # Re-raise the exception after logging


# Import time for the convenience function
import time
