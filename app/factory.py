# app/factory.py
import logging
import time
import config
from core.vectorizer import Vectorizer
from data_access.chroma_db import ChromaVectorDB
from data_access.vector_db_interface import VectorDBInterface
from app.exceptions import InitializationError

logger = logging.getLogger(__name__)

_vectorizer_instance: Vectorizer | None = None
_db_instance: VectorDBInterface | None = None


def create_vectorizer(force_reload: bool = False) -> Vectorizer:
    """
    Crea o recupera una instancia cacheada de Vectorizer.

    Args:
        force_reload: Si es True, ignora la caché y crea una nueva instancia.

    Returns:
        Una instancia de Vectorizer inicializada.

    Raises:
        InitializationError: Si la inicialización del vectorizador falla.
    """
    global _vectorizer_instance
    if _vectorizer_instance is None or force_reload:
        logger.info(
            f"Creating Vectorizer instance (Model: {config.MODEL_NAME}, Device: {config.DEVICE})..."
        )
        try:
            _vectorizer_instance = Vectorizer(
                model_name=config.MODEL_NAME,
                device=config.DEVICE,
                trust_remote_code=config.TRUST_REMOTE_CODE,
            )
            logger.info("Vectorizer instance created successfully.")
        except Exception as e:
            logger.error(f"Failed to create Vectorizer instance: {e}", exc_info=True)
            raise InitializationError(f"Vectorizer initialization failed: {e}") from e
    else:
        logger.debug("Returning cached Vectorizer instance.")
    return _vectorizer_instance


def create_vector_database(force_reload: bool = False) -> VectorDBInterface:
    """
    Crea o recupera una instancia cacheada de VectorDatabase (actualmente ChromaDB).

    Args:
        force_reload: Si es True, ignora la caché y crea una nueva instancia.

    Returns:
        Una instancia inicializada que cumple con VectorDBInterface.

    Raises:
        InitializationError: Si la inicialización de la base de datos falla.
    """
    global _db_instance
    if _db_instance is None or force_reload:
        logger.info(
            f"Creating VectorDatabase instance (Type: ChromaDB, Path: {config.CHROMA_DB_PATH}, Collection: {config.CHROMA_COLLECTION_NAME})..."
        )
        try:
            _db_instance = ChromaVectorDB(
                path=config.CHROMA_DB_PATH,
                collection_name=config.CHROMA_COLLECTION_NAME,
            )
            logger.info("VectorDatabase instance (ChromaDB) created successfully.")
        except Exception as e:
            logger.error(
                f"Failed to create VectorDatabase instance: {e}", exc_info=True
            )
            raise InitializationError(
                f"VectorDatabase initialization failed: {e}"
            ) from e
    else:
        logger.debug("Returning cached VectorDatabase instance.")
    return _db_instance


def get_initialized_components(
    force_reload: bool = False,
) -> tuple[Vectorizer, VectorDBInterface]:
    """
    Función de conveniencia para obtener ambos componentes inicializados.

    Args:
        force_reload: Si es True, fuerza la recarga de ambos componentes.

    Returns:
        Una tupla que contiene (vectorizer, database) inicializados.

    Raises:
        InitializationError: Si la inicialización de cualquiera de los componentes falla.
    """
    logger.info("Initializing application components...")
    start_time = time.time()
    try:
        vectorizer = create_vectorizer(force_reload=force_reload)
        db = create_vector_database(force_reload=force_reload)
        end_time = time.time()
        logger.info(
            f"Components initialized successfully in {end_time - start_time:.2f} seconds."
        )
        return vectorizer, db
    except InitializationError as e:
        logger.critical(
            f"Fatal error during component initialization: {e}", exc_info=True
        )
        raise
    except Exception as e:
        logger.critical(
            f"Unexpected fatal error during component initialization: {e}",
            exc_info=True,
        )
        raise InitializationError(
            f"Unexpected component initialization failure: {e}"
        ) from e
