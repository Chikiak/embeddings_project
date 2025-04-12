import logging
import time
from typing import Optional, Union

import config
from app.exceptions import InitializationError
from core.vectorizer import Vectorizer
from data_access.chroma_db import ChromaVectorDB
from data_access.vector_db_interface import VectorDBInterface

logger = logging.getLogger(__name__)


def create_vectorizer(
    model_name: str, device: str, trust_remote_code: bool
) -> Vectorizer:
    """
    Crea una instancia de Vectorizer para un modelo específico.
    (Sin cambios necesarios aquí)
    """
    logger.info(
        f"Creating Vectorizer instance for model: {model_name} on device: {device}..."
    )
    try:
        vectorizer_instance = Vectorizer(
            model_name=model_name,
            device=device,
            trust_remote_code=trust_remote_code,
        )
        logger.info(
            f"Vectorizer instance for {model_name} created successfully."
        )
        return vectorizer_instance
    except Exception as e:
        logger.error(
            f"Failed to create Vectorizer instance for {model_name}: {e}",
            exc_info=True,
        )
        raise InitializationError(
            f"Vectorizer initialization failed for {model_name}: {e}"
        ) from e


def create_vector_database(
    collection_name: Optional[str] = None,
    expected_dimension_metadata: Optional[Union[int, str]] = None,
) -> VectorDBInterface:
    """
    Crea una instancia de VectorDatabase (ChromaDB) para una colección específica.

    Args:
        collection_name: Nombre deseado para la colección. Si es None, usa el default de config.
        expected_dimension_metadata: La dimensión (int o 'full') que se espera para esta colección.
                                     Se guardará en metadatos al crear la colección.

    Returns:
        Instancia inicializada de VectorDBInterface.
    Raises:
        InitializationError si falla.
    """
    target_collection_name = collection_name or config.CHROMA_COLLECTION_NAME
    logger.info(
        f"Creating/Getting VectorDatabase instance (Type: ChromaDB, Path: {config.CHROMA_DB_PATH}, Collection: {target_collection_name})..."
    )
    if expected_dimension_metadata:
        logger.info(
            f"  Expected dimension metadata for new collection: {expected_dimension_metadata}"
        )

    try:

        db_instance = ChromaVectorDB(
            path=config.CHROMA_DB_PATH,
            collection_name=target_collection_name,
            expected_dimension_metadata=expected_dimension_metadata,
        )

        if not db_instance.is_initialized:

            last_err = getattr(
                db_instance, "_last_init_error", "Unknown initialization issue"
            )
            raise InitializationError(
                f"ChromaDB instance created but failed initialization check for collection '{target_collection_name}'. Last error: {last_err}"
            )

        logger.info(
            f"VectorDatabase instance for collection '{target_collection_name}' obtained successfully."
        )
        return db_instance
    except InitializationError as e:
        logger.error(
            f"InitializationError during DB creation for collection '{target_collection_name}': {e}",
            exc_info=True,
        )
        raise e
    except Exception as e:
        logger.error(
            f"Unexpected error creating VectorDatabase instance for collection '{target_collection_name}': {e}",
            exc_info=True,
        )
        raise InitializationError(
            f"VectorDatabase initialization failed unexpectedly for collection '{target_collection_name}': {e}"
        ) from e


def get_initialized_components(
    model_name: str,
) -> tuple[Vectorizer, VectorDBInterface]:
    """
    Obtiene Vectorizer y la instancia de BD *inicial* (apuntando a la colección por defecto).

    La lógica de la aplicación (indexación/búsqueda) decidirá si usar esta instancia de BD
    o crear/obtener una nueva para una colección específica de dimensión usando create_vector_database().

    Args:
        model_name: El nombre del modelo de embedding a cargar.

    Returns:
        Una tupla que contiene (vectorizer, initial_database) inicializados.

    Raises:
        InitializationError: Si la inicialización de cualquiera de los componentes falla.
    """
    logger.info(f"Initializing components for model: {model_name}...")
    start_time = time.time()
    try:

        vectorizer = create_vectorizer(
            model_name=model_name,
            device=config.DEVICE,
            trust_remote_code=config.TRUST_REMOTE_CODE,
        )

        initial_db = create_vector_database(
            collection_name=None, expected_dimension_metadata=None
        )

        end_time = time.time()
        logger.info(
            f"Initial components for model {model_name} ready in {end_time - start_time:.2f} seconds."
        )

        return vectorizer, initial_db
    except InitializationError as e:
        logger.critical(
            f"Fatal error during component initialization for model {model_name}: {e}",
            exc_info=True,
        )
        raise
    except Exception as e:
        logger.critical(
            f"Unexpected fatal error during component initialization for model {model_name}: {e}",
            exc_info=True,
        )
        raise InitializationError(
            f"Unexpected component initialization failure for model {model_name}: {e}"
        ) from e
