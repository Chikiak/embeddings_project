# app/factory.py
import logging
import time
import config
from core.vectorizer import Vectorizer
from data_access.chroma_db import ChromaVectorDB
from data_access.vector_db_interface import VectorDBInterface
from app.exceptions import InitializationError

logger = logging.getLogger(__name__)

# Removed global cache variables, assuming Streamlit handles caching via st.cache_resource keyed by model_name
# _vectorizer_instance: Vectorizer | None = None
# _db_instance: VectorDBInterface | None = None


def create_vectorizer(model_name: str, device: str, trust_remote_code: bool) -> Vectorizer:
    """
    Crea una instancia de Vectorizer para un modelo específico.

    Args:
        model_name: Nombre del modelo de Hugging Face a cargar.
        device: Dispositivo para la inferencia ('cuda' o 'cpu').
        trust_remote_code: Permitir código remoto del modelo.

    Returns:
        Una instancia de Vectorizer inicializada.

    Raises:
        InitializationError: Si la inicialización del vectorizador falla.
    """
    logger.info(
        f"Creating Vectorizer instance for model: {model_name} on device: {device}..."
    )
    try:
        # Usa el model_name pasado como argumento
        vectorizer_instance = Vectorizer(
            model_name=model_name,
            device=device,
            trust_remote_code=trust_remote_code,
        )
        logger.info(f"Vectorizer instance for {model_name} created successfully.")
        return vectorizer_instance
    except Exception as e:
        logger.error(f"Failed to create Vectorizer instance for {model_name}: {e}", exc_info=True)
        raise InitializationError(f"Vectorizer initialization failed for {model_name}: {e}") from e


def create_vector_database(force_reload: bool = False) -> VectorDBInterface:
    """
    Crea una instancia de VectorDatabase (actualmente ChromaDB).

    Nota: Esta función ahora crea una nueva instancia cada vez que se llama,
          a menos que se implemente una caché externa o se confíe en la caché de Streamlit
          para la función que llama a esta (get_initialized_components).

    Args:
        force_reload: (Actualmente no usado internamente aquí, pero mantenido por consistencia)

    Returns:
        Una instancia inicializada que cumple con VectorDBInterface.

    Raises:
        InitializationError: Si la inicialización de la base de datos falla.
    """
    # No usa caché global _db_instance
    logger.info(
        f"Creating VectorDatabase instance (Type: ChromaDB, Path: {config.CHROMA_DB_PATH}, Collection: {config.CHROMA_COLLECTION_NAME})..."
    )
    try:
        db_instance = ChromaVectorDB(
            path=config.CHROMA_DB_PATH,
            collection_name=config.CHROMA_COLLECTION_NAME,
        )
        logger.info("VectorDatabase instance (ChromaDB) created successfully.")
        return db_instance
    except Exception as e:
        logger.error(
            f"Failed to create VectorDatabase instance: {e}", exc_info=True
        )
        raise InitializationError(
            f"VectorDatabase initialization failed: {e}"
        ) from e


def get_initialized_components(
    model_name: str, # Requiere el nombre del modelo
    force_reload_db: bool = False, # Renombrado para claridad
) -> tuple[Vectorizer, VectorDBInterface]:
    """
    Función de conveniencia para obtener ambos componentes inicializados para un MODELO específico.

    Args:
        model_name: El nombre del modelo de embedding a cargar.
        force_reload_db: Si es True, fuerza la recarga de la base de datos.

    Returns:
        Una tupla que contiene (vectorizer, database) inicializados.

    Raises:
        InitializationError: Si la inicialización de cualquiera de los componentes falla.
    """
    logger.info(f"Initializing application components for model: {model_name}...")
    start_time = time.time()
    try:
        # Llama a create_vectorizer con el model_name específico
        vectorizer = create_vectorizer(
            model_name=model_name,
            device=config.DEVICE,
            trust_remote_code=config.TRUST_REMOTE_CODE
         )
        # La creación de la BD no depende del modelo
        db = create_vector_database(force_reload=force_reload_db)
        end_time = time.time()
        logger.info(
            f"Components for model {model_name} initialized successfully in {end_time - start_time:.2f} seconds."
        )
        return vectorizer, db
    except InitializationError as e:
        logger.critical(
            f"Fatal error during component initialization for model {model_name}: {e}", exc_info=True
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
