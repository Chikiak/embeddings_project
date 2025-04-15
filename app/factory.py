# --- app/factory.py ---
import logging
import time
from typing import Optional, Tuple, Union, Dict

import config # Importa la configuración actualizada
from app.exceptions import InitializationError
from core.vectorizer import Vectorizer
from data_access.chroma_db import ChromaVectorDB # Importa la implementación concreta
from data_access.vector_db_interface import VectorDBInterface

logger = logging.getLogger(__name__)

# Cachés simples en memoria (alternativa a st.cache_resource para uso fuera de Streamlit)
# Nota: Estos cachés son por proceso. En un entorno multi-proceso/servidor, se necesitaría un caché externo.
_vectorizer_cache: Dict[str, Vectorizer] = {}
_db_cache: Dict[str, VectorDBInterface] = {}

def create_vectorizer(
    model_name: str, device: str, trust_remote_code: bool
) -> Vectorizer:
    """
    Crea o recupera desde caché una instancia de Vectorizer para un modelo específico.
    """
    cache_key = f"{model_name}_{device}_{trust_remote_code}"
    if cache_key in _vectorizer_cache:
        logger.debug(f"Returning cached Vectorizer instance for key: {cache_key}")
        return _vectorizer_cache[cache_key]

    logger.info(f"Creating new Vectorizer instance for model: {model_name} on device: {device}...")
    try:
        vectorizer_instance = Vectorizer(
            model_name=model_name,
            device=device,
            trust_remote_code=trust_remote_code,
        )
        if not vectorizer_instance.is_ready:
             raise InitializationError(f"Vectorizer for {model_name} failed readiness check after creation.")

        logger.info(f"Vectorizer instance for {model_name} created successfully.")
        _vectorizer_cache[cache_key] = vectorizer_instance # Guarda en caché
        return vectorizer_instance
    except Exception as e:
        logger.error(f"Failed to create Vectorizer instance for {model_name}: {e}", exc_info=True)
        raise InitializationError(f"Vectorizer initialization failed for {model_name}: {e}") from e

def create_vector_database(
    collection_name: str,
    expected_dimension_metadata: Optional[Union[int, str]],
) -> VectorDBInterface:
    """
    Crea o recupera desde caché una instancia de VectorDatabase (ChromaDB) para una colección específica.

    Args:
        collection_name: Nombre deseado para la colección.
        expected_dimension_metadata: Dimensión esperada (int o 'full') para metadatos.

    Returns:
        Instancia inicializada de VectorDBInterface.
    Raises:
        InitializationError si falla la inicialización.
    """
    # La clave de caché incluye la ruta (implícita en config) y el nombre de la colección
    cache_key = f"{config.CHROMA_DB_PATH}_{collection_name}"
    if cache_key in _db_cache:
        # Verifica si la instancia cacheada sigue siendo válida (simplificado)
        cached_db = _db_cache[cache_key]
        if cached_db.is_initialized:
             logger.debug(f"Returning cached DB instance for collection: '{collection_name}'")
             # Opcional: Verificar si expected_dimension_metadata coincide con el cacheado o el real?
             # Por simplicidad, asumimos que si está en caché y inicializado, está bien.
             return cached_db
        else:
             logger.warning(f"Cached DB instance for '{collection_name}' is no longer initialized. Removing from cache.")
             del _db_cache[cache_key]

    logger.info(f"Creating/Getting new DB instance for collection: '{collection_name}' (Expected Dim Meta: {expected_dimension_metadata})...")
    try:
        # Siempre usa la implementación ChromaVectorDB
        db_instance = ChromaVectorDB(
            path=config.CHROMA_DB_PATH,
            collection_name=collection_name,
            expected_dimension_metadata=expected_dimension_metadata,
        )

        # is_initialized ya se verifica dentro del constructor de ChromaVectorDB (o debería)
        # Si el constructor no lanzó error, asumimos que está lista o is_initialized lo reflejará.
        if not db_instance.is_initialized:
            # Intenta obtener el último error si está disponible
            last_err = getattr(db_instance, "_last_init_error", "Unknown initialization issue")
            raise InitializationError(f"DB instance created but failed initialization check for '{collection_name}'. Last error: {last_err}")

        logger.info(f"DB instance for collection '{collection_name}' obtained successfully.")
        _db_cache[cache_key] = db_instance # Guarda en caché
        return db_instance
    except InitializationError as e:
         # Error ya logueado en ChromaVectorDB probablemente
         raise # Relanza para que el llamador lo maneje
    except Exception as e:
        logger.error(f"Unexpected error creating DB instance for '{collection_name}': {e}", exc_info=True)
        raise InitializationError(f"Unexpected DB initialization failure for '{collection_name}': {e}") from e


def get_or_create_db_for_model_and_dim(
    vectorizer: Vectorizer,
    truncate_dim: Optional[int]
) -> VectorDBInterface:
    """
    Obtiene la instancia de BD correcta basada en el modelo y la dimensión de truncamiento.
    Determina el nombre de la colección y los metadatos esperados.

    Args:
        vectorizer: La instancia del Vectorizer inicializada.
        truncate_dim: La dimensión de truncamiento deseada (None para nativa).

    Returns:
        La instancia de VectorDBInterface correcta.

    Raises:
        InitializationError: Si no se puede obtener la instancia de BD.
        ValueError: Si no se puede determinar la dimensión nativa del vectorizador.
    """
    base_collection_name = config.CHROMA_COLLECTION_NAME_BASE
    model_suffix = vectorizer.model_name.split('/')[-1].replace('-', '_') # Sufijo basado en nombre de modelo

    native_dim = vectorizer.native_dimension
    if not native_dim:
        raise ValueError(f"Cannot determine native dimension for vectorizer '{vectorizer.model_name}'.")

    # Determina el sufijo de dimensión y los metadatos esperados
    is_truncated = truncate_dim is not None and 0 < truncate_dim < native_dim
    if is_truncated:
        effective_target_dimension = truncate_dim
        dimension_suffix = f"_dim{truncate_dim}"
        logger.debug(f"Targeting truncated dimension: {truncate_dim}")
    else:
        effective_target_dimension = "full" # Usar 'full' como metadato para dimensión nativa
        dimension_suffix = "_full"
        if truncate_dim is not None and truncate_dim != 0:
             logger.warning(f"Requested dimension {truncate_dim} is invalid or >= native {native_dim}. Using full dimension.")
        logger.debug(f"Targeting full native dimension: {native_dim}")

    # Construye el nombre final de la colección
    target_collection_name = f"{base_collection_name}_{model_suffix}{dimension_suffix}"
    logger.info(f"Determined target collection name: '{target_collection_name}' (Expected Dim Meta: {effective_target_dimension})")

    # Llama a create_vector_database para obtener/crear/cachear la instancia
    return create_vector_database(target_collection_name, effective_target_dimension)
