# streamlit_app.py (Refactored & Modified for Dimension Handling)
import os
import sys
import time
import streamlit as st
from typing import Optional, Tuple, Union # Añadir Union
import logging

# --- Initial Setup and Path Configuration ---
try:
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path: sys.path.insert(0, project_root)

    import config
    # --- MODIFICADO: Importar create_vector_database directamente ---
    from app.factory import create_vectorizer, create_vector_database
    # -----------------------------------------------------------
    from core.vectorizer import Vectorizer
    from data_access.vector_db_interface import VectorDBInterface
    from app.logging_config import setup_logging
    from app.exceptions import InitializationError, PipelineError # Añadir PipelineError
    from app.ui.state_utils import STATE_SELECTED_MODEL, STATE_SELECTED_DIMENSION
    from app.ui.sidebar import display_database_info
    from app.ui.tabs.indexing_tab import render_indexing_tab
    from app.ui.tabs.text_search_tab import render_text_search_tab
    from app.ui.tabs.image_search_tab import render_image_search_tab
    from app.ui.tabs.hybrid_search_tab import render_hybrid_search_tab
    # --- Importar módulo de búsqueda para usar en la UI principal si es necesario ---
    from app import searching

except ImportError as e:
    # ... (manejo de error crítico inicial) ...
    error_msg = f"Error crítico al importar módulos: {e}..."
    print(error_msg); sys.exit(error_msg)
except Exception as e:
    # ... (manejo de error crítico inicial) ...
     error_msg = f"Error inesperado durante la configuración inicial: {e}"
     print(error_msg); sys.exit(error_msg)

# --- Logging & Page Config ---
setup_logging()
logger = logging.getLogger(__name__)
st.set_page_config(page_title="Buscador de Imágenes Vectorial", page_icon="🔍", layout="wide")

# --- Custom CSS ---
# ... (CSS existente) ...
st.markdown("""<style>...</style>""", unsafe_allow_html=True) # Mantener CSS

# --- Component Initialization Cache (MODIFICADO) ---

# Cache para el Vectorizer (depende solo del nombre del modelo)
@st.cache_resource(show_spinner="Cargando modelo de vectorización...")
def cached_get_vectorizer(model_name_to_load: str) -> Optional[Vectorizer]:
    """Inicializa y cachea el Vectorizer para un modelo específico."""
    logger.info(f"Attempting to load/cache Vectorizer for model: {model_name_to_load}")
    try:
        vectorizer = create_vectorizer(
            model_name=model_name_to_load,
            device=config.DEVICE,
            trust_remote_code=config.TRUST_REMOTE_CODE
        )
        if vectorizer.is_ready:
            logger.info(f"Vectorizer for {model_name_to_load} initialized and cached successfully.")
            return vectorizer
        else:
            logger.error(f"Vectorizer initialization failed for {model_name_to_load} (is_ready is False).")
            return None
    except InitializationError as e:
        logger.error(f"InitializationError caching Vectorizer for {model_name_to_load}: {e}", exc_info=True)
        st.error(f"Error al inicializar el vectorizador '{model_name_to_load}': {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error caching Vectorizer for {model_name_to_load}: {e}", exc_info=True)
        st.error(f"Error inesperado al inicializar el vectorizador '{model_name_to_load}': {e}")
        return None

# Cache para la instancia de BD (depende del nombre de la colección y metadatos esperados)
# Usamos una clave compuesta para el caché
@st.cache_resource(show_spinner="Accediendo a base de datos...")
def cached_get_db_instance(
    collection_name: str,
    expected_dimension_metadata: Optional[Union[int, str]]
) -> Optional[VectorDBInterface]:
    """Inicializa y cachea la instancia de BD para una colección específica."""
    logger.info(f"Attempting to load/cache DB instance for collection: '{collection_name}' (Expected Dim Meta: {expected_dimension_metadata})")
    try:
        # Llama a la factory para obtener/crear la instancia de BD
        db_instance = create_vector_database(
            collection_name=collection_name,
            expected_dimension_metadata=expected_dimension_metadata
        )
        # is_initialized ya se comprueba dentro de create_vector_database
        logger.info(f"DB instance for '{collection_name}' obtained and cached successfully.")
        return db_instance
    except InitializationError as e:
        logger.error(f"InitializationError caching DB for collection '{collection_name}': {e}", exc_info=True)
        st.error(f"Error al inicializar la base de datos para la colección '{collection_name}': {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error caching DB for collection '{collection_name}': {e}", exc_info=True)
        st.error(f"Error inesperado al inicializar la base de datos para la colección '{collection_name}': {e}")
        return None

# --- Main Application Logic ---
def main():
    st.title("🔍 Buscador de Imágenes por Similitud Vectorial")
    st.markdown("...") # Descripción

    # --- Sidebar Configuration ---
    st.sidebar.header("Configuración Global")
    # ... (Lógica existente para determinar default_model_index) ...
    try:
        default_model_name = st.session_state.get(STATE_SELECTED_MODEL, config.DEFAULT_MODEL_NAME)
        if default_model_name not in config.AVAILABLE_MODELS:
            default_model_name = config.AVAILABLE_MODELS[0] if config.AVAILABLE_MODELS else None
            if default_model_name: st.session_state[STATE_SELECTED_MODEL] = default_model_name
            else: st.sidebar.error("¡No hay modelos configurados!"); st.stop()
        default_model_index = config.AVAILABLE_MODELS.index(default_model_name)
    except Exception as e:
        logger.error(f"Error determining default model index: {e}. Falling back to 0.")
        default_model_index = 0
        if config.AVAILABLE_MODELS: st.session_state[STATE_SELECTED_MODEL] = config.AVAILABLE_MODELS[0]
        else: st.sidebar.error("¡No hay modelos disponibles!"); st.stop()


    selected_model = st.sidebar.selectbox(
        "Modelo de Embedding:", options=config.AVAILABLE_MODELS, index=default_model_index, key=STATE_SELECTED_MODEL,
        help="Elige el modelo. Cambiarlo recargará componentes y puede cambiar la colección de BD activa."
    )
    selected_dimension = st.sidebar.number_input(
        "Dimensión Embedding (0 = Completa):", min_value=0,
        value=st.session_state.get(STATE_SELECTED_DIMENSION, config.VECTOR_DIMENSION or 0),
        step=32, key=STATE_SELECTED_DIMENSION,
        help="Dimensión del vector final (0=Nativa). Cambiarlo puede cambiar la colección de BD activa."
    )
    # Determinar dimensión de truncamiento efectiva (None si es 0 o inválido)
    truncate_dim_value = selected_dimension if selected_dimension > 0 else None

    # --- Inicializar Vectorizador (Cacheado por nombre de modelo) ---
    vectorizer = cached_get_vectorizer(selected_model)

    # --- Detener si el Vectorizador falló ---
    if not vectorizer or not vectorizer.is_ready:
        st.error(f"No se pudo inicializar el Vectorizador para el modelo '{selected_model}'. La aplicación no puede continuar.")
        st.warning("Revisa los logs, la configuración del modelo y las dependencias.")
        st.stop()

    # --- Determinar Colección de BD y Obtener Instancia (Cacheada por nombre + metadatos) ---
    db_instance: Optional[VectorDBInterface] = None # Inicializar
    try:
        base_collection_name = config.CHROMA_COLLECTION_NAME.split("_dim")[0].split("_full")[0]
        native_dim = vectorizer.native_dimension

        if not native_dim:
            st.error(f"No se pudo determinar la dimensión nativa para el modelo '{selected_model}'. No se puede seleccionar la colección de BD correcta.")
            st.stop()

        is_truncated = truncate_dim_value is not None and truncate_dim_value > 0 and truncate_dim_value < native_dim
        if is_truncated:
            effective_target_dimension = truncate_dim_value
            target_collection_suffix = f"_dim{truncate_dim_value}"
        else:
            effective_target_dimension = 'full' # Representa dimensión nativa
            target_collection_suffix = "_full" # O "_dim{native_dim}"

        target_collection_name = f"{base_collection_name}{target_collection_suffix}"

        # Obtener/Crear la instancia de BD para la colección objetivo (cacheada)
        db_instance = cached_get_db_instance(target_collection_name, effective_target_dimension)

    except Exception as db_select_err:
         logger.error(f"Error determining target DB collection: {db_select_err}", exc_info=True)
         st.error(f"Error al determinar la colección de base de datos correcta: {db_select_err}")
         st.stop()

    # --- Detener si la Instancia de BD falló ---
    if not db_instance or not db_instance.is_initialized:
        st.error(f"No se pudo inicializar la instancia de Base de Datos para la colección '{target_collection_name}'. La aplicación no puede continuar.")
        st.warning("Revisa los logs y la configuración de la base de datos.")
        st.stop()

    # --- Mostrar Info en Sidebar (usando la instancia de BD correcta) ---
    st.sidebar.header("Información del Sistema")
    # ... (info de Python, Dispositivo, CUDA como antes) ...
    st.sidebar.caption(f"Dispositivo Usado: `{vectorizer.device}`")
    # ...

    st.sidebar.header("Configuración Activa")
    st.sidebar.caption(f"Modelo Cargado: `{selected_model}`")
    st.sidebar.caption(f"Dimensión Objetivo: `{truncate_dim_value or 'Nativa'}` ({effective_target_dimension})")
    st.sidebar.caption(f"Colección Activa: `{getattr(db_instance, 'collection_name', 'N/A')}`") # Mostrar nombre de la colección activa

    # Mostrar Info de la BD activa
    display_database_info(db_instance, st.sidebar) # Pasa la instancia correcta

    # --- Definición y Renderizado de Pestañas ---
    tab_titles = ["💾 Indexar Imágenes", "📝 Buscar por Texto", "🖼️ Buscar por Imagen", "🧬 Búsqueda Híbrida"]
    tab1, tab2, tab3, tab4 = st.tabs(tab_titles)

    # Pasar el vectorizador y la INSTANCIA DE BD CORRECTA a cada pestaña
    with tab1:
        render_indexing_tab(vectorizer, db_instance, truncate_dim_value)
    with tab2:
        render_text_search_tab(vectorizer, db_instance, truncate_dim_value)
    with tab3:
        render_image_search_tab(vectorizer, db_instance, truncate_dim_value)
    with tab4:
         render_hybrid_search_tab(vectorizer, db_instance, truncate_dim_value)

# --- Entry Point ---
if __name__ == "__main__":
    main()
