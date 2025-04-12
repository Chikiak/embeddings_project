# streamlit_app.py (Refactored & Modified for Clustering)
import os
import sys
import time
import torch
import streamlit as st
from typing import Optional, Tuple, Union # Añadir Union
import logging

# --- Initial Setup and Path Configuration ---
try:
    # Add project root to Python path
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Import project modules
    import config
    from app.factory import create_vectorizer, create_vector_database
    from core.vectorizer import Vectorizer
    from data_access.vector_db_interface import VectorDBInterface
    from app.logging_config import setup_logging
    from app.exceptions import InitializationError, PipelineError
    # Import state keys and UI components
    from app.ui.state_utils import (
        STATE_SELECTED_MODEL,
        STATE_SELECTED_DIMENSION,
        # Import new state keys needed for clustering tabs
        STATE_CLUSTER_LABELS,
        STATE_CLUSTER_IDS,
        STATE_REDUCED_EMBEDDINGS,
        STATE_REDUCED_IDS
    )
    from app.ui.sidebar import display_database_info
    # Import original search tabs
    from app.ui.tabs.indexing_tab import render_indexing_tab
    from app.ui.tabs.text_search_tab import render_text_search_tab
    from app.ui.tabs.image_search_tab import render_image_search_tab
    from app.ui.tabs.hybrid_search_tab import render_hybrid_search_tab
    # Import NEW clustering tabs
    from app.ui.tabs.clustering.cluster_execution_tab import render_cluster_execution_tab
    from app.ui.tabs.clustering.cluster_optimization_tab import render_cluster_optimization_tab
    from app.ui.tabs.clustering.cluster_visualization_tab import render_cluster_visualization_tab
    # Import searching module if needed directly
    from app import searching

except ImportError as e:
    error_msg = f"Critical error importing modules: {e}. Check PYTHONPATH and dependencies."
    print(error_msg)
    # Display error in Streamlit if possible, otherwise exit
    try:
        st.error(error_msg)
        st.stop()
    except Exception:
        sys.exit(error_msg)
except Exception as e:
    error_msg = f"Unexpected error during initial setup: {e}"
    print(error_msg)
    try:
        st.error(error_msg)
        st.stop()
    except Exception:
        sys.exit(error_msg)

# --- Logging & Page Config ---
setup_logging()
logger = logging.getLogger(__name__)
st.set_page_config(page_title="Buscador de Imágenes Vectorial", page_icon="🔍", layout="wide")

# --- Custom CSS (Keep existing styles) ---
st.markdown("""
<style>
    /* Add custom CSS styles here if needed */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px; /* Adjust tab spacing */
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        white-space: pre-wrap;
        background-color: #F0F2F6; /* Light background for inactive tabs */
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF; /* White background for active tab */
    }
    .result-item {
        border: 1px solid #eee;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
        text-align: center; /* Center content */
    }
    .result-item img {
        max-width: 100%;
        height: auto;
        border-radius: 3px;
        margin-bottom: 5px; /* Space between image and caption */
    }
    .caption {
        font-size: 0.8em;
        color: #555;
    }
    .selected-query-image {
         border: 2px solid #4CAF50; /* Green border for selected image */
         padding: 5px;
         border-radius: 5px;
         margin-bottom: 10px;
         display: inline-block; /* Fit content */
    }
</style>
""", unsafe_allow_html=True)

# --- Component Initialization Cache (MODIFIED) ---

# Cache for the Vectorizer (depends only on model name)
@st.cache_resource(show_spinner="Cargando modelo de vectorización...")
def cached_get_vectorizer(model_name_to_load: str) -> Optional[Vectorizer]:
    """Initializes and caches the Vectorizer for a specific model."""
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
            st.error(f"Error al inicializar el vectorizador '{model_name_to_load}'.") # Show error in UI
            return None
    except InitializationError as e:
        logger.error(f"InitializationError caching Vectorizer for {model_name_to_load}: {e}", exc_info=True)
        st.error(f"Error al inicializar el vectorizador '{model_name_to_load}': {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error caching Vectorizer for {model_name_to_load}: {e}", exc_info=True)
        st.error(f"Error inesperado al inicializar el vectorizador '{model_name_to_load}': {e}")
        return None

# Cache for the DB instance (depends on collection name and expected dimension metadata)
@st.cache_resource(show_spinner="Accediendo a base de datos...")
def cached_get_db_instance(
    collection_name: str,
    expected_dimension_metadata: Optional[Union[int, str]]
) -> Optional[VectorDBInterface]:
    """Initializes and caches the DB instance for a specific collection."""
    logger.info(f"Attempting to load/cache DB instance for collection: '{collection_name}' (Expected Dim Meta: {expected_dimension_metadata})")
    try:
        # Call the factory to get/create the DB instance
        db_instance = create_vector_database(
            collection_name=collection_name,
            expected_dimension_metadata=expected_dimension_metadata
        )
        # is_initialized check is handled within create_vector_database now
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
    st.markdown(
        "Indexa tus imágenes, luego búscalas usando descripciones de texto, "
        "imágenes de ejemplo, o una combinación de ambas. Explora la estructura "
        "de tus datos con herramientas de clustering y visualización."
    )

    # --- Sidebar Configuration ---
    st.sidebar.header("Configuración Global")

    # Determine default model index safely
    try:
        default_model_name = st.session_state.get(STATE_SELECTED_MODEL, config.DEFAULT_MODEL_NAME)
        if default_model_name not in config.AVAILABLE_MODELS:
            default_model_name = config.AVAILABLE_MODELS[0] if config.AVAILABLE_MODELS else None
            if default_model_name:
                st.session_state[STATE_SELECTED_MODEL] = default_model_name
            else:
                st.sidebar.error("¡No hay modelos configurados en config.py!")
                st.stop()
        default_model_index = config.AVAILABLE_MODELS.index(default_model_name)
    except Exception as e:
        logger.error(f"Error determining default model index: {e}. Falling back to 0.")
        default_model_index = 0
        if config.AVAILABLE_MODELS:
            st.session_state[STATE_SELECTED_MODEL] = config.AVAILABLE_MODELS[0]
        else:
            st.sidebar.error("¡No hay modelos disponibles en config.py!")
            st.stop()

    # Model Selection
    selected_model = st.sidebar.selectbox(
        "Modelo de Embedding:",
        options=config.AVAILABLE_MODELS,
        index=default_model_index,
        key=STATE_SELECTED_MODEL,
        help="Elige el modelo. Cambiarlo recargará componentes y puede cambiar la colección de BD activa."
    )

    # Dimension Selection
    selected_dimension = st.sidebar.number_input(
        "Dimensión Embedding (0 = Completa):",
        min_value=0,
        value=st.session_state.get(STATE_SELECTED_DIMENSION, config.VECTOR_DIMENSION or 0),
        step=32, # Common step for embedding dimensions
        key=STATE_SELECTED_DIMENSION,
        help="Dimensión del vector final (0=Nativa del modelo). Cambiarlo puede cambiar la colección de BD activa."
    )
    # Determine effective truncation dimension (None if 0 or invalid)
    truncate_dim_value = selected_dimension if selected_dimension > 0 else None

    # --- Initialize Vectorizer (Cached by model name) ---
    vectorizer = cached_get_vectorizer(selected_model)

    # --- Stop if Vectorizer failed ---
    if not vectorizer or not vectorizer.is_ready:
        st.error(f"No se pudo inicializar el Vectorizador para el modelo '{selected_model}'. La aplicación no puede continuar.")
        st.warning("Revisa los logs, la configuración del modelo (config.py) y las dependencias (requirements.txt).")
        st.stop()

    # --- Determine Target DB Collection and Get Instance (Cached by name + metadata) ---
    db_instance: Optional[VectorDBInterface] = None
    target_collection_name = "N/A" # Default value
    effective_target_dimension = None # Default value
    try:
        # Use the base name from config and append suffix based on dimension
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
            # Use 'full' as metadata key for native dimension
            effective_target_dimension = 'full'
            target_collection_suffix = "_full" # Or use f"_dim{native_dim}"

        target_collection_name = f"{base_collection_name}{target_collection_suffix}"

        # Get/Create the DB instance for the target collection (cached)
        # Pass the expected dimension metadata for potential creation/verification
        db_instance = cached_get_db_instance(target_collection_name, effective_target_dimension)

    except Exception as db_select_err:
         logger.error(f"Error determining target DB collection: {db_select_err}", exc_info=True)
         st.error(f"Error al determinar la colección de base de datos correcta: {db_select_err}")
         st.stop()

    # --- Stop if DB Instance failed ---
    if not db_instance or not db_instance.is_initialized:
        st.error(f"No se pudo inicializar la instancia de Base de Datos para la colección '{target_collection_name}'. La aplicación no puede continuar.")
        st.warning("Revisa los logs y la configuración de la base de datos (config.py).")
        st.stop()

    # --- Sidebar Info Display (using the correct DB instance) ---
    st.sidebar.header("Información del Sistema")
    st.sidebar.caption(f"Python: `{sys.version.split()[0]}`")
    st.sidebar.caption(f"Streamlit: `{st.__version__}`")
    st.sidebar.caption(f"Dispositivo Usado: `{vectorizer.device}`")
    cuda_available = torch.cuda.is_available() if 'torch' in sys.modules else False
    st.sidebar.caption(f"CUDA Disponible: `{'Sí' if cuda_available else 'No'}`")

    st.sidebar.header("Configuración Activa")
    st.sidebar.caption(f"Modelo Cargado: `{selected_model}`")
    st.sidebar.caption(f"Dimensión Nativa: `{vectorizer.native_dimension or 'N/A'}`")
    st.sidebar.caption(f"Dimensión Objetivo: `{truncate_dim_value or 'Nativa'}`")
    st.sidebar.caption(f"Colección Activa: `{getattr(db_instance, 'collection_name', 'N/A')}`") # Show active collection name

    # Display Info for the active DB
    display_database_info(db_instance, st.sidebar) # Pass the correct instance

    # --- Main Tab Definition and Rendering ---
    main_tab_titles = ["📊 Búsqueda Vectorial", "🧩 Clustering y Visualización"]
    main_tab1, main_tab2 = st.tabs(main_tab_titles)

    # --- Main Tab 1: Vector Search ---
    with main_tab1:
        st.header("Herramientas de Búsqueda")
        # --- Search SUB-TABS (Existing ones) ---
        search_tab_titles = ["💾 Indexar", "📝 Buscar Texto", "🖼️ Buscar Imagen", "🧬 Búsqueda Híbrida"]
        s_tab1, s_tab2, s_tab3, s_tab4 = st.tabs(search_tab_titles)

        # Pass vectorizer, db_instance, and truncate_dim_value to each search sub-tab
        with s_tab1:
            render_indexing_tab(vectorizer, db_instance, truncate_dim_value)
        with s_tab2:
            render_text_search_tab(vectorizer, db_instance, truncate_dim_value)
        with s_tab3:
            render_image_search_tab(vectorizer, db_instance, truncate_dim_value)
        with s_tab4:
             render_hybrid_search_tab(vectorizer, db_instance, truncate_dim_value)

    # --- Main Tab 2: Clustering and Visualization ---
    with main_tab2:
        st.header("Análisis de Clusters")
        # --- Clustering SUB-TABS (New ones) ---
        cluster_tab_titles = [
            "⚙️ Ejecutar Clustering",
            "📉 Optimizar (Pre-Cluster)",
            "🎨 Visualizar (Post-Cluster)"
        ]
        c_tab1, c_tab2, c_tab3 = st.tabs(cluster_tab_titles)

        # Pass necessary components to each clustering sub-tab
        # Note: State management (st.session_state) will be used
        # to pass data like cluster results between these tabs.
        with c_tab1:
            render_cluster_execution_tab(vectorizer, db_instance, truncate_dim_value)
        with c_tab2:
            render_cluster_optimization_tab(vectorizer, db_instance, truncate_dim_value)
        with c_tab3:
            render_cluster_visualization_tab(vectorizer, db_instance, truncate_dim_value)

# --- Entry Point ---
if __name__ == "__main__":
    # Basic Python version check
    if sys.version_info < (3, 8):
        logger.warning(
            "Warning: This application is developed and tested with Python 3.8+. "
            "Older versions might encounter compatibility issues."
        )
    main()
