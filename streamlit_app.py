# streamlit_app.py (Refactored)
import os
import sys
import time
import streamlit as st
from typing import Optional, Tuple
import logging

# --- Initial Setup and Path Configuration ---
try:
    # Ensure the project root is in the Python path
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # --- Import project modules AFTER path setup ---
    import config # Core config
    from app.factory import get_initialized_components # Core factory
    from core.vectorizer import Vectorizer # Type hint for cache
    from data_access.vector_db_interface import VectorDBInterface # Type hint for cache
    from app.logging_config import setup_logging # Logging setup
    from app.exceptions import InitializationError # Core exception

    # --- Import UI Modules ---
    # Import state keys used in the main app structure (sidebar)
    from app.ui.state_utils import STATE_SELECTED_MODEL, STATE_SELECTED_DIMENSION
    # Import the function to display sidebar info
    from app.ui.sidebar import display_database_info
    # Import the rendering function for each tab
    from app.ui.tabs.indexing_tab import render_indexing_tab
    from app.ui.tabs.text_search_tab import render_text_search_tab
    from app.ui.tabs.image_search_tab import render_image_search_tab
    from app.ui.tabs.hybrid_search_tab import render_hybrid_search_tab

except ImportError as e:
    # Keep initial critical error handling
    error_msg = (
        f"Error crítico al importar módulos: {e}. Asegúrate de que todos los archivos "
        f"estén presentes y las dependencias instaladas (`pip install -r requirements.txt`). "
        f"Verifica la estructura de directorios (app/ui/, app/ui/tabs/)."
    )
    print(error_msg)
    # Display error in Streamlit if possible
    try:
        st.error(error_msg)
        st.stop()
    except Exception: # Handle case where Streamlit isn't fully loaded
        sys.exit(f"CRITICAL ERROR: {error_msg}") # Exit if Streamlit fails
except Exception as e:
    error_msg = f"Error inesperado durante la configuración inicial: {e}"
    print(error_msg)
    try:
        st.error(error_msg)
        st.stop()
    except Exception:
        sys.exit(f"CRITICAL ERROR: {error_msg}")

# --- Logging Configuration ---
setup_logging()
logger = logging.getLogger(__name__)

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Buscador de Imágenes Vectorial",
    page_icon="🔍",
    layout="wide"
)

# --- Custom CSS Styles (Keep as is) ---
# Includes styles for results grid, items, progress bar, spinner, etc.
st.markdown(
    """
<style>
    .results-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 15px; padding-top: 10px; }
    .result-item { border: 1px solid #ddd; padding: 10px; border-radius: 8px; text-align: center; background-color: #f9f9f9; transition: box-shadow 0.3s ease; display: flex; flex-direction: column; justify-content: space-between; height: 100%; }
    .result-item:hover { box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
    .result-item img { max-width: 100%; height: 150px; border-radius: 4px; margin-bottom: 5px; object-fit: contain; } /* Ensure images fit well */
    .result-item .caption { font-size: 0.8em; color: #555; word-wrap: break-word; margin-top: auto; } /* Ensure caption wraps */
    .stProgress > div > div > div > div { background-color: #19A7CE; } /* Progress bar color */
    .stSpinner > div { text-align: center; } /* Center spinner text */
    .block-container { padding-top: 2rem; padding-bottom: 2rem; } /* Add padding around main content */
    h1, h2, h3 { color: #003366; } /* Heading colors */
    .selected-query-image img { border: 3px solid #19A7CE; border-radius: 5px; } /* Style selected query image */
</style>
""",
    unsafe_allow_html=True,
)

# --- Component Initialization Cache (Keep as is) ---
@st.cache_resource(show_spinner=False) # Cache vectorizer and DB connection
def cached_get_components(model_name_to_load: str) -> Tuple[Optional[Vectorizer], Optional[VectorDBInterface]]:
    """
    Initializes and caches the Vectorizer and VectorDatabase for a SPECIFIC MODEL.
    Shows progress during first initialization. Returns (None, None) if initialization fails.

    Args:
        model_name_to_load: The name of the model to load.

    Returns:
        A tuple (Vectorizer, VectorDBInterface) or (None, None) on error.
    """
    # Container to show initialization progress temporarily
    init_container = st.empty()
    with init_container.container():
        st.info(
            f"⏳ Inicializando componentes para el modelo '{model_name_to_load}'... Esto puede tardar."
        )
        # Layout for progress bar and status text
        pcol1, pcol2 = st.columns([1, 4])
        progress_bar = pcol1.progress(0)
        status_text = pcol2.empty()

    vectorizer = None
    db = None
    initialization_successful = False

    try:
        status_text.text(f"Cargando modelo de vectorización '{model_name_to_load}'...")
        # Call the factory function to get components
        vectorizer, db = get_initialized_components(model_name=model_name_to_load)
        progress_bar.progress(50) # Update progress

        status_text.text("Verificando conexión a base de datos...")
        # Check if database connection is valid
        if db and db.is_initialized:
            db_count = db.count() # Get item count
            logger.info(f"Database connection verified for model '{model_name_to_load}'. Item count: {db_count}")
            progress_bar.progress(100) # Final progress update
            status_text.success(f"✅ Componentes listos para '{model_name_to_load}'.")
            initialization_successful = True
        elif db is None:
             # Handle case where DB object wasn't created
             status_text.error("❌ Falló la creación de la instancia de base de datos.")
             logger.error(f"Database object is None after factory call for model '{model_name_to_load}'.")
        else:
            # Handle case where DB object exists but failed initialization check
            status_text.error("❌ No se pudo inicializar/conectar a la base de datos.")
            logger.error(f"Database component failed initialization check for model '{model_name_to_load}'.")

    except InitializationError as e:
        # Handle specific initialization errors from the factory
        logger.critical(
            f"Fatal error during component initialization for model '{model_name_to_load}': {e}", exc_info=True
        )
        status_text.error(f"❌ Error fatal al inicializar '{model_name_to_load}': {e}")
        st.error( # Display persistent error if init fails
            f"No se pudieron inicializar los componentes para el modelo '{model_name_to_load}'. Verifica la configuración, dependencias, permisos y conexión."
        )
    except Exception as e:
        # Handle any other unexpected errors during initialization
        logger.critical(
            f"Unexpected fatal error during component initialization for model '{model_name_to_load}': {e}",
            exc_info=True,
        )
        status_text.error(f"❌ Error inesperado al inicializar '{model_name_to_load}': {e}")

    finally:
        # Give user time to read the status message before clearing it
        time.sleep(1.5 if initialization_successful else 3.0)
        init_container.empty() # Remove the progress display

    # Return the components (or None if failed)
    return (vectorizer, db) if initialization_successful else (None, None)


# --- UI Helper Functions (MOVED to app/ui/...) ---
# All helper functions like display_results, reset_image_selection_states, etc.
# are now imported from their respective modules in app/ui/

# --- Tab Rendering Functions (MOVED to app/ui/tabs/...) ---
# All render_*_tab functions are now imported from app/ui/tabs/


# --- Main Application Logic ---
def main():
    """Función principal para ejecutar la aplicación Streamlit."""
    st.title("🔍 Buscador de Imágenes por Similitud Vectorial")
    st.markdown(
        """
    Bienvenido/a. Indexa imágenes y búscalas por contenido visual
    usando descripciones de texto, imágenes de ejemplo o una combinación de ambos.
    """
    )

    # --- Sidebar Configuration ---
    st.sidebar.header("Configuración Global")

    # Determine default model index robustly, handling potential errors
    try:
        # Get model from session state or config default
        default_model_name = st.session_state.get(STATE_SELECTED_MODEL, config.DEFAULT_MODEL_NAME)
        # Ensure the default model is actually in the available list
        if default_model_name not in config.AVAILABLE_MODELS:
            logger.warning(f"Default model '{default_model_name}' not in AVAILABLE_MODELS. Using first available.")
            # Use the first available model if the default isn't valid
            default_model_name = config.AVAILABLE_MODELS[0] if config.AVAILABLE_MODELS else None
            if default_model_name:
                st.session_state[STATE_SELECTED_MODEL] = default_model_name # Correct state
            else:
                 st.sidebar.error("¡No hay modelos configurados en AVAILABLE_MODELS!")
                 st.stop() # Stop if no models are available
        # Find the index of the (potentially corrected) default model
        default_model_index = config.AVAILABLE_MODELS.index(default_model_name)
    except (ValueError, IndexError, TypeError) as e:
        # Fallback if AVAILABLE_MODELS is empty or other errors occur
        logger.error(f"Error determining default model index: {e}. Falling back to 0.")
        default_model_index = 0
        # Ensure session state reflects the fallback if possible
        if config.AVAILABLE_MODELS:
            st.session_state[STATE_SELECTED_MODEL] = config.AVAILABLE_MODELS[0]
        else:
            st.sidebar.error("¡No hay modelos disponibles en la configuración!")
            st.stop() # Critical error if no models configured

    # Model selection dropdown
    selected_model = st.sidebar.selectbox(
        "Modelo de Embedding:",
        options=config.AVAILABLE_MODELS,
        index=default_model_index,
        key=STATE_SELECTED_MODEL, # Use the state key to store selection
        help="Elige el modelo para generar los vectores. Cambiarlo recargará componentes."
    )

    # Embedding dimension input
    selected_dimension = st.sidebar.number_input(
        "Dimensión Embedding (0 = Completa):",
        min_value=0,
        # Get value from state or config, default to 0 if config.VECTOR_DIMENSION is None
        value=st.session_state.get(STATE_SELECTED_DIMENSION, config.VECTOR_DIMENSION or 0),
        step=32, # Example step size
        key=STATE_SELECTED_DIMENSION, # Use state key to store selection
        help=(
            "Número de dimensiones del vector final (trunca si es menor que el nativo). "
            "0 usa la dimensión completa del modelo."
            )
    )
    # Determine actual truncate dimension value (None if 0 or empty)
    truncate_dim_value = selected_dimension if selected_dimension > 0 else None

    # --- Initialize Components based on selected model ---
    # Get the currently selected model name from session state
    active_model_name = st.session_state.get(STATE_SELECTED_MODEL)
    # Call the cached function to get/create components for this model
    # Streamlit's cache handles reusing components if the model name hasn't changed
    vectorizer, db = cached_get_components(active_model_name)

    # --- Display System/DB Info in Sidebar ---
    st.sidebar.header("Información del Sistema")
    st.sidebar.caption(f"Python: {sys.version_info.major}.{sys.version_info.minor}")
    # Display device and CUDA info safely
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        # Get device from vectorizer if available and ready
        actual_device = vectorizer.device if vectorizer and vectorizer.is_ready else 'N/A'
        st.sidebar.caption(f"Dispositivo Usado: `{actual_device}`")
        st.sidebar.caption(f"CUDA Disponible: {'✅ Sí' if cuda_available else '❌ No'}")
        if cuda_available:
            try:
                 # Display GPU name if CUDA is available
                 st.sidebar.caption(f"GPU: {torch.cuda.get_device_name(0)}")
            except Exception as gpu_name_err:
                 st.sidebar.caption(f"GPU: Error al obtener nombre ({gpu_name_err})")
    except ImportError:
        st.sidebar.warning("Librería 'torch' no encontrada.")
    except Exception as e:
        st.sidebar.error(f"Error al verificar CUDA/Dispositivo: {e}")

    # Display active configuration
    st.sidebar.header("Configuración Activa")
    st.sidebar.caption(f"Modelo Cargado: `{active_model_name}`")
    st.sidebar.caption(f"Dimensión Objetivo: `{truncate_dim_value or 'Completa'}`")

    # Display Database Info using the imported function from app.ui.sidebar
    display_database_info(db, st.sidebar)


    # --- Stop if initialization failed ---
    # Check if vectorizer or DB failed to initialize properly
    if not vectorizer or not vectorizer.is_ready or not db or not db.is_initialized:
        st.error(
            f"La aplicación no puede continuar. Falló la inicialización de componentes para el modelo '{active_model_name}'."
        )
        st.warning("Revisa los logs para más detalles. Asegúrate de que el modelo existe y las dependencias están instaladas.")
        st.stop() # Stop execution if components are not ready

    # --- Tabs Definition ---
    tab_titles = [
        "💾 Indexar Imágenes",
        "📝 Buscar por Texto",
        "🖼️ Buscar por Imagen",
        "🧬 Búsqueda Híbrida"
    ]
    # Create the tabs
    tab1, tab2, tab3, tab4 = st.tabs(tab_titles)

    # --- Render Tabs using imported functions ---
    # Pass the necessary initialized components and config values to each tab's render function
    with tab1:
        render_indexing_tab(vectorizer, db, truncate_dim_value)
    with tab2:
        render_text_search_tab(vectorizer, db, truncate_dim_value)
    with tab3:
        render_image_search_tab(vectorizer, db, truncate_dim_value)
    with tab4:
         render_hybrid_search_tab(vectorizer, db, truncate_dim_value)


# --- Entry Point ---
if __name__ == "__main__":
    main() # Run the main Streamlit application function
