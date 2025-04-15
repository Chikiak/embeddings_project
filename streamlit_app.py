# --- streamlit_app.py ---
import logging
import os
import sys
from typing import Optional

import streamlit as st
import torch

# --- Inicio: Añadido para asegurar que los módulos locales se importen ---
try:
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    logger_check = logging.getLogger(__name__)
    logger_check.debug(f"Project root added to sys.path: {project_root}")
except Exception as path_e:
    print(f"Error adjusting sys.path: {path_e}")
# --- Fin: Añadido ---

try:
    # Importaciones del proyecto
    import config # Importa config actualizado
    from app.exceptions import InitializationError
    # Importa funciones refactorizadas de factory
    from core.factory import create_vectorizer, get_or_create_db_for_model_and_dim
    from app.logging_config import setup_logging
    from app.ui.sidebar import display_database_info
    from app.ui.state_utils import (
        STATE_SELECTED_DIMENSION,
        STATE_SELECTED_MODEL,
        clear_clustering_states, # Importa función para limpiar estado de clustering
    )
    # Importaciones de las pestañas
    from app.ui.tabs.clustering.cluster_execution_tab import render_cluster_execution_tab
    from app.ui.tabs.clustering.cluster_optimization_tab import render_cluster_optimization_tab
    from app.ui.tabs.clustering.cluster_visualization_tab import render_cluster_visualization_tab
    from app.ui.tabs.search.hybrid_search_tab import render_hybrid_search_tab
    from app.ui.tabs.search.image_search_tab import render_image_search_tab
    from app.ui.tabs.indexing_tab import render_indexing_tab
    from app.ui.tabs.search.text_search_tab import render_text_search_tab
    # Importaciones del núcleo y acceso a datos
    from core.vectorizer import Vectorizer
    from data_access.vector_db_interface import VectorDBInterface

except ImportError as e:
    error_msg = f"Critical error importing modules: {e}. Check PYTHONPATH, dependencies (requirements.txt), and file structure."
    print(error_msg)
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

# Configuración del logging
setup_logging()
logger = logging.getLogger(__name__)

# Configuración de la página de Streamlit
st.set_page_config(
    page_title="Buscador de Imágenes Vectorial",
    page_icon="🖼️",
    layout="wide"
)

# --- Función para cargar CSS externo ---
def load_css(file_name):
    """Carga un archivo CSS externo."""
    try:
        base_path = os.path.dirname(__file__)
        css_path = os.path.join(base_path, file_name)
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        logger.debug(f"CSS loaded from: {css_path}")
    except Exception as e:
        logger.error(f"Error loading CSS from {css_path}: {e}")
        st.warning(f"Could not load CSS file '{file_name}'.")

# --- Funciones de caché para Vectorizer y DB ---
# Usamos las funciones directamente de factory.py ahora, el caché es interno a factory
# @st.cache_resource(...) # Ya no se necesita aquí si factory lo maneja

# --- Función Principal ---
def main():
    """Función principal de la aplicación Streamlit."""
    load_css("app/ui/style.css")

    st.title("🖼️ Buscador de Imágenes por Similitud Vectorial")
    st.markdown("Explora, indexa, busca y analiza tu colección de imágenes mediante vectores.")
    st.divider()

    # --- Configuración de la Sidebar ---
    st.sidebar.title("⚙️ Configuración")
    st.sidebar.divider()

    # Selección del Modelo
    # Usa el valor guardado en sesión o el por defecto de config
    selected_model_key = STATE_SELECTED_MODEL
    if selected_model_key not in st.session_state:
        st.session_state[selected_model_key] = config.MODEL_NAME # Usa el validado en config

    # Callback para limpiar caché y estado si cambia el modelo
    def on_model_change():
        # Limpia cachés internos de factory (si es necesario y posible)
        # factory._vectorizer_cache.clear() # Ejemplo si el caché fuera accesible
        # factory._db_cache.clear()
        # Limpia estados relevantes de Streamlit que dependen del modelo/DB
        clear_clustering_states()
        # Podrías añadir más limpieza de estado aquí si es necesario
        logger.info(f"Model changed to {st.session_state[selected_model_key]}. Caches/states potentially cleared.")
        # No es necesario limpiar st.cache_resource explícitamente aquí,
        # Streamlit lo maneja basado en los argumentos de la función cacheada.

    selected_model = st.sidebar.selectbox(
        "Modelo de Embedding:",
        options=config.AVAILABLE_MODELS,
        key=selected_model_key,
        on_change=on_model_change, # Llama al callback al cambiar
        help="Elige el modelo. Cambiarlo recargará componentes.",
    )

    # Selección de Dimensión
    selected_dim_key = STATE_SELECTED_DIMENSION
    if selected_dim_key not in st.session_state:
         # Usa config.VECTOR_DIMENSION (que es None para nativa) o 0 si es None
         st.session_state[selected_dim_key] = config.VECTOR_DIMENSION if config.VECTOR_DIMENSION is not None else 0

    # Callback para limpiar caché y estado si cambia la dimensión
    def on_dimension_change():
         # Limpia caché de DB, ya que la colección depende de la dimensión
         # factory._db_cache.clear() # Ejemplo si el caché fuera accesible
         # Limpia estados relevantes
         clear_clustering_states()
         logger.info(f"Dimension changed to {st.session_state[selected_dim_key]}. Caches/states potentially cleared.")

    selected_dimension = st.sidebar.number_input(
        "Dimensión Embedding (0 = Nativa):",
        min_value=0,
        step=32,
        key=selected_dim_key,
        on_change=on_dimension_change, # Llama al callback al cambiar
        help="Dimensión final (0 para nativa). Cambiar esto altera la colección de BD.",
    )

    # Determina el valor de truncamiento efectivo
    truncate_dim_value = selected_dimension if selected_dimension > 0 else None

    # --- Carga de Componentes (Vectorizer y DB) ---
    vectorizer: Optional[Vectorizer] = None
    db_instance: Optional[VectorDBInterface] = None
    error_loading = False

    try:
        # Obtiene/Crea Vectorizer (usa caché interno de factory)
        vectorizer = create_vectorizer(
            model_name=selected_model,
            device=config.DEVICE,
            trust_remote_code=config.TRUST_REMOTE_CODE,
        )
        if not vectorizer or not vectorizer.is_ready:
             st.error(f"Error al inicializar el vectorizador '{selected_model}'.")
             error_loading = True

        # Obtiene/Crea la instancia de BD correcta (usa caché interno de factory)
        if not error_loading:
             db_instance = get_or_create_db_for_model_and_dim(vectorizer, truncate_dim_value)
             if not db_instance or not db_instance.is_initialized:
                  st.error(f"Error al inicializar la base de datos para la configuración actual.")
                  error_loading = True

    except (InitializationError, ValueError) as e:
        st.error(f"Error de inicialización: {e}")
        error_loading = True
    except Exception as e:
        st.error(f"Error inesperado durante la inicialización: {e}")
        logger.error("Unexpected initialization error", exc_info=True)
        error_loading = True

    # Si hubo error, detiene la app
    if error_loading:
        st.warning("La aplicación no puede continuar debido a errores de inicialización.")
        st.stop()

    # --- Información Adicional en Sidebar ---
    st.sidebar.divider()
    st.sidebar.header("ℹ️ Información")
    if vectorizer:
        st.sidebar.caption(f"**Modelo:** `{selected_model}`")
        st.sidebar.caption(f"**Dim. Nativa:** `{vectorizer.native_dimension or 'N/A'}`")
        st.sidebar.caption(f"**Dim. Objetivo:** `{truncate_dim_value or 'Nativa'}`")
    if db_instance:
        st.sidebar.caption(f"**Colección DB:** `{getattr(db_instance, 'collection_name', 'N/A')}`")
        # Muestra info detallada de la BD
        display_database_info(db_instance, st.sidebar) # Pasa el contenedor sidebar

    st.sidebar.divider()
    st.sidebar.header("Sistema")
    st.sidebar.caption(f"**Python:** `{sys.version.split()[0]}`")
    st.sidebar.caption(f"**Streamlit:** `{st.__version__}`")
    if vectorizer:
        st.sidebar.caption(f"**Dispositivo ML:** `{vectorizer.device.upper()}`")
    cuda_available = torch.cuda.is_available()
    st.sidebar.caption(f"**CUDA:** `{'✅ Disponible' if cuda_available else '❌ No Disponible'}`")

    # --- Pestañas Principales ---
    main_tab_titles = ["📊 Búsqueda Vectorial", "🧩 Clustering y Análisis"]
    main_tab1, main_tab2 = st.tabs(main_tab_titles)

    # --- Pestaña de Búsqueda ---
    with main_tab1:
        st.header("Herramientas de Búsqueda y Gestión")
        search_tab_titles = ["💾 Indexar", "📝 Buscar Texto", "🖼️ Buscar Imagen", "🧬 Búsqueda Híbrida"]
        s_tab1, s_tab2, s_tab3, s_tab4 = st.tabs(search_tab_titles)

        # Pasa los componentes inicializados a las pestañas
        with s_tab1: render_indexing_tab(vectorizer, db_instance, truncate_dim_value)
        with s_tab2: render_text_search_tab(vectorizer, db_instance, truncate_dim_value)
        with s_tab3: render_image_search_tab(vectorizer, db_instance, truncate_dim_value)
        with s_tab4: render_hybrid_search_tab(vectorizer, db_instance, truncate_dim_value)

    # --- Pestaña de Clustering ---
    with main_tab2:
        st.header("Análisis de Clusters")
        cluster_tab_titles = ["⚙️ Ejecutar", "📉 Optimizar", "🎨 Visualizar"]
        c_tab1, c_tab2, c_tab3 = st.tabs(cluster_tab_titles)

        # Pasa los componentes inicializados a las pestañas
        with c_tab1: render_cluster_execution_tab(vectorizer, db_instance, truncate_dim_value)
        with c_tab2: render_cluster_optimization_tab(vectorizer, db_instance, truncate_dim_value)
        with c_tab3: render_cluster_visualization_tab(vectorizer, db_instance, truncate_dim_value)


# Punto de entrada principal
if __name__ == "__main__":
    if sys.version_info < (3, 8):
        logger.warning("Se recomienda Python 3.8+.")
    main()
