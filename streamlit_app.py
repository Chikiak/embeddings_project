import logging
import os
import sys
from typing import Optional, Union

import streamlit as st
import torch

# --- Inicio: Añadido para asegurar que los módulos locales se importen ---
try:
    # Obtiene la ruta del directorio donde está este script
    project_root = os.path.dirname(os.path.abspath(__file__))
    # Si no está en sys.path, lo añade al principio
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    logger_check = logging.getLogger(__name__)
    logger_check.debug(f"Project root added to sys.path: {project_root}")
    logger_check.debug(f"Current sys.path: {sys.path}")
except Exception as path_e:
    print(f"Error adjusting sys.path: {path_e}")
# --- Fin: Añadido ---

try:
    # Importaciones del proyecto
    import config
    from app import searching # searching no se usa directamente aquí, pero podría ser necesario para otros módulos
    from app.exceptions import InitializationError, PipelineError
    from app.factory import create_vector_database, create_vectorizer
    from app.logging_config import setup_logging
    from app.ui.sidebar import display_database_info
    from app.ui.state_utils import (
        STATE_CLUSTER_IDS, # No se usan directamente aquí, pero se mantienen por si acaso
        STATE_CLUSTER_LABELS,
        STATE_REDUCED_EMBEDDINGS,
        STATE_REDUCED_IDS,
        STATE_SELECTED_DIMENSION,
        STATE_SELECTED_MODEL,
    )
    # Importaciones de las pestañas
    from app.ui.tabs.clustering.cluster_execution_tab import (
        render_cluster_execution_tab,
    )
    from app.ui.tabs.clustering.cluster_optimization_tab import (
        render_cluster_optimization_tab,
    )
    from app.ui.tabs.clustering.cluster_visualization_tab import (
        render_cluster_visualization_tab,
    )
    from app.ui.tabs.hybrid_search_tab import render_hybrid_search_tab
    from app.ui.tabs.image_search_tab import render_image_search_tab
    from app.ui.tabs.indexing_tab import render_indexing_tab
    from app.ui.tabs.text_search_tab import render_text_search_tab
    # Importaciones del núcleo y acceso a datos
    from core.vectorizer import Vectorizer
    from data_access.vector_db_interface import VectorDBInterface

except ImportError as e:
    # Manejo de errores de importación más robusto
    error_msg = f"Critical error importing modules: {e}. Check PYTHONPATH, dependencies (requirements.txt), and file structure relative to streamlit_app.py."
    print(error_msg)
    # Intenta mostrar en Streamlit si es posible
    try:
        st.error(error_msg)
        st.stop()
    except Exception:
        # Si Streamlit aún no está listo, sal del script
        sys.exit(error_msg)
except Exception as e:
    # Captura otros errores inesperados durante la importación/configuración inicial
    error_msg = f"Unexpected error during initial setup: {e}"
    print(error_msg)
    try:
        st.error(error_msg)
        st.stop()
    except Exception:
        sys.exit(error_msg)

# Configuración del logging (debe hacerse lo antes posible)
setup_logging()
logger = logging.getLogger(__name__)

# Configuración de la página de Streamlit
st.set_page_config(
    page_title="Buscador de Imágenes Vectorial",
    page_icon="🖼️", # Icono actualizado
    layout="wide"
)

# --- Inicio: Función para cargar CSS externo ---
def load_css(file_name):
    """Carga un archivo CSS externo y lo inyecta en la app Streamlit."""
    try:
        # Construye la ruta relativa al script actual
        base_path = os.path.dirname(__file__)
        css_path = os.path.join(base_path, file_name)
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        logger.info(f"CSS cargado desde: {css_path}")
    except FileNotFoundError:
         logger.error(f"Archivo CSS no encontrado en: {css_path}")
         st.error(f"Error crítico: No se pudo cargar el archivo de estilos '{file_name}'. La interfaz puede verse afectada.")
    except Exception as e:
         logger.error(f"Error al cargar CSS desde {css_path}: {e}")
         st.warning(f"No se pudo cargar el archivo CSS '{file_name}': {e}")
# --- Fin: Función para cargar CSS externo ---


# --- Inicio: Funciones de caché para Vectorizer y DB ---
# (Sin cambios en la lógica interna, solo aseguramos que estén definidas)
@st.cache_resource(show_spinner="🧠 Cargando modelo de vectorización...")
def cached_get_vectorizer(model_name_to_load: str) -> Optional[Vectorizer]:
    """Initializes and caches the Vectorizer for a specific model."""
    logger.info(
        f"Attempting to load/cache Vectorizer for model: {model_name_to_load}"
    )
    try:
        vectorizer = create_vectorizer(
            model_name=model_name_to_load,
            device=config.DEVICE,
            trust_remote_code=config.TRUST_REMOTE_CODE,
        )
        if vectorizer.is_ready:
            logger.info(
                f"Vectorizer for {model_name_to_load} initialized and cached successfully."
            )
            return vectorizer
        else:
            logger.error(
                f"Vectorizer initialization failed for {model_name_to_load} (is_ready is False)."
            )
            st.error(
                f"Error al inicializar el vectorizador '{model_name_to_load}'."
            )
            return None
    except InitializationError as e:
        logger.error(
            f"InitializationError caching Vectorizer for {model_name_to_load}: {e}",
            exc_info=True,
        )
        st.error(
            f"Error al inicializar el vectorizador '{model_name_to_load}': {e}"
        )
        return None
    except Exception as e:
        logger.error(
            f"Unexpected error caching Vectorizer for {model_name_to_load}: {e}",
            exc_info=True,
        )
        st.error(
            f"Error inesperado al inicializar el vectorizador '{model_name_to_load}': {e}"
        )
        return None

@st.cache_resource(show_spinner="🔗 Accediendo a base de datos...")
def cached_get_db_instance(
    collection_name: str,
    expected_dimension_metadata: Optional[Union[int, str]],
) -> Optional[VectorDBInterface]:
    """Initializes and caches the DB instance for a specific collection."""
    logger.info(
        f"Attempting to load/cache DB instance for collection: '{collection_name}' (Expected Dim Meta: {expected_dimension_metadata})"
    )
    try:
        # Asegúrate de que la función create_vector_database exista y esté importada
        db_instance = create_vector_database(
            collection_name=collection_name,
            expected_dimension_metadata=expected_dimension_metadata,
        )
        # No necesitas verificar is_initialized aquí si create_vector_database ya lo hace o lanza error
        logger.info(
            f"DB instance for '{collection_name}' obtained and cached successfully."
        )
        return db_instance
    except InitializationError as e:
        logger.error(
            f"InitializationError caching DB for collection '{collection_name}': {e}",
            exc_info=True, # Mostrar traceback en logs
        )
        # Muestra un error más específico en la UI
        st.error(
            f"Error al inicializar la base de datos para la colección '{collection_name}': {e}"
        )
        return None
    except Exception as e:
        logger.error(
            f"Unexpected error caching DB for collection '{collection_name}': {e}",
            exc_info=True, # Mostrar traceback en logs
        )
        st.error(
            f"Error inesperado al inicializar la base de datos para la colección '{collection_name}': {e}"
        )
        return None
# --- Fin: Funciones de caché ---


def main():
    """Función principal de la aplicación Streamlit."""

    # Carga el CSS externo al inicio
    load_css("app/ui/style.css") # Ajusta la ruta si style.css está en otro lugar

    # Opcional: Carga Font Awesome si quieres usar iconos (descomentar si es necesario)
    # st.markdown('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">', unsafe_allow_html=True)

    st.title("🖼️ Buscador de Imágenes por Similitud Vectorial")
    st.markdown(
        """
        Explora y busca en tu colección de imágenes utilizando el poder de los embeddings vectoriales.
        Indexa tus imágenes, luego encuéntralas mediante descripciones textuales, imágenes similares
        o una combinación de ambas. También puedes analizar la estructura de tus datos con clustering.
        """,
        unsafe_allow_html=True # Permite usar iconos si se añaden en markdown
    )
    st.divider() # Separador visual

    # --- Configuración de la Sidebar ---
    st.sidebar.title("⚙️ Configuración")
    st.sidebar.divider()

    # Selección del Modelo
    try:
        # Determina el modelo por defecto o el seleccionado previamente
        default_model_name = st.session_state.get(
            STATE_SELECTED_MODEL, config.DEFAULT_MODEL_NAME
        )
        # Verifica si el modelo por defecto está en la lista de disponibles
        if default_model_name not in config.AVAILABLE_MODELS:
            default_model_name = (
                config.AVAILABLE_MODELS[0] if config.AVAILABLE_MODELS else None
            )
            if default_model_name:
                st.session_state[STATE_SELECTED_MODEL] = default_model_name
                logger.warning(f"Default model '{config.DEFAULT_MODEL_NAME}' not in AVAILABLE_MODELS. Falling back to '{default_model_name}'.")
            else:
                # Caso crítico: no hay modelos disponibles
                st.sidebar.error("¡Error Crítico! No hay modelos configurados en `config.py`.")
                st.stop() # Detiene la ejecución de la app

        # Encuentra el índice del modelo seleccionado/por defecto
        default_model_index = config.AVAILABLE_MODELS.index(default_model_name)

    except ValueError:
        # Si el modelo guardado ya no está en la lista
         logger.warning(f"Saved model '{default_model_name}' no longer in AVAILABLE_MODELS. Falling back to first available.")
         default_model_index = 0
         if config.AVAILABLE_MODELS:
             st.session_state[STATE_SELECTED_MODEL] = config.AVAILABLE_MODELS[0]
         else:
             st.sidebar.error("¡Error Crítico! No hay modelos disponibles en `config.py`.")
             st.stop()
    except Exception as e:
        # Captura otros errores inesperados
        logger.error(f"Error determining default model index: {e}. Falling back to 0.")
        default_model_index = 0
        if config.AVAILABLE_MODELS:
            st.session_state[STATE_SELECTED_MODEL] = config.AVAILABLE_MODELS[0]
        else:
            st.sidebar.error("¡Error Crítico! No hay modelos disponibles en `config.py`.")
            st.stop()

    # Widget para seleccionar el modelo
    selected_model = st.sidebar.selectbox(
        "Modelo de Embedding:",
        options=config.AVAILABLE_MODELS,
        index=default_model_index,
        key=STATE_SELECTED_MODEL, # Guarda la selección en el estado de sesión
        help="Elige el modelo de embedding. Cambiarlo recargará componentes y puede afectar la colección de BD activa.",
    )

    # Widget para seleccionar la dimensión
    selected_dimension = st.sidebar.number_input(
        "Dimensión Embedding (0 = Nativa):",
        min_value=0, # 0 significa usar la dimensión nativa
        value=st.session_state.get(
            STATE_SELECTED_DIMENSION, config.VECTOR_DIMENSION or 0 # Usa config o 0
        ),
        step=32, # Pasos razonables para dimensiones de embedding
        key=STATE_SELECTED_DIMENSION, # Guarda la selección
        help="Dimensión final del vector (0 para usar la dimensión nativa del modelo). Cambiar esto puede alterar la colección de BD utilizada.",
    )

    # Determina el valor de truncamiento a pasar a las funciones
    # Si selected_dimension es 0 o None, truncate_dim_value será None (sin truncamiento)
    truncate_dim_value = selected_dimension if selected_dimension > 0 else None

    # --- Carga de Componentes (Vectorizer y DB) ---
    # Usa las funciones cacheadas
    vectorizer = cached_get_vectorizer(selected_model)

    # Si el vectorizador no se carga, la app no puede continuar
    if not vectorizer or not vectorizer.is_ready:
        # El error ya se muestra dentro de cached_get_vectorizer
        st.warning(
            "Revisa los logs de la consola, la configuración del modelo (`config.py`) y las dependencias (`requirements.txt`)."
        )
        st.stop() # Detiene la ejecución

    # Determina la colección de BD correcta y obtén la instancia
    db_instance: Optional[VectorDBInterface] = None
    target_collection_name = "N/A" # Valor por defecto
    effective_target_dimension: Optional[Union[int, str]] = None # Valor por defecto

    try:
        # Nombre base de la colección (sin sufijos _dimXXX o _full)
        base_collection_name = config.CHROMA_COLLECTION_NAME.split("_dim")[
            0
        ].split("_full")[0]

        # Dimensión nativa del modelo cargado
        native_dim = vectorizer.native_dimension
        if not native_dim:
            # Si no se puede determinar, es un problema
            st.error(
                f"Error Crítico: No se pudo determinar la dimensión nativa para el modelo '{selected_model}'. No se puede seleccionar la colección de BD correcta."
            )
            st.stop()

        # Comprueba si se requiere truncamiento y si es válido
        is_truncated = (
            truncate_dim_value is not None          # Hay un valor de truncamiento
            and truncate_dim_value > 0              # Es positivo
            and truncate_dim_value < native_dim     # Es menor que la dimensión nativa
        )

        if is_truncated:
            # Se usará una dimensión truncada
            effective_target_dimension = truncate_dim_value
            target_collection_suffix = f"_dim{truncate_dim_value}"
            logger.info(f"Using truncated dimension: {truncate_dim_value}")
        else:
            # Se usará la dimensión completa (nativa)
            effective_target_dimension = "full" # Metadato para indicar dimensión completa
            target_collection_suffix = "_full"
            if truncate_dim_value is not None and truncate_dim_value >= native_dim:
                logger.warning(f"Requested dimension {truncate_dim_value} >= native {native_dim}. Using full dimension.")
            else:
                 logger.info(f"Using full native dimension: {native_dim}")

        # Construye el nombre final de la colección
        target_collection_name = (
            f"{base_collection_name}{target_collection_suffix}"
        )
        logger.info(f"Target DB collection determined: '{target_collection_name}' (Expected Dim Meta: {effective_target_dimension})")

        # Obtiene la instancia de la BD (usando caché)
        db_instance = cached_get_db_instance(
            target_collection_name, effective_target_dimension
        )

    except Exception as db_select_err:
        # Captura cualquier error durante la determinación de la BD
        logger.error(
            f"Error determining target DB collection: {db_select_err}",
            exc_info=True, # Muestra traceback en logs
        )
        st.error(
            f"Error crítico al determinar la colección de base de datos: {db_select_err}"
        )
        st.stop() # Detiene la ejecución

    # Verifica si la instancia de BD se obtuvo correctamente
    if not db_instance or not db_instance.is_initialized:
        # El error ya se muestra dentro de cached_get_db_instance
        st.warning(
            "Revisa los logs de la consola y la configuración de la base de datos (`config.py`). Asegúrate de que la ruta de la BD sea accesible."
        )
        st.stop() # Detiene la ejecución

    # --- Información Adicional en Sidebar ---
    st.sidebar.divider()
    st.sidebar.header("ℹ️ Información")
    st.sidebar.caption(f"**Modelo Cargado:** `{selected_model}`")
    st.sidebar.caption(f"**Dimensión Nativa:** `{vectorizer.native_dimension or 'Desconocida'}`")
    st.sidebar.caption(f"**Dimensión Objetivo:** `{truncate_dim_value or 'Nativa'}`")
    st.sidebar.caption(f"**Colección Activa:** `{getattr(db_instance, 'collection_name', 'N/A')}`")

    # Muestra información detallada de la BD (si está disponible)
    display_database_info(db_instance, st.sidebar)

    st.sidebar.divider()
    st.sidebar.header("Sistema")
    st.sidebar.caption(f"**Python:** `{sys.version.split()[0]}`")
    st.sidebar.caption(f"**Streamlit:** `{st.__version__}`")
    st.sidebar.caption(f"**Dispositivo ML:** `{vectorizer.device.upper()}`")
    cuda_available = (
        torch.cuda.is_available() if "torch" in sys.modules else False
    )
    st.sidebar.caption(
        f"**CUDA Disponible:** `{'✅ Sí' if cuda_available else '❌ No'}`"
    )


    # --- Definición de Pestañas Principales ---
    main_tab_titles = [
        "📊 Búsqueda Vectorial",
        "🧩 Clustering y Análisis", # Título actualizado
    ]
    main_tab1, main_tab2 = st.tabs(main_tab_titles)

    # --- Contenido de la Pestaña de Búsqueda ---
    with main_tab1:
        st.header("Herramientas de Búsqueda y Gestión") # Título actualizado
        st.markdown("Indexa nuevas imágenes o busca en la colección existente.")
        st.divider()

        search_tab_titles = [
            "💾 Indexar",
            "📝 Buscar Texto",
            "🖼️ Buscar Imagen",
            "🧬 Búsqueda Híbrida",
        ]
        s_tab1, s_tab2, s_tab3, s_tab4 = st.tabs(search_tab_titles)

        # Renderiza el contenido de cada sub-pestaña de búsqueda
        with s_tab1:
            render_indexing_tab(vectorizer, db_instance, truncate_dim_value)
        with s_tab2:
            render_text_search_tab(vectorizer, db_instance, truncate_dim_value)
        with s_tab3:
            render_image_search_tab(
                vectorizer, db_instance, truncate_dim_value
            )
        with s_tab4:
            render_hybrid_search_tab(
                vectorizer, db_instance, truncate_dim_value
            )

    # --- Contenido de la Pestaña de Clustering ---
    with main_tab2:
        st.header("Análisis de Clusters") # Título actualizado
        st.markdown("Agrupa tus imágenes y visualiza la estructura de los datos.")
        st.divider()

        cluster_tab_titles = [
            "⚙️ Ejecutar Clustering",
            "📉 Optimizar Datos (Pre-Cluster)", # Título actualizado
            "🎨 Visualizar Resultados (Post-Cluster)", # Título actualizado
        ]
        c_tab1, c_tab2, c_tab3 = st.tabs(cluster_tab_titles)

        # Renderiza el contenido de cada sub-pestaña de clustering
        with c_tab1:
            render_cluster_execution_tab(
                vectorizer, db_instance, truncate_dim_value
            )
        with c_tab2:
            render_cluster_optimization_tab(
                vectorizer, db_instance, truncate_dim_value
            )
        with c_tab3:
            render_cluster_visualization_tab(
                vectorizer, db_instance, truncate_dim_value
            )


# Punto de entrada principal
if __name__ == "__main__":
    # Advertencia si se usa una versión antigua de Python
    if sys.version_info < (3, 8):
        logger.warning(
            "Advertencia: Esta aplicación se desarrolla y prueba con Python 3.8+. "
            "Versiones anteriores podrían tener problemas de compatibilidad."
        )
    # Ejecuta la función principal de la aplicación
    main()
