# streamlit_app.py
import os
import sys
import time
import streamlit as st
from PIL import Image, UnidentifiedImageError
import tempfile
from typing import Optional, Tuple, List # Added List
import logging

# --- Initial Setup and Path Configuration ---
try:
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # --- Import project modules AFTER path setup ---
    import config
    from app.factory import get_initialized_components # Keep this
    from core.vectorizer import Vectorizer
    from data_access.vector_db_interface import VectorDBInterface
    from app import pipeline
    from app.models import SearchResults, SearchResultItem
    from core.image_processor import find_image_files
    from app.logging_config import setup_logging
    from app.exceptions import InitializationError, PipelineError, DatabaseError

except ImportError as e:
    error_msg = f"Error crítico al importar módulos: {e}. Asegúrate de que todos los archivos estén presentes y las dependencias instaladas (`pip install -r requirements.txt`)."
    print(error_msg)
    st.error(error_msg)
    st.stop()
except Exception as e:
    error_msg = f"Error inesperado durante la configuración inicial: {e}"
    print(error_msg)
    st.error(error_msg)
    st.stop()

# --- Logging Configuration ---
setup_logging()
logger = logging.getLogger(__name__)

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Buscador de Imágenes Vectorial", page_icon="🔍", layout="wide"
)

# --- Custom CSS Styles ---
# (Keep existing styles)
st.markdown(
    """
<style>
    .results-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 15px; padding-top: 10px; }
    .result-item { border: 1px solid #ddd; padding: 10px; border-radius: 8px; text-align: center; background-color: #f9f9f9; transition: box-shadow 0.3s ease; display: flex; flex-direction: column; justify-content: space-between; height: 100%; }
    .result-item:hover { box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
    .result-item img { max-width: 100%; height: 150px; border-radius: 4px; margin-bottom: 5px; object-fit: contain; }
    .result-item .caption { font-size: 0.8em; color: #555; word-wrap: break-word; margin-top: auto; }
    .stProgress > div > div > div > div { background-color: #19A7CE; }
    .stSpinner > div { text-align: center; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    h1, h2, h3 { color: #003366; }
    .selected-query-image img { border: 3px solid #19A7CE; border-radius: 5px; }
</style>
""",
    unsafe_allow_html=True,
)


# --- Session State Keys ---
# Suffixes like '_img' and '_hybrid' might be added dynamically later if needed
STATE_UPLOADED_IMG_PATH_PREFIX = "uploaded_image_temp_path"
STATE_SELECTED_INDEXED_IMG_PREFIX = "selected_indexed_image_path"
STATE_TRIGGER_SEARCH_FLAG_PREFIX = "trigger_search_after_selection" # New flag key
# General state keys
STATE_CONFIRM_CLEAR = "confirm_clear_triggered"
STATE_CONFIRM_DELETE = "confirm_delete_triggered"
STATE_SELECTED_MODEL = "selected_model"
STATE_SELECTED_DIMENSION = "selected_dimension"
STATE_SELECTED_BATCH_SIZE = "selected_batch_size" # Specific to indexing tab

# Helper to get state keys with suffix
def get_state_key(base: str, suffix: str = "") -> str:
    return f"{base}{suffix}"

# --- Component Initialization Cache ---
# Modifica la función cacheada para que dependa del modelo
@st.cache_resource(show_spinner=False)
def cached_get_components(model_name_to_load: str) -> Tuple[Optional[Vectorizer], Optional[VectorDBInterface]]:
    """
    Initializes and caches the Vectorizer and VectorDatabase for a SPECIFIC MODEL.

    Shows progress during first initialization. Returns (None, None)
    if initialization fails.

    Args:
        model_name_to_load: The name of the model to load.

    Returns:
        A tuple (Vectorizer, VectorDBInterface) or (None, None) on error.
    """
    init_container = st.empty()
    with init_container.container():
        st.info(
            f"⏳ Inicializando componentes para el modelo '{model_name_to_load}'... Esto puede tardar."
        )
        pcol1, pcol2 = st.columns([1, 4])
        progress_bar = pcol1.progress(0)
        status_text = pcol2.empty()

    vectorizer = None
    db = None
    initialization_successful = False

    try:
        status_text.text(f"Cargando modelo de vectorización '{model_name_to_load}'...")
        # Llama a la fábrica modificada con el nombre del modelo
        # force_reload_db=False por defecto, la caché de recursos maneja el ciclo de vida del modelo
        vectorizer, db = get_initialized_components(model_name=model_name_to_load)
        progress_bar.progress(50)

        status_text.text("Verificando conexión a base de datos...")
        # Check DB initialization status after creation
        # The db object might be None if factory failed before DB creation
        if db and db.is_initialized:
            db_count = db.count() # Get count after ensuring initialization
            logger.info(f"Database connection verified for model '{model_name_to_load}'. Item count: {db_count}")
            progress_bar.progress(100)
            status_text.success(f"✅ Componentes listos para '{model_name_to_load}'.")
            initialization_successful = True
        elif db is None:
             status_text.error("❌ Falló la creación de la instancia de base de datos.")
             logger.error(f"Database object is None after factory call for model '{model_name_to_load}'.")
        else: # db exists but is not initialized
            status_text.error("❌ No se pudo inicializar/conectar a la base de datos.")
            logger.error(f"Database component failed initialization check for model '{model_name_to_load}'.")


    except InitializationError as e:
        logger.critical(
            f"Fatal error during component initialization for model '{model_name_to_load}': {e}", exc_info=True
        )
        status_text.error(f"❌ Error fatal al inicializar '{model_name_to_load}': {e}")
        st.error(
            f"No se pudieron inicializar los componentes para el modelo '{model_name_to_load}'. Verifica la configuración (config.py), "
            "dependencias (requirements.txt), permisos de acceso a la ruta de la BD "
            f"('{config.CHROMA_DB_PATH}'), y conexión a internet."
        )
    except Exception as e:
        logger.critical(
            f"Unexpected fatal error during component initialization for model '{model_name_to_load}': {e}",
            exc_info=True,
        )
        status_text.error(f"❌ Error inesperado al inicializar '{model_name_to_load}': {e}")

    finally:
        # Give time for user to see success/error message before clearing
        time.sleep(1.5 if initialization_successful else 3.0)
        init_container.empty()

    if initialization_successful:
        return vectorizer, db
    else:
        # Return None for both if initialization failed at any step
        return None, None


# --- UI Helper Functions ---

def display_results(results: Optional[SearchResults], results_container: st.container):
    """
    Displays search results (SearchResults object) in a grid format.
    """
    if results is None:
        results_container.error("❌ La búsqueda falló o no devolvió resultados.")
        return
    if results.is_empty:
        results_container.warning("🤷‍♂️ No se encontraron resultados para tu búsqueda.")
        return

    results_container.success(f"🎉 ¡Encontrados {results.count} resultados!")

    num_columns = 5 # Adjust as preferred
    cols = results_container.columns(num_columns)

    for i, item in enumerate(results.items):
        col = cols[i % num_columns]
        with col.container():
            st.markdown('<div class="result-item">', unsafe_allow_html=True)
            try:
                if item.id and os.path.isfile(item.id):
                    img = Image.open(item.id)
                    st.image(img, use_container_width=True)
                    similarity_str = (
                        f"{item.similarity:.3f}"
                        if item.similarity is not None
                        else "N/A"
                    )
                    st.markdown(
                        f'<div class="caption">{os.path.basename(item.id)}<br>Similitud: {similarity_str}</div>',
                        unsafe_allow_html=True,
                    )
                elif item.id:
                    st.warning(
                        f"Archivo no encontrado:\n{os.path.basename(item.id)}"
                    )
                    st.caption(f"Ruta original: {item.id}")
                    logger.warning(
                        f"Result image file not found at path: {item.id}"
                    )
                else:
                    st.warning("Resultado con ID inválido.")

            except FileNotFoundError:
                st.error(
                    f"Error crítico: Archivo no encontrado en la ruta: {item.id}"
                )
                logger.error(f"FileNotFoundError for image path: {item.id}")
            except UnidentifiedImageError:
                st.error(
                    f"Error: No se pudo abrir/identificar imagen: {os.path.basename(item.id if item.id else 'ID inválido')}"
                )
                logger.warning(f"UnidentifiedImageError for: {item.id}")
            except Exception as e:
                st.error(
                    f"Error al mostrar '{os.path.basename(item.id if item.id else 'ID inválido')}': {str(e)[:100]}..."
                )
                logger.error(
                    f"Error displaying image {item.id}: {e}", exc_info=True
                )
            st.markdown("</div>", unsafe_allow_html=True)


def reset_image_selection_states(suffix: str = ""):
    """Resets session state variables related to image selection and cleans up temp files."""
    selected_key = get_state_key(STATE_SELECTED_INDEXED_IMG_PREFIX, suffix)
    uploaded_key = get_state_key(STATE_UPLOADED_IMG_PATH_PREFIX, suffix)
    trigger_key = get_state_key(STATE_TRIGGER_SEARCH_FLAG_PREFIX, suffix) # New flag

    # Clear selected indexed image path
    if selected_key in st.session_state:
        st.session_state[selected_key] = None

    # Clear uploaded image path and delete temp file
    temp_path = st.session_state.get(uploaded_key)
    if temp_path:
        if os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
                logger.debug(f"Deleted temp file on state reset ({suffix}): {temp_path}")
            except Exception as e_unlink:
                logger.warning(
                    f"Could not delete temp file {temp_path} on state reset ({suffix}): {e_unlink}"
                )
        if uploaded_key in st.session_state:
             st.session_state[uploaded_key] = None

    # Clear the trigger flag
    if trigger_key in st.session_state:
        st.session_state[trigger_key] = False

    uploader_key = f"image_uploader{suffix}"
    processed_name_key = f"{uploader_key}_processed_name"
    if processed_name_key in st.session_state:
        del st.session_state[processed_name_key]


def _display_database_info(db: VectorDBInterface, container: st.sidebar):
    """Displays current database information in the sidebar."""
    container.header("Base de Datos Vectorial")
    if db:
        try:
            is_ready = db.is_initialized
            if is_ready:
                db_count = db.count()
                db_path = getattr(db, "path", "N/A")
                collection_name = getattr(db, "collection_name", "N/A")
                container.info(
                    f"**Colección:** `{collection_name}`\n\n"
                    f"**Ruta:** `{os.path.abspath(db_path) if db_path != 'N/A' else 'N/A'}`\n\n"
                    f"**Imágenes Indexadas:** `{db_count if db_count >= 0 else 'Error al contar'}`"
                )
            else:
                container.warning("La base de datos no está inicializada o no es accesible.")
        except Exception as e:
            container.error(f"Error al obtener info de la BD: {e}")
            logger.error(f"Error getting DB info for sidebar: {e}", exc_info=True)
    else:
        container.error("Instancia de base de datos no disponible.")


def _render_upload_widget(key_suffix: str = "") -> Optional[str]:
    """Renders the file uploader and manages the temporary file. Returns temp file path on success."""
    uploader_key = f"image_uploader{key_suffix}"
    uploaded_state_key = get_state_key(STATE_UPLOADED_IMG_PATH_PREFIX, key_suffix)
    processed_name_key = f"{uploader_key}_processed_name"

    uploaded_file = st.file_uploader(
        "Arrastra o selecciona una imagen:",
        type=[ext.lstrip(".") for ext in config.IMAGE_EXTENSIONS],
        key=uploader_key,
    )

    current_temp_path = st.session_state.get(uploaded_state_key)

    if uploaded_file is None:
        # If the uploader is cleared, reset the state
        if current_temp_path:
            reset_image_selection_states(key_suffix)
        return None

    # If the same file is already processed and exists, return its path
    if current_temp_path and os.path.exists(current_temp_path) and st.session_state.get(processed_name_key) == uploaded_file.name:
         return current_temp_path
    else:
        # If a new file is uploaded or the previous temp file is gone, create a new one
        reset_image_selection_states(key_suffix) # Clean previous state first
        try:
            file_suffix = os.path.splitext(uploaded_file.name)[1]
            # Use a context manager for safer file handling
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                new_temp_path = tmp_file.name
            st.session_state[uploaded_state_key] = new_temp_path
            st.session_state[processed_name_key] = uploaded_file.name # Store the name of the processed file
            logger.debug(f"Created new temp file for upload ({key_suffix}): {new_temp_path}")
            return new_temp_path
        except Exception as e:
            st.error(f"❌ Error al guardar archivo temporal: {e}")
            logger.error(f"Error creating temp file ({key_suffix}): {e}", exc_info=True)
            reset_image_selection_states(key_suffix) # Ensure cleanup on error
            return None


def _render_indexed_selection(db: VectorDBInterface, key_suffix: str = "") -> Optional[str]:
    """Renders the selection grid for indexed images. Returns the selected path."""
    selected_state_key = get_state_key(STATE_SELECTED_INDEXED_IMG_PREFIX, key_suffix)
    trigger_search_key = get_state_key(STATE_TRIGGER_SEARCH_FLAG_PREFIX, key_suffix) # New flag key

    if not db or not db.is_initialized:
         st.warning("⚠️ Base de datos no disponible para seleccionar imágenes.")
         return st.session_state.get(selected_state_key)
    db_count = -1
    try:
        db_count = db.count()
    except Exception as e:
        st.error(f"Error al obtener contador de la BD: {e}")
        logger.error(f"Error getting DB count in _render_indexed_selection: {e}", exc_info=True)
        return st.session_state.get(selected_state_key)

    if db_count <= 0:
        st.warning(f"⚠️ No hay imágenes indexadas en la base de datos (Count: {db_count}).")
        if selected_state_key in st.session_state:
            st.session_state[selected_state_key] = None
        if trigger_search_key in st.session_state:
            st.session_state[trigger_search_key] = False # Ensure flag is off
        return None

    st.info(
        "Mostrando algunas imágenes indexadas. Haz clic en 'Usar esta' para seleccionarla y buscar automáticamente." # Updated help text
    )
    indexed_image_paths: List[str] = []
    selected_image_path: Optional[str] = st.session_state.get(selected_state_key)

    # --- Retrieve and verify indexed image paths (same logic as before) ---
    try:
        limit_get = 25
        logger.debug(f"Attempting to retrieve up to {limit_get} IDs for indexed selection preview...")
        data = db.get_all_embeddings_with_ids(pagination_batch_size=limit_get)

        if data and data[0]:
            all_ids = data[0]
            logger.debug(f"Retrieved {len(all_ids)} IDs from DB. First few: {all_ids[:10]}")
            if not all_ids:
                 logger.warning("Retrieved data but the ID list is empty.")
                 st.warning("No se encontraron IDs en la base de datos.")
                 return st.session_state.get(selected_state_key)

            with st.spinner("Verificando archivos indexados..."):
                verified_paths = []
                checked_count = 0
                found_count = 0
                logger.debug("Starting file existence check for retrieved IDs...")
                for id_path in all_ids:
                    if id_path and isinstance(id_path, str):
                        checked_count += 1
                        try:
                            exists = os.path.isfile(id_path)
                            if exists:
                                verified_paths.append(id_path)
                                found_count += 1
                        except Exception as e_check:
                            logger.error(f"  Error checking file '{id_path}': {e_check}")
                    else:
                        logger.warning(f"  Skipping invalid ID entry: {id_path}")

                logger.debug(f"File check complete. Found {found_count} existing files out of {checked_count} valid IDs checked.")
                indexed_image_paths = verified_paths

            if checked_count > 0 and found_count < checked_count:
                missing_count = checked_count - found_count
                logger.warning(
                    f"Found {missing_count} missing/invalid indexed files during verification (out of {checked_count} checked)."
                )
        elif data is None:
             logger.error("Failed to retrieve data (None returned) from get_all_embeddings_with_ids.")
        else:
             logger.warning("Retrieved data but found no IDs.")

    except DatabaseError as e:
        st.error(f"Error de base de datos al obtener la lista de imágenes indexadas: {e}")
        logger.error(f"DatabaseError getting indexed image list: {e}", exc_info=True)
        return selected_image_path
    except Exception as e:
        st.error(f"Error inesperado al obtener la lista de imágenes indexadas: {e}")
        logger.error(f"Unexpected error getting indexed image list: {e}", exc_info=True)
        return selected_image_path

    if not indexed_image_paths:
        st.warning("No se encontraron imágenes indexadas válidas (archivos podrían haber sido movidos/eliminados o la indexación usó rutas relativas).")
        if selected_state_key in st.session_state:
            st.session_state[selected_state_key] = None
        if trigger_search_key in st.session_state:
            st.session_state[trigger_search_key] = False # Ensure flag is off
        return None
    # --- End of retrieval logic ---

    max_previews = 15
    display_paths = indexed_image_paths[:max_previews]
    st.write(
        f"Selecciona una de las {len(display_paths)} imágenes mostradas (de {len(indexed_image_paths)} encontradas válidas):"
    )

    cols_select = st.columns(5)
    for i, img_path in enumerate(display_paths):
        col = cols_select[i % 5]
        try:
            if not os.path.isfile(img_path):
                 col.caption(f"Archivo movido:\n{os.path.basename(img_path)}")
                 continue

            img_preview = Image.open(img_path)
            col.image(img_preview, caption=os.path.basename(img_path), width=100)
            button_key = f"select_img_{i}{key_suffix}"

            # --- MODIFICATION: Set trigger flag on button click ---
            if col.button("Usar esta", key=button_key):
                st.session_state[selected_state_key] = img_path
                st.session_state[trigger_search_key] = True # Set the flag!
                logger.info(f"User selected indexed image ({key_suffix}), setting trigger flag: {img_path}")
                # Clear any uploaded file state if an indexed one is selected
                uploaded_key = get_state_key(STATE_UPLOADED_IMG_PATH_PREFIX, key_suffix)
                if uploaded_key in st.session_state and st.session_state[uploaded_key] is not None:
                     reset_image_selection_states(key_suffix) # This now also clears the trigger flag, so set it *after* reset if needed, or modify reset
                     # Let's modify reset_image_selection_states to NOT clear the trigger flag itself
                     # Or, more simply, just clear the uploaded path here:
                     temp_path_to_clear = st.session_state.get(uploaded_key)
                     if temp_path_to_clear and os.path.exists(temp_path_to_clear):
                         try:
                             os.unlink(temp_path_to_clear)
                             logger.debug(f"Deleted temp upload file {temp_path_to_clear} after selecting indexed ({key_suffix})")
                         except Exception as e_unlink:
                             logger.warning(f"Could not delete temp file {temp_path_to_clear} after selecting indexed ({key_suffix}): {e_unlink}")
                     st.session_state[uploaded_key] = None # Clear the path state

                # Streamlit reruns automatically on button click.
                # The state and flag are set, the next run will detect the flag.
        except Exception as img_err:
            col.error(f"Error: {os.path.basename(img_path)}")
            logger.warning(f"Error loading indexed image preview {img_path}: {img_err}")

    # Return the currently selected path (might be from a previous run)
    return st.session_state.get(selected_state_key)


def _display_query_image(query_image_path: Optional[str], source_info: str, key_suffix: str = ""):
    """Displays the selected/uploaded query image."""
    if query_image_path and os.path.isfile(query_image_path):
        try:
            with st.container():
                 st.markdown('<div class="selected-query-image">', unsafe_allow_html=True)
                 st.image(
                     Image.open(query_image_path),
                     caption=f"{source_info}: {os.path.basename(query_image_path)}",
                     width=250,
                 )
                 st.markdown("</div>", unsafe_allow_html=True)
            return True
        except Exception as e:
            st.error(
                f"❌ Error al mostrar la imagen de consulta ({os.path.basename(query_image_path)}): {e}"
            )
            logger.error(
                f"Error displaying query image {query_image_path} ({key_suffix}): {e}", exc_info=True
            )
            # Don't reset state here automatically, let the main flow handle it if needed
            # reset_image_selection_states(key_suffix)
            return False
    elif query_image_path:
        # If the path exists in state but the file is gone
        st.error(f"El archivo de la imagen de consulta ya no existe: {query_image_path}")
        # Reset the specific state key that holds the invalid path
        selected_key = get_state_key(STATE_SELECTED_INDEXED_IMG_PREFIX, key_suffix)
        uploaded_key = get_state_key(STATE_UPLOADED_IMG_PATH_PREFIX, key_suffix)
        if st.session_state.get(selected_key) == query_image_path:
            st.session_state[selected_key] = None
        if st.session_state.get(uploaded_key) == query_image_path:
            st.session_state[uploaded_key] = None
        # Also clear the trigger flag if it was associated with this selection
        trigger_key = get_state_key(STATE_TRIGGER_SEARCH_FLAG_PREFIX, key_suffix)
        if trigger_key in st.session_state:
            st.session_state[trigger_key] = False
        return False
    return False


# --- Tab Rendering Functions ---

def render_indexing_tab(vectorizer: Vectorizer, db: VectorDBInterface, truncate_dim: Optional[int]):
    """Renders the content for the 'Indexar Imágenes' tab."""
    st.header("1. Indexar Directorio de Imágenes")
    st.markdown(
        "Selecciona (o introduce la ruta a) una carpeta que contenga las imágenes que deseas indexar. "
        "La aplicación buscará imágenes recursivamente."
    )

    default_image_dir = getattr(config, "DEFAULT_IMAGE_DIR", os.path.abspath("images"))
    image_dir_path = st.text_input(
        "Ruta al directorio de imágenes:",
        value=default_image_dir,
        placeholder="Ej: C:/Users/TuUsuario/Pictures o ./data/images",
        help="Introduce la ruta completa a la carpeta con imágenes.",
        key="index_image_dir_path"
    )

    selected_batch_size = st.number_input(
        "Imágenes a Procesar por Lote (Batch Size):",
        min_value=1,
        max_value=128,
        value=st.session_state.get(STATE_SELECTED_BATCH_SIZE, config.BATCH_SIZE_IMAGES),
        step=4,
        key=STATE_SELECTED_BATCH_SIZE,
        help="Número de imágenes a cargar y vectorizar juntas. Afecta uso de memoria y velocidad."
    )

    if st.button("🔍 Verificar Directorio", key="check_dir"):
        if not image_dir_path:
            st.warning("Por favor, introduce una ruta al directorio.")
        elif not os.path.isdir(image_dir_path):
            st.error(
                f"❌ El directorio no existe o no es accesible: `{image_dir_path}`"
            )
        else:
            st.info(f"Verificando directorio: `{image_dir_path}`...")
            with st.spinner("Buscando imágenes..."):
                try:
                    image_files = find_image_files(image_dir_path)
                except Exception as find_err:
                    st.error(f"Error al buscar archivos: {find_err}")
                    logger.error(f"Error finding files in {image_dir_path}: {find_err}", exc_info=True)
                    image_files = []

            if not image_files:
                st.warning(
                    f"⚠️ No se encontraron archivos de imagen compatibles en `{image_dir_path}`."
                )
            else:
                st.success(f"✅ ¡Encontradas {len(image_files)} imágenes potenciales!")
                with st.expander("Ver ejemplos de imágenes encontradas (máx. 5)"):
                    preview_files = image_files[:5]
                    cols_preview = st.columns(len(preview_files))
                    for i, file_path in enumerate(preview_files):
                        try:
                            if os.path.isfile(file_path):
                                cols_preview[i].image(
                                    Image.open(file_path),
                                    caption=os.path.basename(file_path),
                                    width=100,
                                )
                            else:
                                cols_preview[i].warning(f"Movido:\n{os.path.basename(file_path)}")
                        except Exception as img_err:
                            cols_preview[i].error(
                                f"Error {os.path.basename(file_path)}"
                            )
                            logger.warning(
                                f"Error loading preview image {file_path}: {img_err}"
                            )

    if st.button("⚙️ Procesar e Indexar Directorio", key="process_dir"):
        if not image_dir_path or not os.path.isdir(image_dir_path):
            st.error(
                "❌ Ruta de directorio inválida. Verifica la ruta antes de indexar."
            )
        elif not db or not db.is_initialized:
            st.error("❌ La base de datos no está lista. No se puede indexar.")
        elif not vectorizer:
             st.error("❌ El vectorizador no está listo.")
        else:
            st.info(f"🚀 Iniciando proceso de indexación para: `{image_dir_path}`")
            progress_bar_idx = st.progress(0.0)
            status_text_idx = st.empty()
            start_time_idx = time.time()
            total_processed_successfully = 0

            def update_streamlit_progress(current: int, total: int):
                if total > 0:
                    progress_percent = min(float(current) / float(total), 1.0)
                    try:
                        progress_bar_idx.progress(progress_percent)
                    except Exception: pass # Avoid errors if widget removed
                else:
                     try: progress_bar_idx.progress(0.0)
                     except Exception: pass

            def update_streamlit_status(message: str):
                 try: status_text_idx.info(message)
                 except Exception: pass # Avoid errors if widget removed

            try:
                logger.info(
                    f"Starting indexing via pipeline for directory: {image_dir_path}"
                )
                total_processed_successfully = pipeline.process_directory(
                    directory_path=image_dir_path,
                    vectorizer=vectorizer,
                    db=db,
                    batch_size=selected_batch_size,
                    truncate_dim=truncate_dim,
                    progress_callback=update_streamlit_progress,
                    status_callback=update_streamlit_status,
                )
                end_time_idx = time.time()
                elapsed_time = end_time_idx - start_time_idx
                status_text_idx.success(
                    f"✅ Indexación completada en {elapsed_time:.2f}s. Imágenes guardadas/actualizadas: {total_processed_successfully}."
                )
                logger.info(
                    f"Indexing finished successfully in {elapsed_time:.2f}s. Stored/updated: {total_processed_successfully}"
                )
                # Update sidebar info after indexing
                _display_database_info(db, st.sidebar)

            except PipelineError as e:
                status_text_idx.error(f"❌ Error en el pipeline de indexación: {e}")
                logger.error(
                    f"PipelineError during Streamlit indexing call: {e}", exc_info=True
                )
            except Exception as e:
                status_text_idx.error(
                    f"❌ Error inesperado durante el proceso de indexación: {e}"
                )
                logger.error(
                    f"Unexpected error during Streamlit indexing call: {e}",
                    exc_info=True,
                )

    # --- Advanced Options (Clear/Delete) ---
    with st.expander("⚠️ Opciones Avanzadas / Peligrosas"):
        st.warning(
            "¡Precaución! Estas acciones modifican permanentemente la base de datos."
        )
        col_adv1, col_adv2 = st.columns(2)
        with col_adv1:
             if st.button("🗑️ Limpiar TODA la Colección Actual", key="clear_collection_btn"):
                 st.session_state[STATE_CONFIRM_CLEAR] = True
                 st.session_state[STATE_CONFIRM_DELETE] = False
                 st.rerun() # Rerun to show confirmation
        with col_adv2:
             if st.button(
                 "❌ ELIMINAR TODA la Colección Actual", key="delete_collection_btn"
             ):
                 st.session_state[STATE_CONFIRM_DELETE] = True
                 st.session_state[STATE_CONFIRM_CLEAR] = False
                 st.rerun() # Rerun to show confirmation

        # Confirmation logic for Clear
        if st.session_state.get(STATE_CONFIRM_CLEAR, False):
            st.markdown(
                f"🚨 **¡Atención!** Vas a eliminar **TODOS** los elementos de la colección `{getattr(db, 'collection_name', 'N/A')}`. Esta acción no se puede deshacer."
            )
            confirm_clear_check = st.checkbox(
                "Sí, entiendo y quiero limpiar la colección.", key="confirm_clear_check"
            )
            if confirm_clear_check:
                if st.button("CONFIRMAR LIMPIEZA", key="confirm_clear_final_btn"):
                    if db and db.is_initialized:
                        with st.spinner("Vaciando la colección..."):
                            try:
                                count_before = db.count()
                                success = db.clear_collection()
                                if success:
                                    st.success(
                                        f"✅ Colección limpiada. Elementos eliminados: {count_before}."
                                    )
                                    _display_database_info(db, st.sidebar) # Update sidebar
                                else:
                                    st.error("❌ Falló la operación de limpieza (ver logs).")
                            except DatabaseError as e:
                                st.error(f"❌ Error de base de datos al limpiar: {e}")
                                logger.error(
                                    "DatabaseError clearing collection via Streamlit",
                                    exc_info=True,
                                )
                            except Exception as e:
                                st.error(
                                    f"❌ Error inesperado al limpiar la colección: {e}"
                                )
                                logger.error(
                                    "Error clearing collection via Streamlit",
                                    exc_info=True,
                                )
                    else:
                        st.error("❌ La base de datos no está lista.")
                    st.session_state[STATE_CONFIRM_CLEAR] = False # Reset flag
                    st.rerun() # Rerun to clear confirmation UI
            else:
                st.info("Marca la casilla para habilitar el botón de confirmación.")

        # Confirmation logic for Delete
        if st.session_state.get(STATE_CONFIRM_DELETE, False):
            st.markdown(
                f"🚨 **¡PELIGRO MÁXIMO!** Vas a **ELIMINAR PERMANENTEMENTE** toda la colección `{getattr(db, 'collection_name', 'N/A')}` y sus datos. ¡No hay vuelta atrás!"
            )
            confirm_delete_check = st.checkbox(
                "Sí, entiendo que esto es IRREVERSIBLE y quiero eliminar la colección.",
                key="confirm_delete_check",
            )
            if confirm_delete_check:
                confirm_text = st.text_input(
                    "Escribe 'ELIMINAR' para confirmar:", key="confirm_delete_text"
                )
                if confirm_text == "ELIMINAR":
                    if st.button(
                        "CONFIRMAR ELIMINACIÓN PERMANENTE",
                        key="confirm_delete_final_btn",
                    ):
                        if db:
                            with st.spinner("Eliminando la colección..."):
                                try:
                                    collection_name_deleted = getattr(
                                        db, "collection_name", "N/A"
                                    )
                                    success = db.delete_collection()
                                    if success:
                                        st.success(
                                            f"✅ Colección '{collection_name_deleted}' eliminada permanentemente."
                                        )
                                        # Update sidebar to reflect deletion
                                        st.sidebar.empty() # Clear old info
                                        st.sidebar.header("Base de Datos Vectorial")
                                        st.sidebar.warning(f"Colección '{collection_name_deleted}' eliminada.")
                                        st.sidebar.info("Reinicia la aplicación o indexa datos para crear una nueva colección.")
                                    else:
                                        st.error("❌ Falló la operación de eliminación (ver logs).")
                                except DatabaseError as e:
                                    st.error(f"❌ Error de base de datos al eliminar: {e}")
                                    logger.error("DatabaseError deleting collection via Streamlit", exc_info=True)
                                except Exception as e:
                                    st.error(f"❌ Error inesperado al eliminar la colección: {e}")
                                    logger.error("Error deleting collection via Streamlit", exc_info=True)
                        else:
                            st.error("❌ Objeto de base de datos no disponible.")
                        st.session_state[STATE_CONFIRM_DELETE] = False # Reset flag
                        st.rerun() # Rerun to clear confirmation UI and update sidebar
                else:
                    st.warning("Escribe 'ELIMINAR' en el campo de texto para habilitar el botón.")
            else:
                st.info("Marca la casilla y escribe 'ELIMINAR' para habilitar el botón de confirmación.")


def render_text_search_tab(vectorizer: Vectorizer, db: VectorDBInterface, truncate_dim: Optional[int]):
    """Renders the content for the 'Buscar por Texto' tab."""
    st.header("2. Buscar Imágenes por Descripción Textual")
    st.markdown(
        "Escribe una descripción de la imagen que buscas (ej: 'perro jugando en la playa', 'atardecer sobre montañas')."
    )

    query_text = st.text_input(
        "Descripción de la imagen:",
        placeholder="Ej: gato durmiendo sobre un teclado",
        key="text_query_input",
    )

    num_results_text = st.slider(
        "Número máximo de resultados:",
        min_value=1, max_value=50,
        value=config.DEFAULT_N_RESULTS,
        key="num_results_text_slider",
    )

    results_container_text = st.container() # Define container for results

    if st.button("🔎 Buscar por Texto", key="search_text_button"):
        if not query_text.strip():
            st.warning("⚠️ Por favor, introduce una descripción para buscar.")
        elif not db or not db.is_initialized:
            st.warning("⚠️ La base de datos no está lista. Indexa imágenes primero.")
        elif db.count() <= 0:
             st.warning("⚠️ La base de datos está vacía. Indexa imágenes primero.")
        elif not vectorizer:
             st.error("❌ El vectorizador no está listo.")
        else:
            with st.spinner(f"🧠 Buscando imágenes similares a: '{query_text}'..."):
                try:
                    logger.info(f"Performing text search for: '{query_text}'")
                    results: Optional[SearchResults] = pipeline.search_by_text(
                        query_text=query_text,
                        vectorizer=vectorizer,
                        db=db,
                        n_results=num_results_text,
                        truncate_dim=truncate_dim,
                    )
                    display_results(results, results_container_text) # Display in the container
                    if results:
                        logger.info(
                            f"Text search completed. Found {results.count} results."
                        )

                except PipelineError as e:
                    st.error(f"❌ Error en el pipeline de búsqueda por texto: {e}")
                    logger.error(
                        f"PipelineError during text search via Streamlit: {e}",
                        exc_info=True,
                    )
                except Exception as e:
                    st.error(
                        f"❌ Ocurrió un error inesperado durante la búsqueda por texto: {e}"
                    )
                    logger.error(
                        f"Unexpected error during text search via Streamlit: {e}",
                        exc_info=True,
                    )


def render_image_search_tab(vectorizer: Vectorizer, db: VectorDBInterface, truncate_dim: Optional[int]):
    """Renders the content for the 'Buscar por Imagen' tab."""
    st.header("3. Buscar Imágenes por Similitud Visual")
    st.markdown(
        "Sube una imagen de ejemplo o selecciona una de las imágenes ya indexadas."
    )

    key_suffix = "_img_search"
    uploaded_state_key = get_state_key(STATE_UPLOADED_IMG_PATH_PREFIX, key_suffix)
    selected_state_key = get_state_key(STATE_SELECTED_INDEXED_IMG_PREFIX, key_suffix)
    trigger_search_key = get_state_key(STATE_TRIGGER_SEARCH_FLAG_PREFIX, key_suffix) # New flag key

    # Initialize state keys if they don't exist
    if uploaded_state_key not in st.session_state:
        st.session_state[uploaded_state_key] = None
    if selected_state_key not in st.session_state:
        st.session_state[selected_state_key] = None
    if trigger_search_key not in st.session_state:
        st.session_state[trigger_search_key] = False

    # --- Automatic Search Trigger Check ---
    # Check if the trigger flag was set in the previous run
    search_triggered_automatically = False
    if st.session_state.get(trigger_search_key, False):
        logger.debug(f"Detected trigger flag set for key: {trigger_search_key}")
        st.session_state[trigger_search_key] = False # Reset the flag immediately
        query_image_path_auto = st.session_state.get(selected_state_key)

        if query_image_path_auto and os.path.isfile(query_image_path_auto):
            if not db or not db.is_initialized or db.count() <= 0:
                st.warning("⚠️ La base de datos no está lista o está vacía. No se puede buscar automáticamente.")
            elif not vectorizer:
                st.error("❌ El vectorizador no está listo para la búsqueda automática.")
            else:
                num_results_auto = st.session_state.get(f"num_results_img_slider{key_suffix}", config.DEFAULT_N_RESULTS)
                results_container_auto = st.container() # Container for auto-search results
                with st.spinner("🖼️ Buscando imágenes similares (automático)..."):
                    try:
                        logger.info(f"Performing AUTOMATIC image search using selected indexed image: {query_image_path_auto}")
                        results_auto: Optional[SearchResults] = pipeline.search_by_image(
                            query_image_path=query_image_path_auto,
                            vectorizer=vectorizer,
                            db=db,
                            n_results=num_results_auto,
                            truncate_dim=truncate_dim,
                        )
                        # Display results immediately after automatic search
                        display_results(results_auto, results_container_auto)
                        if results_auto:
                            logger.info(f"Automatic image search completed. Found {results_auto.count} similar images.")
                        search_triggered_automatically = True # Mark that search happened
                    except PipelineError as e:
                        st.error(f"❌ Error en el pipeline de búsqueda automática por imagen: {e}")
                        logger.error(f"PipelineError during automatic image search: {e}", exc_info=True)
                    except Exception as e:
                        st.error(f"❌ Error inesperado durante la búsqueda automática por imagen: {e}")
                        logger.error(f"Unexpected error during automatic image search: {e}", exc_info=True)
        elif query_image_path_auto:
             st.error(f"La imagen seleccionada para búsqueda automática ya no existe: {query_image_path_auto}")
             logger.warning(f"Selected image for automatic search no longer exists: {query_image_path_auto}")
             reset_image_selection_states(key_suffix) # Clean up invalid state
        else:
             logger.warning("Trigger flag was set, but no selected image path found in state.")
    # --- End of Automatic Search Trigger Check ---


    # --- UI Rendering ---
    search_mode = st.radio(
        "Elige el origen de la imagen de consulta:",
        ("Subir una imagen nueva", "Seleccionar una imagen ya indexada"),
        key=f"image_search_mode{key_suffix}",
        horizontal=True,
        # Reset state when changing modes to avoid confusion
        on_change=reset_image_selection_states, args=(key_suffix,)
    )

    num_results_img = st.slider(
        "Número máximo de resultados:",
        min_value=1, max_value=50,
        value=config.DEFAULT_N_RESULTS,
        key=f"num_results_img_slider{key_suffix}",
    )

    query_image_path_to_use: Optional[str] = None
    query_image_source_info: str = ""

    if search_mode == "Subir una imagen nueva":
        # Ensure selected indexed state is clear if switching to upload
        if st.session_state.get(selected_state_key) is not None:
            st.session_state[selected_state_key] = None
            st.session_state[trigger_search_key] = False # Also clear flag
        temp_path = _render_upload_widget(key_suffix=key_suffix)
        if temp_path:
            query_image_path_to_use = temp_path
            query_image_source_info = "Imagen Subida"
    elif search_mode == "Seleccionar una imagen ya indexada":
        # Ensure uploaded state is clear if switching to select
        if st.session_state.get(uploaded_state_key) is not None:
             reset_image_selection_states(key_suffix) # Clears uploaded path and resets flag
        selected_path = _render_indexed_selection(db, key_suffix=key_suffix)
        if selected_path:
            query_image_path_to_use = selected_path
            query_image_source_info = "Imagen Indexada Seleccionada"

    # Display the currently selected/uploaded image
    query_image_displayed = _display_query_image(
        query_image_path_to_use, query_image_source_info, key_suffix=key_suffix
    )

    # Container for manual search results (distinct from auto-search container)
    results_container_manual = st.container()

    # Only show the manual search button for the "Upload" mode
    if search_mode == "Subir una imagen nueva":
        search_button_disabled = not query_image_displayed
        if st.button(
            "🖼️ Buscar Imágenes Similares",
            key=f"search_image_button{key_suffix}",
            disabled=search_button_disabled,
        ):
            if not db or not db.is_initialized:
                 st.warning("⚠️ La base de datos no está lista. Indexa imágenes primero.")
            elif db.count() <= 0:
                 st.warning("⚠️ La base de datos está vacía. Indexa imágenes primero.")
            elif not vectorizer:
                 st.error("❌ El vectorizador no está listo.")
            elif query_image_path_to_use:
                with st.spinner("🖼️ Buscando imágenes similares..."):
                    try:
                        logger.info(
                            f"Performing MANUAL image search using query image: {query_image_path_to_use}"
                        )
                        results: Optional[SearchResults] = pipeline.search_by_image(
                            query_image_path=query_image_path_to_use,
                            vectorizer=vectorizer,
                            db=db,
                            n_results=num_results_img,
                            truncate_dim=truncate_dim,
                        )
                        # Display results in the manual container
                        display_results(results, results_container_manual)
                        if results:
                            logger.info(
                                f"Manual image search completed. Found {results.count} similar images."
                            )

                    except PipelineError as e:
                        st.error(f"❌ Error en el pipeline de búsqueda por imagen: {e}")
                        logger.error(
                            f"PipelineError during manual image search via Streamlit: {e}",
                            exc_info=True,
                        )
                    except Exception as e:
                        st.error(
                            f"❌ Ocurrió un error inesperado durante la búsqueda por imagen: {e}"
                        )
                        logger.error(
                            f"Unexpected error during manual image search via Streamlit: {e}",
                            exc_info=True,
                        )

        elif search_button_disabled:
            st.info("Sube una imagen válida para activar el botón de búsqueda.")
    elif search_mode == "Seleccionar una imagen ya indexada":
        # In this mode, search happens automatically via the trigger flag check above.
        # We might add a message if no search was triggered yet.
        if not search_triggered_automatically and query_image_displayed:
             st.info("La imagen está seleccionada. La búsqueda se realizó automáticamente al hacer clic en 'Usar esta'.")
        elif not query_image_displayed:
             st.info("Selecciona una imagen de la lista de abajo para buscar automáticamente.")


def render_hybrid_search_tab(vectorizer: Vectorizer, db: VectorDBInterface, truncate_dim: Optional[int]):
    """Renders the content for the 'Búsqueda Híbrida' tab (Upload Only)."""
    st.header("4. Búsqueda Híbrida (Texto + Imagen)")
    st.markdown("Combina una descripción textual y una **imagen subida** para encontrar resultados.")

    key_suffix = "_hybrid"
    uploaded_state_key = get_state_key(STATE_UPLOADED_IMG_PATH_PREFIX, key_suffix)

    # Initialize state
    if uploaded_state_key not in st.session_state:
        st.session_state[uploaded_state_key] = None

    col1, col2 = st.columns(2)
    with col1:
        hybrid_query_text = st.text_input(
            "Descripción Textual:",
            placeholder="Ej: persona sonriendo con gafas de sol",
            key=f"hybrid_text_query{key_suffix}"
        )
        hybrid_alpha = st.slider(
            "Peso Texto vs Imagen (0=Solo Imagen, 1=Solo Texto):",
            0.0, 1.0, 0.5,
            key=f"hybrid_alpha{key_suffix}",
            help="Controla la influencia de cada modalidad en la búsqueda."
        )
        num_results_hybrid = st.slider(
            "Número máximo de resultados:", 1, 50, config.DEFAULT_N_RESULTS,
            key=f"num_results_hybrid{key_suffix}"
        )

    with col2:
        st.subheader("Imagen de Ejemplo (Subir):")
        query_image_path_hybrid = None
        query_image_source_hybrid = ""

        # Use the standard upload widget
        temp_path_hybrid = _render_upload_widget(key_suffix=key_suffix)
        if temp_path_hybrid:
            query_image_path_hybrid = temp_path_hybrid
            query_image_source_hybrid = "Imagen Subida"

        # Display the uploaded image
        query_image_displayed_hybrid = _display_query_image(
            query_image_path_hybrid, query_image_source_hybrid, key_suffix=key_suffix
        )

    results_container_hybrid = st.container() # Define container for results

    # Enable search button only if both text and image are provided
    search_button_disabled_hybrid = not (hybrid_query_text.strip() and query_image_displayed_hybrid)
    if st.button("🧬 Buscar Híbrido", key=f"search_hybrid_button{key_suffix}", disabled=search_button_disabled_hybrid):
        if not db or not db.is_initialized:
             st.warning("⚠️ La base de datos no está lista.")
        elif db.count() <= 0:
             st.warning("⚠️ La base de datos está vacía.")
        elif not vectorizer:
             st.error("❌ El vectorizador no está listo.")
        elif query_image_path_hybrid:
             with st.spinner(f"🧬 Realizando búsqueda híbrida (Alfa={hybrid_alpha:.2f})..."):
                  try:
                       logger.info(f"Performing hybrid search: Text='{hybrid_query_text}', Image='{query_image_path_hybrid}', Alpha={hybrid_alpha}")
                       results = pipeline.search_hybrid(
                            query_text=hybrid_query_text,
                            query_image_path=query_image_path_hybrid,
                            vectorizer=vectorizer,
                            db=db,
                            n_results=num_results_hybrid,
                            truncate_dim=truncate_dim,
                            alpha=hybrid_alpha
                       )
                       display_results(results, results_container_hybrid) # Display results
                       if results:
                            logger.info(f"Hybrid search completed. Found {results.count} results.")
                  except PipelineError as e:
                       st.error(f"❌ Error en pipeline de búsqueda híbrida: {e}")
                       logger.error(f"PipelineError during hybrid search via Streamlit: {e}", exc_info=True)
                  except Exception as e:
                       st.error(f"❌ Error inesperado en búsqueda híbrida: {e}")
                       logger.error(f"Unexpected error during hybrid search via Streamlit: {e}", exc_info=True)

    elif search_button_disabled_hybrid:
         st.info("Introduce texto Y sube una imagen válida para activar la búsqueda híbrida.")


# --- Main Application ---
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

    # Model Selection
    try:
        # Use session state to remember selection, fallback to config default
        default_model_index = config.AVAILABLE_MODELS.index(st.session_state.get(STATE_SELECTED_MODEL, config.DEFAULT_MODEL_NAME))
    except ValueError:
        default_model_index = 0 # Fallback if saved model is no longer available

    selected_model = st.sidebar.selectbox(
        "Modelo de Embedding:",
        options=config.AVAILABLE_MODELS,
        index=default_model_index,
        key=STATE_SELECTED_MODEL, # Store selection in session state
        help="Elige el modelo para generar los vectores. Cambiarlo recargará el modelo."
    )

    # Dimension Selection
    selected_dimension = st.sidebar.number_input(
        "Dimensión Embedding (0 = Completa):",
        min_value=0,
        value=st.session_state.get(STATE_SELECTED_DIMENSION, config.VECTOR_DIMENSION or 0), # Use session state
        step=32,
        key=STATE_SELECTED_DIMENSION, # Store selection in session state
        help="Número de dimensiones del vector final. 0 usa la dimensión nativa del modelo."
    )
    truncate_dim_value = selected_dimension if selected_dimension > 0 else None

    # --- Initialize Components based on selected model ---
    # The key for caching is the selected model name from session state
    active_model_name = st.session_state.get(STATE_SELECTED_MODEL, config.DEFAULT_MODEL_NAME)
    vectorizer, db = cached_get_components(active_model_name)

    # --- Display System/DB Info ---
    st.sidebar.header("Información del Sistema")
    st.sidebar.caption(f"Python: {sys.version_info.major}.{sys.version_info.minor}")
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        # Get actual device from the initialized vectorizer if available
        actual_device = vectorizer.device if vectorizer else 'N/A'
        st.sidebar.caption(f"Dispositivo Usado: `{actual_device}`")
        st.sidebar.caption(f"CUDA Disponible: {'✅ Sí' if cuda_available else '❌ No'}")
        if cuda_available:
            try:
                 st.sidebar.caption(f"GPU: {torch.cuda.get_device_name(0)}")
            except Exception as gpu_name_err:
                 st.sidebar.caption(f"GPU: Error al obtener nombre ({gpu_name_err})")

    except ImportError:
        st.sidebar.warning("Librería 'torch' no encontrada.")
    except Exception as e:
        st.sidebar.error(f"Error al verificar CUDA/Dispositivo: {e}")

    st.sidebar.header("Configuración Activa")
    st.sidebar.caption(f"Modelo Cargado: `{active_model_name}`")
    st.sidebar.caption(f"Dimensión Objetivo: `{truncate_dim_value or 'Completa'}`")

    # Display DB info (uses the potentially initialized db object)
    if db:
         _display_database_info(db, st.sidebar)
    else:
         # This case implies initialization failed severely
         st.sidebar.error("Base de datos no inicializada.")


    # --- Stop if initialization failed ---
    if not vectorizer or not db:
        st.error(
            f"La aplicación no puede continuar. Falló la inicialización de componentes para el modelo '{active_model_name}'."
        )
        st.stop() # Stop script execution

    # --- Tabs ---
    tab_titles = [
        "💾 Indexar Imágenes",
        "📝 Buscar por Texto",
        "🖼️ Buscar por Imagen",
        "🧬 Búsqueda Híbrida"
    ]
    tab1, tab2, tab3, tab4 = st.tabs(tab_titles)

    # Render content for each tab, passing the initialized components
    with tab1:
        render_indexing_tab(vectorizer, db, truncate_dim_value)
    with tab2:
        render_text_search_tab(vectorizer, db, truncate_dim_value)
    with tab3:
        render_image_search_tab(vectorizer, db, truncate_dim_value)
    with tab4:
         render_hybrid_search_tab(vectorizer, db, truncate_dim_value)


if __name__ == "__main__":
    main()
