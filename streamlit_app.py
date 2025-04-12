# streamlit_app.py
import os
import sys
import time
import streamlit as st
from PIL import Image, UnidentifiedImageError
import tempfile
from typing import Optional, Tuple, Dict, Any, List, Callable
import logging

# --- Initial Setup and Path Configuration ---
try:
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Import project modules AFTER path setup
    import config
    from app.factory import get_initialized_components  # Use factory
    from core.vectorizer import Vectorizer  # Still need type hint
    from data_access.vector_db_interface import VectorDBInterface  # Use interface
    from app import pipeline
    from app.models import SearchResults, SearchResultItem  # Use data models
    from core.image_processor import find_image_files  # Keep specific imports needed

except ImportError as e:
    # Critical error during import - display and stop
    error_msg = f"Error crítico al importar módulos: {e}. Asegúrate de que todos los archivos estén presentes y las dependencias instaladas (`pip install -r requirements.txt`)."
    print(error_msg)  # Log to console as well
    st.error(error_msg)
    st.stop()  # Stop execution if imports fail
except Exception as e:
    error_msg = f"Error inesperado durante la configuración inicial: {e}"
    print(error_msg)
    st.error(error_msg)
    st.stop()

# --- Logging Configuration ---
# Basic logging setup for Streamlit app
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Buscador de Imágenes Vectorial",
    page_icon="🔍",
    layout="wide"
)

# --- Custom CSS Styles ---
# (Keep existing styles or enhance them)
st.markdown("""
<style>
    .results-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 15px; padding-top: 10px; }
    .result-item { border: 1px solid #ddd; padding: 10px; border-radius: 8px; text-align: center; background-color: #f9f9f9; transition: box-shadow 0.3s ease; display: flex; flex-direction: column; justify-content: space-between; height: 100%; }
    .result-item:hover { box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
    .result-item img { max-width: 100%; height: 150px; /* Fixed height */ border-radius: 4px; margin-bottom: 5px; object-fit: contain; /* Ensure image fits */ }
    .result-item .caption { font-size: 0.8em; color: #555; word-wrap: break-word; margin-top: auto; /* Push caption to bottom */ }
    .stProgress > div > div > div > div { background-color: #19A7CE; }
    .stSpinner > div { text-align: center; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    h1, h2, h3 { color: #003366; }
    /* Style for selected image preview */
    .selected-query-image img { border: 3px solid #19A7CE; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# --- Session State Keys ---
# Define constants for session state keys to avoid typos
STATE_UPLOADED_IMG_PATH = "uploaded_image_temp_path"
STATE_SELECTED_INDEXED_IMG = "selected_indexed_image_path"
STATE_CONFIRM_CLEAR = "confirm_clear_triggered"
STATE_CONFIRM_DELETE = "confirm_delete_triggered"  # Added for delete confirmation


# --- UI Utility Functions ---

@st.cache_resource(show_spinner=False)  # Cache the initialized components
def cached_get_components() -> Tuple[Optional[Vectorizer], Optional[VectorDBInterface]]:
    """
    Initializes and caches the Vectorizer and VectorDatabase using the factory.
    Shows progress during the first initialization.
    Returns (None, None) if initialization fails.
    """
    # Use an empty container for messages/progress bars that can be cleared
    init_container = st.empty()
    with init_container.container():
        st.info("⏳ Inicializando componentes (modelo y BD)... Esto puede tardar la primera vez.")
        # Use columns for better layout of progress bar and text
        pcol1, pcol2 = st.columns([1, 4])
        progress_bar = pcol1.progress(0)
        status_text = pcol2.empty()

    vectorizer = None
    db = None
    initialization_successful = False

    try:
        status_text.text("Cargando modelo de vectorización...")
        # Use factory to get components
        vectorizer, db = get_initialized_components()
        # Simulate progress
        progress_bar.progress(50)
        status_text.text("Verificando conexión a base de datos...")
        # Check if DB is usable (e.g., by getting count)
        if db and db.is_initialized:
            db_count = db.count()  # Simple check
            logger.info(f"Database connection verified. Item count: {db_count}")
            progress_bar.progress(100)
            status_text.success("✅ Componentes listos.")
            initialization_successful = True
        else:
            status_text.error("❌ No se pudo inicializar/conectar a la base de datos.")
            logger.error("Database component failed initialization check.")

    except RuntimeError as e:
        logger.critical(f"Fatal error during component initialization: {e}", exc_info=True)
        status_text.error(f"❌ Error fatal al inicializar: {e}")
        st.error(
            "No se pudieron inicializar los componentes necesarios. Verifica la configuración (config.py), "
            "dependencias (requirements.txt), permisos de acceso a la ruta de la BD "
            f"('{config.CHROMA_DB_PATH}'), y conexión a internet (para descargar el modelo)."
        )
    except Exception as e:
        logger.critical(f"Unexpected fatal error during component initialization: {e}", exc_info=True)
        status_text.error(f"❌ Error inesperado al inicializar: {e}")

    finally:
        # Clear the initialization messages/progress bar after a short delay
        if initialization_successful:
            time.sleep(1.5)  # Give user time to see success message
        init_container.empty()  # Remove the container

    if initialization_successful:
        return vectorizer, db
    else:
        return None, None  # Return None if failed


def display_results(results: Optional[SearchResults], results_container: st.container):
    """
    Displays search results (SearchResults object) in a grid format.

    Args:
        results: A SearchResults object or None.
        results_container: The Streamlit container to display results in.
    """
    if results is None:
        results_container.error("❌ La búsqueda falló o no devolvió resultados.")
        return
    if results.is_empty:
        results_container.warning("🤷‍♂️ No se encontraron resultados para tu búsqueda.")
        return

    results_container.success(f"🎉 ¡Encontrados {results.count} resultados!")

    # Define number of columns for the grid
    num_columns = 5  # Adjust as needed
    cols = results_container.columns(num_columns)

    for i, item in enumerate(results.items):
        col = cols[i % num_columns]
        with col:  # Use context manager for better structure within column
            # Use a container within the column for the result item div
            with st.container():
                st.markdown('<div class="result-item">', unsafe_allow_html=True)
                try:
                    if os.path.isfile(item.id):
                        img = Image.open(item.id)
                        st.image(img, use_container_width=True)
                        # Use the similarity property from SearchResultItem
                        similarity_str = f"{item.similarity:.3f}" if item.similarity is not None else "N/A"
                        # Display filename and similarity in the caption div
                        st.markdown(
                            f'<div class="caption">{os.path.basename(item.id)}<br>Similitud: {similarity_str}</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        # Handle case where the image file is missing
                        st.warning(f"Archivo no encontrado:\n{os.path.basename(item.id)}")
                        st.caption(f"Ruta original: {item.id}")
                        logger.warning(f"Result image file not found at path: {item.id}")

                except FileNotFoundError:
                    # Should be caught by isfile, but handle defensively
                    st.error(f"Error crítico: Archivo no encontrado en la ruta: {item.id}")
                    logger.error(f"FileNotFoundError for image path: {item.id}")
                except UnidentifiedImageError:
                    st.error(f"Error: No se pudo abrir/identificar imagen: {os.path.basename(item.id)}")
                    logger.warning(f"UnidentifiedImageError for: {item.id}")
                except Exception as e:
                    # Catch-all for other display errors
                    st.error(f"Error al mostrar '{os.path.basename(item.id)}': {str(e)[:100]}...")
                    logger.error(f"Error displaying image {item.id}: {e}", exc_info=True)

                st.markdown("</div>", unsafe_allow_html=True)  # Close result-item div


def reset_image_selection_states():
    """Resets session state variables related to image selection and cleans up temp files."""
    st.session_state[STATE_SELECTED_INDEXED_IMG] = None
    temp_path = st.session_state.get(STATE_UPLOADED_IMG_PATH)
    if temp_path and os.path.exists(temp_path):
        try:
            os.unlink(temp_path)
            logger.debug(f"Deleted temp file on state reset: {temp_path}")
        except Exception as e_unlink:
            logger.warning(f"Could not delete temp file {temp_path} on state reset: {e_unlink}")
    st.session_state[STATE_UPLOADED_IMG_PATH] = None


def _display_database_info(db: VectorDBInterface, container: st.sidebar):
    """Displays current database information in the sidebar."""
    container.header("Base de Datos Vectorial")
    if db and db.is_initialized:
        db_count = db.count()
        # Assuming ChromaDB implementation for path/name, might need adjustment if interface changes
        db_path = getattr(db, 'path', 'N/A')
        collection_name = getattr(db, 'collection_name', 'N/A')
        container.info(
            f"**Colección:** `{collection_name}`\n\n"
            f"**Ruta:** `{os.path.abspath(db_path) if db_path != 'N/A' else 'N/A'}`\n\n"
            f"**Imágenes Indexadas:** `{db_count if db_count >= 0 else 'Error'}`"
        )
    else:
        container.warning("La base de datos no está inicializada o no es accesible.")


# --- Tab Rendering Functions ---

def render_indexing_tab(vectorizer: Vectorizer, db: VectorDBInterface):
    """Renders the content for the 'Index Images' tab."""
    st.header("1. Indexar Directorio de Imágenes")
    st.markdown(
        "Selecciona (o introduce la ruta a) una carpeta que contenga las imágenes que deseas indexar. "
        "La aplicación buscará imágenes recursivamente."
    )

    # Use default from config if available, otherwise a sensible default
    default_image_dir = getattr(config, 'DEFAULT_IMAGE_DIR', os.path.abspath("images"))
    image_dir_path = st.text_input(
        "Ruta al directorio de imágenes:",
        value=default_image_dir,
        placeholder="Ej: C:/Users/TuUsuario/Pictures o ./data/images",
        help="Introduce la ruta completa a la carpeta con imágenes.",
    )

    # --- Directory Verification ---
    if st.button("🔍 Verificar Directorio", key="check_dir"):
        if not image_dir_path:
            st.warning("Por favor, introduce una ruta al directorio.")
        elif not os.path.isdir(image_dir_path):
            st.error(f"❌ El directorio no existe o no es accesible: `{image_dir_path}`")
        else:
            st.info(f"Verificando directorio: `{image_dir_path}`...")
            with st.spinner("Buscando imágenes..."):
                image_files = find_image_files(image_dir_path)
            if not image_files:
                st.warning(f"⚠️ No se encontraron archivos de imagen compatibles en `{image_dir_path}`.")
            else:
                st.success(f"✅ ¡Encontradas {len(image_files)} imágenes potenciales!")
                # Show preview of first few images found
                with st.expander("Ver ejemplos de imágenes encontradas (máx. 5)"):
                    preview_files = image_files[:5]
                    cols_preview = st.columns(len(preview_files))
                    for i, file_path in enumerate(preview_files):
                        try:
                            cols_preview[i].image(
                                Image.open(file_path),
                                caption=os.path.basename(file_path),
                                width=100,
                            )
                        except Exception as img_err:
                            cols_preview[i].error(f"Error {os.path.basename(file_path)}")
                            logger.warning(f"Error loading preview image {file_path}: {img_err}")

    # --- Indexing Button and Logic ---
    if st.button("⚙️ Procesar e Indexar Directorio", key="process_dir"):
        if not image_dir_path or not os.path.isdir(image_dir_path):
            st.error("❌ Ruta de directorio inválida. Verifica la ruta antes de indexar.")
        elif not db or not db.is_initialized:
            st.error("❌ La base de datos no está lista. No se puede indexar.")
        else:
            st.info(f"🚀 Iniciando proceso de indexación para: `{image_dir_path}`")
            # Use placeholders for progress bar and status text
            progress_bar_idx = st.progress(0.0)  # Start at 0.0 (float)
            status_text_idx = st.empty()
            start_time_idx = time.time()
            total_processed_successfully = 0

            # Define callbacks for pipeline progress
            def update_streamlit_progress(current: int, total: int):
                if total > 0:
                    progress_percent = min(float(current) / float(total), 1.0)
                    progress_bar_idx.progress(progress_percent)
                else:
                    progress_bar_idx.progress(0.0)

            def update_streamlit_status(message: str):
                status_text_idx.info(message)  # Use info or other levels as needed

            try:
                logger.info(f"Starting indexing via pipeline for directory: {image_dir_path}")
                # Call pipeline function with callbacks
                total_processed_successfully = pipeline.process_directory(
                    directory_path=image_dir_path,
                    vectorizer=vectorizer,
                    db=db,
                    batch_size=config.BATCH_SIZE_IMAGES,
                    truncate_dim=config.VECTOR_DIMENSION,  # Use configured dimension
                    progress_callback=update_streamlit_progress,
                    status_callback=update_streamlit_status,
                )
                end_time_idx = time.time()
                elapsed_time = end_time_idx - start_time_idx
                st.success(
                    f"✅ Indexación completada en {elapsed_time:.2f}s. Imágenes guardadas/actualizadas: {total_processed_successfully}.")
                logger.info(
                    f"Indexing finished successfully in {elapsed_time:.2f}s. Stored/updated: {total_processed_successfully}")
                # Update sidebar info after indexing
                _display_database_info(db, st.sidebar)

            except Exception as e:
                status_text_idx.error(f"❌ Error durante el proceso de indexación: {e}")
                logger.error(f"Error during Streamlit indexing call: {e}", exc_info=True)
            finally:
                # Clear progress/status after completion or error
                time.sleep(2)  # Allow user to see final message
                progress_bar_idx.empty()
                status_text_idx.empty()
                # Consider st.rerun() if you want the page state to fully refresh

    # --- Advanced / Dangerous Options ---
    with st.expander("⚠️ Opciones Avanzadas / Peligrosas"):
        st.warning("¡Precaución! Estas acciones modifican permanentemente la base de datos.")

        # Clear Collection Button
        if st.button("🗑️ Limpiar TODA la Colección Actual", key="clear_collection_btn"):
            st.session_state[STATE_CONFIRM_CLEAR] = True
            # Reset delete confirmation if clear is clicked
            st.session_state[STATE_CONFIRM_DELETE] = False
            st.rerun()  # Rerun to show confirmation controls

        # Delete Collection Button
        if st.button("❌ ELIMINAR TODA la Colección Actual", key="delete_collection_btn"):
            st.session_state[STATE_CONFIRM_DELETE] = True
            # Reset clear confirmation if delete is clicked
            st.session_state[STATE_CONFIRM_CLEAR] = False
            st.rerun()  # Rerun to show confirmation controls

        # Confirmation logic for Clear
        if st.session_state.get(STATE_CONFIRM_CLEAR, False):
            st.markdown(
                f"🚨 **¡Atención!** Vas a eliminar **TODOS** los elementos de la colección `{getattr(db, 'collection_name', 'N/A')}`. Esta acción no se puede deshacer.")
            confirm_clear_check = st.checkbox("Sí, entiendo y quiero limpiar la colección.", key="confirm_clear_check")
            if confirm_clear_check:
                if st.button("CONFIRMAR LIMPIEZA", key="confirm_clear_final_btn"):
                    if db and db.is_initialized:
                        with st.spinner("Vaciando la colección..."):
                            try:
                                count_before = db.count()
                                success = db.clear_collection()
                                if success:
                                    st.success(f"✅ Colección limpiada. Elementos eliminados: {count_before}.")
                                    _display_database_info(db, st.sidebar)  # Update sidebar
                                else:
                                    st.error("❌ Falló la operación de limpieza.")
                            except Exception as e:
                                st.error(f"❌ Error al limpiar la colección: {e}")
                                logger.error("Error clearing collection via Streamlit", exc_info=True)
                    else:
                        st.error("❌ La base de datos no está lista.")
                    # Reset confirmation state and rerun
                    st.session_state[STATE_CONFIRM_CLEAR] = False
                    st.rerun()
            else:
                st.info("Marca la casilla para habilitar el botón de confirmación.")

        # Confirmation logic for Delete
        if st.session_state.get(STATE_CONFIRM_DELETE, False):
            st.markdown(
                f"🚨 **¡PELIGRO MÁXIMO!** Vas a **ELIMINAR PERMANENTEMENTE** toda la colección `{getattr(db, 'collection_name', 'N/A')}` y sus datos. ¡No hay vuelta atrás!")
            confirm_delete_check = st.checkbox("Sí, entiendo que esto es IRREVERSIBLE y quiero eliminar la colección.",
                                               key="confirm_delete_check")
            if confirm_delete_check:
                confirm_text = st.text_input("Escribe 'ELIMINAR' para confirmar:", key="confirm_delete_text")
                if confirm_text == "ELIMINAR":
                    if st.button("CONFIRMAR ELIMINACIÓN PERMANENTE", key="confirm_delete_final_btn"):
                        if db:  # Need db object even if not initialized to call delete
                            with st.spinner("Eliminando la colección..."):
                                try:
                                    collection_name_deleted = getattr(db, 'collection_name', 'N/A')
                                    success = db.delete_collection()
                                    if success:
                                        st.success(
                                            f"✅ Colección '{collection_name_deleted}' eliminada permanentemente.")
                                        # DB object is likely invalid now, display message
                                        st.sidebar.warning(
                                            "La colección ha sido eliminada. Recarga la página o reinicia la app para crear una nueva.")
                                        # Optionally, try to re-initialize or stop the app here
                                    else:
                                        st.error("❌ Falló la operación de eliminación.")
                                except Exception as e:
                                    st.error(f"❌ Error al eliminar la colección: {e}")
                                    logger.error("Error deleting collection via Streamlit", exc_info=True)
                        else:
                            st.error("❌ Objeto de base de datos no disponible.")
                        # Reset confirmation state and rerun
                        st.session_state[STATE_CONFIRM_DELETE] = False
                        st.rerun()
                else:
                    st.warning("Escribe 'ELIMINAR' en el campo de texto para habilitar el botón.")
            else:
                st.info("Marca la casilla y escribe 'ELIMINAR' para habilitar el botón de confirmación.")


def render_text_search_tab(vectorizer: Vectorizer, db: VectorDBInterface):
    """Renders the content for the 'Search by Text' tab."""
    st.header("2. Buscar Imágenes por Descripción Textual")
    st.markdown(
        "Escribe una descripción de la imagen que buscas (ej: 'perro jugando en la playa', 'atardecer sobre montañas').")

    query_text = st.text_input(
        "Descripción de la imagen:",
        placeholder="Ej: gato durmiendo sobre un teclado",
        key="text_query_input",
    )

    num_results_text = st.slider(
        "Número máximo de resultados:",
        min_value=1, max_value=50, value=config.DEFAULT_N_RESULTS, key="num_results_text_slider",
    )

    if st.button("🔎 Buscar por Texto", key="search_text_button"):
        if not query_text.strip():
            st.warning("⚠️ Por favor, introduce una descripción para buscar.")
        elif not db or not db.is_initialized or db.count() == 0:
            st.warning("⚠️ La base de datos está vacía o no está lista. Indexa imágenes primero.")
        else:
            results_container_text = st.container()  # Container for results
            with st.spinner(f"🧠 Buscando imágenes similares a: '{query_text}'..."):
                try:
                    logger.info(f"Performing text search for: '{query_text}'")
                    # Call pipeline, expect SearchResults or None
                    results: Optional[SearchResults] = pipeline.search_by_text(
                        query_text=query_text,
                        vectorizer=vectorizer,
                        db=db,
                        n_results=num_results_text,
                        truncate_dim=config.VECTOR_DIMENSION,
                    )
                    # Display results using the utility function
                    display_results(results, results_container_text)
                    if results:
                        logger.info(f"Text search completed. Found {results.count} results.")

                except Exception as e:
                    st.error(f"❌ Ocurrió un error durante la búsqueda por texto: {e}")
                    logger.error(f"Error during text search via Streamlit: {e}", exc_info=True)


# --- Helper functions for Image Search Tab ---
def _render_upload_widget() -> Optional[str]:
    """Renders the file uploader and manages the temporary file. Returns temp file path if successful."""
    uploaded_file = st.file_uploader(
        "Arrastra o selecciona una imagen:",
        type=[ext.lstrip(".") for ext in config.IMAGE_EXTENSIONS],
        key="image_uploader",
        # Use on_change to clear previous state if a new file is uploaded/removed
        on_change=reset_image_selection_states,
    )

    if uploaded_file is None:
        # Ensure cleanup if file is removed via the 'x'
        reset_image_selection_states()
        return None

    # Reuse existing temp file path from session state if it belongs to this upload instance
    temp_file_path = st.session_state.get(STATE_UPLOADED_IMG_PATH)
    create_new_temp = True

    if temp_file_path and os.path.exists(temp_file_path):
        # Basic check: if filename part matches, assume it's the same upload instance
        # This isn't foolproof but avoids recreating temp file on every rerun for the same upload
        if os.path.basename(temp_file_path).endswith(os.path.splitext(uploaded_file.name)[1]):
            create_new_temp = False
        else:
            # New file uploaded, reset state (which includes deleting old temp file)
            reset_image_selection_states()
            temp_file_path = None  # Clear path as old file is deleted

    if create_new_temp:
        try:
            # Create a new temporary file
            with tempfile.NamedTemporaryFile(
                    delete=False, suffix=os.path.splitext(uploaded_file.name)[1]
            ) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                st.session_state[STATE_UPLOADED_IMG_PATH] = tmp_file.name
                temp_file_path = tmp_file.name  # Update path for current run
                logger.debug(f"Created new temp file for upload: {temp_file_path}")
        except Exception as e:
            st.error(f"❌ Error al guardar archivo temporal: {e}")
            logger.error(f"Error creating temp file: {e}", exc_info=True)
            reset_image_selection_states()  # Clean up on error
            return None

    # Return the path if it's valid and exists
    if temp_file_path and os.path.exists(temp_file_path):
        return temp_file_path
    else:
        logger.error(f"Temporary file path invalid or file missing: {temp_file_path}")
        reset_image_selection_states()  # Clean up state if file is missing
        return None


def _render_indexed_selection(db: VectorDBInterface) -> Optional[str]:
    """Renders the selection grid for indexed images. Returns the selected path."""
    if not db or not db.is_initialized or db.count() == 0:
        st.warning("⚠️ No hay imágenes indexadas en la base de datos para seleccionar.")
        return None

    st.info("Mostrando algunas imágenes indexadas. Haz clic en 'Usar esta' para seleccionarla.")
    indexed_image_paths: List[str] = []
    selected_image_path: Optional[str] = st.session_state.get(STATE_SELECTED_INDEXED_IMG)

    try:
        limit_get = 25  # Show more examples
        # Get IDs from DB
        result = db.collection.get(limit=limit_get, include=[])  # Only need IDs
        if result and "ids" in result:
            all_ids = result["ids"]
            # Filter for existing files only AFTER getting results
            # This is crucial as files might be deleted externally
            with st.spinner("Verificando archivos indexados..."):
                indexed_image_paths = [id_path for id_path in all_ids if os.path.isfile(id_path)]
            if len(indexed_image_paths) < len(all_ids):
                logger.warning(f"Found {len(all_ids) - len(indexed_image_paths)} missing indexed files during preview.")
        else:
            logger.warning("Could not retrieve IDs from the database collection.")

    except Exception as e:
        st.error(f"Error al obtener la lista de imágenes indexadas: {e}")
        logger.error(f"Error getting indexed image list: {e}", exc_info=True)
        return selected_image_path  # Return current selection on error

    if not indexed_image_paths:
        st.warning("No se encontraron imágenes indexadas válidas (archivos podrían haber sido movidos/eliminados).")
        return selected_image_path  # Return current selection

    # Display selection grid
    max_previews = 15  # Limit display grid size
    display_paths = indexed_image_paths[:max_previews]
    st.write(
        f"Selecciona una de las {len(display_paths)} imágenes mostradas (de {len(indexed_image_paths)} encontradas válidas):")

    cols_select = st.columns(5)  # Grid columns
    for i, img_path in enumerate(display_paths):
        col = cols_select[i % 5]
        try:
            # Display image preview
            img_preview = Image.open(img_path)
            col.image(img_preview, caption=os.path.basename(img_path), width=100)
            # Button to select this image
            button_key = f"select_img_{i}"
            if col.button("Usar esta", key=button_key):
                st.session_state[STATE_SELECTED_INDEXED_IMG] = img_path
                logger.info(f"User selected indexed image: {img_path}")
                # Clear uploaded file state if an indexed image is selected
                reset_image_selection_states()
                st.rerun()  # Rerun to update the displayed selection
        except Exception as img_err:
            col.caption(f"Error: {os.path.basename(img_path)}")
            logger.warning(f"Error loading indexed image preview {img_path}: {img_err}")

    return st.session_state.get(STATE_SELECTED_INDEXED_IMG)  # Return the potentially updated selection


def _display_query_image(query_image_path: Optional[str], source_info: str):
    """Displays the selected/uploaded query image."""
    if query_image_path and os.path.isfile(query_image_path):
        try:
            # Use a subheader and custom class for styling
            st.subheader("Imagen de Consulta:")
            st.markdown('<div class="selected-query-image">', unsafe_allow_html=True)
            st.image(Image.open(query_image_path), caption=f"{source_info}: {os.path.basename(query_image_path)}",
                     width=250)
            st.markdown('</div>', unsafe_allow_html=True)
            return True  # Indicate success
        except Exception as e:
            st.error(f"❌ Error al mostrar la imagen de consulta ({os.path.basename(query_image_path)}): {e}")
            logger.error(f"Error displaying query image {query_image_path}: {e}", exc_info=True)
            return False  # Indicate failure
    elif query_image_path:  # Path exists but file doesn't
        st.error(f"El archivo de la imagen de consulta ya no existe: {query_image_path}")
        return False
    return False  # No path provided


def render_image_search_tab(vectorizer: Vectorizer, db: VectorDBInterface):
    """Renders the content for the 'Search by Image' tab."""
    st.header("3. Buscar Imágenes por Similitud Visual")
    st.markdown("Sube una imagen de ejemplo o selecciona una de las imágenes ya indexadas.")

    # Initialize session state if needed (safer)
    if STATE_UPLOADED_IMG_PATH not in st.session_state:
        st.session_state[STATE_UPLOADED_IMG_PATH] = None
    if STATE_SELECTED_INDEXED_IMG not in st.session_state:
        st.session_state[STATE_SELECTED_INDEXED_IMG] = None

    search_mode = st.radio(
        "Elige el origen de la imagen de consulta:",
        ("Subir una imagen nueva", "Seleccionar una imagen ya indexada"),
        key="image_search_mode",
        horizontal=True,
        # Reset selection state when mode changes
        on_change=reset_image_selection_states,
    )

    num_results_img = st.slider(
        "Número máximo de resultados:",
        min_value=1, max_value=50, value=config.DEFAULT_N_RESULTS, key="num_results_img_slider",
    )

    query_image_path_to_use: Optional[str] = None
    query_image_source_info: str = ""

    # --- Handle Image Source ---
    if search_mode == "Subir una imagen nueva":
        temp_path = _render_upload_widget()
        if temp_path:
            query_image_path_to_use = temp_path
            query_image_source_info = "Imagen Subida"

    elif search_mode == "Seleccionar una imagen ya indexada":
        selected_path = _render_indexed_selection(db)
        if selected_path:
            query_image_path_to_use = selected_path
            query_image_source_info = "Imagen Indexada Seleccionada"

    # --- Display Query Image Preview ---
    query_image_displayed = _display_query_image(query_image_path_to_use, query_image_source_info)

    # --- Search Button ---
    search_button_disabled = not query_image_displayed  # Disable if no valid query image is shown
    if st.button("🖼️ Buscar Imágenes Similares", key="search_image_button", disabled=search_button_disabled):
        if not db or not db.is_initialized or db.count() == 0:
            st.warning("⚠️ La base de datos está vacía o no está lista. Indexa imágenes primero.")
        elif query_image_path_to_use:  # Should be true if button is enabled, but double-check
            results_container_img = st.container()
            with st.spinner("🖼️ Buscando imágenes similares..."):
                try:
                    logger.info(f"Performing image search using query image: {query_image_path_to_use}")
                    # Call pipeline, expect SearchResults or None
                    results: Optional[SearchResults] = pipeline.search_by_image(
                        query_image_path=query_image_path_to_use,
                        vectorizer=vectorizer,
                        db=db,
                        n_results=num_results_img,
                        truncate_dim=config.VECTOR_DIMENSION,
                    )
                    # Display results
                    display_results(results, results_container_img)
                    if results:
                        logger.info(f"Image search completed. Found {results.count} similar images.")

                except Exception as e:
                    st.error(f"❌ Ocurrió un error durante la búsqueda por imagen: {e}")
                    logger.error(f"Error during image search via Streamlit: {e}", exc_info=True)
                finally:
                    # Clean up temp file AFTER search only if it was the one used
                    # Check if the path used was the one from the upload state
                    temp_path_in_state = st.session_state.get(STATE_UPLOADED_IMG_PATH)
                    if temp_path_in_state and query_image_path_to_use == temp_path_in_state:
                        reset_image_selection_states()  # This will handle unlinking the file


    elif search_button_disabled:
        # Provide guidance if button is disabled
        if search_mode == "Subir una imagen nueva":
            st.info("Sube una imagen válida para activar el botón de búsqueda.")
        elif search_mode == "Seleccionar una imagen ya indexada":
            st.info("Selecciona una de las imágenes indexadas mostradas arriba para activar el botón.")


# --- Main Application Logic ---
def main():
    """Main function to run the Streamlit application."""
    st.title("🔍 Buscador de Imágenes por Similitud Vectorial")
    st.markdown(
        """
    Bienvenido/a. Indexa imágenes y búscalas por contenido visual
    usando descripciones de texto o imágenes de ejemplo.
    """
    )

    # Initialize components using cached factory function
    vectorizer, db = cached_get_components()

    # Stop execution if components failed to initialize
    if not vectorizer or not db:
        st.error("La aplicación no puede continuar sin los componentes esenciales inicializados.")
        st.stop()

    # --- Sidebar Information ---
    st.sidebar.header("Información del Sistema")
    st.sidebar.caption(f"Python: {sys.version_info.major}.{sys.version_info.minor}")
    # Display CUDA info if torch is available
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        st.sidebar.caption(f"CUDA Disponible: {'✅ Sí' if cuda_available else '❌ No'}")
        if cuda_available: st.sidebar.caption(f"GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        st.sidebar.warning("Librería 'torch' no encontrada.")
    except Exception as e:
        st.sidebar.error(f"Error al verificar CUDA: {e}")

    st.sidebar.header("Configuración Clave")
    st.sidebar.caption(f"Modelo: `{config.MODEL_NAME}`")
    st.sidebar.caption(f"Dispositivo: `{config.DEVICE}`")
    st.sidebar.caption(f"Dimensión Vector: `{config.VECTOR_DIMENSION or 'Completa'}`")

    # Display Database Info in Sidebar
    _display_database_info(db, st.sidebar)

    # --- Main Area Tabs ---
    tab1, tab2, tab3 = st.tabs([
        "💾 Indexar Imágenes",
        "📝 Buscar por Texto",
        "🖼️ Buscar por Imagen"
    ])

    with tab1:
        render_indexing_tab(vectorizer, db)

    with tab2:
        render_text_search_tab(vectorizer, db)

    with tab3:
        render_image_search_tab(vectorizer, db)


# --- Application Entry Point ---
if __name__ == "__main__":
    main()
