import os
import sys
import time
import streamlit as st
from PIL import Image, UnidentifiedImageError
import tempfile
from typing import Optional, Tuple, Dict, Any, List, Callable
import logging

# --- Configuración Inicial y Rutas ---
# Asegurarse de que el directorio raíz del proyecto esté en sys.path
try:
    # Use __file__ to get the directory of the current script
    project_root = os.path.dirname(os.path.abspath(__file__))
    # Construct the path to the parent directory (vector_image_project)
    # Assuming streamlit_app.py is directly inside vector_image_project
    # If it's in a subdirectory like 'app', adjust the path accordingly.
    # Example if streamlit_app.py is in 'vector_image_project/app/':
    # project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Now import project modules
    import config
    from core.vectorizer import Vectorizer
    from data_access.vector_db import VectorDatabase
    from app import pipeline  # Importar el módulo pipeline completo
    from core.image_processor import (
        find_image_files,
        load_image,
    )
except ImportError as e:
    print(f"Error crítico al importar módulos del proyecto: {e}")
    st.error(f"Error crítico al importar módulos del proyecto: {e}")
    st.stop()
except Exception as e:
    print(f"Error inesperado durante la configuración inicial: {e}")
    st.error(f"Error inesperado durante la configuración inicial: {e}")
    st.stop()


# --- Configuración de Logging para Streamlit ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# --- Configuración de Página Streamlit ---
st.set_page_config(
    page_title="Buscador de Imágenes Vectorial", page_icon="🔍", layout="wide"
)

# --- Estilos CSS Personalizados ---
st.markdown(
    """
<style>
    /* Estilos para mejorar la apariencia */
    .results-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 15px; padding-top: 10px; }
    .result-item { border: 1px solid #ddd; padding: 10px; border-radius: 8px; text-align: center; background-color: #f9f9f9; transition: box-shadow 0.3s ease; }
    .result-item:hover { box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
    .result-item img { max-width: 100%; height: auto; border-radius: 4px; margin-bottom: 5px; object-fit: contain; max-height: 150px; } /* Limitar altura imagen */
    .result-item .caption { font-size: 0.8em; color: #555; word-wrap: break-word; } /* Estilo para caption */
    .stProgress > div > div > div > div { background-color: #19A7CE; } /* Color barra progreso */
    .stSpinner > div { text-align: center; } /* Centrar texto spinner */
    .block-container { padding-top: 2rem; padding-bottom: 2rem; } /* Más padding general */
    h1, h2, h3 { color: #003366; } /* Color títulos */
</style>
""",
    unsafe_allow_html=True,
)


# --- Funciones de Utilidad y Componentes de la UI ---

@st.cache_resource(show_spinner=False)
def initialize_components() -> Tuple[Optional[Vectorizer], Optional[VectorDatabase]]:
    """
    Inicializa y cachea el Vectorizer y la VectorDatabase.
    Muestra el progreso durante la inicialización.
    """
    init_message = st.empty()
    init_message.info(
        "⏳ Inicializando componentes (modelo y base de datos)... Esto puede tardar la primera vez."
    )
    col1, col2 = st.columns([1, 3])
    progress_bar = col1.progress(0)
    status_text = col2.empty()
    vectorizer = None
    db = None

    try:
        status_text.text("Cargando modelo de vectorización...")
        logger.info("Loading vectorizer model...")
        vectorizer = Vectorizer(
            model_name=config.MODEL_NAME,
            device=config.DEVICE,
            trust_remote_code=config.TRUST_REMOTE_CODE,
        )
        progress_bar.progress(50)
        status_text.text("Conectando a la base de datos vectorial...")
        logger.info("Connecting to vector database...")
        db = VectorDatabase(
            path=config.CHROMA_DB_PATH, collection_name=config.CHROMA_COLLECTION_NAME
        )
        progress_bar.progress(100)
        status_text.success("✅ Componentes listos.")
        logger.info("Components initialized successfully.")
        time.sleep(1.5)
        status_text.empty()
        progress_bar.empty()
        init_message.empty()
        return vectorizer, db
    except Exception as e:
        logger.error(f"Fatal error during component initialization: {e}", exc_info=True)
        status_text.error(f"❌ Error fatal al inicializar: {e}")
        st.error(
            "No se pudieron inicializar los componentes necesarios. Verifica la configuración (config.py), "
            "las dependencias (requirements.txt), los permisos de acceso a la ruta de la BD "
            f"('{config.CHROMA_DB_PATH}') y la conexión a internet (para descargar el modelo la primera vez)."
        )
        progress_bar.empty()
        init_message.empty()
        return None, None


def display_results(results: Optional[Dict[str, Any]], results_container: st.container):
    """
    Muestra los resultados de la búsqueda en un formato de cuadrícula.
    """
    if not results or not results.get("ids"):
        results_container.warning("🤷‍♂️ No se encontraron resultados para tu búsqueda.")
        return

    ids = results.get("ids", [])
    distances = results.get("distances", [])
    metadatas = (
        results.get("metadatas")
        if results.get("metadatas") is not None
        else [{}] * len(ids)
    )
    if len(metadatas) < len(ids):
        metadatas.extend([{}] * (len(ids) - len(metadatas)))

    results_container.success(f"🎉 ¡Encontrados {len(ids)} resultados!")
    num_columns = 5
    cols = results_container.columns(num_columns)

    for i, img_id in enumerate(ids):
        col = cols[i % num_columns]
        with col, st.container():
            st.markdown('<div class="result-item">', unsafe_allow_html=True)
            try:
                if os.path.isfile(img_id):
                    img = Image.open(img_id)
                    st.image(img, use_container_width=True)
                    similarity = (
                        1 - distances[i]
                        if distances and i < len(distances) and distances[i] is not None
                        else 0
                    )
                    st.markdown(
                        f'<div class="caption">{os.path.basename(img_id)}<br>Similitud: {similarity:.3f}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.warning(f"Archivo no encontrado:\n{os.path.basename(img_id)}")
                    st.caption(f"Ruta original: {img_id}")
                    logger.warning(f"Result image file not found at path: {img_id}")
            except FileNotFoundError:
                st.error(f"Error crítico: Archivo no encontrado en la ruta: {img_id}")
                logger.error(f"File not found error for image path: {img_id}")
            except UnidentifiedImageError:
                st.error(
                    f"Error: No se pudo abrir o identificar la imagen: {os.path.basename(img_id)}"
                )
                logger.warning(f"UnidentifiedImageError for: {img_id}")
            except Exception as e:
                st.error(
                    f"Error al mostrar '{os.path.basename(img_id)}': {str(e)[:100]}..."
                )
                logger.error(f"Error displaying image {img_id}: {e}", exc_info=True)
            st.markdown("</div>", unsafe_allow_html=True)


# Callback para resetear estados de selección de imagen
def reset_image_selection_states():
    """Resets session state variables related to image selection."""
    st.session_state.selected_indexed_image_path = None
    # Limpiar archivo temporal si existe
    if st.session_state.get("uploaded_image_temp_path") and os.path.exists(
        st.session_state.uploaded_image_temp_path
    ):
        try:
            os.unlink(st.session_state.uploaded_image_temp_path)
            logger.debug(
                f"Deleted temp file on mode change: {st.session_state.uploaded_image_temp_path}"
            )
        except Exception as e_unlink:
            logger.warning(f"Could not delete temp file on mode change: {e_unlink}")
    st.session_state.uploaded_image_temp_path = None


# --- Funciones Refactorizadas para cada Pestaña ---

def render_indexing_tab(vectorizer: Vectorizer, db: VectorDatabase):
    """Renders the content for the 'Index Images' tab."""
    st.header("1. Indexar Directorio de Imágenes")
    st.markdown(
        "Selecciona (o introduce la ruta a) una carpeta que contenga las imágenes que deseas indexar. "
        "La aplicación buscará imágenes recursivamente en esa carpeta."
    )

    default_image_dir = os.path.abspath("images")
    image_dir_path = st.text_input(
        "Ruta al directorio de imágenes:",
        value=default_image_dir,
        placeholder="Ej: C:/Users/TuUsuario/Pictures o ./data/images",
        help="Introduce la ruta completa a la carpeta con imágenes.",
    )

    if st.button("🔍 Verificar Directorio", key="check_dir"):
        if not image_dir_path:
            st.warning("Por favor, introduce una ruta al directorio.")
        elif not os.path.isdir(image_dir_path):
            st.error(
                f"❌ El directorio no existe o no es accesible en la ruta: `{image_dir_path}`"
            )
        else:
            st.info(f"Verificando directorio: `{image_dir_path}`...")
            image_files = find_image_files(image_dir_path)
            if not image_files:
                st.warning(
                    f"⚠️ No se encontraron archivos de imagen compatibles en `{image_dir_path}` o sus subdirectorios."
                )
            else:
                st.success(f"✅ ¡Encontradas {len(image_files)} imágenes potenciales!")
                with st.expander("Ver ejemplos de imágenes encontradas (máximo 5)"):
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
                            cols_preview[i].error(
                                f"Error al cargar {os.path.basename(file_path)}"
                            )
                            logger.warning(
                                f"Error loading preview image {file_path}: {img_err}"
                            )

    if st.button("⚙️ Procesar e Indexar Directorio", key="process_dir"):
        if not image_dir_path or not os.path.isdir(image_dir_path):
            st.error(
                "❌ Ruta de directorio inválida o no especificada. Verifica la ruta antes de indexar."
            )
        else:
            st.info(f"🚀 Iniciando proceso de indexación para: `{image_dir_path}`")
            progress_bar_idx = st.progress(0)
            status_text_idx = st.empty()
            start_time_idx = time.time()
            total_processed_successfully = 0

            def update_streamlit_progress(current: int, total: int):
                if total > 0:
                    progress_percent = min(float(current) / float(total), 1.0)
                    progress_bar_idx.progress(progress_percent)
                else:
                    progress_bar_idx.progress(0.0)

            def update_streamlit_status(message: str):
                status_text_idx.text(message)

            try:
                logger.info(
                    f"Starting indexing via pipeline for directory: {image_dir_path}"
                )
                total_processed_successfully = pipeline.process_directory(
                    directory_path=image_dir_path,
                    vectorizer=vectorizer,
                    db=db,
                    batch_size=config.BATCH_SIZE_IMAGES,
                    truncate_dim=config.VECTOR_DIMENSION,
                    progress_callback=update_streamlit_progress,
                    status_callback=update_streamlit_status,
                )
                end_time_idx = time.time()
                elapsed_time = end_time_idx - start_time_idx
                st.success(
                    f"✅ Indexación completada en {elapsed_time:.2f} segundos. Total de imágenes guardadas/actualizadas: {total_processed_successfully}."
                )
                logger.info(
                    f"Indexing finished successfully in {elapsed_time:.2f}s. Stored/updated: {total_processed_successfully}"
                )
                # Actualizar sidebar
                if db and db.collection:
                    num_items = db.collection.count()
                    st.sidebar.info(
                        f"**BD Actual:** `{config.CHROMA_COLLECTION_NAME}`\n\n"
                        f"**Ruta:** `{os.path.abspath(config.CHROMA_DB_PATH)}`\n\n"
                        f"**Imágenes Indexadas:** `{num_items}`"
                    )

            except Exception as e:
                status_text_idx.error(f"❌ Error durante el proceso de indexación: {e}")
                logger.error(f"Error during Streamlit indexing call: {e}", exc_info=True)
            finally:
                time.sleep(3)
                progress_bar_idx.empty()
                status_text_idx.empty()
                st.rerun()

    # --- Opciones Avanzadas (Limpiar Colección) ---
    with st.expander("⚠️ Opciones Avanzadas / Peligrosas"):
        st.warning(
            "¡Precaución! Estas acciones modifican permanentemente la base de datos."
        )

        if st.button("🗑️ Limpiar TODA la Colección Actual", key="clear_collection"):
            st.markdown(
                f"🚨 **¡Atención!** Esta acción eliminará **TODOS** los elementos de la colección `{config.CHROMA_COLLECTION_NAME}` en `{config.CHROMA_DB_PATH}`. Esta acción no se puede deshacer."
            )
            st.session_state.confirm_clear_triggered = True

        if st.session_state.get("confirm_clear_triggered", False):
            confirm_check = st.checkbox(
                "Sí, entiendo las consecuencias y quiero limpiar la colección.",
                key="confirm_clear_check",
            )
            if confirm_check:
                if st.button(
                    "CONFIRMAR LIMPIEZA AHORA", key="confirm_clear_button_final"
                ):
                    try:
                        with st.spinner("Vaciando la colección..."):
                            logger.warning(
                                f"User initiated CLEAR ALL for collection: {db.collection_name}"
                            )
                            count_before = db.collection.count() if db.collection else 0
                            success = db.clear_collection()
                            count_after = db.collection.count() if db.collection else 0
                            if success:
                                logger.info(
                                    f"Collection cleared. Items before: {count_before}, Items after: {count_after}"
                                )
                                st.success(
                                    f"✅ Colección `{config.CHROMA_COLLECTION_NAME}` limpiada. Elementos eliminados: {count_before}."
                                )
                                # Actualizar sidebar
                                st.sidebar.info(
                                    f"**BD Actual:** `{config.CHROMA_COLLECTION_NAME}`\n\n"
                                    f"**Ruta:** `{os.path.abspath(config.CHROMA_DB_PATH)}`\n\n"
                                    f"**Imágenes Indexadas:** `{count_after}`"
                                )
                            else:
                                st.error("❌ Falló la operación de limpieza de la colección.")
                                logger.error("Failed to clear collection.")

                    except Exception as e:
                        st.error(f"❌ Error al intentar limpiar la colección: {e}")
                        logger.error(
                            f"Error clearing collection via Streamlit: {e}",
                            exc_info=True,
                        )
                    finally:
                        st.session_state.confirm_clear_triggered = False
                        st.rerun()
            else:
                st.info(
                    "Marca la casilla de confirmación para habilitar el botón de limpieza."
                )


def render_text_search_tab(vectorizer: Vectorizer, db: VectorDatabase):
    """Renders the content for the 'Search by Text' tab."""
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
        "Número máximo de resultados a mostrar:",
        min_value=1,
        max_value=50,
        value=10,
        key="num_results_text_slider",
    )

    if st.button("🔎 Buscar por Texto", key="search_text_button"):
        if not query_text.strip():
            st.warning("⚠️ Por favor, introduce una descripción para buscar.")
        elif db.collection is None or db.collection.count() == 0:
            st.warning("⚠️ La base de datos está vacía. Indexa algunas imágenes primero.")
        else:
            results_container_text = st.container()
            with st.spinner(f"🧠 Buscando imágenes similares a: '{query_text}'..."):
                try:
                    logger.info(f"Performing text search for: '{query_text}'")
                    results_text = pipeline.search_by_text(
                        query_text=query_text,
                        vectorizer=vectorizer,
                        db=db,
                        n_results=num_results_text,
                        truncate_dim=config.VECTOR_DIMENSION,
                    )
                    display_results(results_text, results_container_text)
                    logger.info(
                        f"Text search completed. Found {len(results_text.get('ids', [])) if results_text else 0} results."
                    )
                except Exception as e:
                    st.error(f"❌ Ocurrió un error durante la búsqueda por texto: {e}")
                    logger.error(
                        f"Error during text search via Streamlit: {e}", exc_info=True
                    )


def render_image_search_tab(vectorizer: Vectorizer, db: VectorDatabase):
    """Renders the content for the 'Search by Image' tab."""
    st.header("3. Buscar Imágenes por Similitud con Otra Imagen")
    st.markdown(
        "Puedes subir una imagen de ejemplo o seleccionar una de las imágenes que ya has indexado."
    )

    # Initialize session state if not present
    if "selected_indexed_image_path" not in st.session_state:
        st.session_state.selected_indexed_image_path = None
    if "uploaded_image_temp_path" not in st.session_state:
        st.session_state.uploaded_image_temp_path = None

    search_mode = st.radio(
        "Elige el origen de la imagen de consulta:",
        ("Subir una imagen nueva", "Seleccionar una imagen ya indexada"),
        key="image_search_mode",
        horizontal=True,
        on_change=reset_image_selection_states,
    )

    num_results_img = st.slider(
        "Número máximo de resultados:",
        min_value=1,
        max_value=50,
        value=10,
        key="num_results_img_slider",
    )

    query_image_path_to_use: Optional[str] = None

    # --- Mode: Upload Image ---
    if search_mode == "Subir una imagen nueva":
        uploaded_file = st.file_uploader(
            "Arrastra o selecciona una imagen de tu dispositivo:",
            type=[ext.lstrip(".") for ext in config.IMAGE_EXTENSIONS],
            key="image_uploader",
            on_change=reset_image_selection_states, # Reset if a new file is uploaded
        )

        if uploaded_file is not None:
            # Use the existing temp path if it's still valid for this upload instance
            # Otherwise, create a new one
            temp_file_path = st.session_state.get("uploaded_image_temp_path")
            create_new_temp = True
            if temp_file_path and os.path.exists(temp_file_path):
                 # Simple check: if the file name matches, assume it's the same upload instance
                 if os.path.basename(temp_file_path).endswith(os.path.splitext(uploaded_file.name)[1]):
                      create_new_temp = False # Reuse existing temp file for this instance
                 else: # New file name, delete old temp file first
                      reset_image_selection_states()

            if create_new_temp:
                try:
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=os.path.splitext(uploaded_file.name)[1]
                    ) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        st.session_state.uploaded_image_temp_path = tmp_file.name
                        temp_file_path = tmp_file.name # Update path
                        logger.debug(f"Created new temp file: {temp_file_path}")

                except Exception as e:
                    st.error(f"❌ Error al crear archivo temporal: {e}")
                    logger.error(f"Error creating temp file: {e}", exc_info=True)
                    reset_image_selection_states() # Clean up on error
                    temp_file_path = None

            # Display if temp file path is valid
            if temp_file_path and os.path.exists(temp_file_path):
                 try:
                    img_uploaded = Image.open(temp_file_path)
                    st.image(
                        img_uploaded,
                        caption=f"Imagen subida: {uploaded_file.name}",
                        width=250,
                    )
                    query_image_path_to_use = temp_file_path
                 except Exception as e:
                    st.error(f"❌ Error al procesar la imagen subida: {e}")
                    logger.error(f"Error processing uploaded image: {e}", exc_info=True)
                    reset_image_selection_states() # Clean up on error
                    query_image_path_to_use = None
            else:
                 # Handle case where temp file creation failed or file disappeared
                 if temp_file_path: # Only log if path existed but file doesn't
                     logger.error(f"Temp file path exists but file not found: {temp_file_path}")
                 query_image_path_to_use = None


    # --- Mode: Select Indexed Image ---
    elif search_mode == "Seleccionar una imagen ya indexada":
        if db.collection is None or db.collection.count() == 0:
            st.warning("⚠️ No hay imágenes indexadas en la base de datos para seleccionar.")
        else:
            st.info(
                "Mostrando algunas imágenes indexadas. Haz clic en 'Usar esta' para seleccionarla como consulta."
            )
            indexed_image_paths: List[str] = []
            try:
                limit_get = 100
                # Use offset for potential pagination in the future if needed
                # offset = st.session_state.get("indexed_offset", 0)
                result = db.collection.get(
                    limit=limit_get,
                    # offset=offset, # Add offset if implementing pagination
                    include=[]
                )
                if result and "ids" in result:
                    # Filter for existing files only AFTER getting results
                    all_ids = result["ids"]
                    indexed_image_paths = [
                        id_path for id_path in all_ids if os.path.isfile(id_path)
                    ]
                    # Log if some files were missing
                    if len(indexed_image_paths) < len(all_ids):
                        logger.warning(f"Found {len(all_ids) - len(indexed_image_paths)} missing indexed files during preview.")

                else:
                    logger.warning("Could not retrieve IDs from the database collection.")

            except Exception as e:
                st.error(f"Error al obtener la lista de imágenes indexadas: {e}")
                logger.error(f"Error getting indexed image list: {e}", exc_info=True)

            if indexed_image_paths:
                max_previews = 15
                display_paths = indexed_image_paths[:max_previews]
                st.write(
                    f"Selecciona una de las {len(display_paths)} imágenes mostradas (de {len(indexed_image_paths)} encontradas válidas):"
                )

                cols_select = st.columns(5)
                for i, img_path in enumerate(display_paths):
                    col = cols_select[i % 5]
                    try:
                        img_preview = Image.open(img_path)
                        col.image(
                            img_preview, caption=os.path.basename(img_path), width=100
                        )
                        button_key = f"select_img_{i}"
                        if col.button("Usar esta", key=button_key):
                            st.session_state.selected_indexed_image_path = img_path
                            logger.info(f"User selected indexed image: {img_path}")
                            st.rerun()
                    except Exception as img_err:
                        col.caption(f"Error: {os.path.basename(img_path)}")
                        logger.warning(
                            f"Error loading indexed image preview {img_path}: {img_err}"
                        )

                # Display selected image
                selected_path_from_state = st.session_state.get("selected_indexed_image_path")
                if selected_path_from_state:
                    if os.path.isfile(selected_path_from_state):
                        st.success("Imagen de consulta seleccionada:")
                        try:
                            st.image(Image.open(selected_path_from_state), width=250)
                            query_image_path_to_use = selected_path_from_state
                        except Exception as e:
                            st.error(f"Error al mostrar la imagen seleccionada: {e}")
                            logger.error(
                                f"Error displaying selected indexed image {selected_path_from_state}: {e}"
                            )
                            st.session_state.selected_indexed_image_path = None
                            query_image_path_to_use = None
                    else:
                        st.error(
                            f"El archivo de la imagen seleccionada ya no existe: {selected_path_from_state}"
                        )
                        logger.warning(
                            f"Selected indexed image file no longer exists: {selected_path_from_state}"
                        )
                        st.session_state.selected_indexed_image_path = None
                        query_image_path_to_use = None
            else:
                 st.warning(
                    "No se encontraron imágenes indexadas válidas para seleccionar (archivos podrían haber sido movidos o eliminados)."
                )

    # --- Search Button (Common to both modes) ---
    search_button_disabled = query_image_path_to_use is None
    if st.button(
        "🖼️ Buscar Imágenes Similares",
        key="search_image_button",
        disabled=search_button_disabled,
    ):
        if db.collection is None or db.collection.count() == 0:
            st.warning("⚠️ La base de datos está vacía. Indexa algunas imágenes primero.")
        elif query_image_path_to_use: # Double check path is set
            results_container_img = st.container()
            with st.spinner("🖼️ Buscando imágenes similares..."):
                try:
                    logger.info(
                        f"Performing image search using query image: {query_image_path_to_use}"
                    )
                    results_img = pipeline.search_by_image(
                        query_image_path=query_image_path_to_use,
                        vectorizer=vectorizer,
                        db=db,
                        n_results=num_results_img,
                        truncate_dim=config.VECTOR_DIMENSION,
                    )
                    display_results(results_img, results_container_img)
                    logger.info(
                        f"Image search completed. Found {len(results_img.get('ids', [])) if results_img else 0} similar images."
                    )
                except Exception as e:
                    st.error(f"❌ Ocurrió un error durante la búsqueda por imagen: {e}")
                    logger.error(
                        f"Error during image search via Streamlit: {e}", exc_info=True
                    )
                finally:
                    # Clean up temp file AFTER search if it was used (check state again)
                    temp_path_in_state = st.session_state.get("uploaded_image_temp_path")
                    # Check if the path used for search was indeed the temp path from state
                    if temp_path_in_state and query_image_path_to_use == temp_path_in_state:
                        if os.path.exists(temp_path_in_state):
                            try:
                                os.unlink(temp_path_in_state)
                                st.session_state.uploaded_image_temp_path = None
                                logger.debug(f"Deleted temporary uploaded file after search: {temp_path_in_state}")
                            except Exception as e_unlink:
                                logger.warning(
                                    f"Could not delete temporary uploaded file {temp_path_in_state} after search: {e_unlink}"
                                )
                        else:
                             # If file doesn't exist but path is in state, just clear state
                             st.session_state.uploaded_image_temp_path = None
                             logger.debug(f"Cleared temp file path from state (file already gone): {temp_path_in_state}")


    elif search_button_disabled:
        if search_mode == "Subir una imagen nueva":
            st.info("Sube una imagen válida para activar el botón de búsqueda.")
        elif search_mode == "Seleccionar una imagen ya indexada":
            st.info(
                "Selecciona una de las imágenes indexadas mostradas arriba para activar el botón de búsqueda."
            )


# --- Lógica Principal de la Aplicación ---
def main():
    """Main function to run the Streamlit application."""
    st.title("🔍 Buscador de Imágenes por Similitud Vectorial")
    st.markdown(
        """
    Bienvenido/a. Esta aplicación te permite indexar imágenes y luego buscarlas
    basándose en su contenido visual, usando descripciones de texto o imágenes de ejemplo.
    Utiliza modelos de embeddings (como CLIP) y una base de datos vectorial (ChromaDB).
    """
    )

    # Initialize components (model and DB)
    vectorizer, db = initialize_components()

    if not vectorizer or not db:
        st.error(
            "La aplicación no puede continuar sin los componentes esenciales inicializados."
        )
        st.stop()

    # Display DB info in sidebar
    try:
        if db and db.collection:
            num_items = db.collection.count()
            st.sidebar.info(
                f"**BD Actual:** `{config.CHROMA_COLLECTION_NAME}`\n\n"
                f"**Ruta:** `{os.path.abspath(config.CHROMA_DB_PATH)}`\n\n"
                f"**Imágenes Indexadas:** `{num_items}`"
            )
        else:
            st.sidebar.warning(
                "La colección de la base de datos no parece estar disponible."
            )
            logger.warning("Database collection is not available after initialization.")
    except Exception as e:
        st.sidebar.error(f"Error al obtener info de la BD: {e}")
        logger.error(f"Error getting DB info for sidebar: {e}", exc_info=True)

    # Create tabs
    tab1, tab2, tab3 = st.tabs(
        ["💾 Indexar Imágenes", "📝 Buscar por Texto", "🖼️ Buscar por Imagen"]
    )

    # Render content for each tab using refactored functions
    with tab1:
        render_indexing_tab(vectorizer, db)

    with tab2:
        render_text_search_tab(vectorizer, db)

    with tab3:
        render_image_search_tab(vectorizer, db)


# --- Punto de Entrada ---
if __name__ == "__main__":
    # --- Verificaciones Previas e Info en Sidebar ---
    st.sidebar.header("Información del Sistema")
    st.sidebar.caption(
        f"Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        st.sidebar.caption(f"CUDA Disponible: {'✅ Sí' if cuda_available else '❌ No'}")
        if cuda_available:
            st.sidebar.caption(f"GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        st.sidebar.warning("Librería 'torch' no encontrada.")
    except Exception as e:
        st.sidebar.error(f"Error al verificar CUDA: {e}")

    st.sidebar.header("Configuración Clave")
    st.sidebar.caption(f"Modelo: `{config.MODEL_NAME}`")
    st.sidebar.caption(f"Dispositivo: `{config.DEVICE}`")
    st.sidebar.caption(
        f"Dimensión Vector: `{config.VECTOR_DIMENSION or 'Completa (Auto)'}`"
    )
    st.sidebar.caption(f"Ruta BD: `{os.path.abspath(config.CHROMA_DB_PATH)}`")
    st.sidebar.caption(f"Colección BD: `{config.CHROMA_COLLECTION_NAME}`")

    # Run the main Streamlit app logic
    main()
