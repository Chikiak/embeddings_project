import os
import sys
import time
import streamlit as st
from PIL import Image, UnidentifiedImageError
import tempfile
import glob
from typing import Optional, Tuple, Dict, Any, List

# --- Configuración Inicial y Rutas ---
try:
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    import config
    from core.vectorizer import Vectorizer
    from data_access.vector_db import VectorDatabase
    from app import pipeline
    from core.image_processor import find_image_files, load_image
except ImportError as e:
    st.error(f"Error crítico al importar módulos del proyecto: {e}")
    st.stop()
except Exception as e:
    st.error(f"Error inesperado durante la configuración inicial: {e}")
    st.stop()

# --- Configuración de Página Streamlit ---
st.set_page_config(
    page_title="Buscador de Imágenes Vectorial", page_icon="🔍", layout="wide"
)

# --- Estilos CSS Personalizados (Opcional) ---
st.markdown(
    """
<style>
    /* ... (Estilos CSS como antes) ... */
    .results-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 15px; padding-top: 10px; }
    .result-item { border: 1px solid #ddd; padding: 10px; border-radius: 8px; text-align: center; background-color: #f9f9f9; }
    .result-item img { max-width: 100%; height: auto; border-radius: 4px; margin-bottom: 5px; }
    .stProgress > div > div > div > div { background-color: #19A7CE; }
    .stSpinner > div { text-align: center; }
</style>
""",
    unsafe_allow_html=True,
)

# --- Funciones de Utilidad y Componentes de la UI ---


@st.cache_resource
def initialize_components() -> Tuple[Optional[Vectorizer], Optional[VectorDatabase]]:
    # ... (Igual que antes) ...
    st.write(
        "⏳ Inicializando componentes (modelo y base de datos)... Esto puede tardar la primera vez."
    )
    progress_bar = st.progress(0)
    status_text = st.empty()
    try:
        status_text.text("Cargando modelo de vectorización...")
        vectorizer = Vectorizer(
            model_name=config.MODEL_NAME,
            device=config.DEVICE,
            trust_remote_code=config.TRUST_REMOTE_CODE,
        )
        progress_bar.progress(50)
        status_text.text("Conectando a la base de datos vectorial...")
        db = VectorDatabase(
            path=config.CHROMA_DB_PATH, collection_name=config.CHROMA_COLLECTION_NAME
        )
        progress_bar.progress(100)
        status_text.success("✅ Componentes listos.")
        time.sleep(1)
        status_text.empty()
        progress_bar.empty()
        return vectorizer, db
    except Exception as e:
        status_text.error(f"❌ Error fatal al inicializar: {e}")
        st.error(
            "Verifica la configuración (config.py), las dependencias (requirements.txt) y los permisos."
        )
        progress_bar.empty()
        return None, None


def display_results(results: Optional[Dict[str, Any]], results_container: st.container):
    # ... (Igual que antes, usando use_container_width) ...
    if not results or not results.get("ids"):
        results_container.warning("🤷‍♂️ No se encontraron resultados para tu búsqueda.")
        return
    ids = results.get("ids", [])
    distances = results.get("distances", [])
    metadatas = results.get("metadatas", [])
    results_container.success(f"🎉 ¡Encontrados {len(ids)} resultados!")
    cols = results_container.columns(5)
    for i, img_id in enumerate(ids):
        col = cols[i % 5]
        with col.container():
            st.markdown(f'<div class="result-item">', unsafe_allow_html=True)
            try:
                if os.path.isfile(img_id):
                    img = Image.open(img_id)
                    col.image(
                        img,
                        caption=f"{os.path.basename(img_id)}",
                        use_container_width=True,
                    )
                    similarity = (
                        1 - distances[i]
                        if distances and i < len(distances) and distances[i] is not None
                        else 0
                    )
                    col.write(f"Similitud: {similarity:.3f}")
                else:
                    col.warning(f"Archivo no encontrado:\n{os.path.basename(img_id)}")
                    col.caption(f"Ruta original: {img_id}")
            except FileNotFoundError:
                col.error(f"Error: Archivo no encontrado en la ruta: {img_id}")
            except UnidentifiedImageError:
                col.error(
                    f"Error: No se pudo abrir o identificar la imagen: {os.path.basename(img_id)}"
                )
            except Exception as e:
                col.error(
                    f"Error al mostrar '{os.path.basename(img_id)}': {str(e)[:100]}..."
                )
            st.markdown("</div>", unsafe_allow_html=True)


# --- Lógica Principal de la Aplicación ---
def main():
    st.title("🔍 Buscador de Imágenes por Similitud Vectorial")
    st.markdown(
        """
    Bienvenido/a. Esta aplicación te permite indexar imágenes y luego buscarlas
    basándose en su contenido visual, usando descripciones de texto o imágenes de ejemplo.
    """
    )

    vectorizer, db = initialize_components()
    if not vectorizer or not db:
        st.error("La aplicación no puede continuar sin los componentes inicializados.")
        st.stop()

    try:
        if db and db.collection:
            num_items = db.collection.count()
            st.sidebar.info(
                f"**BD Actual:** `{config.CHROMA_COLLECTION_NAME}`\n\n**Imágenes Indexadas:** `{num_items}`"
            )
        else:
            st.sidebar.warning(
                "La colección de la base de datos no parece estar disponible."
            )
    except Exception as e:
        st.sidebar.error(f"Error al obtener info de la BD: {e}")

    tab1, tab2, tab3 = st.tabs(
        ["💾 Indexar Imágenes", "📝 Buscar por Texto", "🖼️ Buscar por Imagen"]
    )

    # === Pestaña 1: Indexar Imágenes ===
    with tab1:
        # ... (Código de indexación igual que antes) ...
        st.header("1. Indexar Directorio de Imágenes")
        st.markdown("Selecciona una carpeta...")
        default_image_dir = "images"
        image_dir_path = st.text_input(
            "Ruta al directorio de imágenes:",
            value=default_image_dir,
            placeholder="Ej: C:/Users/TuUsuario/Pictures o ./data/images",
            help="Escribe la ruta completa.",
        )
        col1_idx, col2_idx = st.columns([1, 2])
        with col1_idx:
            if st.button("🔍 Verificar Directorio", key="check_dir"):
                if not image_dir_path:
                    st.warning("Introduce una ruta.")
                elif not os.path.isdir(image_dir_path):
                    st.error(f"❌ Directorio no existe.")
                else:
                    st.info(f"Verificando en: `{image_dir_path}`...")
                    image_files = find_image_files(image_dir_path)
                    if not image_files:
                        st.warning(f"⚠️ No se encontraron imágenes.")
                    else:
                        st.success(f"✅ ¡Encontradas {len(image_files)} imágenes!")
                        with st.expander("Ver ejemplos (máx. 5)"):
                            preview_files = image_files[:5]
                            cols_preview = st.columns(len(preview_files))
                            for i, file_path in enumerate(preview_files):
                                try:
                                    cols_preview[i].image(
                                        Image.open(file_path),
                                        caption=os.path.basename(file_path),
                                        width=100,
                                    )
                                except Exception:
                                    cols_preview[i].error(f"Error")
        with col2_idx:
            if st.button("⚙️ Procesar e Indexar Directorio", key="process_dir"):
                if not image_dir_path or not os.path.isdir(image_dir_path):
                    st.error("❌ Ruta inválida.")
                else:
                    st.info(f"🚀 Iniciando indexación para: `{image_dir_path}`")
                    progress_bar_idx = st.progress(0)
                    status_text_idx = st.empty()
                    status_text_idx.text("Buscando archivos...")
                    processed_count = 0
                    total_processed_successfully = 0
                    start_time_idx = time.time()
                    try:
                        image_paths = find_image_files(image_dir_path)
                        total_images = len(image_paths)
                        if not image_paths:
                            status_text_idx.warning("No hay imágenes.")
                            progress_bar_idx.empty()
                        else:
                            status_text_idx.text(
                                f"Encontradas {total_images}. Procesando..."
                            )
                            batch_size = config.BATCH_SIZE_IMAGES
                            num_batches = (total_images + batch_size - 1) // batch_size
                            for i in range(num_batches):
                                batch_start_idx = i * batch_size
                                batch_end_idx = min((i + 1) * batch_size, total_images)
                                current_batch_paths = image_paths[
                                    batch_start_idx:batch_end_idx
                                ]
                                current_batch_num = i + 1
                                status_text_idx.text(
                                    f" Lote {current_batch_num}/{num_batches}: Cargando {len(current_batch_paths)}..."
                                )
                                loaded_images, loaded_paths = (
                                    pipeline.batch_load_images(current_batch_paths)
                                )
                                if not loaded_images:
                                    status_text_idx.text(
                                        f" Lote {current_batch_num}: No válidas."
                                    )
                                    processed_count += len(current_batch_paths)
                                else:
                                    status_text_idx.text(
                                        f" Lote {current_batch_num}: Vectorizando {len(loaded_images)}..."
                                    )
                                    embeddings = vectorizer.vectorize_images(
                                        loaded_images,
                                        batch_size=len(loaded_images),
                                        truncate_dim=config.VECTOR_DIMENSION,
                                    )
                                    if embeddings and len(embeddings) == len(
                                        loaded_paths
                                    ):
                                        status_text_idx.text(
                                            f" Lote {current_batch_num}: Guardando {len(loaded_paths)}..."
                                        )
                                        ids = loaded_paths
                                        metadatas = [
                                            {"source_path": path}
                                            for path in loaded_paths
                                        ]
                                        db.add_embeddings(
                                            ids=ids,
                                            embeddings=embeddings,
                                            metadatas=metadatas,
                                        )
                                        total_processed_successfully += len(ids)
                                    else:
                                        status_text_idx.warning(
                                            f" Lote {current_batch_num}: Error vectorizar/guardar."
                                        )
                                    processed_count += len(current_batch_paths)
                                if total_images > 0:
                                    progress_bar_idx.progress(
                                        min(processed_count / total_images, 1.0)
                                    )
                            end_time_idx = time.time()
                            elapsed_time = end_time_idx - start_time_idx
                            status_text_idx.success(
                                f"✅ Indexación completada en {elapsed_time:.2f}s. {total_processed_successfully}/{total_images} guardadas."
                            )
                            num_items = db.collection.count()
                            st.sidebar.info(
                                f"**BD Actual:** `{config.CHROMA_COLLECTION_NAME}`\n\n**Imágenes Indexadas:** `{num_items}`"
                            )
                    except Exception as e:
                        status_text_idx.error(f"❌ Error indexación: {e}")
                    finally:
                        time.sleep(2)
                        progress_bar_idx.empty()
        with st.expander("⚠️ Opciones Avanzadas"):
            if st.button("🗑️ Limpiar TODA la Colección", key="clear_collection"):
                st.warning(
                    f"🚨 ¡Atención! Eliminará TODO de `{config.CHROMA_COLLECTION_NAME}`."
                )
                if "confirm_clear" not in st.session_state:
                    st.session_state.confirm_clear = False
                if st.checkbox("Sí, entiendo y quiero limpiar", key="confirm_checkbox"):
                    st.session_state.confirm_clear = True
                else:
                    st.session_state.confirm_clear = False
                if st.session_state.confirm_clear:
                    if st.button("CONFIRMAR LIMPIEZA", key="confirm_clear_button"):
                        try:
                            with st.spinner("Vaciando..."):
                                count_before = db.collection.count()
                                db.clear_collection()
                                count_after = db.collection.count()
                            st.success(
                                f"✅ Colección limpiada. {count_before} -> {count_after} elementos."
                            )
                            st.sidebar.info(
                                f"**BD Actual:** `{config.CHROMA_COLLECTION_NAME}`\n\n**Imágenes Indexadas:** `{count_after}`"
                            )
                        except Exception as e:
                            st.error(f"❌ Error limpieza: {e}")
                        finally:
                            st.session_state.confirm_clear = False
                            st.rerun()
                else:
                    st.info("Marca la casilla si estás seguro.")

    # === Pestaña 2: Buscar por Texto ===
    with tab2:
        # ... (Código de búsqueda por texto igual que antes) ...
        st.header("2. Buscar Imágenes por Descripción Textual")
        st.markdown("Escribe una descripción...")
        query_text = st.text_input(
            "Descripción:", placeholder="Ej: gato durmiendo", key="text_query_input"
        )
        num_results_text = st.slider(
            "Resultados:", 1, 50, 10, key="num_results_text_slider"
        )
        if st.button("🔎 Buscar por Texto", key="search_text_button"):
            if not query_text.strip():
                st.warning("⚠️ Introduce descripción.")
            else:
                results_container_text = st.container()
                with st.spinner(f"🧠 Buscando: '{query_text}'..."):
                    try:
                        results_text = pipeline.search_by_text(
                            query_text,
                            vectorizer,
                            db,
                            num_results_text,
                            config.VECTOR_DIMENSION,
                        )
                        display_results(results_text, results_container_text)
                    except Exception as e:
                        st.error(f"❌ Error búsqueda texto: {e}")

    # === Pestaña 3: Buscar por Imagen ===
    with tab3:
        st.header("3. Buscar Imágenes por Similitud con Otra Imagen")
        st.markdown("Sube una imagen o selecciona una ya indexada.")

        # <--- CAMBIO: Inicializar session_state si no existe --->
        if "selected_indexed_image_path" not in st.session_state:
            st.session_state.selected_indexed_image_path = None

        # <--- CAMBIO: Añadir callback para resetear si cambia el modo --->
        def reset_selected_path():
            st.session_state.selected_indexed_image_path = None

        search_mode = st.radio(
            "Elige el modo de búsqueda por imagen:",
            ("Subir una imagen nueva", "Seleccionar una imagen ya indexada"),
            key="image_search_mode",
            horizontal=True,
            on_change=reset_selected_path,  # <--- CAMBIO: Llamar a reset en cambio --->
        )

        num_results_img = st.slider(
            "Número máximo de resultados:",
            min_value=1,
            max_value=50,
            value=10,
            key="num_results_img_slider",
        )

        query_image_path_img: Optional[str] = None
        temp_file_to_delete: Optional[str] = None

        # --- Modo: Subir Imagen ---
        if search_mode == "Subir una imagen nueva":
            # <--- CAMBIO: Asegurarse de que el state está reseteado --->
            # st.session_state.selected_indexed_image_path = None # Ya se hace con on_change

            uploaded_file = st.file_uploader(
                "Arrastra o selecciona una imagen:",
                type=list(ext.strip(".") for ext in config.IMAGE_EXTENSIONS),
                key="image_uploader",
            )

            if uploaded_file is not None:
                try:
                    img_uploaded = Image.open(uploaded_file)
                    st.image(
                        img_uploaded,
                        caption=f"Imagen subida: {uploaded_file.name}",
                        width=250,
                    )
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=os.path.splitext(uploaded_file.name)[1]
                    ) as tmp_file:
                        img_uploaded.save(tmp_file.name)
                        query_image_path_img = tmp_file.name
                        temp_file_to_delete = tmp_file.name
                except Exception as e:
                    st.error(f"❌ Error al procesar la imagen subida: {e}")
                    query_image_path_img = None

        # --- Modo: Seleccionar Imagen Indexada ---
        elif search_mode == "Seleccionar una imagen ya indexada":
            st.info("Mostrando algunas imágenes indexadas para seleccionar.")
            indexed_image_paths: List[str] = []
            try:
                if db and db.collection and db.collection.count() > 0:
                    result = db.collection.get(include=[])
                    if result and "ids" in result:
                        indexed_image_paths = [
                            id_path
                            for id_path in result["ids"]
                            if os.path.isfile(id_path)
                        ]
                else:
                    st.warning("Colección vacía.")
            except Exception as e:
                st.error(f"Error al obtener lista indexada: {e}")

            if indexed_image_paths:
                max_previews = 15
                display_paths = indexed_image_paths[:max_previews]
                st.write(
                    f"Selecciona una de las {len(display_paths)} (de {len(indexed_image_paths)}) mostradas:"
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
                        # <--- CAMBIO: Guardar en session_state al hacer clic --->
                        if col.button("Usar esta", key=button_key):
                            st.session_state.selected_indexed_image_path = img_path
                            # st.rerun() # Opcional: Forzar rerun para mostrar inmediatamente abajo
                    except Exception:
                        col.caption(f"Error {os.path.basename(img_path)}")

                # <--- CAMBIO: Leer de session_state para mostrar y preparar búsqueda --->
                if st.session_state.selected_indexed_image_path:
                    selected_path_from_state = (
                        st.session_state.selected_indexed_image_path
                    )
                    # Verificar si el archivo todavía existe antes de usarlo
                    if os.path.isfile(selected_path_from_state):
                        st.success(
                            f"Imagen seleccionada: `{os.path.basename(selected_path_from_state)}`"
                        )
                        try:
                            st.image(Image.open(selected_path_from_state), width=250)
                            query_image_path_img = (
                                selected_path_from_state  # Preparar para la búsqueda
                            )
                        except Exception as e:
                            st.error(f"Error al mostrar la imagen seleccionada: {e}")
                            st.session_state.selected_indexed_image_path = (
                                None  # Resetear si hay error
                            )
                    else:
                        st.error(
                            f"El archivo seleccionado ya no existe: {selected_path_from_state}"
                        )
                        st.session_state.selected_indexed_image_path = None  # Resetear

                # else:
                # No mostrar mensaje de info aquí para no ser repetitivo si no se ha seleccionado nada aún

            else:
                st.warning("No hay imágenes indexadas válidas para seleccionar.")

        # --- Botón de Búsqueda por Imagen (común a ambos modos) ---
        # Ahora query_image_path_img se establece correctamente antes de esta verificación
        if query_image_path_img:
            if st.button("🖼️ Buscar Imágenes Similares", key="search_image_button"):
                results_container_img = st.container()
                with st.spinner("🖼️ Buscando imágenes similares..."):
                    try:
                        # ... (Lógica de búsqueda y filtrado igual que antes) ...
                        results_img = pipeline.search_by_image(
                            query_image_path_img,
                            vectorizer,
                            db,
                            num_results_img + 1,
                            config.VECTOR_DIMENSION,
                        )
                        if results_img and results_img.get("ids"):
                            try:
                                query_basename = os.path.basename(query_image_path_img)
                                ids_list = list(results_img["ids"])
                                distances_list = list(results_img["distances"])
                                metadatas_list = list(results_img["metadatas"])
                                indices_to_remove = [
                                    idx
                                    for idx, res_id in enumerate(ids_list)
                                    if idx < len(distances_list)
                                    and os.path.basename(res_id) == query_basename
                                    and distances_list[idx] < 1e-6
                                ]
                                if indices_to_remove:
                                    for idx in sorted(indices_to_remove, reverse=True):
                                        if idx < len(ids_list):
                                            ids_list.pop(idx)
                                        if idx < len(distances_list):
                                            distances_list.pop(idx)
                                        if idx < len(metadatas_list):
                                            metadatas_list.pop(idx)
                                    results_img["ids"] = ids_list
                                    results_img["distances"] = distances_list
                                    results_img["metadatas"] = metadatas_list
                                results_img["ids"] = results_img["ids"][
                                    :num_results_img
                                ]
                                results_img["distances"] = results_img["distances"][
                                    :num_results_img
                                ]
                                results_img["metadatas"] = results_img["metadatas"][
                                    :num_results_img
                                ]
                            except Exception as filter_e:
                                st.warning(f"No se pudo filtrar consulta: {filter_e}")
                        display_results(results_img, results_container_img)
                    except Exception as e:
                        st.error(f"❌ Error búsqueda imagen: {e}")
                    finally:
                        if temp_file_to_delete and os.path.exists(temp_file_to_delete):
                            try:
                                os.unlink(temp_file_to_delete)
                            except Exception as e_unlink:
                                st.warning(f"No se pudo eliminar tmp: {e_unlink}")
        # Mensajes informativos si no hay imagen de consulta lista
        elif search_mode == "Subir una imagen nueva" and uploaded_file is None:
            st.info("Sube una imagen para activar la búsqueda.")
        elif (
            search_mode == "Seleccionar una imagen ya indexada"
            and not st.session_state.selected_indexed_image_path
        ):
            st.info(
                "Selecciona una de las imágenes indexadas mostradas arriba para activar la búsqueda."
            )


# --- Punto de Entrada ---
if __name__ == "__main__":
    # --- Verificaciones Previas (Opcional pero útil) ---
    st.sidebar.header("Información del Sistema")
    # ... (igual que antes) ...
    st.sidebar.caption(f"Python: {sys.version_info.major}.{sys.version_info.minor}")
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        st.sidebar.caption(f"CUDA Disponible: {'✅ Sí' if cuda_available else '❌ No'}")
    except ImportError:
        st.sidebar.warning("'torch' no encontrado.")
    except Exception as e:
        st.sidebar.error(f"Error check CUDA: {e}")
    st.sidebar.header("Configuración Clave")
    st.sidebar.caption(f"Modelo: `{config.MODEL_NAME}`")
    st.sidebar.caption(f"Dispositivo: `{config.DEVICE}`")
    st.sidebar.caption(f"Dimensión Vector: `{config.VECTOR_DIMENSION or 'Auto'}`")
    st.sidebar.caption(f"Ruta BD: `{config.CHROMA_DB_PATH}`")

    main()
