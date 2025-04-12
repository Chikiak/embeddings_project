import logging
import os
import tempfile
from typing import List, Optional

import streamlit as st
from PIL import Image, UnidentifiedImageError

import config
from app.exceptions import DatabaseError
from data_access.vector_db_interface import VectorDBInterface

from .state_utils import (
    STATE_SELECTED_INDEXED_IMG_PREFIX,
    STATE_TRIGGER_SEARCH_FLAG_PREFIX,
    STATE_UPLOADED_IMG_PATH_PREFIX,
    get_state_key,
    reset_image_selection_states,
)

logger = logging.getLogger(__name__)


def _render_upload_widget(key_suffix: str = "") -> Optional[str]:
    """
    Renderiza el widget st.file_uploader y gestiona el archivo temporal asociado.

    Al subir un archivo nuevo, crea un archivo temporal y guarda su ruta en
    el estado de sesión. Si se elimina el archivo subido (o se cambia el modo),
    limpia el estado y elimina el archivo temporal usando reset_image_selection_states.

    Args:
        key_suffix: Sufijo para crear una clave única para el estado de este widget.

    Returns:
        La ruta al archivo temporal si se subió un archivo con éxito, None en caso contrario.
    """

    uploader_key = f"image_uploader{key_suffix}"
    uploaded_state_key = get_state_key(
        STATE_UPLOADED_IMG_PATH_PREFIX, key_suffix
    )

    processed_name_key = f"{uploader_key}_processed_name"

    allowed_extensions = [ext.lstrip(".") for ext in config.IMAGE_EXTENSIONS]

    uploaded_file = st.file_uploader(
        "Arrastra o selecciona una imagen:",
        type=allowed_extensions,
        key=uploader_key,
        help=f"Formatos soportados: {', '.join(allowed_extensions)}",
    )

    current_temp_path = st.session_state.get(uploaded_state_key)

    if uploaded_file is None:

        if current_temp_path:
            logger.debug(
                f"Widget uploader ({uploader_key}) está vacío, reseteando estado..."
            )

            reset_image_selection_states(key_suffix)
        return None

    else:

        if (
            current_temp_path
            and os.path.exists(current_temp_path)
            and st.session_state.get(processed_name_key) == uploaded_file.name
        ):
            logger.debug(
                f"Retornando ruta temporal existente para {uploaded_file.name} ({key_suffix})"
            )
            return current_temp_path
        else:

            logger.debug(
                f"Archivo nuevo detectado ({uploaded_file.name}) o estado necesita reseteo ({key_suffix})."
            )
            reset_image_selection_states(key_suffix)

            try:

                file_suffix = os.path.splitext(uploaded_file.name)[1]

                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=file_suffix
                ) as tmp_file:

                    tmp_file.write(uploaded_file.getvalue())
                    new_temp_path = tmp_file.name

                st.session_state[uploaded_state_key] = new_temp_path

                st.session_state[processed_name_key] = uploaded_file.name
                logger.debug(
                    f"Archivo temporal creado para subida ({key_suffix}): {new_temp_path}"
                )
                return new_temp_path
            except Exception as e:

                st.error(f"❌ Error al guardar archivo temporal: {e}")
                logger.error(
                    f"Error creando archivo temporal ({key_suffix}): {e}",
                    exc_info=True,
                )

                reset_image_selection_states(key_suffix)
                return None


def _render_indexed_selection(
    db: VectorDBInterface, key_suffix: str = ""
) -> Optional[str]:
    """
    Renderiza una cuadrícula para seleccionar imágenes ya indexadas en la BD.

    Al seleccionar una imagen, guarda su ruta en el estado de sesión y activa
    un flag para búsqueda automática. Limpia el estado de imagen subida.

    Args:
        db: La instancia de la interfaz de base de datos vectorial.
        key_suffix: Sufijo para crear claves de estado únicas.

    Returns:
        La ruta de la imagen indexada seleccionada actualmente, o None si no hay selección.
    """

    selected_state_key = get_state_key(
        STATE_SELECTED_INDEXED_IMG_PREFIX, key_suffix
    )
    trigger_search_key = get_state_key(
        STATE_TRIGGER_SEARCH_FLAG_PREFIX, key_suffix
    )
    uploaded_state_key = get_state_key(
        STATE_UPLOADED_IMG_PATH_PREFIX, key_suffix
    )

    uploader_key = f"image_uploader{key_suffix}"
    processed_name_key = f"{uploader_key}_processed_name"

    if not db or not db.is_initialized:
        st.warning("⚠️ Base de datos no disponible para seleccionar imágenes.")
        return st.session_state.get(selected_state_key)

    db_count = -1
    try:
        db_count = db.count()
    except Exception as e:
        st.error(f"Error al obtener contador de la BD: {e}")
        logger.error(
            f"Error getting DB count in _render_indexed_selection: {e}",
            exc_info=True,
        )
        return st.session_state.get(selected_state_key)

    if db_count <= 0:
        st.info(
            f"ℹ️ No hay imágenes indexadas en la base de datos para seleccionar."
        )
        if (
            selected_state_key in st.session_state
            and st.session_state[selected_state_key]
        ):
            logger.debug(
                "Limpiando selección indexada porque la BD está vacía."
            )
            st.session_state[selected_state_key] = None
            st.session_state[trigger_search_key] = False

        return None

    st.info(
        "Mostrando algunas imágenes indexadas. Haz clic en 'Usar esta' para seleccionarla como consulta."
    )
    indexed_image_paths: List[str] = []
    selected_image_path: Optional[str] = st.session_state.get(
        selected_state_key
    )

    try:
        limit_get = 25
        logger.debug(
            f"Intentando obtener hasta {limit_get} IDs para vista previa..."
        )
        data = db.get_all_embeddings_with_ids(pagination_batch_size=limit_get)

        if data and data[0]:
            all_ids = data[0]
            logger.debug(
                f"Obtenidos {len(all_ids)} IDs. Primeros: {all_ids[:5]}"
            )
            with st.spinner("Verificando archivos indexados..."):
                verified_paths = []
                missing_count = 0
                for id_path in all_ids:
                    if id_path and isinstance(id_path, str):
                        try:
                            if os.path.isfile(id_path):
                                verified_paths.append(id_path)
                            else:
                                logger.debug(
                                    f"  Archivo indexado no encontrado: {id_path}"
                                )
                                missing_count += 1
                        except Exception as e_check:
                            logger.error(
                                f"  Error comprobando archivo '{id_path}': {e_check}"
                            )
                            missing_count += 1
                    else:
                        logger.warning(
                            f"  Ignorando entrada de ID inválida: {id_path}"
                        )
                indexed_image_paths = verified_paths
                logger.debug(
                    f"Comprobación completa. Encontrados {len(verified_paths)} archivos de {len(all_ids)} IDs."
                )
                if missing_count > 0:
                    logger.warning(
                        f"Se encontraron {missing_count} archivos faltantes/inválidos."
                    )
        else:
            logger.warning(
                "No se obtuvieron IDs de la BD para la vista previa."
            )

    except DatabaseError as e:
        st.error(f"Error de BD al obtener imágenes indexadas: {e}")
        logger.error(
            f"DatabaseError getting indexed image list: {e}", exc_info=True
        )
        return selected_image_path
    except Exception as e:
        st.error(f"Error inesperado al obtener imágenes indexadas: {e}")
        logger.error(
            f"Unexpected error getting indexed image list: {e}", exc_info=True
        )
        return selected_image_path

    if not indexed_image_paths:
        st.warning("No se encontraron imágenes indexadas válidas.")
        if (
            selected_state_key in st.session_state
            and st.session_state[selected_state_key]
        ):
            logger.debug(
                "Limpiando selección indexada porque no hay archivos válidos."
            )
            st.session_state[selected_state_key] = None
            st.session_state[trigger_search_key] = False

        return None

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
            col.image(
                img_preview, caption=os.path.basename(img_path), width=100
            )
            button_key = f"select_img_{i}{key_suffix}"
            is_selected = selected_image_path == img_path

            if not is_selected:

                if col.button("Usar esta", key=button_key):

                    st.session_state[selected_state_key] = img_path

                    st.session_state[trigger_search_key] = True
                    logger.info(
                        f"Usuario seleccionó imagen indexada ({key_suffix}), activando flag: {img_path}"
                    )

                    uploaded_temp_path = st.session_state.get(
                        uploaded_state_key
                    )
                    if uploaded_temp_path:
                        if os.path.exists(uploaded_temp_path):
                            try:
                                os.unlink(uploaded_temp_path)
                                logger.debug(
                                    f"Limpiando estado de subida: Archivo temporal eliminado {uploaded_temp_path}"
                                )
                            except Exception as e_unlink:
                                logger.warning(
                                    f"No se pudo eliminar archivo temporal {uploaded_temp_path} al seleccionar indexada: {e_unlink}"
                                )
                        if uploaded_state_key in st.session_state:
                            st.session_state[uploaded_state_key] = None
                            logger.debug(
                                f"Limpiando estado de subida: Clave {uploaded_state_key} reseteada."
                            )

                    if processed_name_key in st.session_state:
                        del st.session_state[processed_name_key]
                        logger.debug(
                            f"Limpiando estado de subida: Clave {processed_name_key} eliminada."
                        )

                    st.rerun()

        except Exception as img_err:
            col.error(f"Error: {os.path.basename(img_path)}")
            logger.warning(
                f"Error cargando vista previa de imagen indexada {img_path}: {img_err}"
            )

    return st.session_state.get(selected_state_key)


def _display_query_image(
    query_image_path: Optional[str], source_info: str, key_suffix: str = ""
) -> bool:
    """
    Muestra la imagen de consulta seleccionada (subida o indexada).

    Args:
        query_image_path: Ruta a la imagen a mostrar.
        source_info: Descripción del origen ("Imagen Subida", "Imagen Indexada Seleccionada").
        key_suffix: Sufijo para diferenciar contextos si es necesario para reseteo.

    Returns:
        True si la imagen se mostró con éxito, False en caso contrario.
    """

    if (
        query_image_path
        and isinstance(query_image_path, str)
        and os.path.isfile(query_image_path)
    ):
        try:

            with st.container():

                st.markdown(
                    '<div class="selected-query-image">',
                    unsafe_allow_html=True,
                )

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
                f"Error displaying query image {query_image_path} ({key_suffix}): {e}",
                exc_info=True,
            )

            reset_image_selection_states(key_suffix)
            st.rerun()
            return False

    elif query_image_path and isinstance(query_image_path, str):
        st.error(
            f"El archivo de la imagen de consulta ya no existe: {query_image_path}"
        )
        logger.warning(
            f"Query image file path in state is invalid: {query_image_path}"
        )

        reset_image_selection_states(key_suffix)
        st.rerun()
        return False

    else:
        return False
