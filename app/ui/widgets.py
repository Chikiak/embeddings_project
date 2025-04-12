# app/ui/widgets.py
import streamlit as st
from PIL import Image, UnidentifiedImageError
import os
import logging
import tempfile
from typing import Optional, List

# Importaciones relativas/absolutas correctas
import config
from data_access.vector_db_interface import VectorDBInterface
from app.exceptions import DatabaseError
# Importar TODAS las claves y la función de reseteo general
from .state_utils import (
    get_state_key,
    reset_image_selection_states, # Aún se usa en _render_upload_widget y on_change
    STATE_UPLOADED_IMG_PATH_PREFIX,
    STATE_SELECTED_INDEXED_IMG_PREFIX,
    STATE_TRIGGER_SEARCH_FLAG_PREFIX,
)

logger = logging.getLogger(__name__)

# _render_upload_widget (sin cambios respecto a la versión anterior)
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
    # Claves únicas para este widget específico
    uploader_key = f"image_uploader{key_suffix}"
    uploaded_state_key = get_state_key(STATE_UPLOADED_IMG_PATH_PREFIX, key_suffix)
    # Clave para rastrear el nombre del último archivo procesado por este uploader
    processed_name_key = f"{uploader_key}_processed_name"

    # Obtener extensiones permitidas desde la configuración
    allowed_extensions = [ext.lstrip(".") for ext in config.IMAGE_EXTENSIONS]

    # Renderizar el widget file_uploader
    uploaded_file = st.file_uploader(
        "Arrastra o selecciona una imagen:",
        type=allowed_extensions, # Especificar tipos permitidos
        key=uploader_key,
        help=f"Formatos soportados: {', '.join(allowed_extensions)}",
    )

    # Obtener la ruta temporal actual (si existe) del estado
    current_temp_path = st.session_state.get(uploaded_state_key)

    # --- Lógica de Manejo del Estado y Archivo Temporal ---

    # Caso 1: No hay archivo subido en el widget
    if uploaded_file is None:
        # Si *había* un archivo temporal asociado a este widget, significa que
        # el usuario lo eliminó o cambió de modo. Debemos limpiar el estado.
        if current_temp_path:
            logger.debug(f"Widget uploader ({uploader_key}) está vacío, reseteando estado...")
            # reset_image_selection_states se encarga de eliminar el archivo temporal
            reset_image_selection_states(key_suffix)
        return None # No hay archivo subido

    # Caso 2: Hay un archivo subido en el widget
    else:
        # Comprobar si este archivo es el mismo que ya procesamos y guardamos
        # Esto evita recrear el archivo temporal si Streamlit re-ejecuta pero el archivo no cambió
        if current_temp_path and os.path.exists(current_temp_path) and st.session_state.get(processed_name_key) == uploaded_file.name:
            logger.debug(f"Retornando ruta temporal existente para {uploaded_file.name} ({key_suffix})")
            return current_temp_path # Usar el archivo temporal existente
        else:
            # Es un archivo nuevo, o el archivo temporal anterior fue eliminado.
            # Primero, limpiar cualquier estado/archivo antiguo asociado a este widget.
            logger.debug(f"Archivo nuevo detectado ({uploaded_file.name}) o estado necesita reseteo ({key_suffix}).")
            reset_image_selection_states(key_suffix)

            # Crear un nuevo archivo temporal para el archivo recién subido
            try:
                # Obtener la extensión del archivo original para mantenerla
                file_suffix = os.path.splitext(uploaded_file.name)[1]
                # Crear archivo temporal Nombrado (NamedTemporaryFile) que persiste después de cerrar
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp_file:
                    # Escribir el contenido del archivo subido al archivo temporal
                    tmp_file.write(uploaded_file.getvalue())
                    new_temp_path = tmp_file.name # Obtener la ruta del archivo temporal creado
                # Guardar la ruta del nuevo archivo temporal en el estado de sesión
                st.session_state[uploaded_state_key] = new_temp_path
                # Marcar el nombre de este archivo como procesado para este uploader
                st.session_state[processed_name_key] = uploaded_file.name
                logger.debug(f"Archivo temporal creado para subida ({key_suffix}): {new_temp_path}")
                return new_temp_path # Devolver la ruta del nuevo archivo temporal
            except Exception as e:
                # Manejar errores durante la creación del archivo temporal
                st.error(f"❌ Error al guardar archivo temporal: {e}")
                logger.error(f"Error creando archivo temporal ({key_suffix}): {e}", exc_info=True)
                # Asegurarse de que el estado quede limpio después de un fallo
                reset_image_selection_states(key_suffix)
                return None # Indicar fallo


# _render_indexed_selection (CON CAMBIOS EN LA LÓGICA DEL BOTÓN)
def _render_indexed_selection(db: VectorDBInterface, key_suffix: str = "") -> Optional[str]:
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
    # Claves de estado relevantes
    selected_state_key = get_state_key(STATE_SELECTED_INDEXED_IMG_PREFIX, key_suffix)
    trigger_search_key = get_state_key(STATE_TRIGGER_SEARCH_FLAG_PREFIX, key_suffix)
    uploaded_state_key = get_state_key(STATE_UPLOADED_IMG_PATH_PREFIX, key_suffix)
    # Clave del widget uploader para limpiar su estado asociado
    uploader_key = f"image_uploader{key_suffix}"
    processed_name_key = f"{uploader_key}_processed_name"

    # Verificar si la BD está lista
    if not db or not db.is_initialized:
         st.warning("⚠️ Base de datos no disponible para seleccionar imágenes.")
         return st.session_state.get(selected_state_key)

    # Obtener contador
    db_count = -1
    try:
        db_count = db.count()
    except Exception as e:
        st.error(f"Error al obtener contador de la BD: {e}")
        logger.error(f"Error getting DB count in _render_indexed_selection: {e}", exc_info=True)
        return st.session_state.get(selected_state_key)

    # Si no hay imágenes indexadas
    if db_count <= 0:
        st.info(f"ℹ️ No hay imágenes indexadas en la base de datos para seleccionar.")
        if selected_state_key in st.session_state and st.session_state[selected_state_key]:
            logger.debug("Limpiando selección indexada porque la BD está vacía.")
            st.session_state[selected_state_key] = None
            st.session_state[trigger_search_key] = False
            # st.rerun() # Opcional
        return None

    # --- Obtener y Verificar Rutas de Imágenes Indexadas ---
    st.info(
        "Mostrando algunas imágenes indexadas. Haz clic en 'Usar esta' para seleccionarla como consulta."
    )
    indexed_image_paths: List[str] = []
    selected_image_path: Optional[str] = st.session_state.get(selected_state_key)

    try:
        limit_get = 25
        logger.debug(f"Intentando obtener hasta {limit_get} IDs para vista previa...")
        data = db.get_all_embeddings_with_ids(pagination_batch_size=limit_get)

        if data and data[0]:
            all_ids = data[0]
            logger.debug(f"Obtenidos {len(all_ids)} IDs. Primeros: {all_ids[:5]}")
            with st.spinner("Verificando archivos indexados..."):
                verified_paths = []
                missing_count = 0
                for id_path in all_ids:
                    if id_path and isinstance(id_path, str):
                        try:
                            if os.path.isfile(id_path):
                                verified_paths.append(id_path)
                            else:
                                logger.debug(f"  Archivo indexado no encontrado: {id_path}")
                                missing_count += 1
                        except Exception as e_check:
                            logger.error(f"  Error comprobando archivo '{id_path}': {e_check}")
                            missing_count += 1
                    else:
                        logger.warning(f"  Ignorando entrada de ID inválida: {id_path}")
                indexed_image_paths = verified_paths
                logger.debug(f"Comprobación completa. Encontrados {len(verified_paths)} archivos de {len(all_ids)} IDs.")
                if missing_count > 0:
                     logger.warning(f"Se encontraron {missing_count} archivos faltantes/inválidos.")
        else:
            logger.warning("No se obtuvieron IDs de la BD para la vista previa.")

    except DatabaseError as e:
        st.error(f"Error de BD al obtener imágenes indexadas: {e}")
        logger.error(f"DatabaseError getting indexed image list: {e}", exc_info=True)
        return selected_image_path
    except Exception as e:
        st.error(f"Error inesperado al obtener imágenes indexadas: {e}")
        logger.error(f"Unexpected error getting indexed image list: {e}", exc_info=True)
        return selected_image_path

    if not indexed_image_paths:
        st.warning("No se encontraron imágenes indexadas válidas.")
        if selected_state_key in st.session_state and st.session_state[selected_state_key]:
             logger.debug("Limpiando selección indexada porque no hay archivos válidos.")
             st.session_state[selected_state_key] = None
             st.session_state[trigger_search_key] = False
             # st.rerun() # Opcional
        return None

    # --- Renderizar Cuadrícula de Selección ---
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
            is_selected = (selected_image_path == img_path)

            if not is_selected:
                # ******** INICIO DE LA CORRECCIÓN ********
                if col.button("Usar esta", key=button_key):
                    # 1. Establecer la imagen seleccionada
                    st.session_state[selected_state_key] = img_path
                    # 2. Establecer el flag para la búsqueda automática
                    st.session_state[trigger_search_key] = True
                    logger.info(f"Usuario seleccionó imagen indexada ({key_suffix}), activando flag: {img_path}")

                    # 3. Limpiar MANUALMENTE solo el estado de la imagen SUBIDA
                    #    (No llamar a reset_image_selection_states aquí)
                    uploaded_temp_path = st.session_state.get(uploaded_state_key)
                    if uploaded_temp_path:
                        if os.path.exists(uploaded_temp_path):
                            try:
                                os.unlink(uploaded_temp_path)
                                logger.debug(f"Limpiando estado de subida: Archivo temporal eliminado {uploaded_temp_path}")
                            except Exception as e_unlink:
                                logger.warning(f"No se pudo eliminar archivo temporal {uploaded_temp_path} al seleccionar indexada: {e_unlink}")
                        if uploaded_state_key in st.session_state:
                            st.session_state[uploaded_state_key] = None
                            logger.debug(f"Limpiando estado de subida: Clave {uploaded_state_key} reseteada.")
                    # Limpiar también el marcador del nombre procesado del uploader
                    if processed_name_key in st.session_state:
                         del st.session_state[processed_name_key]
                         logger.debug(f"Limpiando estado de subida: Clave {processed_name_key} eliminada.")

                    # 4. Forzar re-ejecución para que la pestaña procese el flag
                    st.rerun()
                # ******** FIN DE LA CORRECCIÓN ********

        except Exception as img_err:
            col.error(f"Error: {os.path.basename(img_path)}")
            logger.warning(f"Error cargando vista previa de imagen indexada {img_path}: {img_err}")

    return st.session_state.get(selected_state_key)


# _display_query_image (sin cambios respecto a la versión anterior)
def _display_query_image(query_image_path: Optional[str], source_info: str, key_suffix: str = "") -> bool:
    """
    Muestra la imagen de consulta seleccionada (subida o indexada).

    Args:
        query_image_path: Ruta a la imagen a mostrar.
        source_info: Descripción del origen ("Imagen Subida", "Imagen Indexada Seleccionada").
        key_suffix: Sufijo para diferenciar contextos si es necesario para reseteo.

    Returns:
        True si la imagen se mostró con éxito, False en caso contrario.
    """
    # Verificar si la ruta es válida y el archivo existe
    if query_image_path and isinstance(query_image_path, str) and os.path.isfile(query_image_path):
        try:
            # Usar un contenedor para aplicar estilos
            with st.container():
                 # Aplicar clase CSS para posible borde o estilo
                 st.markdown('<div class="selected-query-image">', unsafe_allow_html=True)
                 # Mostrar la imagen
                 st.image(
                     Image.open(query_image_path),
                     caption=f"{source_info}: {os.path.basename(query_image_path)}",
                     width=250, # Ancho fijo para la vista previa
                 )
                 st.markdown("</div>", unsafe_allow_html=True)
            return True # Imagen mostrada con éxito
        except Exception as e:
            # Manejar errores al abrir o mostrar la imagen
            st.error(
                f"❌ Error al mostrar la imagen de consulta ({os.path.basename(query_image_path)}): {e}"
            )
            logger.error(
                f"Error displaying query image {query_image_path} ({key_suffix}): {e}", exc_info=True
            )
            # Intentar resetear el estado si la imagen no se puede mostrar
            reset_image_selection_states(key_suffix)
            st.rerun() # Forzar re-ejecución para limpiar la UI
            return False
    # Caso donde la ruta existe en el estado pero el archivo no se encuentra
    elif query_image_path and isinstance(query_image_path, str):
        st.error(f"El archivo de la imagen de consulta ya no existe: {query_image_path}")
        logger.warning(f"Query image file path in state is invalid: {query_image_path}")
        # Resetear el estado asociado a esta ruta inválida
        reset_image_selection_states(key_suffix)
        st.rerun() # Forzar re-ejecución para limpiar la UI
        return False
    # Caso donde no hay ruta de imagen de consulta
    else:
        return False # No hay imagen que mostrar
