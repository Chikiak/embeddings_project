# app/ui/state_utils.py
import os
import logging
import streamlit as st

logger = logging.getLogger(__name__)

# --- Constantes para Claves de Estado de Sesión ---
# Prefijos para evitar colisiones entre pestañas si es necesario
STATE_UPLOADED_IMG_PATH_PREFIX = "uploaded_image_temp_path"
STATE_SELECTED_INDEXED_IMG_PREFIX = "selected_indexed_image_path"
STATE_TRIGGER_SEARCH_FLAG_PREFIX = "trigger_search_after_selection" # Flag para búsqueda automática

# Claves de estado globales/compartidas
STATE_CONFIRM_CLEAR = "confirm_clear_triggered"
STATE_CONFIRM_DELETE = "confirm_delete_triggered"
STATE_SELECTED_MODEL = "selected_model" # Modelo elegido en la sidebar
STATE_SELECTED_DIMENSION = "selected_dimension" # Dimensión elegida en la sidebar

# Claves específicas de pestañas (ejemplos)
STATE_SELECTED_BATCH_SIZE = "selected_batch_size" # Para la pestaña de indexación
STATE_HYBRID_METHOD = "hybrid_search_method" # Para la pestaña híbrida

# --- Funciones Auxiliares para el Estado ---

def get_state_key(base: str, suffix: str = "") -> str:
    """
    Genera una clave única para el estado de sesión añadiendo un sufijo.

    Args:
        base: La clave base (ej: STATE_UPLOADED_IMG_PATH_PREFIX).
        suffix: Un sufijo opcional para diferenciar instancias (ej: "_img_search").

    Returns:
        La clave de estado completa.
    """
    return f"{base}{suffix}"

def reset_image_selection_states(suffix: str = ""):
    """
    Reinicia las variables de estado relacionadas con la selección de imágenes
    (subidas o indexadas) para un contexto específico (identificado por suffix)
    y elimina los archivos temporales asociados.
    """
    selected_key = get_state_key(STATE_SELECTED_INDEXED_IMG_PREFIX, suffix)
    uploaded_key = get_state_key(STATE_UPLOADED_IMG_PATH_PREFIX, suffix)
    trigger_key = get_state_key(STATE_TRIGGER_SEARCH_FLAG_PREFIX, suffix)
    uploader_key = f"image_uploader{suffix}" # Clave base del widget uploader
    # Clave interna que usamos para rastrear si el archivo del uploader fue procesado
    processed_name_key = f"{uploader_key}_processed_name"

    # 1. Limpiar la ruta de la imagen indexada seleccionada
    if selected_key in st.session_state:
        if st.session_state[selected_key] is not None:
            logger.debug(f"Reseteando estado: Limpiando imagen indexada seleccionada ({selected_key})")
            st.session_state[selected_key] = None

    # 2. Limpiar la ruta de la imagen subida y eliminar el archivo temporal
    temp_path = st.session_state.get(uploaded_key)
    if temp_path:
        # Eliminar el archivo temporal si existe
        if os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
                logger.debug(f"Reseteando estado: Archivo temporal eliminado ({suffix}): {temp_path}")
            except Exception as e_unlink:
                logger.warning(f"No se pudo eliminar el archivo temporal {temp_path} al resetear estado ({suffix}): {e_unlink}")
        # Eliminar la clave del estado de sesión
        if uploaded_key in st.session_state:
             logger.debug(f"Reseteando estado: Limpiando ruta de imagen subida ({uploaded_key})")
             st.session_state[uploaded_key] = None

    # 3. Limpiar el flag de activación de búsqueda automática
    if trigger_key in st.session_state:
        if st.session_state[trigger_key]:
            logger.debug(f"Reseteando estado: Limpiando flag de activación ({trigger_key})")
            st.session_state[trigger_key] = False

    # 4. Limpiar el estado asociado al widget file_uploader
    # Esto ayuda a asegurar que una nueva subida cree un nuevo archivo temporal
    if processed_name_key in st.session_state:
        logger.debug(f"Reseteando estado: Eliminando marcador de archivo procesado ({processed_name_key}) para uploader '{uploader_key}'")
        del st.session_state[processed_name_key]

    # Nota: Resetear visualmente el widget st.file_uploader es complejo.
    # Estas acciones limpian el estado *lógico* y los archivos temporales.
    # Para un reseteo visual completo, se podría necesitar cambiar la `key` del widget
    # o usar st.empty() para renderizarlo condicionalmente.
