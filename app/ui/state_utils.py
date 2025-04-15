import logging
import os
from typing import List, Optional # Añadir List, Optional

import streamlit as st

logger = logging.getLogger(__name__)

# --- Claves de Estado para Selección de Imágenes (Sin cambios) ---
STATE_UPLOADED_IMG_PATH_PREFIX = "uploaded_image_temp_path"
STATE_SELECTED_INDEXED_IMG_PREFIX = "selected_indexed_image_path"
STATE_TRIGGER_SEARCH_FLAG_PREFIX = "trigger_search_after_selection"

# --- Claves de Estado para Configuración General (Sin cambios) ---
STATE_CONFIRM_CLEAR = "confirm_clear_triggered"
STATE_CONFIRM_DELETE = "confirm_delete_triggered"
STATE_SELECTED_MODEL = "selected_model"
STATE_SELECTED_DIMENSION = "selected_dimension"

# --- Claves de Estado para Widgets Específicos (Sin cambios) ---
STATE_SELECTED_BATCH_SIZE = "selected_batch_size"
STATE_HYBRID_METHOD = "hybrid_search_method"

# --- Claves de Estado para Clustering ---
STATE_CLUSTER_LABELS = "cluster_labels_result" # Etiquetas numéricas asignadas
STATE_CLUSTER_IDS = "cluster_ids_result"       # IDs de los elementos clusterizados
STATE_REDUCED_EMBEDDINGS = "reduced_embeddings_for_clustering" # Embeddings reducidos (si se usaron)
STATE_REDUCED_IDS = "reduced_embeddings_ids"           # IDs correspondientes a los embeddings reducidos
# --- NUEVA CLAVE ---
STATE_GENERATED_LABELS = 'generated_cluster_labels' # Diccionario {cluster_num: text_label}

def get_state_key(base: str, suffix: str = "") -> str:
    """Generates a unique session state key by appending a suffix."""
    return f"{base}{suffix}"


def reset_image_selection_states(suffix: str = ""):
    """
    Resets state variables related to image selection (uploaded or indexed)
    for a specific context (identified by suffix) and cleans up associated temp files.
    (Sin cambios en la lógica interna)
    """
    selected_key = get_state_key(STATE_SELECTED_INDEXED_IMG_PREFIX, suffix)
    uploaded_key = get_state_key(STATE_UPLOADED_IMG_PATH_PREFIX, suffix)
    trigger_key = get_state_key(STATE_TRIGGER_SEARCH_FLAG_PREFIX, suffix)
    uploader_key = f"image_uploader{suffix}"
    processed_name_key = f"{uploader_key}_processed_name"

    # Limpiar estado de imagen seleccionada indexada
    if selected_key in st.session_state and st.session_state[selected_key] is not None:
        logger.debug(f"Resetting state: Clearing selected indexed image ({selected_key})")
        st.session_state[selected_key] = None

    # Limpiar estado y archivo temporal de imagen subida
    temp_path = st.session_state.get(uploaded_key)
    if temp_path:
        if os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
                logger.debug(f"Resetting state: Temp file deleted ({suffix}): {temp_path}")
            except Exception as e_unlink:
                logger.warning(f"Could not delete temp file {temp_path} on state reset ({suffix}): {e_unlink}")
        if uploaded_key in st.session_state:
            logger.debug(f"Resetting state: Clearing uploaded image path ({uploaded_key})")
            st.session_state[uploaded_key] = None

    # Limpiar flag de búsqueda automática
    if trigger_key in st.session_state and st.session_state[trigger_key]:
        logger.debug(f"Resetting state: Clearing trigger flag ({trigger_key})")
        st.session_state[trigger_key] = False

    # Limpiar marcador de archivo procesado por el uploader
    if processed_name_key in st.session_state:
        logger.debug(f"Resetting state: Removing processed file marker ({processed_name_key}) for uploader '{uploader_key}'")
        del st.session_state[processed_name_key]


def clear_clustering_states():
    """Clears all session state variables related to clustering results, optimization, and labeling."""
    # --- MODIFICADO: Incluir la nueva clave de etiquetas generadas ---
    keys_to_clear = [
        STATE_CLUSTER_LABELS,
        STATE_CLUSTER_IDS,
        STATE_REDUCED_EMBEDDINGS,
        STATE_REDUCED_IDS,
        STATE_GENERATED_LABELS
    ]
    cleared_count = 0
    logger.debug(f"Attempting to clear clustering states: {keys_to_clear}")
    for key in keys_to_clear:
        if key in st.session_state:
            try:
                del st.session_state[key]
                cleared_count += 1
                logger.debug(f"  Cleared state key: {key}")
            except Exception as e:
                 logger.warning(f"  Could not delete state key {key}: {e}")

    if cleared_count > 0:
        logger.info(f"Cleared {cleared_count} clustering-related session state variables.")
    else:
         logger.debug("No clustering-related session state variables found to clear.")
