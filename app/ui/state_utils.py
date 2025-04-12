# app/ui/state_utils.py
import os
import logging
import streamlit as st

logger = logging.getLogger(__name__)

# --- Constants for Session State Keys ---

# Prefixes for image selection widgets to avoid collisions between tabs
STATE_UPLOADED_IMG_PATH_PREFIX = "uploaded_image_temp_path"
STATE_SELECTED_INDEXED_IMG_PREFIX = "selected_indexed_image_path"
STATE_TRIGGER_SEARCH_FLAG_PREFIX = "trigger_search_after_selection" # Flag for automatic search

# Global/shared state keys
STATE_CONFIRM_CLEAR = "confirm_clear_triggered"
STATE_CONFIRM_DELETE = "confirm_delete_triggered"
STATE_SELECTED_MODEL = "selected_model" # Model chosen in sidebar
STATE_SELECTED_DIMENSION = "selected_dimension" # Dimension chosen in sidebar

# Tab-specific state keys
STATE_SELECTED_BATCH_SIZE = "selected_batch_size" # For indexing tab
STATE_HYBRID_METHOD = "hybrid_search_method" # For hybrid search tab

# --- NEW State Keys for Clustering Tabs ---
STATE_CLUSTER_LABELS = "cluster_labels_result"      # Stores the labels array from the last clustering run
STATE_CLUSTER_IDS = "cluster_ids_result"            # Stores the IDs corresponding to the labels
STATE_REDUCED_EMBEDDINGS = "reduced_embeddings_for_clustering" # Stores PCA/UMAP reduced data
STATE_REDUCED_IDS = "reduced_embeddings_ids"        # Stores IDs corresponding to reduced data

# --- Helper Functions for State ---

def get_state_key(base: str, suffix: str = "") -> str:
    """
    Generates a unique session state key by appending a suffix.

    Args:
        base: The base key (e.g., STATE_UPLOADED_IMG_PATH_PREFIX).
        suffix: An optional suffix to differentiate instances (e.g., "_img_search").

    Returns:
        The complete state key string.
    """
    return f"{base}{suffix}"

def reset_image_selection_states(suffix: str = ""):
    """
    Resets state variables related to image selection (uploaded or indexed)
    for a specific context (identified by suffix) and cleans up associated temp files.
    """
    selected_key = get_state_key(STATE_SELECTED_INDEXED_IMG_PREFIX, suffix)
    uploaded_key = get_state_key(STATE_UPLOADED_IMG_PATH_PREFIX, suffix)
    trigger_key = get_state_key(STATE_TRIGGER_SEARCH_FLAG_PREFIX, suffix)
    uploader_key = f"image_uploader{suffix}" # Base key of the file_uploader widget
    # Internal key used to track if the uploader's file was processed
    processed_name_key = f"{uploader_key}_processed_name"

    # 1. Clear selected indexed image path
    if selected_key in st.session_state:
        if st.session_state[selected_key] is not None:
            logger.debug(f"Resetting state: Clearing selected indexed image ({selected_key})")
            st.session_state[selected_key] = None

    # 2. Clear uploaded image path and delete the temporary file
    temp_path = st.session_state.get(uploaded_key)
    if temp_path:
        # Delete the temp file if it exists
        if os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
                logger.debug(f"Resetting state: Temp file deleted ({suffix}): {temp_path}")
            except Exception as e_unlink:
                logger.warning(f"Could not delete temp file {temp_path} on state reset ({suffix}): {e_unlink}")
        # Remove the key from session state
        if uploaded_key in st.session_state:
             logger.debug(f"Resetting state: Clearing uploaded image path ({uploaded_key})")
             st.session_state[uploaded_key] = None

    # 3. Clear the automatic search trigger flag
    if trigger_key in st.session_state:
        if st.session_state[trigger_key]:
            logger.debug(f"Resetting state: Clearing trigger flag ({trigger_key})")
            st.session_state[trigger_key] = False

    # 4. Clear the state associated with the file_uploader widget
    # This helps ensure a new upload creates a new temp file correctly
    if processed_name_key in st.session_state:
        logger.debug(f"Resetting state: Removing processed file marker ({processed_name_key}) for uploader '{uploader_key}'")
        del st.session_state[processed_name_key]

    # Note: Visually resetting the st.file_uploader widget itself is complex.
    # These actions clean the *logical* state and associated temp files.
    # For a full visual reset, changing the widget's `key` or using st.empty()
    # for conditional rendering might be needed, but often isn't necessary.

# --- Function to clear clustering-related states ---
def clear_clustering_states():
    """Clears all session state variables related to clustering results and optimization."""
    keys_to_clear = [
        STATE_CLUSTER_LABELS,
        STATE_CLUSTER_IDS,
        STATE_REDUCED_EMBEDDINGS,
        STATE_REDUCED_IDS,
        # Add any other clustering-related keys here
        "centroid_search_results" # Example of another related key used in label_centroid_tab
    ]
    cleared_count = 0
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
            cleared_count += 1
    if cleared_count > 0:
        logger.debug(f"Cleared {cleared_count} clustering-related session state variables.")

# You might call clear_clustering_states() when:
# - The database collection changes (due to model or dimension change)
# - The user explicitly requests to clear results
# - An error occurs that invalidates previous results
