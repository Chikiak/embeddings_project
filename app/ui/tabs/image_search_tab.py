import logging
import os
from typing import Optional

import streamlit as st

import config
from app import searching
from app.exceptions import PipelineError
from app.models import SearchResults
from core.vectorizer import Vectorizer
from data_access.vector_db_interface import VectorDBInterface

from ..common import display_results
from ..state_utils import (
    STATE_SELECTED_INDEXED_IMG_PREFIX,
    STATE_TRIGGER_SEARCH_FLAG_PREFIX,
    STATE_UPLOADED_IMG_PATH_PREFIX,
    get_state_key,
    reset_image_selection_states,
)
from ..widgets import (
    _display_query_image,
    _render_indexed_selection,
    _render_upload_widget,
)

logger = logging.getLogger(__name__)


def render_image_search_tab(
    vectorizer: Vectorizer, db: VectorDBInterface, truncate_dim: Optional[int]
):
    """
    Renderiza el contenido y maneja la lógica para la pestaña 'Buscar por Imagen'.
    Incluye lógica para búsqueda automática al seleccionar una imagen indexada.

    Args:
        vectorizer: Instancia del Vectorizer inicializado.
        db: Instancia de la interfaz de BD vectorial inicializada.
        truncate_dim: Dimensión de truncamiento seleccionada (puede ser None).
    """
    st.header("3. Buscar Imágenes por Similitud Visual")
    st.markdown(
        "Sube una imagen de ejemplo o selecciona una de las imágenes ya indexadas para buscar imágenes visualmente similares."
    )

    key_suffix = "_img_search"
    uploaded_state_key = get_state_key(
        STATE_UPLOADED_IMG_PATH_PREFIX, key_suffix
    )
    selected_state_key = get_state_key(
        STATE_SELECTED_INDEXED_IMG_PREFIX, key_suffix
    )
    trigger_search_key = get_state_key(
        STATE_TRIGGER_SEARCH_FLAG_PREFIX, key_suffix
    )

    if uploaded_state_key not in st.session_state:
        st.session_state[uploaded_state_key] = None
    if selected_state_key not in st.session_state:
        st.session_state[selected_state_key] = None
    if trigger_search_key not in st.session_state:
        st.session_state[trigger_search_key] = False

    results_container = st.container()

    search_triggered_automatically = False
    if st.session_state.get(trigger_search_key, False):
        logger.debug(
            f"Detectado flag de activación para búsqueda automática: {trigger_search_key}"
        )

        st.session_state[trigger_search_key] = False

        query_image_path_auto = st.session_state.get(selected_state_key)

        if query_image_path_auto and os.path.isfile(query_image_path_auto):

            if not db or not db.is_initialized or db.count() <= 0:
                st.warning(
                    "⚠️ La base de datos no está lista o está vacía. No se puede buscar automáticamente."
                )
            elif not vectorizer or not vectorizer.is_ready:
                st.error(
                    "❌ El vectorizador no está listo para la búsqueda automática."
                )
            else:

                num_results_auto_key = f"num_results_img_slider{key_suffix}"
                num_results_auto = st.session_state.get(
                    num_results_auto_key, config.DEFAULT_N_RESULTS
                )

                with results_container:
                    with st.spinner(
                        "🖼️ Buscando imágenes similares (automático)..."
                    ):
                        try:
                            logger.info(
                                f"Realizando búsqueda AUTOMÁTICA por imagen usando: {query_image_path_auto}"
                            )

                            results_auto: Optional[SearchResults] = (
                                searching.search_by_image(
                                    query_image_path=query_image_path_auto,
                                    vectorizer=vectorizer,
                                    db=db,
                                    n_results=num_results_auto,
                                    truncate_dim=truncate_dim,
                                )
                            )

                            display_results(results_auto, results_container)
                            if results_auto:
                                logger.info(
                                    f"Búsqueda automática completada. Encontrados {results_auto.count} imágenes similares."
                                )
                            search_triggered_automatically = True
                        except PipelineError as e:
                            st.error(
                                f"❌ Error en el pipeline de búsqueda automática por imagen: {e}"
                            )
                            logger.error(
                                f"PipelineError during automatic image search: {e}",
                                exc_info=True,
                            )
                        except Exception as e:
                            st.error(
                                f"❌ Error inesperado durante la búsqueda automática por imagen: {e}"
                            )
                            logger.error(
                                f"Unexpected error during automatic image search: {e}",
                                exc_info=True,
                            )

        elif query_image_path_auto:
            st.error(
                f"La imagen seleccionada para búsqueda automática ya no existe: {query_image_path_auto}"
            )
            logger.warning(
                f"Selected image for automatic search no longer exists: {query_image_path_auto}"
            )

            reset_image_selection_states(key_suffix)

        else:

            logger.warning(
                "Flag de activación detectado, pero no se encontró ruta de imagen seleccionada en el estado."
            )

    search_mode = st.radio(
        "Elige el origen de la imagen de consulta:",
        ("Subir una imagen nueva", "Seleccionar una imagen ya indexada"),
        key=f"image_search_mode{key_suffix}",
        horizontal=True,
        on_change=reset_image_selection_states,
        args=(key_suffix,),
    )

    num_results_img = st.slider(
        "Número máximo de resultados:",
        min_value=1,
        max_value=50,
        value=config.DEFAULT_N_RESULTS,
        key=f"num_results_img_slider{key_suffix}",
    )

    query_image_path_to_use: Optional[str] = None
    query_image_source_info: str = ""

    if search_mode == "Subir una imagen nueva":

        temp_path = _render_upload_widget(key_suffix=key_suffix)
        if temp_path:
            query_image_path_to_use = temp_path
            query_image_source_info = "Imagen Subida"
    elif search_mode == "Seleccionar una imagen ya indexada":

        selected_path = _render_indexed_selection(db, key_suffix=key_suffix)
        if selected_path:
            query_image_path_to_use = selected_path
            query_image_source_info = "Imagen Indexada Seleccionada"

    query_image_displayed = _display_query_image(
        query_image_path_to_use, query_image_source_info, key_suffix=key_suffix
    )

    if search_mode == "Subir una imagen nueva":

        search_button_disabled = not query_image_displayed
        if st.button(
            "🖼️ Buscar Imágenes Similares",
            key=f"search_image_button{key_suffix}",
            disabled=search_button_disabled,
            help=(
                "Sube una imagen primero para habilitar la búsqueda."
                if search_button_disabled
                else ""
            ),
        ):

            if not db or not db.is_initialized:
                st.warning(
                    "⚠️ La base de datos no está lista. Indexa imágenes primero."
                )
            elif db.count() <= 0:
                st.warning(
                    "⚠️ La base de datos está vacía. Indexa imágenes primero."
                )
            elif not vectorizer or not vectorizer.is_ready:
                st.error("❌ El vectorizador no está listo.")
            elif query_image_path_to_use:

                with results_container:
                    with st.spinner("🖼️ Buscando imágenes similares..."):
                        try:
                            logger.info(
                                f"Realizando búsqueda MANUAL por imagen usando: {query_image_path_to_use}"
                            )

                            results: Optional[SearchResults] = (
                                searching.search_by_image(
                                    query_image_path=query_image_path_to_use,
                                    vectorizer=vectorizer,
                                    db=db,
                                    n_results=num_results_img,
                                    truncate_dim=truncate_dim,
                                )
                            )

                            display_results(results, results_container)
                            if results:
                                logger.info(
                                    f"Búsqueda manual completada. Encontrados {results.count} imágenes similares."
                                )
                        except PipelineError as e:
                            st.error(
                                f"❌ Error en el pipeline de búsqueda por imagen: {e}"
                            )
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
            else:

                st.warning(
                    "⚠️ No hay una imagen de consulta subida para buscar."
                )

    elif search_mode == "Seleccionar una imagen ya indexada":

        if query_image_displayed and search_triggered_automatically:
            st.info("Resultados de la búsqueda automática mostrados arriba.")

        elif query_image_displayed and not search_triggered_automatically:
            st.info(
                "Imagen seleccionada. Si deseas buscar de nuevo, selecciona otra imagen de la lista."
            )

        elif not query_image_displayed:
            st.info(
                "Selecciona una imagen de la lista de abajo para buscar automáticamente."
            )
