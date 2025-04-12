import logging
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
    STATE_HYBRID_METHOD,
    STATE_UPLOADED_IMG_PATH_PREFIX,
    get_state_key,
    reset_image_selection_states,
)
from ..widgets import _display_query_image, _render_upload_widget

logger = logging.getLogger(__name__)


def render_hybrid_search_tab(
    vectorizer: Vectorizer, db: VectorDBInterface, truncate_dim: Optional[int]
):
    """
    Renderiza el contenido y maneja la lógica para la pestaña 'Búsqueda Híbrida'.

    Args:
        vectorizer: Instancia del Vectorizer inicializado.
        db: Instancia de la interfaz de BD vectorial inicializada.
        truncate_dim: Dimensión de truncamiento seleccionada (puede ser None).
    """
    st.header("4. Búsqueda Híbrida (Texto + Imagen)")
    st.markdown(
        "Combina una descripción textual y una imagen de ejemplo para encontrar resultados relevantes para ambos."
    )

    key_suffix = "_hybrid"
    uploaded_state_key = get_state_key(
        STATE_UPLOADED_IMG_PATH_PREFIX, key_suffix
    )
    hybrid_method_key = STATE_HYBRID_METHOD

    if uploaded_state_key not in st.session_state:
        st.session_state[uploaded_state_key] = None

    if hybrid_method_key not in st.session_state:

        st.session_state[hybrid_method_key] = "Fusión RRF"

    hybrid_method = st.radio(
        "Método de Búsqueda Híbrida:",
        options=["Fusión RRF", "Interpolar Embeddings"],
        key=hybrid_method_key,
        horizontal=True,
        help=(
            "**Fusión RRF:** Realiza búsquedas separadas para texto e imagen y combina los rankings (recomendado, ignora Alpha).\n\n"
            "**Interpolar Embeddings:** Combina los vectores de texto e imagen ANTES de buscar (requiere ajustar Alpha)."
        ),
    )

    col1, col2 = st.columns(2)

    with col1:

        hybrid_query_text = st.text_input(
            "Descripción Textual:",
            placeholder="Ej: persona sonriendo con gafas de sol",
            key=f"hybrid_text_query{key_suffix}",
        )

        alpha_disabled = hybrid_method != "Interpolar Embeddings"

        if not alpha_disabled:
            hybrid_alpha = st.slider(
                "Peso Texto vs Imagen (Alpha):",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                key=f"hybrid_alpha{key_suffix}",
                help="Controla la influencia de cada modalidad (0=Solo Imagen, 1=Solo Texto). Solo para 'Interpolar Embeddings'.",
                disabled=alpha_disabled,
            )
        else:

            hybrid_alpha = st.session_state.get(
                f"hybrid_alpha{key_suffix}", 0.5
            )
            st.caption(
                "El deslizador Alpha se ignora cuando se usa Fusión RRF."
            )

        num_results_hybrid = st.slider(
            "Número máximo de resultados:",
            min_value=1,
            max_value=50,
            value=config.DEFAULT_N_RESULTS,
            key=f"num_results_hybrid{key_suffix}",
        )

    with col2:

        st.subheader("Imagen de Ejemplo (Subir):")

        temp_path_hybrid = _render_upload_widget(key_suffix=key_suffix)
        query_image_path_hybrid = (
            temp_path_hybrid if temp_path_hybrid else None
        )

        query_image_displayed_hybrid = _display_query_image(
            query_image_path_hybrid, "Imagen Subida", key_suffix=key_suffix
        )

    results_container_hybrid = st.container()

    search_button_disabled_hybrid = not (
        hybrid_query_text.strip() and query_image_displayed_hybrid
    )
    if st.button(
        "🧬 Buscar Híbrido",
        key=f"search_hybrid_button{key_suffix}",
        disabled=search_button_disabled_hybrid,
    ):

        if not db or not db.is_initialized:
            st.warning("⚠️ La base de datos no está lista.")
        elif db.count() <= 0:
            st.warning("⚠️ La base de datos está vacía.")
        elif not vectorizer or not vectorizer.is_ready:
            st.error("❌ El vectorizador no está listo.")
        elif query_image_path_hybrid:

            search_method_name = st.session_state[hybrid_method_key]

            with results_container_hybrid:
                with st.spinner(
                    f"🧬 Realizando búsqueda híbrida ({search_method_name})..."
                ):
                    try:
                        results: Optional[SearchResults] = None

                        if search_method_name == "Interpolar Embeddings":
                            logger.info(
                                f"Performing hybrid search (Interpolate): Text='{hybrid_query_text}', Image='{query_image_path_hybrid}', Alpha={hybrid_alpha}"
                            )

                            results = searching.search_hybrid(
                                query_text=hybrid_query_text,
                                query_image_path=query_image_path_hybrid,
                                vectorizer=vectorizer,
                                db=db,
                                n_results=num_results_hybrid,
                                truncate_dim=truncate_dim,
                                alpha=hybrid_alpha,
                            )
                        elif search_method_name == "Fusión RRF":
                            logger.info(
                                f"Performing hybrid search (RRF): Text='{hybrid_query_text}', Image='{query_image_path_hybrid}'"
                            )

                            results = searching.search_hybrid_rrf(
                                query_text=hybrid_query_text,
                                query_image_path=query_image_path_hybrid,
                                vectorizer=vectorizer,
                                db=db,
                                n_results=num_results_hybrid,
                                truncate_dim=truncate_dim,
                            )
                        else:

                            st.error(
                                f"Método híbrido desconocido seleccionado: {search_method_name}"
                            )

                        if results is not None:
                            display_results(results, results_container_hybrid)
                            logger.info(
                                f"Hybrid search ({search_method_name}) completed. Found {results.count} results."
                            )

                    except PipelineError as e:

                        st.error(
                            f"❌ Error en pipeline de búsqueda híbrida ({search_method_name}): {e}"
                        )
                        logger.error(
                            f"PipelineError during hybrid search ({search_method_name}) via Streamlit: {e}",
                            exc_info=True,
                        )
                    except Exception as e:

                        st.error(
                            f"❌ Error inesperado en búsqueda híbrida ({search_method_name}): {e}"
                        )
                        logger.error(
                            f"Unexpected error during hybrid search ({search_method_name}) via Streamlit: {e}",
                            exc_info=True,
                        )
        else:

            st.warning(
                "⚠️ No se proporcionó texto o imagen para la búsqueda híbrida."
            )

    elif search_button_disabled_hybrid:
        st.info(
            "ℹ️ Introduce texto Y sube una imagen válida para activar la búsqueda híbrida."
        )
