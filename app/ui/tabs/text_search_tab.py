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

logger = logging.getLogger(__name__)


def render_text_search_tab(
    vectorizer: Vectorizer, db: VectorDBInterface, truncate_dim: Optional[int]
):
    """
    Renderiza el contenido y maneja la lógica para la pestaña 'Buscar por Texto'.

    Args:
        vectorizer: Instancia del Vectorizer inicializado.
        db: Instancia de la interfaz de BD vectorial inicializada.
        truncate_dim: Dimensión de truncamiento seleccionada (puede ser None).
    """
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
        "Número máximo de resultados:",
        min_value=1,
        max_value=50,
        value=config.DEFAULT_N_RESULTS,
        key="num_results_text_slider",
    )

    results_container_text = st.container()

    if st.button("🔎 Buscar por Texto", key="search_text_button"):

        if not query_text.strip():
            st.warning("⚠️ Por favor, introduce una descripción para buscar.")
        elif not db or not db.is_initialized:
            st.warning(
                "⚠️ La base de datos no está lista. Indexa imágenes primero."
            )
        elif db.count() <= 0:

            st.warning(
                "⚠️ La base de datos está vacía. Indexa imágenes primero."
            )
        elif not vectorizer or not vectorizer.is_ready:

            st.error("❌ El vectorizador no está listo.")
        else:

            with results_container_text:

                with st.spinner(
                    f"🧠 Buscando imágenes similares a: '{query_text}'..."
                ):
                    try:
                        logger.info(
                            f"Performing text search for: '{query_text}'"
                        )

                        results: Optional[SearchResults] = (
                            searching.search_by_text(
                                query_text=query_text,
                                vectorizer=vectorizer,
                                db=db,
                                n_results=num_results_text,
                                truncate_dim=truncate_dim,
                            )
                        )

                        display_results(results, results_container_text)
                        if results:
                            logger.info(
                                f"Text search completed. Found {results.count} results."
                            )

                    except PipelineError as e:

                        st.error(
                            f"❌ Error en el pipeline de búsqueda por texto: {e}"
                        )
                        logger.error(
                            f"PipelineError during text search via Streamlit: {e}",
                            exc_info=True,
                        )
                    except Exception as e:

                        st.error(
                            f"❌ Ocurrió un error inesperado durante la búsqueda por texto: {e}"
                        )
                        logger.error(
                            f"Unexpected error during text search via Streamlit: {e}",
                            exc_info=True,
                        )
