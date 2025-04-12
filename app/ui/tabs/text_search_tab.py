# --- app/ui/tabs/text_search_tab.py ---
import logging
from typing import Optional

import streamlit as st

import config
from app import searching
from app.exceptions import PipelineError
from app.models import SearchResults # Asegúrate de que SearchResults esté disponible
from core.vectorizer import Vectorizer
from data_access.vector_db_interface import VectorDBInterface

# Importa la función común para mostrar resultados
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
    st.subheader("2. Buscar Imágenes por Descripción Textual") # Subheader
    st.markdown(
        """
        Escribe una descripción de la imagen que buscas en lenguaje natural.
        Por ejemplo: *'perro jugando en la playa con una pelota roja'*,
        *'atardecer sobre montañas nevadas'*, *'logotipo abstracto azul y naranja'*.
        """
    )

    # Input para la consulta de texto
    query_text = st.text_input(
        "Descripción de la imagen:",
        placeholder="Ej: gato durmiendo sobre un teclado",
        key="text_query_input", # Clave única para este input
        help="Introduce tu consulta textual aquí."
    )

    # Slider para el número de resultados
    num_results_text = st.slider(
        "Número máximo de resultados:",
        min_value=1,
        max_value=50, # Límite razonable
        value=config.DEFAULT_N_RESULTS, # Valor por defecto desde config
        key="num_results_text_slider", # Clave única
        help="Selecciona cuántas imágenes similares quieres ver."
    )

    st.divider() # Separador

    # Contenedor para mostrar los resultados
    results_container_text = st.container()

    # Botón para iniciar la búsqueda
    if st.button("📝 Buscar por Texto", key="search_text_button", type="primary"):

        # Validaciones antes de buscar
        if not query_text.strip():
            st.warning("⚠️ Por favor, introduce una descripción para buscar.")
        elif not db or not db.is_initialized:
            st.warning(
                f"⚠️ La base de datos '{getattr(db, 'collection_name', 'N/A')}' no está lista. Indexa imágenes primero."
            )
        elif db.count() <= 0:
            st.warning(
                f"⚠️ La base de datos '{getattr(db, 'collection_name', 'N/A')}' está vacía. Indexa imágenes primero."
            )
        elif not vectorizer or not vectorizer.is_ready:
            st.error("❌ El vectorizador no está listo. No se puede procesar la consulta.")
        else:
            # Si todo está bien, procede con la búsqueda
            with results_container_text:
                # Mostrar spinner mientras se busca
                with st.spinner(
                    f"🧠 Buscando imágenes similares a: '{query_text}'..."
                ):
                    try:
                        logger.info(
                            f"Performing text search for: '{query_text}' in collection '{getattr(db, 'collection_name', 'N/A')}'"
                        )
                        # Llamada a la función de búsqueda del backend
                        results: Optional[SearchResults] = (
                            searching.search_by_text(
                                query_text=query_text,
                                vectorizer=vectorizer,
                                db=db,
                                n_results=num_results_text,
                                truncate_dim=truncate_dim, # Pasa la dimensión
                            )
                        )

                        # Muestra los resultados usando la función común
                        display_results(results, results_container_text)

                        if results:
                            logger.info(
                                f"Text search completed. Found {results.count} results."
                            )
                        else:
                             logger.warning("Text search returned None.")


                    except PipelineError as e:
                        # Error controlado del pipeline
                        st.error(
                            f"❌ Error en el pipeline de búsqueda por texto: {e}"
                        )
                        logger.error(
                            f"PipelineError during text search via Streamlit: {e}",
                            exc_info=True, # Muestra traceback en logs
                        )
                    except Exception as e:
                        # Error inesperado
                        st.error(
                            f"❌ Ocurrió un error inesperado durante la búsqueda por texto: {e}"
                        )
                        logger.error(
                            f"Unexpected error during text search via Streamlit: {e}",
                            exc_info=True, # Muestra traceback en logs
                        )
