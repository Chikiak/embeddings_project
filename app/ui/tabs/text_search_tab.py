# app/ui/tabs/text_search_tab.py
import streamlit as st
import logging
from typing import Optional

# Importaciones relativas/absolutas correctas
import config # Acceso a configuraciones por defecto (ej: DEFAULT_N_RESULTS)
# Componentes principales pasados como argumentos
from core.vectorizer import Vectorizer
from data_access.vector_db_interface import VectorDBInterface
# Módulo de lógica de negocio para búsqueda
from app import searching
# Excepciones relevantes
from app.exceptions import PipelineError
# Modelo de datos para resultados
from app.models import SearchResults
# Helper de UI compartido para mostrar resultados
from ..common import display_results

logger = logging.getLogger(__name__)

def render_text_search_tab(vectorizer: Vectorizer, db: VectorDBInterface, truncate_dim: Optional[int]):
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

    # --- Input para Consulta de Texto ---
    query_text = st.text_input(
        "Descripción de la imagen:",
        placeholder="Ej: gato durmiendo sobre un teclado",
        key="text_query_input", # Clave única para este input
    )

    # --- Slider para Número de Resultados ---
    num_results_text = st.slider(
        "Número máximo de resultados:",
        min_value=1,
        max_value=50, # Límite superior razonable
        value=config.DEFAULT_N_RESULTS, # Valor por defecto desde config
        key="num_results_text_slider", # Clave única para este slider
    )

    # --- Contenedor para Mostrar Resultados ---
    # Crear un contenedor vacío que se llenará cuando se realice la búsqueda
    results_container_text = st.container()

    # --- Botón de Búsqueda y Lógica Asociada ---
    if st.button("🔎 Buscar por Texto", key="search_text_button"):
        # Validaciones antes de buscar
        if not query_text.strip():
            st.warning("⚠️ Por favor, introduce una descripción para buscar.")
        elif not db or not db.is_initialized:
            st.warning("⚠️ La base de datos no está lista. Indexa imágenes primero.")
        elif db.count() <= 0:
             # Comprobar si la BD está vacía
             st.warning("⚠️ La base de datos está vacía. Indexa imágenes primero.")
        elif not vectorizer or not vectorizer.is_ready:
             # Comprobar si el vectorizador está listo
             st.error("❌ El vectorizador no está listo.")
        else:
            # Si todo está bien, realizar la búsqueda
            # Usar el contenedor de resultados definido previamente
            with results_container_text:
                # Mostrar spinner mientras se busca
                with st.spinner(f"🧠 Buscando imágenes similares a: '{query_text}'..."):
                    try:
                        logger.info(f"Performing text search for: '{query_text}'")
                        # Llamar a la función de lógica de negocio para buscar por texto
                        results: Optional[SearchResults] = searching.search_by_text(
                            query_text=query_text,
                            vectorizer=vectorizer,
                            db=db,
                            n_results=num_results_text, # Usar valor del slider
                            truncate_dim=truncate_dim, # Usar valor global
                        )
                        # Mostrar los resultados usando el helper común
                        display_results(results, results_container_text)
                        if results:
                            logger.info(
                                f"Text search completed. Found {results.count} results."
                            )

                    except PipelineError as e:
                        # Manejar errores específicos del pipeline
                        st.error(f"❌ Error en el pipeline de búsqueda por texto: {e}")
                        logger.error(
                            f"PipelineError during text search via Streamlit: {e}",
                            exc_info=True,
                        )
                    except Exception as e:
                        # Manejar otros errores inesperados
                        st.error(
                            f"❌ Ocurrió un error inesperado durante la búsqueda por texto: {e}"
                        )
                        logger.error(
                            f"Unexpected error during text search via Streamlit: {e}",
                            exc_info=True,
                        )
