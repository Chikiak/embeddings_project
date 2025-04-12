# --- app/ui/tabs/hybrid_search_tab.py ---
import logging
from typing import Optional

import streamlit as st
import time

import config
from app import searching
from app.exceptions import PipelineError
from app.models import SearchResults # Asegúrate de que SearchResults esté disponible
from core.vectorizer import Vectorizer
from data_access.vector_db_interface import VectorDBInterface

# Importaciones de UI comunes y widgets
from ..common import display_results
from ..state_utils import (
    STATE_HYBRID_METHOD,
    STATE_UPLOADED_IMG_PATH_PREFIX,
    get_state_key,
    reset_image_selection_states, # Usado por el widget de subida
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
    st.subheader("4. Búsqueda Híbrida (Texto + Imagen)") # Subheader
    st.markdown(
        """
        Combina una **descripción textual** y una **imagen de ejemplo** para encontrar
        resultados relevantes para ambos criterios. Útil para búsquedas más específicas.
        """
    )

    # Sufijo para claves de estado únicas para esta pestaña
    key_suffix = "_hybrid"
    # Claves de estado
    uploaded_state_key = get_state_key(
        STATE_UPLOADED_IMG_PATH_PREFIX, key_suffix
    )
    hybrid_method_key = STATE_HYBRID_METHOD # Clave global para el método

    # Inicializar claves en el estado de sesión si no existen
    if uploaded_state_key not in st.session_state:
        st.session_state[uploaded_state_key] = None
    # Método por defecto: Fusión RRF
    if hybrid_method_key not in st.session_state:
        st.session_state[hybrid_method_key] = "Fusión RRF"

    # Selección del método de búsqueda híbrida
    hybrid_method = st.radio(
        "Método de Búsqueda Híbrida:",
        options=["Fusión RRF", "Interpolar Embeddings"],
        key=hybrid_method_key, # Usa la clave de estado
        horizontal=True, # Opciones en horizontal
        help=(
            "**Fusión RRF:** Realiza búsquedas separadas para texto e imagen y combina los rankings "
            "(generalmente recomendado, ignora el peso Alpha).\n\n"
            "**Interpolar Embeddings:** Combina los vectores de texto e imagen *antes* de buscar "
            "(requiere ajustar el peso Alpha para balancear la influencia)."
        ),
    )

    st.divider() # Separador

    # Layout en dos columnas para inputs de texto/imagen
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Componente Textual:**")
        # Input para la consulta de texto
        hybrid_query_text = st.text_input(
            "Descripción Textual:",
            placeholder="Ej: persona sonriendo con gafas de sol",
            key=f"hybrid_text_query{key_suffix}", # Clave única
            help="Describe características adicionales o el contexto deseado."
        )

        st.divider() # Separador dentro de la columna

        st.markdown("**Configuración de Búsqueda:**")
        # Slider para el peso Alpha (solo relevante para 'Interpolar Embeddings')
        alpha_disabled = hybrid_method != "Interpolar Embeddings"
        hybrid_alpha = st.slider(
            "Peso Texto vs Imagen (Alpha):",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.get(f"hybrid_alpha{key_suffix}", 0.5), # Guarda/recupera valor
            step=0.05,
            key=f"hybrid_alpha{key_suffix}", # Clave única
            help="Solo para 'Interpolar Embeddings'. 0 = Solo Imagen, 0.5 = Equilibrado, 1 = Solo Texto.",
            disabled=alpha_disabled, # Deshabilitar si se usa RRF
        )
        if alpha_disabled:
            st.caption("ℹ️ El peso Alpha se ignora cuando se usa 'Fusión RRF'.")

        # Slider para el número de resultados
        num_results_hybrid = st.slider(
            "Número máximo de resultados:",
            min_value=1,
            max_value=50,
            value=config.DEFAULT_N_RESULTS,
            key=f"num_results_hybrid{key_suffix}", # Clave única
            help="Selecciona cuántas imágenes similares quieres ver."
        )

    with col2:
        st.markdown("**Componente Visual (Subir Imagen):**")
        # Widget para subir la imagen de ejemplo
        temp_path_hybrid = _render_upload_widget(key_suffix=key_suffix)
        query_image_path_hybrid = (
            temp_path_hybrid if temp_path_hybrid else None
        )

        st.divider() # Separador dentro de la columna

        # Muestra la imagen subida (si existe)
        st.markdown("**Imagen de Ejemplo Seleccionada:**")
        query_image_displayed_hybrid = _display_query_image(
            query_image_path_hybrid, "Imagen Subida para Híbrida", key_suffix=key_suffix
        )
        if not query_image_displayed_hybrid:
            st.caption("Sube una imagen para la parte visual de la búsqueda.")

    st.divider() # Separador antes del botón y resultados

    # Contenedor para los resultados
    results_container_hybrid = st.container()

    # Botón para iniciar la búsqueda híbrida
    # Habilitado solo si hay texto Y una imagen subida
    search_button_disabled_hybrid = not (
        hybrid_query_text.strip() and query_image_displayed_hybrid
    )
    if st.button(
        "🧬 Buscar Híbrido",
        key=f"search_hybrid_button{key_suffix}", # Clave única
        disabled=search_button_disabled_hybrid,
        type="primary", # Botón primario
        help=(
            "Introduce texto Y sube una imagen para activar la búsqueda."
            if search_button_disabled_hybrid
            else f"Realiza la búsqueda combinando texto e imagen usando el método '{hybrid_method}'."
        )
    ):
        # Validaciones de BD y vectorizador
        if not db or not db.is_initialized:
            st.warning(f"⚠️ La base de datos '{getattr(db, 'collection_name', 'N/A')}' no está lista.")
        elif db.count() <= 0:
            st.warning(f"⚠️ La base de datos '{getattr(db, 'collection_name', 'N/A')}' está vacía.")
        elif not vectorizer or not vectorizer.is_ready:
            st.error("❌ El vectorizador no está listo.")
        elif query_image_path_hybrid: # Asegurarse de que hay ruta de imagen
            # Realiza la búsqueda según el método seleccionado
            search_method_name = st.session_state[hybrid_method_key]

            with results_container_hybrid:
                with st.spinner(
                    f"🧬 Realizando búsqueda híbrida ({search_method_name})..."
                ):
                    try:
                        results: Optional[SearchResults] = None
                        start_time = time.time() # Medir tiempo

                        # Lógica según el método
                        if search_method_name == "Interpolar Embeddings":
                            logger.info(
                                f"Performing hybrid search (Interpolate): Text='{hybrid_query_text}', Image='{query_image_path_hybrid}', Alpha={hybrid_alpha}, Collection='{getattr(db, 'collection_name', 'N/A')}'"
                            )
                            results = searching.search_hybrid(
                                query_text=hybrid_query_text,
                                query_image_path=query_image_path_hybrid,
                                vectorizer=vectorizer,
                                db=db,
                                n_results=num_results_hybrid,
                                truncate_dim=truncate_dim, # Pasa dimensión
                                alpha=hybrid_alpha, # Pasa alpha
                            )
                        elif search_method_name == "Fusión RRF":
                            logger.info(
                                f"Performing hybrid search (RRF): Text='{hybrid_query_text}', Image='{query_image_path_hybrid}', Collection='{getattr(db, 'collection_name', 'N/A')}'"
                            )
                            results = searching.search_hybrid_rrf(
                                query_text=hybrid_query_text,
                                query_image_path=query_image_path_hybrid,
                                vectorizer=vectorizer,
                                db=db,
                                n_results=num_results_hybrid,
                                truncate_dim=truncate_dim, # Pasa dimensión
                                # k_rrf se usa por defecto en la función searching
                            )
                        else:
                            # Caso improbable si se añade otro método y no se maneja
                            st.error(
                                f"Método híbrido desconocido seleccionado: {search_method_name}"
                            )

                        end_time = time.time() # Fin medición tiempo

                        # Muestra resultados si la búsqueda fue exitosa
                        if results is not None:
                            display_results(results, results_container_hybrid)
                            logger.info(
                                f"Hybrid search ({search_method_name}) completed in {end_time - start_time:.2f}s. Found {results.count} results."
                            )
                        else:
                            # Si la función de búsqueda devolvió None (error interno)
                             logger.error(f"Hybrid search function ({search_method_name}) returned None.")
                             st.error("❌ La búsqueda híbrida falló internamente.")

                    except PipelineError as e:
                        # Error controlado del pipeline
                        st.error(
                            f"❌ Error en pipeline de búsqueda híbrida ({search_method_name}): {e}"
                        )
                        logger.error(
                            f"PipelineError during hybrid search ({search_method_name}) via Streamlit: {e}",
                            exc_info=True,
                        )
                    except Exception as e:
                        # Error inesperado
                        st.error(
                            f"❌ Error inesperado en búsqueda híbrida ({search_method_name}): {e}"
                        )
                        logger.error(
                            f"Unexpected error during hybrid search ({search_method_name}) via Streamlit: {e}",
                            exc_info=True,
                        )
        else:
            # Esto no debería ocurrir si el botón está bien deshabilitado
            st.warning(
                "⚠️ No se proporcionó texto o imagen para la búsqueda híbrida."
            )

    # Mensaje si el botón está deshabilitado
    elif search_button_disabled_hybrid:
        st.info(
            "ℹ️ Introduce texto en la descripción Y sube una imagen de ejemplo para activar la búsqueda híbrida."
        )

