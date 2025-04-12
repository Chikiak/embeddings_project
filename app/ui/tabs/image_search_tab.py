# app/ui/tabs/image_search_tab.py
import streamlit as st
import os
import logging
from typing import Optional

# Importaciones relativas/absolutas correctas
import config # Acceso a configuraciones por defecto
# Componentes principales pasados como argumentos
from core.vectorizer import Vectorizer
from data_access.vector_db_interface import VectorDBInterface
# Módulo de lógica de negocio para búsqueda
from app import searching
# Excepciones relevantes
from app.exceptions import PipelineError
# Modelo de datos para resultados
from app.models import SearchResults
# Helpers de UI compartidos y específicos
from ..common import display_results
from ..widgets import _render_upload_widget, _render_indexed_selection, _display_query_image
# Utilidades de estado para manejar selección y flags
from ..state_utils import (
    get_state_key,
    reset_image_selection_states,
    STATE_UPLOADED_IMG_PATH_PREFIX,
    STATE_SELECTED_INDEXED_IMG_PREFIX,
    STATE_TRIGGER_SEARCH_FLAG_PREFIX,
)

logger = logging.getLogger(__name__)

def render_image_search_tab(vectorizer: Vectorizer, db: VectorDBInterface, truncate_dim: Optional[int]):
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

    # Sufijo único para las claves de estado de esta pestaña
    key_suffix = "_img_search"
    uploaded_state_key = get_state_key(STATE_UPLOADED_IMG_PATH_PREFIX, key_suffix)
    selected_state_key = get_state_key(STATE_SELECTED_INDEXED_IMG_PREFIX, key_suffix)
    trigger_search_key = get_state_key(STATE_TRIGGER_SEARCH_FLAG_PREFIX, key_suffix)

    # --- Inicializar estado si no existe ---
    # Es importante inicializar antes de usarlos, especialmente el flag
    if uploaded_state_key not in st.session_state:
        st.session_state[uploaded_state_key] = None
    if selected_state_key not in st.session_state:
        st.session_state[selected_state_key] = None
    if trigger_search_key not in st.session_state:
        st.session_state[trigger_search_key] = False

    # --- Contenedor para Mostrar Resultados ---
    # Se usará tanto para la búsqueda automática como la manual
    results_container = st.container()

    # --- Lógica de Búsqueda Automática (se ejecuta al inicio del renderizado de la pestaña) ---
    # Comprobar si el flag de activación está puesto
    search_triggered_automatically = False # Variable para rastrear si se ejecutó
    if st.session_state.get(trigger_search_key, False):
        logger.debug(f"Detectado flag de activación para búsqueda automática: {trigger_search_key}")
        # Resetear el flag INMEDIATAMENTE para evitar bucles
        st.session_state[trigger_search_key] = False
        # Obtener la ruta de la imagen seleccionada que activó el flag
        query_image_path_auto = st.session_state.get(selected_state_key)

        # Verificar si la ruta es válida y el archivo existe
        if query_image_path_auto and os.path.isfile(query_image_path_auto):
            # Validar estado de BD y vectorizador
            if not db or not db.is_initialized or db.count() <= 0:
                st.warning("⚠️ La base de datos no está lista o está vacía. No se puede buscar automáticamente.")
            elif not vectorizer or not vectorizer.is_ready:
                st.error("❌ El vectorizador no está listo para la búsqueda automática.")
            else:
                # Obtener el número de resultados del slider (usando su clave de estado)
                num_results_auto_key = f"num_results_img_slider{key_suffix}"
                num_results_auto = st.session_state.get(num_results_auto_key, config.DEFAULT_N_RESULTS)

                # Realizar la búsqueda automática
                with results_container: # Usar el contenedor principal
                    with st.spinner("🖼️ Buscando imágenes similares (automático)..."):
                        try:
                            logger.info(f"Realizando búsqueda AUTOMÁTICA por imagen usando: {query_image_path_auto}")
                            # Llamar a la función de lógica de negocio
                            results_auto: Optional[SearchResults] = searching.search_by_image(
                                query_image_path=query_image_path_auto,
                                vectorizer=vectorizer,
                                db=db,
                                n_results=num_results_auto,
                                truncate_dim=truncate_dim,
                            )
                            # Mostrar resultados en el contenedor principal
                            display_results(results_auto, results_container)
                            if results_auto:
                                logger.info(f"Búsqueda automática completada. Encontrados {results_auto.count} imágenes similares.")
                            search_triggered_automatically = True # Marcar que la búsqueda se ejecutó
                        except PipelineError as e:
                            st.error(f"❌ Error en el pipeline de búsqueda automática por imagen: {e}")
                            logger.error(f"PipelineError during automatic image search: {e}", exc_info=True)
                        except Exception as e:
                            st.error(f"❌ Error inesperado durante la búsqueda automática por imagen: {e}")
                            logger.error(f"Unexpected error during automatic image search: {e}", exc_info=True)
        # Manejar caso donde la ruta seleccionada ya no es válida
        elif query_image_path_auto:
             st.error(f"La imagen seleccionada para búsqueda automática ya no existe: {query_image_path_auto}")
             logger.warning(f"Selected image for automatic search no longer exists: {query_image_path_auto}")
             # Limpiar el estado inválido
             reset_image_selection_states(key_suffix)
             # No re-ejecutar aquí, permitir que la UI muestre el error
        else:
             # Caso raro donde el flag está activo pero no hay ruta seleccionada
             logger.warning("Flag de activación detectado, pero no se encontró ruta de imagen seleccionada en el estado.")
    # --- Fin Lógica de Búsqueda Automática ---


    # --- Renderizado de Widgets de UI ---
    # Radio button para elegir modo (subir vs seleccionar)
    # Importante: on_change resetea el estado para evitar conflictos entre modos
    search_mode = st.radio(
        "Elige el origen de la imagen de consulta:",
        ("Subir una imagen nueva", "Seleccionar una imagen ya indexada"),
        key=f"image_search_mode{key_suffix}", # Clave única
        horizontal=True, # Mostrar horizontalmente
        # Al cambiar de modo, llamar a reset_image_selection_states para limpiar
        on_change=reset_image_selection_states, args=(key_suffix,)
    )

    # Slider para número de resultados (común a ambos modos)
    num_results_img = st.slider(
        "Número máximo de resultados:",
        min_value=1, max_value=50,
        value=config.DEFAULT_N_RESULTS,
        key=f"num_results_img_slider{key_suffix}", # Guardar valor en estado
    )

    # Variables para guardar la ruta de la imagen a usar y su origen
    query_image_path_to_use: Optional[str] = None
    query_image_source_info: str = ""

    # Renderizar el widget apropiado según el modo seleccionado
    if search_mode == "Subir una imagen nueva":
        # Renderizar el widget de subida
        temp_path = _render_upload_widget(key_suffix=key_suffix)
        if temp_path:
            query_image_path_to_use = temp_path
            query_image_source_info = "Imagen Subida"
    elif search_mode == "Seleccionar una imagen ya indexada":
        # Renderizar la cuadrícula de selección de imágenes indexadas
        selected_path = _render_indexed_selection(db, key_suffix=key_suffix)
        if selected_path:
            query_image_path_to_use = selected_path
            query_image_source_info = "Imagen Indexada Seleccionada"

    # Mostrar la imagen de consulta elegida (si existe)
    query_image_displayed = _display_query_image(
        query_image_path_to_use, query_image_source_info, key_suffix=key_suffix
    )

    # --- Botón de Búsqueda Manual (Solo para modo 'Subir') ---
    if search_mode == "Subir una imagen nueva":
        # Habilitar botón solo si se ha subido y mostrado una imagen
        search_button_disabled = not query_image_displayed
        if st.button(
            "🖼️ Buscar Imágenes Similares",
            key=f"search_image_button{key_suffix}",
            disabled=search_button_disabled,
            # Texto de ayuda si está deshabilitado
            help="Sube una imagen primero para habilitar la búsqueda." if search_button_disabled else ""
        ):
            # Validaciones antes de la búsqueda manual
            if not db or not db.is_initialized:
                 st.warning("⚠️ La base de datos no está lista. Indexa imágenes primero.")
            elif db.count() <= 0:
                 st.warning("⚠️ La base de datos está vacía. Indexa imágenes primero.")
            elif not vectorizer or not vectorizer.is_ready:
                 st.error("❌ El vectorizador no está listo.")
            elif query_image_path_to_use: # Asegurarse de que hay una ruta válida
                # Realizar la búsqueda manual
                with results_container: # Usar el contenedor principal
                    with st.spinner("🖼️ Buscando imágenes similares..."):
                        try:
                            logger.info(
                                f"Realizando búsqueda MANUAL por imagen usando: {query_image_path_to_use}"
                            )
                            # Llamar a la función de lógica de negocio
                            results: Optional[SearchResults] = searching.search_by_image(
                                query_image_path=query_image_path_to_use,
                                vectorizer=vectorizer,
                                db=db,
                                n_results=num_results_img, # Usar valor actual del slider
                                truncate_dim=truncate_dim,
                            )
                            # Mostrar resultados
                            display_results(results, results_container)
                            if results:
                                logger.info(
                                    f"Búsqueda manual completada. Encontrados {results.count} imágenes similares."
                                )
                        except PipelineError as e:
                            st.error(f"❌ Error en el pipeline de búsqueda por imagen: {e}")
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
                 # Fallback si el botón se activa sin imagen (no debería pasar)
                 st.warning("⚠️ No hay una imagen de consulta subida para buscar.")

    # --- Feedback para modo 'Seleccionar' ---
    elif search_mode == "Seleccionar una imagen ya indexada":
        # Informar al usuario si la búsqueda automática ya se ejecutó
        if query_image_displayed and search_triggered_automatically:
            st.info("Resultados de la búsqueda automática mostrados arriba.")
        # Informar si hay una imagen seleccionada pero la búsqueda no se activó (estado raro)
        elif query_image_displayed and not search_triggered_automatically:
             st.info("Imagen seleccionada. Si deseas buscar de nuevo, selecciona otra imagen de la lista.")
        # Informar si no hay imagen seleccionada
        elif not query_image_displayed:
             st.info("Selecciona una imagen de la lista de abajo para buscar automáticamente.")
