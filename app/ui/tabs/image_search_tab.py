# --- app/ui/tabs/image_search_tab.py ---
import logging
import os
from typing import Optional

import streamlit as st

import config
from app import searching
from app.exceptions import PipelineError
from app.models import SearchResults # Asegúrate de que SearchResults esté disponible
from core.vectorizer import Vectorizer
from data_access.vector_db_interface import VectorDBInterface

# Importaciones de UI comunes y widgets
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
    st.subheader("3. Buscar Imágenes por Similitud Visual") # Subheader
    st.markdown(
        """
        Encuentra imágenes visualmente parecidas a una imagen de referencia.
        Puedes **subir una imagen nueva** desde tu dispositivo o
        **seleccionar una imagen que ya esté indexada** en la base de datos activa.
        """
    )

    # Sufijo para claves de estado únicas para esta pestaña
    key_suffix = "_img_search"
    # Obtener claves de estado usando la función helper
    uploaded_state_key = get_state_key(
        STATE_UPLOADED_IMG_PATH_PREFIX, key_suffix
    )
    selected_state_key = get_state_key(
        STATE_SELECTED_INDEXED_IMG_PREFIX, key_suffix
    )
    trigger_search_key = get_state_key(
        STATE_TRIGGER_SEARCH_FLAG_PREFIX, key_suffix
    )

    # Inicializar claves en el estado de sesión si no existen
    if uploaded_state_key not in st.session_state:
        st.session_state[uploaded_state_key] = None
    if selected_state_key not in st.session_state:
        st.session_state[selected_state_key] = None
    if trigger_search_key not in st.session_state:
        st.session_state[trigger_search_key] = False

    # Contenedor principal para los resultados
    results_container = st.container()

    # --- Lógica para Búsqueda Automática ---
    # Se activa si se seleccionó una imagen indexada en el ciclo anterior
    search_triggered_automatically = False
    if st.session_state.get(trigger_search_key, False):
        logger.debug(
            f"Detectado flag de activación para búsqueda automática: {trigger_search_key}"
        )
        # Resetea el flag para evitar búsquedas repetidas
        st.session_state[trigger_search_key] = False

        # Obtiene la ruta de la imagen seleccionada del estado
        query_image_path_auto = st.session_state.get(selected_state_key)

        # Verifica que la ruta sea válida y el archivo exista
        if query_image_path_auto and os.path.isfile(query_image_path_auto):
            # Validaciones de BD y vectorizador
            if not db or not db.is_initialized or db.count() <= 0:
                st.warning(
                    f"⚠️ La base de datos '{getattr(db, 'collection_name', 'N/A')}' no está lista o está vacía. No se puede buscar automáticamente."
                )
            elif not vectorizer or not vectorizer.is_ready:
                st.error(
                    "❌ El vectorizador no está listo para la búsqueda automática."
                )
            else:
                # Obtiene el número de resultados del slider (o usa default)
                num_results_auto_key = f"num_results_img_slider{key_suffix}"
                num_results_auto = st.session_state.get(
                    num_results_auto_key, config.DEFAULT_N_RESULTS
                )

                # Realiza la búsqueda automática
                with results_container:
                    with st.spinner(
                        f"🖼️ Buscando imágenes similares a '{os.path.basename(query_image_path_auto)}' (automático)..."
                    ):
                        try:
                            logger.info(
                                f"Realizando búsqueda AUTOMÁTICA por imagen usando: {query_image_path_auto} en '{getattr(db, 'collection_name', 'N/A')}'"
                            )
                            # Llamada a la función de búsqueda del backend
                            results_auto: Optional[SearchResults] = (
                                searching.search_by_image(
                                    query_image_path=query_image_path_auto,
                                    vectorizer=vectorizer,
                                    db=db,
                                    n_results=num_results_auto,
                                    truncate_dim=truncate_dim, # Pasa la dimensión
                                )
                            )
                            # Muestra los resultados
                            display_results(results_auto, results_container)
                            if results_auto:
                                logger.info(
                                    f"Búsqueda automática completada. Encontrados {results_auto.count} imágenes similares."
                                )
                            search_triggered_automatically = True # Marca que se hizo búsqueda auto
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
            # Si la ruta estaba en el estado pero el archivo ya no existe
            st.error(
                f"La imagen seleccionada para búsqueda automática ya no existe: {query_image_path_auto}"
            )
            logger.warning(
                f"Selected image for automatic search no longer exists: {query_image_path_auto}"
            )
            # Limpia los estados relacionados
            reset_image_selection_states(key_suffix)
        else:
            # Si el flag estaba activo pero no había ruta
            logger.warning(
                "Flag de activación detectado, pero no se encontró ruta de imagen seleccionada en el estado."
            )
    # --- Fin Lógica para Búsqueda Automática ---

    st.divider() # Separador

    # Selección del modo de búsqueda (Subir vs Seleccionar)
    search_mode = st.radio(
        "Elige el origen de la imagen de consulta:",
        ("Subir una imagen nueva", "Seleccionar una imagen ya indexada"),
        key=f"image_search_mode{key_suffix}", # Clave única
        horizontal=True, # Mostrar opciones en horizontal
        on_change=reset_image_selection_states, # Limpia estados si cambia el modo
        args=(key_suffix,), # Pasa el sufijo a la función on_change
        help="Elige si quieres buscar usando una imagen de tu equipo o una ya existente en la BD."
    )

    # Slider para número de resultados (común a ambos modos)
    num_results_img = st.slider(
        "Número máximo de resultados:",
        min_value=1,
        max_value=50,
        value=config.DEFAULT_N_RESULTS,
        key=f"num_results_img_slider{key_suffix}", # Clave única
        help="Selecciona cuántas imágenes similares quieres ver."
    )

    st.divider() # Separador

    query_image_path_to_use: Optional[str] = None
    query_image_source_info: str = ""

    # Renderiza el widget correspondiente según el modo seleccionado
    if search_mode == "Subir una imagen nueva":
        st.markdown("**Sube tu imagen de referencia:**")
        # Llama al widget de subida
        temp_path = _render_upload_widget(key_suffix=key_suffix)
        if temp_path:
            query_image_path_to_use = temp_path
            query_image_source_info = "Imagen Subida"
    elif search_mode == "Seleccionar una imagen ya indexada":
        st.markdown("**Selecciona una imagen de la base de datos:**")
        # Llama al widget de selección de imágenes indexadas
        selected_path = _render_indexed_selection(db, key_suffix=key_suffix)
        if selected_path:
            query_image_path_to_use = selected_path
            query_image_source_info = "Imagen Indexada Seleccionada"

    # Muestra la imagen de consulta (si hay alguna seleccionada/subida)
    query_image_displayed = _display_query_image(
        query_image_path_to_use, query_image_source_info, key_suffix=key_suffix
    )

    st.divider() # Separador

    # --- Lógica para Búsqueda Manual (solo si se subió imagen) ---
    if search_mode == "Subir una imagen nueva":
        # Habilita el botón solo si se ha subido y mostrado una imagen
        search_button_disabled = not query_image_displayed
        if st.button(
            "🖼️ Buscar Imágenes Similares",
            key=f"search_image_button{key_suffix}", # Clave única
            disabled=search_button_disabled,
            type="primary", # Botón primario
            help=(
                "Sube una imagen primero para habilitar la búsqueda."
                if search_button_disabled
                else "Busca imágenes visualmente similares a la imagen subida."
            ),
        ):
            # Validaciones antes de buscar
            if not db or not db.is_initialized:
                st.warning(
                    f"⚠️ La base de datos '{getattr(db, 'collection_name', 'N/A')}' no está lista. Indexa imágenes primero."
                )
            elif db.count() <= 0:
                st.warning(
                    f"⚠️ La base de datos '{getattr(db, 'collection_name', 'N/A')}' está vacía. Indexa imágenes primero."
                )
            elif not vectorizer or not vectorizer.is_ready:
                st.error("❌ El vectorizador no está listo.")
            elif query_image_path_to_use: # Asegurarse de que hay una ruta
                # Realiza la búsqueda manual
                with results_container:
                    with st.spinner(f"🖼️ Buscando imágenes similares a '{os.path.basename(query_image_path_to_use)}'..."):
                        try:
                            logger.info(
                                f"Realizando búsqueda MANUAL por imagen usando: {query_image_path_to_use} en '{getattr(db, 'collection_name', 'N/A')}'"
                            )
                            # Llamada a la función de búsqueda del backend
                            results: Optional[SearchResults] = (
                                searching.search_by_image(
                                    query_image_path=query_image_path_to_use,
                                    vectorizer=vectorizer,
                                    db=db,
                                    n_results=num_results_img,
                                    truncate_dim=truncate_dim, # Pasa la dimensión
                                )
                            )
                            # Muestra los resultados
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
                # Esto no debería ocurrir si el botón está bien deshabilitado
                st.warning(
                    "⚠️ No hay una imagen de consulta subida para buscar."
                )

    # --- Mensajes informativos para el modo de selección indexada ---
    elif search_mode == "Seleccionar una imagen ya indexada":
        if query_image_displayed and search_triggered_automatically:
            # Si se mostró la imagen y ya se hizo la búsqueda automática
            st.success("✅ Resultados de la búsqueda automática mostrados arriba.")
        elif query_image_displayed and not search_triggered_automatically:
            # Si se mostró la imagen pero no se activó la búsqueda (ej: recarga de página)
            st.info(
                "ℹ️ Imagen seleccionada. La búsqueda se realiza automáticamente al hacer clic en 'Usar esta'. Si deseas buscar de nuevo, selecciona otra imagen de la lista."
            )
        elif not query_image_displayed:
            # Si no hay ninguna imagen seleccionada
            st.info(
                "👇 Selecciona una imagen de la lista de abajo para buscar automáticamente imágenes similares."
            )
