# app/ui/tabs/hybrid_search_tab.py
import streamlit as st
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
from ..widgets import _render_upload_widget, _display_query_image
# Utilidades de estado para manejar selección y configuración
from ..state_utils import (
    get_state_key,
    STATE_UPLOADED_IMG_PATH_PREFIX,
    STATE_HYBRID_METHOD, # Clave para el método híbrido
    reset_image_selection_states # Para limpiar subida si falla
)

logger = logging.getLogger(__name__)

def render_hybrid_search_tab(vectorizer: Vectorizer, db: VectorDBInterface, truncate_dim: Optional[int]):
    """
    Renderiza el contenido y maneja la lógica para la pestaña 'Búsqueda Híbrida'.

    Args:
        vectorizer: Instancia del Vectorizer inicializado.
        db: Instancia de la interfaz de BD vectorial inicializada.
        truncate_dim: Dimensión de truncamiento seleccionada (puede ser None).
    """
    st.header("4. Búsqueda Híbrida (Texto + Imagen)")
    st.markdown("Combina una descripción textual y una imagen de ejemplo para encontrar resultados relevantes para ambos.")

    # Sufijo único para las claves de estado de esta pestaña
    key_suffix = "_hybrid"
    uploaded_state_key = get_state_key(STATE_UPLOADED_IMG_PATH_PREFIX, key_suffix)
    hybrid_method_key = STATE_HYBRID_METHOD # Clave definida en state_utils

    # --- Inicializar estado si no existe ---
    if uploaded_state_key not in st.session_state:
        st.session_state[uploaded_state_key] = None
    # Establecer método híbrido por defecto si no está en el estado
    if hybrid_method_key not in st.session_state:
        # Default a RRF ya que generalmente funciona bien sin ajuste de alpha
        st.session_state[hybrid_method_key] = "Fusión RRF"

    # --- Selección del Método Híbrido ---
    hybrid_method = st.radio(
        "Método de Búsqueda Híbrida:",
        # Opciones disponibles
        options=["Fusión RRF", "Interpolar Embeddings"],
        key=hybrid_method_key, # Guardar selección en estado
        horizontal=True, # Mostrar horizontalmente
        help=(
             "**Fusión RRF:** Realiza búsquedas separadas para texto e imagen y combina los rankings (recomendado, ignora Alpha).\n\n"
             "**Interpolar Embeddings:** Combina los vectores de texto e imagen ANTES de buscar (requiere ajustar Alpha)."
        )
    )

    # --- Layout de Inputs (Texto e Imagen) ---
    col1, col2 = st.columns(2) # Dos columnas para organizar inputs

    with col1:
        # Input para consulta textual
        hybrid_query_text = st.text_input(
            "Descripción Textual:",
            placeholder="Ej: persona sonriendo con gafas de sol",
            key=f"hybrid_text_query{key_suffix}" # Clave única
        )

        # Slider para Alpha (solo relevante para 'Interpolar Embeddings')
        alpha_disabled = (hybrid_method != "Interpolar Embeddings")
        # Mostrar slider solo si el método es Interpolar
        if not alpha_disabled:
            hybrid_alpha = st.slider(
                "Peso Texto vs Imagen (Alpha):",
                min_value=0.0,
                max_value=1.0,
                value=0.5, # Valor inicial por defecto
                step=0.05, # Pasos del slider
                key=f"hybrid_alpha{key_suffix}", # Clave única
                help="Controla la influencia de cada modalidad (0=Solo Imagen, 1=Solo Texto). Solo para 'Interpolar Embeddings'.",
                disabled=alpha_disabled # Deshabilitado si no es Interpolar
            )
        else:
             # Si es RRF, obtener el valor de alpha del estado (o default) pero no mostrar slider
             hybrid_alpha = st.session_state.get(f"hybrid_alpha{key_suffix}", 0.5)
             st.caption("El deslizador Alpha se ignora cuando se usa Fusión RRF.")

        # Slider para número de resultados
        num_results_hybrid = st.slider(
            "Número máximo de resultados:",
            min_value=1,
            max_value=50,
            value=config.DEFAULT_N_RESULTS, # Default desde config
            key=f"num_results_hybrid{key_suffix}" # Clave única
        )

    with col2:
        # Sección para subir la imagen de ejemplo
        st.subheader("Imagen de Ejemplo (Subir):")
        # Usar el widget de subida reutilizable
        temp_path_hybrid = _render_upload_widget(key_suffix=key_suffix)
        query_image_path_hybrid = temp_path_hybrid if temp_path_hybrid else None

        # Mostrar la imagen subida usando el widget reutilizable
        query_image_displayed_hybrid = _display_query_image(
            query_image_path_hybrid, "Imagen Subida", key_suffix=key_suffix
        )

    # --- Contenedor para Resultados ---
    results_container_hybrid = st.container()

    # --- Botón de Búsqueda Híbrida y Lógica ---
    # Habilitar botón solo si hay texto Y una imagen subida válida
    search_button_disabled_hybrid = not (hybrid_query_text.strip() and query_image_displayed_hybrid)
    if st.button("🧬 Buscar Híbrido", key=f"search_hybrid_button{key_suffix}", disabled=search_button_disabled_hybrid):
        # Validaciones antes de buscar
        if not db or not db.is_initialized:
             st.warning("⚠️ La base de datos no está lista.")
        elif db.count() <= 0:
             st.warning("⚠️ La base de datos está vacía.")
        elif not vectorizer or not vectorizer.is_ready:
             st.error("❌ El vectorizador no está listo.")
        elif query_image_path_hybrid: # Asegurarse de que la ruta de imagen es válida
             # Obtener método seleccionado del estado
             search_method_name = st.session_state[hybrid_method_key]
             # Realizar búsqueda dentro del contenedor de resultados
             with results_container_hybrid:
                 with st.spinner(f"🧬 Realizando búsqueda híbrida ({search_method_name})..."):
                      try:
                           results: Optional[SearchResults] = None
                           # Ejecutar la lógica de búsqueda según el método seleccionado
                           if search_method_name == "Interpolar Embeddings":
                                logger.info(f"Performing hybrid search (Interpolate): Text='{hybrid_query_text}', Image='{query_image_path_hybrid}', Alpha={hybrid_alpha}")
                                # Llamar a la función de búsqueda híbrida interpolada
                                results = searching.search_hybrid(
                                     query_text=hybrid_query_text,
                                     query_image_path=query_image_path_hybrid,
                                     vectorizer=vectorizer,
                                     db=db,
                                     n_results=num_results_hybrid,
                                     truncate_dim=truncate_dim,
                                     alpha=hybrid_alpha # Pasar el valor de alpha
                                )
                           elif search_method_name == "Fusión RRF":
                                logger.info(f"Performing hybrid search (RRF): Text='{hybrid_query_text}', Image='{query_image_path_hybrid}'")
                                # Llamar a la función de búsqueda híbrida RRF
                                results = searching.search_hybrid_rrf(
                                     query_text=hybrid_query_text,
                                     query_image_path=query_image_path_hybrid,
                                     vectorizer=vectorizer,
                                     db=db,
                                     n_results=num_results_hybrid,
                                     truncate_dim=truncate_dim
                                     # k_rrf usa el valor por defecto definido en searching.py
                                )
                           else:
                                # Manejar caso de método desconocido (no debería ocurrir)
                                st.error(f"Método híbrido desconocido seleccionado: {search_method_name}")

                           # Mostrar resultados si la búsqueda se realizó
                           if results is not None:
                               display_results(results, results_container_hybrid)
                               logger.info(f"Hybrid search ({search_method_name}) completed. Found {results.count} results.")

                      except PipelineError as e:
                           # Manejar errores específicos del pipeline híbrido
                           st.error(f"❌ Error en pipeline de búsqueda híbrida ({search_method_name}): {e}")
                           logger.error(f"PipelineError during hybrid search ({search_method_name}) via Streamlit: {e}", exc_info=True)
                      except Exception as e:
                           # Manejar otros errores inesperados
                           st.error(f"❌ Error inesperado en búsqueda híbrida ({search_method_name}): {e}")
                           logger.error(f"Unexpected error during hybrid search ({search_method_name}) via Streamlit: {e}", exc_info=True)
        else:
             # Fallback si el botón se activa sin imagen (no debería pasar)
             st.warning("⚠️ No se proporcionó texto o imagen para la búsqueda híbrida.")

    # Mensaje de ayuda si el botón está deshabilitado
    elif search_button_disabled_hybrid:
         st.info("ℹ️ Introduce texto Y sube una imagen válida para activar la búsqueda híbrida.")
