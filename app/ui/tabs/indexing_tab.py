# --- app/ui/tabs/indexing_tab.py ---
import logging
import os
import time
from typing import Optional

import streamlit as st

import config
from ...services import indexing
from app.exceptions import DatabaseError, PipelineError
from core.image_processor import find_image_files
from core.vectorizer import Vectorizer
from data_access.vector_db_interface import VectorDBInterface

# Importar funciones/constantes necesarias de otros módulos UI
from ..sidebar import display_database_info
from ..state_utils import (
    STATE_CONFIRM_CLEAR,
    STATE_CONFIRM_DELETE,
    STATE_SELECTED_BATCH_SIZE,
)

logger = logging.getLogger(__name__)


def render_indexing_tab(
    vectorizer: Vectorizer, db: VectorDBInterface, truncate_dim: Optional[int]
):
    """
    Renderiza el contenido y maneja la lógica para la pestaña 'Indexar Imágenes'.

    Args:
        vectorizer: Instancia del Vectorizer inicializado.
        db: Instancia de la interfaz de BD vectorial inicializada.
        truncate_dim: Dimensión de truncamiento seleccionada (puede ser None).
    """
    st.subheader("1. Indexar Directorio de Imágenes") # Usar subheader para consistencia
    st.markdown(
        """
        Selecciona o introduce la ruta a una carpeta que contenga las imágenes
        que deseas añadir o actualizar en la base de datos vectorial activa.
        La aplicación buscará imágenes recursivamente en subcarpetas.
        Las imágenes ya existentes en la colección actual **no** se volverán a procesar.
        """
    )

    # Input para la ruta del directorio
    default_image_dir = getattr(
        config, "DEFAULT_IMAGE_DIR", os.path.abspath("images")
    )
    image_dir_path = st.text_input(
        "Ruta al directorio de imágenes:",
        value=default_image_dir,
        placeholder="Ej: C:/Users/TuUsuario/Pictures o ./data/images",
        help="Introduce la ruta completa a la carpeta con imágenes.",
        key="index_image_dir_path",
    )

    # Input para el tamaño del lote
    selected_batch_size = st.number_input(
        "Imágenes a Procesar por Lote (Batch Size):",
        min_value=1,
        max_value=128, # Límite superior razonable
        value=st.session_state.get(
            STATE_SELECTED_BATCH_SIZE, config.BATCH_SIZE_IMAGES
        ),
        step=4, # Pasos de 4
        key=STATE_SELECTED_BATCH_SIZE, # Usar constante para la clave
        help="Número de imágenes a cargar y vectorizar juntas. Afecta uso de memoria y velocidad.",
    )

    st.divider() # Separador visual

    # Botón para verificar el directorio
    if st.button("🔍 Verificar Directorio", key="check_dir"):
        if not image_dir_path:
            st.warning("Por favor, introduce una ruta al directorio.")
        elif not os.path.isdir(image_dir_path):
            st.error(
                f"❌ El directorio no existe o no es accesible: `{image_dir_path}`"
            )
        else:
            st.info(f"Verificando directorio: `{image_dir_path}`...")
            with st.spinner("Buscando imágenes..."):
                try:
                    image_files = find_image_files(image_dir_path)
                except Exception as find_err:
                    st.error(f"Error al buscar archivos: {find_err}")
                    logger.error(
                        f"Error finding files in {image_dir_path}: {find_err}",
                        exc_info=True,
                    )
                    image_files = []

            if not image_files:
                st.warning(
                    f"⚠️ No se encontraron archivos de imagen compatibles en `{image_dir_path}` y sus subdirectorios."
                )
            else:
                st.success(
                    f"✅ ¡Encontradas {len(image_files)} imágenes potenciales!"
                )
                # Expander para previsualizar imágenes
                with st.expander(
                    "Ver ejemplos de imágenes encontradas (máx. 5)"
                ):
                    preview_files = image_files[:5]
                    cols_preview = st.columns(len(preview_files))
                    for i, file_path in enumerate(preview_files):
                        try:
                            if os.path.isfile(file_path):
                                # Usa st.image directamente para la previsualización
                                cols_preview[i].image(
                                    file_path, # Pasa la ruta directamente
                                    caption=os.path.basename(file_path),
                                    width=100, # Ancho pequeño para previsualización
                                )
                            else:
                                # Si el archivo ya no existe
                                cols_preview[i].warning(
                                    f"Movido:\n{os.path.basename(file_path)}"
                                )
                        except Exception as img_err:
                            # Error al cargar la imagen de previsualización
                            cols_preview[i].error(
                                f"Error {os.path.basename(file_path)}"
                            )
                            logger.warning(
                                f"Error loading preview image {file_path}: {img_err}"
                            )

    st.divider() # Otro separador

    # Botón para iniciar la indexación
    if st.button("⚙️ Procesar e Indexar Directorio", key="process_dir", type="primary"):
        # Validaciones previas
        if not image_dir_path or not os.path.isdir(image_dir_path):
            st.error(
                "❌ Ruta de directorio inválida. Verifica la ruta antes de indexar."
            )
        elif not db or not db.is_initialized:
            st.error("❌ La base de datos no está lista. No se puede indexar.")
        elif not vectorizer or not vectorizer.is_ready:
            st.error("❌ El vectorizador no está listo.")
        else:
            # Iniciar proceso de indexación
            st.info(
                f"🚀 Iniciando proceso de indexación para: `{image_dir_path}`"
            )
            # Elementos para mostrar progreso y estado
            progress_bar_idx = st.progress(0.0, text="Iniciando...")
            status_text_idx = st.empty() # Para mostrar mensajes de estado
            start_time_idx = time.time()
            total_processed_successfully = 0

            # --- Callbacks para actualizar la UI ---
            def update_streamlit_progress(current: int, total: int):
                """Actualiza la barra de progreso de Streamlit."""
                progress_percent = 0.0
                if total > 0:
                    progress_percent = min(float(current) / float(total), 1.0)
                try:
                    # Texto opcional en la barra de progreso
                    progress_text = f"Procesando: {current}/{total} archivos"
                    progress_bar_idx.progress(progress_percent, text=progress_text)
                except Exception as e:
                    # Ignorar errores si la UI se cierra mientras se actualiza
                    logger.debug(
                        f"Error updating progress bar (may be closed): {e}"
                    )

            def update_streamlit_status(message: str):
                """Actualiza el texto de estado de Streamlit."""
                try:
                    status_text_idx.info(message) # Usar st.info para mensajes de estado
                except Exception as e:
                    logger.debug(
                        f"Error updating status text (may be closed): {e}"
                    )
            # --- Fin Callbacks ---

            try:
                logger.info(
                    f"Starting indexing via Streamlit for directory: {image_dir_path}"
                )
                # Llamada a la función de indexación del backend
                total_processed_successfully = indexing.process_directory(
                    directory_path=image_dir_path,
                    vectorizer=vectorizer,
                    db=db,
                    batch_size=selected_batch_size,
                    truncate_dim=truncate_dim,
                    progress_callback=update_streamlit_progress, # Pasar callback
                    status_callback=update_streamlit_status,     # Pasar callback
                )

                # Mensaje final de éxito
                end_time_idx = time.time()
                elapsed_time = end_time_idx - start_time_idx
                final_message = (
                    f"✅ Indexación completada en {elapsed_time:.2f}s. "
                    f"Imágenes nuevas guardadas/actualizadas: {total_processed_successfully}."
                )
                status_text_idx.success(final_message) # Usar st.success al final
                progress_bar_idx.progress(1.0, text="Completado") # Marcar progreso como 100%
                logger.info(
                    f"Indexing finished successfully in {elapsed_time:.2f}s. Stored/updated: {total_processed_successfully}"
                )

                # Actualizar info de la BD en la sidebar
                display_database_info(db, st.sidebar)

            except PipelineError as e:
                # Error controlado del pipeline
                error_message = f"❌ Error en el pipeline de indexación: {e}"
                status_text_idx.error(error_message)
                progress_bar_idx.progress(1.0, text="Error") # Indicar error en progreso
                logger.error(
                    f"PipelineError during Streamlit indexing call: {e}",
                    exc_info=True, # Mostrar traceback en logs
                )
            except Exception as e:
                # Error inesperado
                error_message = f"❌ Error inesperado durante la indexación: {e}"
                status_text_idx.error(error_message)
                progress_bar_idx.progress(1.0, text="Error") # Indicar error en progreso
                logger.error(
                    f"Unexpected error during Streamlit indexing call: {e}",
                    exc_info=True, # Mostrar traceback en logs
                )

    st.divider() # Separador antes de opciones avanzadas

    # --- Opciones Avanzadas / Peligrosas ---
    with st.expander("⚠️ Opciones Avanzadas / Peligrosas"):
        st.warning(
            "¡Precaución! Estas acciones modifican permanentemente la base de datos activa."
        )
        col_adv1, col_adv2 = st.columns(2)

        # Nombre de la colección activa para mostrar en mensajes
        current_collection_name = (
            getattr(db, "collection_name", "N/A") if db else "N/A"
        )

        with col_adv1:
            # Botón para iniciar el proceso de limpieza
            st.button(
                "🗑️ Limpiar Colección Actual",
                key="clear_collection_btn",
                on_click=lambda: st.session_state.update(
                    {STATE_CONFIRM_CLEAR: True, STATE_CONFIRM_DELETE: False}
                ),
                help=f"Elimina todos los vectores de '{current_collection_name}', pero mantiene la colección."
            )

        with col_adv2:
            # Botón para iniciar el proceso de eliminación
            st.button(
                "❌ Eliminar Colección Actual",
                key="delete_collection_btn",
                type="secondary", # Usar estilo secundario/peligroso si se define en CSS
                on_click=lambda: st.session_state.update(
                    {STATE_CONFIRM_DELETE: True, STATE_CONFIRM_CLEAR: False}
                ),
                 help=f"Elimina permanentemente toda la colección '{current_collection_name}'."
            )

        # --- Lógica de Confirmación para Limpiar ---
        if st.session_state.get(STATE_CONFIRM_CLEAR, False):
            st.subheader("Confirmar Limpieza")
            st.error( # Usar st.error para máxima atención
                f"🚨 **¡Atención!** Estás a punto de eliminar **TODOS** los elementos de la colección `{current_collection_name}`. Esta acción no se puede deshacer."
            )
            confirm_clear_check = st.checkbox(
                "Sí, entiendo y quiero limpiar esta colección.",
                key="confirm_clear_check",
            )

            # Botón de confirmación final, deshabilitado hasta marcar el checkbox
            if st.button(
                "CONFIRMAR LIMPIEZA",
                key="confirm_clear_final_btn",
                disabled=not confirm_clear_check,
                type="primary" # Botón primario para la acción final
            ):
                if db and db.is_initialized:
                    with st.spinner(
                        f"Vaciando la colección '{current_collection_name}'..."
                    ):
                        try:
                            count_before = db.count()
                            success = db.clear_collection()
                            if success:
                                st.success(
                                    f"✅ Colección '{current_collection_name}' limpiada. Elementos eliminados: {count_before}."
                                )
                                # Actualizar info en sidebar
                                display_database_info(db, st.sidebar)
                            else:
                                st.error(
                                    f"❌ Falló la operación de limpieza de '{current_collection_name}' (revisa los logs)."
                                )
                        except DatabaseError as e:
                            st.error(
                                f"❌ Error de base de datos al limpiar '{current_collection_name}': {e}"
                            )
                            logger.error(
                                f"DatabaseError clearing collection '{current_collection_name}' via Streamlit",
                                exc_info=True,
                            )
                        except Exception as e:
                            st.error(
                                f"❌ Error inesperado al limpiar '{current_collection_name}': {e}"
                            )
                            logger.error(
                                f"Error clearing collection '{current_collection_name}' via Streamlit",
                                exc_info=True,
                            )
                else:
                    st.error("❌ La base de datos no está lista.")

                # Resetea el estado de confirmación y re-ejecuta para limpiar la UI
                st.session_state[STATE_CONFIRM_CLEAR] = False
                st.rerun()

            # Botón para cancelar
            if st.button("Cancelar Limpieza", key="cancel_clear_btn"):
                st.session_state[STATE_CONFIRM_CLEAR] = False
                st.rerun()

        # --- Lógica de Confirmación para Eliminar ---
        if st.session_state.get(STATE_CONFIRM_DELETE, False):
            st.subheader("Confirmar Eliminación Permanente")
            st.error( # Usar st.error para máxima atención
                f"🚨 **¡PELIGRO MÁXIMO!** Vas a **ELIMINAR PERMANENTEMENTE** toda la colección `{current_collection_name}` y sus datos. ¡No hay vuelta atrás!"
            )
            confirm_delete_check = st.checkbox(
                "Sí, entiendo que esto es IRREVERSIBLE y quiero eliminar esta colección.",
                key="confirm_delete_check",
            )
            # Campo de texto para doble confirmación
            confirm_text = st.text_input(
                "Escribe 'ELIMINAR' (en mayúsculas) para confirmar:",
                key="confirm_delete_text",
                placeholder="ELIMINAR"
            )

            # Habilitar botón solo si ambas condiciones se cumplen
            delete_button_disabled = not (
                confirm_delete_check and confirm_text == "ELIMINAR"
            )

            if st.button(
                "CONFIRMAR ELIMINACIÓN PERMANENTE",
                key="confirm_delete_final_btn",
                disabled=delete_button_disabled,
                type="primary" # Botón primario para la acción final
            ):
                if db: # No necesita is_initialized para intentar eliminar
                    with st.spinner(
                        f"Eliminando la colección '{current_collection_name}'..."
                    ):
                        try:
                            collection_name_deleted = getattr(
                                db, "collection_name", "N/A"
                            )
                            success = db.delete_collection()
                            if success:
                                st.success(
                                    f"✅ Colección '{collection_name_deleted}' eliminada permanentemente."
                                )
                                # Limpiar y actualizar la sidebar
                                st.sidebar.empty() # Limpia contenido anterior de la sidebar
                                st.sidebar.title("⚙️ Configuración") # Restaura título
                                st.sidebar.divider()
                                # ... (restaurar selectores de modelo/dimensión si es necesario) ...
                                st.sidebar.header("ℹ️ Información")
                                st.sidebar.warning( # Mensaje claro en sidebar
                                    f"Colección '{collection_name_deleted}' eliminada."
                                )
                                st.sidebar.info(
                                    "Indexa datos para crear una nueva colección o cambia la configuración."
                                )
                                # Forzar recarga de la instancia de BD en el próximo ciclo
                                # (Streamlit podría necesitar un reinicio completo para reflejar esto si la instancia está cacheadada)
                                # Intenta limpiar claves de caché si es posible (esto es avanzado y puede no funcionar)
                                # st.cache_resource.clear() # ¡Cuidado! Limpia *todo* el caché de recursos
                                st.warning("La colección ha sido eliminada. Puede que necesites refrescar la página o reiniciar la app para reflejar todos los cambios.")

                            else:
                                st.error(
                                    f"❌ Falló la operación de eliminación de '{collection_name_deleted}' (revisa los logs)."
                                )
                        except DatabaseError as e:
                            st.error(
                                f"❌ Error de base de datos al eliminar '{current_collection_name}': {e}"
                            )
                            logger.error(
                                f"DatabaseError deleting collection '{current_collection_name}' via Streamlit",
                                exc_info=True,
                            )
                        except Exception as e:
                            st.error(
                                f"❌ Error inesperado al eliminar '{current_collection_name}': {e}"
                            )
                            logger.error(
                                f"Error deleting collection '{current_collection_name}' via Streamlit",
                                exc_info=True,
                            )
                else:
                    st.error("❌ Objeto de base de datos no disponible.")

                # Resetea estado y re-ejecuta
                st.session_state[STATE_CONFIRM_DELETE] = False
                st.rerun()

            # Ayuda contextual si el botón está deshabilitado
            if delete_button_disabled:
                if not confirm_delete_check:
                    st.caption("ℹ️ Marca la casilla de confirmación para proceder.")
                elif confirm_text != "ELIMINAR":
                    st.caption(
                        "ℹ️ Escribe 'ELIMINAR' exactamente como se muestra para habilitar el botón."
                    )

            # Botón de cancelar
            if st.button("Cancelar Eliminación", key="cancel_delete_btn"):
                st.session_state[STATE_CONFIRM_DELETE] = False
                st.rerun()
