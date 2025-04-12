import logging
import os
import time
from typing import Optional

import streamlit as st
from PIL import Image

import config
from app import indexing
from app.exceptions import DatabaseError, PipelineError
from core.image_processor import find_image_files
from core.vectorizer import Vectorizer
from data_access.vector_db_interface import VectorDBInterface

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
    st.header("1. Indexar Directorio de Imágenes")
    st.markdown(
        "Selecciona (o introduce la ruta a) una carpeta que contenga las imágenes que deseas indexar. "
        "La aplicación buscará imágenes recursivamente en subcarpetas."
    )

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

    selected_batch_size = st.number_input(
        "Imágenes a Procesar por Lote (Batch Size):",
        min_value=1,
        max_value=128,
        value=st.session_state.get(
            STATE_SELECTED_BATCH_SIZE, config.BATCH_SIZE_IMAGES
        ),
        step=4,
        key=STATE_SELECTED_BATCH_SIZE,
        help="Número de imágenes a cargar y vectorizar juntas. Afecta uso de memoria y velocidad.",
    )

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

                with st.expander(
                    "Ver ejemplos de imágenes encontradas (máx. 5)"
                ):
                    preview_files = image_files[:5]

                    cols_preview = st.columns(len(preview_files))
                    for i, file_path in enumerate(preview_files):
                        try:

                            if os.path.isfile(file_path):
                                cols_preview[i].image(
                                    Image.open(file_path),
                                    caption=os.path.basename(file_path),
                                    width=100,
                                )
                            else:

                                cols_preview[i].warning(
                                    f"Movido:\n{os.path.basename(file_path)}"
                                )
                        except Exception as img_err:

                            cols_preview[i].error(
                                f"Error {os.path.basename(file_path)}"
                            )
                            logger.warning(
                                f"Error loading preview image {file_path}: {img_err}"
                            )

    if st.button("⚙️ Procesar e Indexar Directorio", key="process_dir"):

        if not image_dir_path or not os.path.isdir(image_dir_path):
            st.error(
                "❌ Ruta de directorio inválida. Verifica la ruta antes de indexar."
            )
        elif not db or not db.is_initialized:
            st.error("❌ La base de datos no está lista. No se puede indexar.")
        elif not vectorizer or not vectorizer.is_ready:
            st.error("❌ El vectorizador no está listo.")
        else:

            st.info(
                f"🚀 Iniciando proceso de indexación para: `{image_dir_path}`"
            )

            progress_bar_idx = st.progress(0.0)
            status_text_idx = st.empty()
            start_time_idx = time.time()
            total_processed_successfully = 0

            def update_streamlit_progress(current: int, total: int):
                """Actualiza la barra de progreso de Streamlit."""
                progress_percent = 0.0
                if total > 0:

                    progress_percent = min(float(current) / float(total), 1.0)
                try:

                    progress_bar_idx.progress(progress_percent)
                except Exception as e:
                    logger.debug(
                        f"Error updating progress bar (may be closed): {e}"
                    )

            def update_streamlit_status(message: str):
                """Actualiza el texto de estado de Streamlit."""
                try:

                    status_text_idx.info(message)
                except Exception as e:
                    logger.debug(
                        f"Error updating status text (may be closed): {e}"
                    )

            try:
                logger.info(
                    f"Starting indexing via Streamlit for directory: {image_dir_path}"
                )

                total_processed_successfully = indexing.process_directory(
                    directory_path=image_dir_path,
                    vectorizer=vectorizer,
                    db=db,
                    batch_size=selected_batch_size,
                    truncate_dim=truncate_dim,
                    progress_callback=update_streamlit_progress,
                    status_callback=update_streamlit_status,
                )

                end_time_idx = time.time()
                elapsed_time = end_time_idx - start_time_idx
                status_text_idx.success(
                    f"✅ Indexación completada en {elapsed_time:.2f}s. Imágenes guardadas/actualizadas: {total_processed_successfully}."
                )
                logger.info(
                    f"Indexing finished successfully in {elapsed_time:.2f}s. Stored/updated: {total_processed_successfully}"
                )

                display_database_info(db, st.sidebar)

            except PipelineError as e:

                status_text_idx.error(
                    f"❌ Error en el pipeline de indexación: {e}"
                )
                logger.error(
                    f"PipelineError during Streamlit indexing call: {e}",
                    exc_info=True,
                )
            except Exception as e:

                status_text_idx.error(
                    f"❌ Error inesperado durante el proceso de indexación: {e}"
                )
                logger.error(
                    f"Unexpected error during Streamlit indexing call: {e}",
                    exc_info=True,
                )

    with st.expander("⚠️ Opciones Avanzadas / Peligrosas"):
        st.warning(
            "¡Precaución! Estas acciones modifican permanentemente la base de datos."
        )
        col_adv1, col_adv2 = st.columns(2)

        current_collection_name = (
            getattr(db, "collection_name", "N/A") if db else "N/A"
        )

        with col_adv1:

            st.button(
                "🗑️ Limpiar TODA la Colección Actual",
                key="clear_collection_btn",
                on_click=lambda: st.session_state.update(
                    {STATE_CONFIRM_CLEAR: True, STATE_CONFIRM_DELETE: False}
                ),
            )

        with col_adv2:

            st.button(
                "❌ ELIMINAR TODA la Colección Actual",
                key="delete_collection_btn",
                on_click=lambda: st.session_state.update(
                    {STATE_CONFIRM_DELETE: True, STATE_CONFIRM_CLEAR: False}
                ),
            )

        if st.session_state.get(STATE_CONFIRM_CLEAR, False):
            st.markdown(
                f"🚨 **¡Atención!** Vas a eliminar **TODOS** los elementos de la colección `{current_collection_name}`. Esta acción no se puede deshacer."
            )

            confirm_clear_check = st.checkbox(
                "Sí, entiendo y quiero limpiar la colección.",
                key="confirm_clear_check",
            )

            if st.button(
                "CONFIRMAR LIMPIEZA",
                key="confirm_clear_final_btn",
                disabled=not confirm_clear_check,
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
                                display_database_info(db, st.sidebar)
                            else:
                                st.error(
                                    "❌ Falló la operación de limpieza (ver logs)."
                                )
                        except DatabaseError as e:
                            st.error(
                                f"❌ Error de base de datos al limpiar: {e}"
                            )
                            logger.error(
                                "DatabaseError clearing collection via Streamlit",
                                exc_info=True,
                            )
                        except Exception as e:
                            st.error(
                                f"❌ Error inesperado al limpiar la colección: {e}"
                            )
                            logger.error(
                                "Error clearing collection via Streamlit",
                                exc_info=True,
                            )
                else:
                    st.error("❌ La base de datos no está lista.")

                st.session_state[STATE_CONFIRM_CLEAR] = False
                st.rerun()

            if st.button("Cancelar Limpieza", key="cancel_clear_btn"):
                st.session_state[STATE_CONFIRM_CLEAR] = False
                st.rerun()

        if st.session_state.get(STATE_CONFIRM_DELETE, False):
            st.markdown(
                f"🚨 **¡PELIGRO MÁXIMO!** Vas a **ELIMINAR PERMANENTEMENTE** toda la colección `{current_collection_name}` y sus datos. ¡No hay vuelta atrás!"
            )

            confirm_delete_check = st.checkbox(
                "Sí, entiendo que esto es IRREVERSIBLE y quiero eliminar la colección.",
                key="confirm_delete_check",
            )

            confirm_text = st.text_input(
                "Escribe 'ELIMINAR' para confirmar:", key="confirm_delete_text"
            )

            delete_button_disabled = not (
                confirm_delete_check and confirm_text == "ELIMINAR"
            )

            if st.button(
                "CONFIRMAR ELIMINACIÓN PERMANENTE",
                key="confirm_delete_final_btn",
                disabled=delete_button_disabled,
            ):

                if db:
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

                                st.sidebar.empty()
                                st.sidebar.header("Base de Datos Vectorial")
                                st.sidebar.warning(
                                    f"Colección '{collection_name_deleted}' eliminada."
                                )
                                st.sidebar.info(
                                    "Reinicia la aplicación o indexa datos para crear una nueva colección."
                                )
                            else:
                                st.error(
                                    "❌ Falló la operación de eliminación (ver logs)."
                                )
                        except DatabaseError as e:
                            st.error(
                                f"❌ Error de base de datos al eliminar: {e}"
                            )
                            logger.error(
                                "DatabaseError deleting collection via Streamlit",
                                exc_info=True,
                            )
                        except Exception as e:
                            st.error(
                                f"❌ Error inesperado al eliminar la colección: {e}"
                            )
                            logger.error(
                                "Error deleting collection via Streamlit",
                                exc_info=True,
                            )
                else:
                    st.error("❌ Objeto de base de datos no disponible.")

                st.session_state[STATE_CONFIRM_DELETE] = False
                st.rerun()

            if delete_button_disabled:
                if not confirm_delete_check:
                    st.info("Marca la casilla para proceder.")
                elif confirm_text != "ELIMINAR":
                    st.warning(
                        "Escribe 'ELIMINAR' en el campo de texto para habilitar el botón."
                    )

            if st.button("Cancelar Eliminación", key="cancel_delete_btn"):
                st.session_state[STATE_CONFIRM_DELETE] = False
                st.rerun()
