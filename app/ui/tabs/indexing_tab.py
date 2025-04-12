# app/ui/tabs/indexing_tab.py
import streamlit as st
import os
import time
import logging
from typing import Optional
from PIL import Image # Necesario para mostrar vistas previas

# Importaciones relativas/absolutas correctas
import config # Acceso a configuraciones por defecto
# Componentes principales pasados como argumentos
from core.vectorizer import Vectorizer
from data_access.vector_db_interface import VectorDBInterface
# Módulo de lógica de negocio para indexación
from app import indexing
# Excepciones relevantes
from app.exceptions import PipelineError, DatabaseError
# Helper para encontrar imágenes
from core.image_processor import find_image_files
# Utilidades de estado y UI compartida
from ..state_utils import STATE_SELECTED_BATCH_SIZE, STATE_CONFIRM_CLEAR, STATE_CONFIRM_DELETE
from ..sidebar import display_database_info # Para actualizar sidebar

logger = logging.getLogger(__name__)

def render_indexing_tab(vectorizer: Vectorizer, db: VectorDBInterface, truncate_dim: Optional[int]):
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

    # --- Input para Directorio de Imágenes ---
    # Usar valor por defecto de config, permitir al usuario cambiarlo
    default_image_dir = getattr(config, "DEFAULT_IMAGE_DIR", os.path.abspath("images"))
    image_dir_path = st.text_input(
        "Ruta al directorio de imágenes:",
        value=default_image_dir, # Valor inicial
        placeholder="Ej: C:/Users/TuUsuario/Pictures o ./data/images",
        help="Introduce la ruta completa a la carpeta con imágenes.",
        key="index_image_dir_path" # Clave única para este input
    )

    # --- Input para Tamaño de Lote (Batch Size) ---
    # Usar valor del estado de sesión o el por defecto de config
    selected_batch_size = st.number_input(
        "Imágenes a Procesar por Lote (Batch Size):",
        min_value=1,
        max_value=128, # Límite práctico, ajustar según memoria GPU/CPU
        value=st.session_state.get(STATE_SELECTED_BATCH_SIZE, config.BATCH_SIZE_IMAGES),
        step=4, # Pasos razonables
        key=STATE_SELECTED_BATCH_SIZE, # Guardar en estado de sesión
        help="Número de imágenes a cargar y vectorizar juntas. Afecta uso de memoria y velocidad."
    )

    # --- Botón para Verificar Directorio ---
    if st.button("🔍 Verificar Directorio", key="check_dir"):
        # Validar la ruta introducida
        if not image_dir_path:
            st.warning("Por favor, introduce una ruta al directorio.")
        elif not os.path.isdir(image_dir_path):
            st.error(
                f"❌ El directorio no existe o no es accesible: `{image_dir_path}`"
            )
        else:
            # Si la ruta es válida, buscar imágenes
            st.info(f"Verificando directorio: `{image_dir_path}`...")
            with st.spinner("Buscando imágenes..."):
                try:
                    # Usar la función de core.image_processor
                    image_files = find_image_files(image_dir_path)
                except Exception as find_err:
                    st.error(f"Error al buscar archivos: {find_err}")
                    logger.error(f"Error finding files in {image_dir_path}: {find_err}", exc_info=True)
                    image_files = []

            # Mostrar resultado de la búsqueda de archivos
            if not image_files:
                st.warning(
                    f"⚠️ No se encontraron archivos de imagen compatibles en `{image_dir_path}` y sus subdirectorios."
                )
            else:
                st.success(f"✅ ¡Encontradas {len(image_files)} imágenes potenciales!")
                # Mostrar vistas previas de algunas imágenes encontradas
                with st.expander("Ver ejemplos de imágenes encontradas (máx. 5)"):
                    preview_files = image_files[:5]
                    # Crear columnas para las vistas previas
                    cols_preview = st.columns(len(preview_files))
                    for i, file_path in enumerate(preview_files):
                        try:
                            # Comprobar si el archivo aún existe antes de abrirlo
                            if os.path.isfile(file_path):
                                cols_preview[i].image(
                                    Image.open(file_path),
                                    caption=os.path.basename(file_path),
                                    width=100, # Tamaño pequeño para vista previa
                                )
                            else:
                                # Indicar si el archivo fue movido/eliminado
                                cols_preview[i].warning(f"Movido:\n{os.path.basename(file_path)}")
                        except Exception as img_err:
                            # Manejar errores al abrir la imagen de vista previa
                            cols_preview[i].error(
                                f"Error {os.path.basename(file_path)}"
                            )
                            logger.warning(
                                f"Error loading preview image {file_path}: {img_err}"
                            )

    # --- Botón para Procesar e Indexar ---
    if st.button("⚙️ Procesar e Indexar Directorio", key="process_dir"):
        # Validaciones previas a la indexación
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
            st.info(f"🚀 Iniciando proceso de indexación para: `{image_dir_path}`")
            # Crear elementos UI para mostrar progreso y estado
            progress_bar_idx = st.progress(0.0)
            status_text_idx = st.empty() # Contenedor para mensajes de estado
            start_time_idx = time.time()
            total_processed_successfully = 0

            # --- Callbacks para actualizar la UI desde el proceso de indexación ---
            def update_streamlit_progress(current: int, total: int):
                """Actualiza la barra de progreso de Streamlit."""
                progress_percent = 0.0
                if total > 0:
                    # Calcular porcentaje, asegurando que no exceda 1.0
                    progress_percent = min(float(current) / float(total), 1.0)
                try:
                    # Actualizar la barra de progreso (puede fallar si el widget desaparece)
                    progress_bar_idx.progress(progress_percent)
                except Exception as e:
                    logger.debug(f"Error updating progress bar (may be closed): {e}")

            def update_streamlit_status(message: str):
                 """Actualiza el texto de estado de Streamlit."""
                 try:
                     # Mostrar mensaje en el contenedor de texto
                     status_text_idx.info(message)
                 except Exception as e:
                    logger.debug(f"Error updating status text (may be closed): {e}")
            # --- Fin Callbacks ---

            try:
                logger.info(
                    f"Starting indexing via Streamlit for directory: {image_dir_path}"
                )
                # Llamar a la función de lógica de negocio para indexar
                # Pasando los callbacks para actualizar la UI
                total_processed_successfully = indexing.process_directory(
                    directory_path=image_dir_path,
                    vectorizer=vectorizer,
                    db=db,
                    batch_size=selected_batch_size, # Usar valor seleccionado
                    truncate_dim=truncate_dim, # Usar valor seleccionado
                    progress_callback=update_streamlit_progress,
                    status_callback=update_streamlit_status,
                )
                # Indexación completada con éxito
                end_time_idx = time.time()
                elapsed_time = end_time_idx - start_time_idx
                status_text_idx.success(
                    f"✅ Indexación completada en {elapsed_time:.2f}s. Imágenes guardadas/actualizadas: {total_processed_successfully}."
                )
                logger.info(
                    f"Indexing finished successfully in {elapsed_time:.2f}s. Stored/updated: {total_processed_successfully}"
                )
                # Actualizar la información de la BD en la sidebar
                display_database_info(db, st.sidebar)
                # Opcional: Limpiar barra de progreso después de un tiempo
                # time.sleep(2)
                # progress_bar_idx.empty()

            except PipelineError as e:
                # Manejar errores específicos del pipeline de indexación
                status_text_idx.error(f"❌ Error en el pipeline de indexación: {e}")
                logger.error(
                    f"PipelineError during Streamlit indexing call: {e}", exc_info=True
                )
            except Exception as e:
                # Manejar otros errores inesperados
                status_text_idx.error(
                    f"❌ Error inesperado durante el proceso de indexación: {e}"
                )
                logger.error(
                    f"Unexpected error during Streamlit indexing call: {e}",
                    exc_info=True,
                )

    # --- Opciones Avanzadas (Limpiar/Eliminar Colección) ---
    with st.expander("⚠️ Opciones Avanzadas / Peligrosas"):
        st.warning(
            "¡Precaución! Estas acciones modifican permanentemente la base de datos."
        )
        col_adv1, col_adv2 = st.columns(2)

        # Obtener nombre actual de la colección para mostrar en mensajes
        current_collection_name = getattr(db, "collection_name", "N/A") if db else "N/A"

        # --- Botón Limpiar Colección ---
        with col_adv1:
             # El botón solo cambia el estado, la lógica se maneja más abajo
             st.button(
                 "🗑️ Limpiar TODA la Colección Actual",
                 key="clear_collection_btn",
                 # Usar on_click para actualizar el estado de forma segura
                 on_click=lambda: st.session_state.update({STATE_CONFIRM_CLEAR: True, STATE_CONFIRM_DELETE: False})
            )

        # --- Botón Eliminar Colección ---
        with col_adv2:
             # El botón solo cambia el estado
             st.button(
                 "❌ ELIMINAR TODA la Colección Actual",
                 key="delete_collection_btn",
                 on_click=lambda: st.session_state.update({STATE_CONFIRM_DELETE: True, STATE_CONFIRM_CLEAR: False})
            )

        # --- Lógica de Confirmación para Limpiar ---
        # Se muestra solo si el flag STATE_CONFIRM_CLEAR es True
        if st.session_state.get(STATE_CONFIRM_CLEAR, False):
            st.markdown(
                f"🚨 **¡Atención!** Vas a eliminar **TODOS** los elementos de la colección `{current_collection_name}`. Esta acción no se puede deshacer."
            )
            # Checkbox de confirmación
            confirm_clear_check = st.checkbox(
                "Sí, entiendo y quiero limpiar la colección.", key="confirm_clear_check"
            )
            # Botón final de confirmación, habilitado por el checkbox
            if st.button("CONFIRMAR LIMPIEZA", key="confirm_clear_final_btn", disabled=not confirm_clear_check):
                    # Verificar si la BD está lista antes de actuar
                    if db and db.is_initialized:
                        with st.spinner(f"Vaciando la colección '{current_collection_name}'..."):
                            try:
                                count_before = db.count() # Contar antes de limpiar
                                success = db.clear_collection() # Llamar al método de la interfaz
                                if success:
                                    st.success(
                                        f"✅ Colección '{current_collection_name}' limpiada. Elementos eliminados: {count_before}."
                                    )
                                    display_database_info(db, st.sidebar) # Actualizar sidebar
                                else:
                                    st.error("❌ Falló la operación de limpieza (ver logs).")
                            except DatabaseError as e:
                                st.error(f"❌ Error de base de datos al limpiar: {e}")
                                logger.error("DatabaseError clearing collection via Streamlit", exc_info=True)
                            except Exception as e:
                                st.error(f"❌ Error inesperado al limpiar la colección: {e}")
                                logger.error("Error clearing collection via Streamlit", exc_info=True)
                    else:
                        st.error("❌ La base de datos no está lista.")
                    # Resetear el estado de confirmación y re-ejecutar para limpiar UI
                    st.session_state[STATE_CONFIRM_CLEAR] = False
                    st.rerun()

            # Botón para cancelar la acción
            if st.button("Cancelar Limpieza", key="cancel_clear_btn"):
                st.session_state[STATE_CONFIRM_CLEAR] = False
                st.rerun()

        # --- Lógica de Confirmación para Eliminar ---
        # Se muestra solo si el flag STATE_CONFIRM_DELETE es True
        if st.session_state.get(STATE_CONFIRM_DELETE, False):
            st.markdown(
                f"🚨 **¡PELIGRO MÁXIMO!** Vas a **ELIMINAR PERMANENTEMENTE** toda la colección `{current_collection_name}` y sus datos. ¡No hay vuelta atrás!"
            )
            # Checkbox de confirmación
            confirm_delete_check = st.checkbox(
                "Sí, entiendo que esto es IRREVERSIBLE y quiero eliminar la colección.",
                key="confirm_delete_check",
            )
            # Input de texto para doble confirmación
            confirm_text = st.text_input(
                 "Escribe 'ELIMINAR' para confirmar:", key="confirm_delete_text"
            )

            # Habilitar botón solo si ambas confirmaciones son correctas
            delete_button_disabled = not (confirm_delete_check and confirm_text == "ELIMINAR")

            if st.button("CONFIRMAR ELIMINACIÓN PERMANENTE", key="confirm_delete_final_btn", disabled=delete_button_disabled):
                    # Necesitamos el objeto DB para llamar a delete, incluso si no está 'inicializado'
                    if db:
                        with st.spinner(f"Eliminando la colección '{current_collection_name}'..."):
                            try:
                                collection_name_deleted = getattr(db, "collection_name", "N/A") # Guardar nombre
                                success = db.delete_collection() # Llamar al método de la interfaz
                                if success:
                                    st.success(
                                        f"✅ Colección '{collection_name_deleted}' eliminada permanentemente."
                                    )
                                    # Actualizar sidebar para reflejar eliminación
                                    # Limpiar y mostrar mensaje apropiado
                                    st.sidebar.empty() # Limpiar contenido anterior de la sidebar
                                    st.sidebar.header("Base de Datos Vectorial") # Re-añadir cabecera
                                    st.sidebar.warning(f"Colección '{collection_name_deleted}' eliminada.")
                                    st.sidebar.info("Reinicia la aplicación o indexa datos para crear una nueva colección.")
                                else:
                                    st.error("❌ Falló la operación de eliminación (ver logs).")
                            except DatabaseError as e:
                                st.error(f"❌ Error de base de datos al eliminar: {e}")
                                logger.error("DatabaseError deleting collection via Streamlit", exc_info=True)
                            except Exception as e:
                                st.error(f"❌ Error inesperado al eliminar la colección: {e}")
                                logger.error("Error deleting collection via Streamlit", exc_info=True)
                    else:
                        st.error("❌ Objeto de base de datos no disponible.")
                    # Resetear estado de confirmación y re-ejecutar
                    st.session_state[STATE_CONFIRM_DELETE] = False
                    st.rerun()

            # Mostrar ayuda si el botón está deshabilitado
            if delete_button_disabled:
                 if not confirm_delete_check:
                     st.info("Marca la casilla para proceder.")
                 elif confirm_text != "ELIMINAR":
                     st.warning("Escribe 'ELIMINAR' en el campo de texto para habilitar el botón.")

            # Botón para cancelar la acción
            if st.button("Cancelar Eliminación", key="cancel_delete_btn"):
                 st.session_state[STATE_CONFIRM_DELETE] = False
                 st.rerun()

