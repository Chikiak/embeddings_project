# app/ui/sidebar.py
import streamlit as st
import os
import logging
from typing import Optional

# Importaciones relativas/absolutas correctas
# Necesita la interfaz de BD para obtener información
from data_access.vector_db_interface import VectorDBInterface
# Necesita la excepción de BD para manejo de errores
from app.exceptions import DatabaseError

logger = logging.getLogger(__name__)

def display_database_info(db: Optional[VectorDBInterface], container: st.sidebar):
    """
    Muestra información sobre la base de datos vectorial actual en la sidebar.

    Args:
        db: La instancia de la interfaz de base de datos vectorial (puede ser None).
        container: El objeto sidebar de Streamlit (st.sidebar).
    """
    container.header("Base de Datos Vectorial")
    # Verificar si la instancia de BD existe
    if db:
        try:
            # Comprobar si la BD está inicializada
            is_ready = db.is_initialized
            if is_ready:
                # Obtener contador de ítems (puede lanzar DatabaseError)
                db_count = db.count()
                # Intentar obtener ruta y nombre de colección de forma segura
                # (puede que no existan en todas las implementaciones de la interfaz)
                db_path = getattr(db, "path", "N/A") if hasattr(db, "path") else "N/A"
                collection_name = getattr(db, "collection_name", "N/A") if hasattr(db, "collection_name") else "N/A"

                # Mostrar la información formateada
                container.info(
                    f"**Colección:** `{collection_name}`\n\n"
                    # Mostrar ruta absoluta si es una ruta válida
                    f"**Ruta:** `{os.path.abspath(db_path) if db_path != 'N/A' and isinstance(db_path, str) else 'N/A'}`\n\n"
                    # Mostrar contador o mensaje de error
                    f"**Imágenes Indexadas:** `{db_count if db_count >= 0 else 'Error al contar'}`"
                )
            else:
                # Mensaje si la BD no está inicializada
                container.warning("La base de datos no está inicializada o no es accesible.")
                # Opcional: Mostrar último error conocido si la implementación lo guarda
                last_error = getattr(db, "_last_init_error", None)
                if last_error:
                    container.caption(f"Último error conocido: {last_error}")

        except DatabaseError as e:
            # Manejar errores específicos de la BD al obtener información
            container.error(f"Error de BD al obtener info: {e}")
            logger.error(f"DatabaseError getting DB info for sidebar: {e}", exc_info=False) # Log menos verboso
        except Exception as e:
            # Manejar otros errores inesperados
            container.error(f"Error inesperado al obtener info de la BD: {e}")
            logger.error(f"Unexpected error getting DB info for sidebar: {e}", exc_info=True)
    else:
        # Mensaje si la instancia de BD es None (falló la inicialización)
        container.error("Instancia de base de datos no disponible (None).")

