import logging
import os
from typing import Optional

import streamlit as st

from app.exceptions import DatabaseError
from data_access.vector_db_interface import VectorDBInterface

logger = logging.getLogger(__name__)


def display_database_info(
    db: Optional[VectorDBInterface], container: st.sidebar
):
    """
    Muestra información sobre la base de datos vectorial actual en la sidebar.

    Args:
        db: La instancia de la interfaz de base de datos vectorial (puede ser None).
        container: El objeto sidebar de Streamlit (st.sidebar).
    """
    container.header("Base de Datos Vectorial")

    if db:
        try:

            is_ready = db.is_initialized
            if is_ready:

                db_count = db.count()

                db_path = (
                    getattr(db, "path", "N/A")
                    if hasattr(db, "path")
                    else "N/A"
                )
                collection_name = (
                    getattr(db, "collection_name", "N/A")
                    if hasattr(db, "collection_name")
                    else "N/A"
                )

                container.info(
                    f"**Colección:** `{collection_name}`\n\n"
                    f"**Ruta:** `{os.path.abspath(db_path) if db_path != 'N/A' and isinstance(db_path, str) else 'N/A'}`\n\n"
                    f"**Imágenes Indexadas:** `{db_count if db_count >= 0 else 'Error al contar'}`"
                )
            else:

                container.warning(
                    "La base de datos no está inicializada o no es accesible."
                )

                last_error = getattr(db, "_last_init_error", None)
                if last_error:
                    container.caption(f"Último error conocido: {last_error}")

        except DatabaseError as e:

            container.error(f"Error de BD al obtener info: {e}")
            logger.error(
                f"DatabaseError getting DB info for sidebar: {e}",
                exc_info=False,
            )
        except Exception as e:

            container.error(f"Error inesperado al obtener info de la BD: {e}")
            logger.error(
                f"Unexpected error getting DB info for sidebar: {e}",
                exc_info=True,
            )
    else:

        container.error("Instancia de base de datos no disponible (None).")
