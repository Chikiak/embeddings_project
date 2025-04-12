import logging
import os
from typing import Optional

import streamlit as st
from PIL import Image, UnidentifiedImageError

from app.models import SearchResultItem, SearchResults

logger = logging.getLogger(__name__)


def display_results(
    results: Optional[SearchResults], results_container: st.container
):
    """
    Muestra los resultados de la búsqueda (objeto SearchResults) en un formato de cuadrícula.
    Maneja la posible visualización de puntuaciones RRF en lugar de distancias.

    Args:
        results: El objeto SearchResults que contiene los ítems encontrados.
        results_container: El contenedor de Streamlit donde se mostrarán los resultados.
    """

    if results is None:
        results_container.error(
            "❌ La búsqueda falló o no devolvió resultados."
        )
        return

    if results.is_empty:
        results_container.warning(
            "🤷‍♂️ No se encontraron resultados para tu búsqueda."
        )
        return

    results_container.success(f"🎉 ¡Encontrados {results.count} resultados!")

    num_columns = max(1, min(results.count, 5))
    cols = results_container.columns(num_columns)

    for i, item in enumerate(results.items):

        col = cols[i % num_columns]

        with col.container():

            st.markdown('<div class="result-item">', unsafe_allow_html=True)
            try:

                if (
                    item.id
                    and isinstance(item.id, str)
                    and os.path.isfile(item.id)
                ):

                    img = Image.open(item.id)
                    st.image(img, use_container_width=True)

                    score_label = "Score"
                    score_value_str = "N/A"
                    if item.distance is not None:

                        is_rrf = item.metadata.get("fusion_method") == "RRF"

                        if is_rrf:
                            score_label = "Score RRF"
                            score_value_str = f"{item.distance:.4f}"

                        elif 0 <= item.distance <= 2.0:

                            score_label = "Distancia"
                            score_value_str = f"{item.distance:.3f}"

                        else:
                            score_label = "Distancia"
                            score_value_str = f"{item.distance:.3f}"

                    st.markdown(
                        f'<div class="caption">{os.path.basename(item.id)}<br>{score_label}: {score_value_str}</div>',
                        unsafe_allow_html=True,
                    )

                elif item.id and isinstance(item.id, str):
                    st.warning(
                        f"Archivo no encontrado:\n{os.path.basename(item.id)}"
                    )
                    st.caption(f"Ruta registrada: {item.id}")
                    logger.warning(
                        f"Result image file not found at path: {item.id}"
                    )

                else:
                    st.warning("Resultado con ID inválido.")

            except FileNotFoundError:
                st.error(
                    f"Error crítico: Archivo no encontrado en la ruta: {item.id}"
                )
                logger.error(f"FileNotFoundError for image path: {item.id}")
            except UnidentifiedImageError:
                st.error(
                    f"Error: No se pudo abrir/identificar imagen: {os.path.basename(item.id if item.id else 'ID inválido')}"
                )
                logger.warning(f"UnidentifiedImageError for: {item.id}")
            except Exception as e:
                st.error(
                    f"Error al mostrar '{os.path.basename(item.id if item.id else 'ID inválido')}': {str(e)[:100]}..."
                )
                logger.error(
                    f"Error displaying image {item.id}: {e}", exc_info=True
                )

            st.markdown("</div>", unsafe_allow_html=True)
