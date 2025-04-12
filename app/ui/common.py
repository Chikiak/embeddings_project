# app/ui/common.py
import streamlit as st
from PIL import Image, UnidentifiedImageError
import os
import logging
from typing import Optional

# Importaciones relativas correctas desde la raíz del proyecto
# Asume que 'app' está en la raíz o en sys.path
from app.models import SearchResults, SearchResultItem

logger = logging.getLogger(__name__)

def display_results(results: Optional[SearchResults], results_container: st.container):
    """
    Muestra los resultados de la búsqueda (objeto SearchResults) en un formato de cuadrícula.
    Maneja la posible visualización de puntuaciones RRF en lugar de distancias.

    Args:
        results: El objeto SearchResults que contiene los ítems encontrados.
        results_container: El contenedor de Streamlit donde se mostrarán los resultados.
    """
    # Manejar caso de fallo en la búsqueda
    if results is None:
        results_container.error("❌ La búsqueda falló o no devolvió resultados.")
        return
    # Manejar caso de no encontrar resultados
    if results.is_empty:
        results_container.warning("🤷‍♂️ No se encontraron resultados para tu búsqueda.")
        return

    # Mostrar mensaje de éxito y número de resultados
    results_container.success(f"🎉 ¡Encontrados {results.count} resultados!")

    # Ajustar dinámicamente el número de columnas (ej: entre 1 y 5)
    num_columns = max(1, min(results.count, 5))
    cols = results_container.columns(num_columns)

    # Iterar sobre cada ítem de resultado
    for i, item in enumerate(results.items):
        # Asignar el ítem a una columna (ciclando)
        col = cols[i % num_columns]
        # Usar un contenedor dentro de la columna para aplicar estilos
        with col.container():
            # Aplicar clase CSS para el estilo del ítem
            st.markdown('<div class="result-item">', unsafe_allow_html=True)
            try:
                # Verificar si el ID es una ruta válida y el archivo existe
                # Asume que los IDs son rutas absolutas después de la indexación
                if item.id and isinstance(item.id, str) and os.path.isfile(item.id):
                    # Abrir y mostrar la imagen
                    img = Image.open(item.id)
                    st.image(img, use_container_width=True) # Ajustar imagen al ancho del contenedor

                    # Lógica mejorada para mostrar la puntuación/distancia
                    score_label = "Score" # Etiqueta por defecto
                    score_value_str = "N/A" # Valor por defecto
                    if item.distance is not None:
                        # Comprobar si se usó RRF (buscando en metadatos, si se añadió)
                        is_rrf = item.metadata.get("fusion_method") == "RRF"

                        if is_rrf:
                             score_label = "Score RRF" # Puntuación RRF (mayor es mejor)
                             score_value_str = f"{item.distance:.4f}"
                        # ChromaDB con 'cosine' devuelve distancia = 1 - similitud
                        # Rango esperado para distancia coseno: [0, 2], más común [0, 1]
                        elif 0 <= item.distance <= 2.0:
                            # Asumimos que es distancia coseno (menor es mejor)
                            score_label = "Distancia"
                            score_value_str = f"{item.distance:.3f}"
                            # Opcional: Calcular y mostrar similitud = 1 - distancia
                            # similarity = 1.0 - item.distance
                            # score_label = "Similitud"
                            # score_value_str = f"{similarity:.3f}"
                        else: # Otras métricas (ej: L2) o valores inesperados
                            score_label = "Distancia" # Etiqueta genérica
                            score_value_str = f"{item.distance:.3f}"

                    # Mostrar nombre de archivo y puntuación/distancia
                    st.markdown(
                        f'<div class="caption">{os.path.basename(item.id)}<br>{score_label}: {score_value_str}</div>',
                        unsafe_allow_html=True,
                    )
                # Manejar caso donde el ID existe pero el archivo no
                elif item.id and isinstance(item.id, str):
                    st.warning(f"Archivo no encontrado:\n{os.path.basename(item.id)}")
                    st.caption(f"Ruta registrada: {item.id}")
                    logger.warning(f"Result image file not found at path: {item.id}")
                # Manejar caso de ID inválido
                else:
                    st.warning("Resultado con ID inválido.")

            # Manejar errores específicos al procesar/mostrar la imagen
            except FileNotFoundError:
                st.error(f"Error crítico: Archivo no encontrado en la ruta: {item.id}")
                logger.error(f"FileNotFoundError for image path: {item.id}")
            except UnidentifiedImageError:
                st.error(f"Error: No se pudo abrir/identificar imagen: {os.path.basename(item.id if item.id else 'ID inválido')}")
                logger.warning(f"UnidentifiedImageError for: {item.id}")
            except Exception as e:
                st.error(f"Error al mostrar '{os.path.basename(item.id if item.id else 'ID inválido')}': {str(e)[:100]}...")
                logger.error(f"Error displaying image {item.id}: {e}", exc_info=True)
            # Cerrar el div del ítem
            st.markdown("</div>", unsafe_allow_html=True)
