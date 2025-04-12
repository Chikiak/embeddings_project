# --- app/ui/tabs/clustering/cluster_optimization_tab.py ---
import logging
import time
from typing import Optional

import numpy as np
import streamlit as st

from app import clustering
from app.exceptions import DatabaseError, InitializationError, PipelineError
from app.ui.state_utils import (
    STATE_CLUSTER_IDS, # Para limpiar si se optimiza
    STATE_CLUSTER_LABELS, # Para limpiar si se optimiza
    STATE_REDUCED_EMBEDDINGS,
    STATE_REDUCED_IDS,
)
from core.vectorizer import Vectorizer # No se usa directamente, pero podría ser útil
from data_access.vector_db_interface import VectorDBInterface

logger = logging.getLogger(__name__)


def render_cluster_optimization_tab(
    vectorizer: Vectorizer, db: VectorDBInterface, truncate_dim: Optional[int]
):
    """
    Renderiza el contenido y maneja la lógica para la pestaña 'Optimizar Datos (Pre-Clustering)'.

    Args:
        vectorizer: Instancia del Vectorizer (puede ser útil para info).
        db: Instancia de la interfaz de BD vectorial inicializada.
        truncate_dim: Dimensión de truncamiento seleccionada (puede ser None).
    """
    st.subheader("📉 Optimizar Datos (Reducción de Dimensionalidad)") # Subheader
    st.markdown(
        """
        Aplica técnicas de reducción de dimensionalidad (**PCA** o **UMAP**) a los embeddings
        **antes** de ejecutar el clustering. Esto puede mejorar el rendimiento y, a veces,
        la calidad de algunos algoritmos de clustering, especialmente en datos de muy alta dimensión.

        **Importante:** Si aplicas una técnica aquí, los datos reducidos se guardarán
        temporalmente y se usarán **automáticamente** en la pestaña 'Ejecutar Clustering'.
        Si seleccionas 'Ninguna', se usarán los datos originales.
        """
    )

    # Validaciones iniciales de la base de datos
    if not db or not db.is_initialized:
        st.warning("⚠️ Base de datos no inicializada. No se puede optimizar.")
        return
    db_count = db.count()
    if db_count <= 0:
        st.warning(
            f"⚠️ Base de datos '{getattr(db, 'collection_name', 'N/A')}' vacía o inaccesible (Count={db_count}). No hay datos para optimizar."
        )
        # Limpiar estado de reducción si la BD está vacía
        if STATE_REDUCED_EMBEDDINGS in st.session_state:
            del st.session_state[STATE_REDUCED_EMBEDDINGS]
        if STATE_REDUCED_IDS in st.session_state:
            del st.session_state[STATE_REDUCED_IDS]
        return

    st.divider() # Separador

    # Selección de la técnica de reducción
    reduction_method = st.selectbox(
        "Técnica de Reducción de Dimensionalidad:",
        options=["Ninguna (Usar datos originales)", "PCA", "UMAP"],
        key="reduction_method_select", # Clave única
        index=0, # Por defecto, no hacer nada
        help="**Ninguna:** Usa los embeddings tal cual están en la BD. "
             "**PCA:** Rápido, bueno para capturar varianza global, lineal. "
             "**UMAP:** Suele preservar mejor la estructura local/global, no lineal, puede ser más lento."
    )

    # Parámetros específicos de la técnica seleccionada
    params = {}
    target_dim_display = "N/A" # Para mostrar en el botón

    if reduction_method == "PCA":
        # Parámetro n_components para PCA
        pca_n_components = st.slider(
            "Dimensiones Objetivo (PCA):",
            min_value=2,
            # Máximo: dimensión original - 1 (o 512 como límite práctico), y menor que el número de muestras
            max_value=min(512, db_count - 1 if db_count > 1 else 2),
            value=min(32, db_count - 1 if db_count > 1 else 2), # Valor inicial razonable
            step=1, # Pasos de 1 dimensión
            key="pca_n_components", # Clave única
            help="Número de dimensiones a las que reducir usando PCA. Debe ser menor que la dimensión original.",
        )
        params["n_components"] = pca_n_components
        target_dim_display = str(pca_n_components)
    elif reduction_method == "UMAP":
        # Verificar si UMAP está disponible
        if clustering.umap is None:
            st.error(
                "❌ La biblioteca 'umap-learn' no está instalada. No se puede usar UMAP. "
                "Instálala con: `pip install umap-learn`"
            )
            return # No continuar si UMAP no está

        # Parámetro n_components para UMAP
        umap_n_components = st.slider(
            "Dimensiones Objetivo (UMAP):",
            min_value=2,
            # Máximo: dimensión original - 1 (o 128 como límite práctico), y menor que el número de muestras
            max_value=min(128, db_count - 1 if db_count > 1 else 2),
            value=min(16, db_count - 1 if db_count > 1 else 2), # Valor inicial común
            step=1,
            key="umap_n_components", # Clave única
            help="Número de dimensiones a las que reducir usando UMAP. Debe ser menor que la dimensión original.",
        )
        params["n_components"] = umap_n_components
        target_dim_display = str(umap_n_components)

        # Otros parámetros de UMAP
        params["n_neighbors"] = st.slider(
            "Vecinos Cercanos (n_neighbors):",
            min_value=2,
            max_value=min(200, db_count - 1 if db_count > 1 else 2), # Límite superior
            value=min(15, db_count - 1 if db_count > 1 else 2), # Valor por defecto común
            step=1,
            key="umap_n_neighbors", # Clave única
            help="Controla el balance entre estructura local (valores bajos <10) y global (valores altos >50).",
        )
        params["min_dist"] = st.slider(
            "Distancia Mínima (min_dist):",
            min_value=0.0,
            max_value=0.99,
            value=0.1, # Valor por defecto común
            step=0.05,
            key="umap_min_dist", # Clave única
            help="Controla cuán agrupados quedan los puntos en el espacio reducido. Valores bajos = más agrupados.",
        )
        params["metric"] = st.selectbox(
            "Métrica de Distancia (metric):",
            options=["cosine", "euclidean", "l2", "manhattan"], # Opciones comunes
            index=0, # Cosine por defecto para embeddings
            key="umap_metric", # Clave única
            help="Métrica usada por UMAP para encontrar vecinos cercanos en el espacio original.",
        )

    st.divider() # Separador

    # Botón para aplicar la reducción (si se seleccionó una)
    if reduction_method != "Ninguna (Usar datos originales)":
        if st.button(
            f"📉 Aplicar {reduction_method} (a {target_dim_display} dims) y Guardar",
            key=f"apply_{reduction_method}_btn", # Clave única por método
            type="primary" # Botón primario
        ):
            collection_name = getattr(db, "collection_name", "N/A")
            st.info(
                f"Iniciando reducción con {reduction_method} sobre los datos de '{collection_name}'..."
            )
            # Mostrar spinner durante la operación
            with st.spinner(
                f"⏳ Aplicando {reduction_method}... Esto puede tardar, especialmente UMAP en datos grandes."
            ):
                try:
                    # 1. Obtener los embeddings originales de la BD
                    start_fetch_time = time.time()
                    logger.info(
                        f"Fetching original embeddings from '{collection_name}' for {reduction_method}..."
                    )
                    # Usar la función de la BD para obtener todos los IDs y embeddings
                    ids, embeddings = db.get_all_embeddings_with_ids(
                        pagination_batch_size=5000 # Tamaño de página razonable
                    )
                    end_fetch_time = time.time()

                    # Validar datos obtenidos
                    if (
                        embeddings is None
                        or not isinstance(embeddings, np.ndarray)
                        or embeddings.shape[0] == 0
                        or not isinstance(ids, list)
                        or len(ids) != embeddings.shape[0]
                    ):
                        st.error(
                            f"❌ No se pudieron obtener embeddings originales válidos de la base de datos '{collection_name}'."
                        )
                        logger.error(f"Failed to fetch valid original embeddings from '{collection_name}'.")
                        return # Detener si falla la carga

                    original_dim = embeddings.shape[1]
                    num_points = embeddings.shape[0]
                    logger.info(
                        f"Fetched {num_points} embeddings (Original Dim: {original_dim}) in {end_fetch_time - start_fetch_time:.2f}s."
                    )

                    # Validar dimensión objetivo contra dimensión original
                    target_dim_int = params.get("n_components")
                    if isinstance(target_dim_int, int) and target_dim_int >= original_dim:
                        st.error(
                            f"❌ La dimensión objetivo ({target_dim_int}) debe ser menor que la dimensión original ({original_dim})."
                        )
                        logger.error(f"Target dimension {target_dim_int} >= original dimension {original_dim}.")
                        return # Detener si la dimensión es inválida

                    # 2. Aplicar la reducción de dimensionalidad seleccionada
                    start_reduce_time = time.time()
                    reduced_embeddings = None
                    if reduction_method == "PCA":
                        reduced_embeddings = clustering.apply_pca(
                            embeddings, **params
                        )
                    elif reduction_method == "UMAP":
                        reduced_embeddings = clustering.apply_umap(
                            embeddings, **params
                        )
                    end_reduce_time = time.time()
                    reduction_duration = end_reduce_time - start_reduce_time

                    # 3. Guardar resultados en el estado de sesión si la reducción fue exitosa
                    if reduced_embeddings is not None and reduced_embeddings.shape[0] == num_points:
                        new_dim = reduced_embeddings.shape[1]
                        st.success(
                            f"¡{reduction_method} aplicado en {reduction_duration:.2f}s! "
                            f"Nuevas dimensiones: {new_dim}"
                        )
                        # Guardar en el estado de sesión
                        st.session_state[STATE_REDUCED_EMBEDDINGS] = reduced_embeddings
                        st.session_state[STATE_REDUCED_IDS] = ids
                        logger.info(f"Reduced embeddings (Shape: {reduced_embeddings.shape}) and IDs saved to session state.")
                        st.info(
                            f"✅ Embeddings reducidos a **{new_dim} dimensiones** guardados. "
                            "Ve a 'Ejecutar Clustering' para usarlos."
                        )
                        # Limpiar resultados de clustering anteriores, ya que los datos cambiaron
                        if STATE_CLUSTER_LABELS in st.session_state:
                            del st.session_state[STATE_CLUSTER_LABELS]
                            logger.debug("Cleared previous cluster labels from session state.")
                        if STATE_CLUSTER_IDS in st.session_state:
                            del st.session_state[STATE_CLUSTER_IDS]
                            logger.debug("Cleared previous cluster IDs from session state.")
                        # Forzar re-renderizado para mostrar estado actualizado
                        st.rerun()

                    else:
                        # Si la función de reducción falló o devolvió datos inconsistentes
                        st.error(f"❌ Falló la aplicación de {reduction_method}.")
                        logger.error(f"Reduction function {reduction_method} failed or returned inconsistent data.")

                # Manejo de errores específicos del pipeline
                except (PipelineError, DatabaseError, ValueError, ImportError) as e:
                    st.error(
                        f"❌ Error durante la reducción ({reduction_method}): {e}"
                    )
                    logger.error(
                        f"Error applying {reduction_method}", exc_info=True
                    )
                # Manejo de errores inesperados
                except Exception as e:
                    st.error(
                        f"❌ Error inesperado durante la reducción ({reduction_method}): {e}"
                    )
                    logger.error(
                        f"Unexpected error applying {reduction_method}",
                        exc_info=True,
                    )

    # Lógica si se selecciona "Ninguna"
    elif reduction_method == "Ninguna (Usar datos originales)":
        # Si había datos reducidos guardados, ofrecer limpiarlos
        if STATE_REDUCED_EMBEDDINGS in st.session_state:
            st.warning("⚠️ Hay datos reducidos guardados de una ejecución anterior.")
            if st.button(
                "🗑️ Limpiar Datos Reducidos Guardados",
                key="clear_reduced_data_explicit",
                help="Elimina los datos reducidos para asegurar que se usen los originales."
            ):
                del st.session_state[STATE_REDUCED_EMBEDDINGS]
                if STATE_REDUCED_IDS in st.session_state:
                    del st.session_state[STATE_REDUCED_IDS]
                logger.info("User explicitly cleared reduced data from session state.")
                st.success(
                    "Datos reducidos eliminados. Se usarán los datos originales para clustering."
                )
                # Forzar re-renderizado para reflejar el cambio
                st.rerun()

    st.divider() # Separador final

    # Mostrar estado actual (si hay datos reducidos o no)
    if (
        STATE_REDUCED_EMBEDDINGS in st.session_state
        and st.session_state[STATE_REDUCED_EMBEDDINGS] is not None
    ):
        # Si hay datos reducidos activos
        reduced_data_preview = st.session_state[STATE_REDUCED_EMBEDDINGS]
        num_pts = reduced_data_preview.shape[0]
        num_dims = reduced_data_preview.shape[1]
        st.success( # Usar success para indicar estado activo
            f"✅ **Datos Reducidos Activos:** {num_pts} puntos, {num_dims} dimensiones. "
            "Estos se usarán en 'Ejecutar Clustering'."
        )
        # Botón para limpiar estos datos activos
        if st.button(
            "🗑️ Limpiar Datos Reducidos Activos", key="clear_reduced_data_active",
            help="Vuelve a usar los datos originales eliminando los reducidos."
        ):
            del st.session_state[STATE_REDUCED_EMBEDDINGS]
            if STATE_REDUCED_IDS in st.session_state:
                del st.session_state[STATE_REDUCED_IDS]
            logger.info("User cleared active reduced data from session state.")
            st.success("Datos reducidos eliminados de la sesión.")
            # Forzar re-renderizado
            st.rerun()
    else:
        # Si no hay datos reducidos activos
        st.info(
            "ℹ️ No hay datos reducidos activos. 'Ejecutar Clustering' usará los datos originales de la base de datos."
        )
