# app/ui/tabs/clustering/cluster_optimization_tab.py
import streamlit as st
import logging
import numpy as np
import time
from typing import Optional

# Project imports
from core.vectorizer import Vectorizer
from data_access.vector_db_interface import VectorDBInterface
from app import clustering # Import the clustering logic module
from app.exceptions import PipelineError, DatabaseError, InitializationError
from app.ui.state_utils import ( # Import state keys
    STATE_REDUCED_EMBEDDINGS,
    STATE_REDUCED_IDS,
    STATE_CLUSTER_LABELS, # To clear if optimization is applied
    STATE_CLUSTER_IDS
)

logger = logging.getLogger(__name__)


def render_cluster_optimization_tab(vectorizer: Vectorizer, db: VectorDBInterface, truncate_dim: Optional[int]):
    """Renders the tab for pre-clustering optimization (dimensionality reduction)."""
    st.header("📉 Optimizar Datos (Pre-Clustering)")
    st.markdown(
        "Aplica técnicas de reducción de dimensionalidad (PCA o UMAP) a los embeddings **antes** de ejecutar el clustering. "
        "Esto puede mejorar el rendimiento y la calidad de algunos algoritmos de clustering en datos de alta dimensión, "
        "especialmente si la dimensionalidad original es muy alta."
        "\n\n**Importante:** Si aplicas una técnica aquí, los datos reducidos se guardarán en la sesión y se usarán "
        "automáticamente en la pestaña 'Ejecutar Clustering'."
    )

    # --- Pre-checks ---
    if not db or not db.is_initialized:
        st.warning("⚠️ Base de datos no inicializada. No se puede optimizar.")
        return
    db_count = db.count()
    if db_count <= 0:
        st.warning(f"⚠️ Base de datos '{getattr(db, 'collection_name', 'N/A')}' vacía o inaccesible (Count={db_count}). No hay datos para optimizar.")
        # Clear potentially stale state if DB is empty
        if STATE_REDUCED_EMBEDDINGS in st.session_state: del st.session_state[STATE_REDUCED_EMBEDDINGS]
        if STATE_REDUCED_IDS in st.session_state: del st.session_state[STATE_REDUCED_IDS]
        return

    # --- Technique Selection ---
    reduction_method = st.selectbox(
        "Técnica de Reducción:",
        options=["Ninguna (Usar datos originales)", "PCA", "UMAP"],
        key="reduction_method_select",
        help="UMAP suele preservar mejor la estructura local/global que PCA, pero PCA es más rápido y determinista."
    )

    params = {}
    target_dim = "N/A"

    # --- Reduction Parameters ---
    if reduction_method == "PCA":
        params['n_components'] = st.slider(
            "Dimensiones Objetivo (PCA):",
            min_value=2,
            max_value=min(512, db_count -1 if db_count > 1 else 2), # Limit max components reasonably
            value=min(32, db_count -1 if db_count > 1 else 2), # Sensible default
            step=1,
            key="pca_n_components",
            help="Número de dimensiones a las que reducir usando PCA."
        )
        target_dim = params['n_components']
    elif reduction_method == "UMAP":
        if clustering.umap is None: # Check if UMAP library is available
             st.error("La biblioteca 'umap-learn' no está instalada. No se puede usar UMAP. Instálala con 'pip install umap-learn'.")
             return # Stop rendering UMAP options if library is missing

        params['n_components'] = st.slider(
            "Dimensiones Objetivo (UMAP):",
            min_value=2,
            max_value=min(128, db_count -1 if db_count > 1 else 2), # Limit max components reasonably
            value=min(16, db_count -1 if db_count > 1 else 2), # Sensible default
            step=1,
            key="umap_n_components",
            help="Número de dimensiones a las que reducir usando UMAP."
        )
        params['n_neighbors'] = st.slider(
            "Vecinos Cercanos (UMAP):",
            min_value=2,
            max_value=min(200, db_count -1 if db_count > 1 else 2), # Limit max neighbors
            value=min(15, db_count -1 if db_count > 1 else 2), # Sensible default
            step=1,
            key="umap_n_neighbors",
            help="Controla el balance estructura local vs global. Valores bajos (<10) enfocan local, altos (>50) global."
        )
        params['min_dist'] = st.slider(
            "Distancia Mínima (UMAP):",
            min_value=0.0,
            max_value=0.99,
            value=0.1,
            step=0.05,
            key="umap_min_dist",
            help="Controla cuán agrupados quedan los puntos en el espacio reducido. Valores bajos = más agrupados."
        )
        params['metric'] = st.selectbox(
            "Métrica (UMAP):",
            options=['cosine', 'euclidean', 'l2', 'manhattan'], # Common metrics
            index=0,
            key="umap_metric",
            help="Métrica de distancia usada por UMAP para encontrar vecinos."
        )
        target_dim = params['n_components']

    # --- Apply Reduction Button ---
    st.divider()
    if reduction_method != "Ninguna (Usar datos originales)":
        if st.button(f"📊 Aplicar {reduction_method} y Guardar en Sesión", key=f"apply_{reduction_method}_btn", type="primary"):
            collection_name = getattr(db, 'collection_name', 'N/A')
            st.info(f"Iniciando reducción con {reduction_method} sobre los datos de '{collection_name}'...")
            with st.spinner(f"Aplicando {reduction_method}... Esto puede tardar unos minutos para UMAP/t-SNE."):
                try:
                    # 1. Get original embeddings
                    start_fetch_time = time.time()
                    logger.info(f"Fetching original embeddings from '{collection_name}' for {reduction_method}...")
                    ids, embeddings = db.get_all_embeddings_with_ids(pagination_batch_size=5000) # Adjust batch size
                    end_fetch_time = time.time()
                    if embeddings is None or embeddings.shape[0] == 0:
                        st.error("No se pudieron obtener embeddings originales de la base de datos.")
                        return # Stop execution
                    logger.info(f"Fetched {embeddings.shape[0]} embeddings (Dim: {embeddings.shape[1]}) in {end_fetch_time - start_fetch_time:.2f}s.")
                    original_dim = embeddings.shape[1]

                    # Check if target dimension is valid
                    if isinstance(target_dim, int) and target_dim >= original_dim:
                        st.error(f"La dimensión objetivo ({target_dim}) debe ser menor que la dimensión original ({original_dim}).")
                        return # Stop execution

                    # 2. Apply selected reduction
                    start_reduce_time = time.time()
                    reduced_embeddings = None
                    if reduction_method == "PCA":
                        reduced_embeddings = clustering.apply_pca(embeddings, **params)
                    elif reduction_method == "UMAP":
                        reduced_embeddings = clustering.apply_umap(embeddings, **params)
                    end_reduce_time = time.time()

                    if reduced_embeddings is not None:
                         st.success(f"¡{reduction_method} aplicado en {end_reduce_time - start_reduce_time:.2f}s! Nuevas dimensiones: {reduced_embeddings.shape[1]}")
                         # Save reduced data to session state for the clustering tab
                         st.session_state[STATE_REDUCED_EMBEDDINGS] = reduced_embeddings
                         st.session_state[STATE_REDUCED_IDS] = ids # Save corresponding IDs
                         st.info(f"Embeddings reducidos a {reduced_embeddings.shape[1]} dimensiones guardados en sesión. "
                                 "Ve a 'Ejecutar Clustering' para usarlos.")
                         # Clear previous clustering results as they are now invalid
                         if STATE_CLUSTER_LABELS in st.session_state: del st.session_state[STATE_CLUSTER_LABELS]
                         if STATE_CLUSTER_IDS in st.session_state: del st.session_state[STATE_CLUSTER_IDS]
                    else:
                         st.error(f"Falló la aplicación de {reduction_method}.")

                except (PipelineError, DatabaseError, ValueError, ImportError) as e:
                    st.error(f"Error durante la reducción ({reduction_method}): {e}")
                    logger.error(f"Error applying {reduction_method}", exc_info=True)
                except Exception as e:
                    st.error(f"Error inesperado durante la reducción ({reduction_method}): {e}")
                    logger.error(f"Unexpected error applying {reduction_method}", exc_info=True)
    elif reduction_method == "Ninguna (Usar datos originales)":
         # If 'None' is selected, clear any existing reduced data from session
         if STATE_REDUCED_EMBEDDINGS in st.session_state:
             if st.button("🗑️ Limpiar Datos Reducidos Guardados", key="clear_reduced_data_explicit"):
                 del st.session_state[STATE_REDUCED_EMBEDDINGS]
                 if STATE_REDUCED_IDS in st.session_state: del st.session_state[STATE_REDUCED_IDS]
                 st.success("Datos reducidos eliminados de la sesión. Se usarán los datos originales para clustering.")
                 st.rerun()


    # --- Display info about saved reduced data ---
    st.divider()
    if STATE_REDUCED_EMBEDDINGS in st.session_state and st.session_state[STATE_REDUCED_EMBEDDINGS] is not None:
        reduced_data_preview = st.session_state[STATE_REDUCED_EMBEDDINGS]
        st.success(f"✅ Datos reducidos activos en memoria ({reduced_data_preview.shape[0]} puntos, {reduced_data_preview.shape[1]} dims). "
                "Estos se usarán en 'Ejecutar Clustering'.")
        if st.button("🗑️ Limpiar Datos Reducidos", key="clear_reduced_data_active"):
             del st.session_state[STATE_REDUCED_EMBEDDINGS]
             if STATE_REDUCED_IDS in st.session_state: del st.session_state[STATE_REDUCED_IDS]
             st.success("Datos reducidos eliminados de la sesión.")
             st.rerun()
    else:
         st.info("ℹ️ No hay datos reducidos activos. 'Ejecutar Clustering' usará los datos originales.")

