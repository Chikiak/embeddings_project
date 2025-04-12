import logging
import time
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

from app import clustering
from app.exceptions import DatabaseError, PipelineError
from app.ui.state_utils import (
    STATE_CLUSTER_IDS,
    STATE_CLUSTER_LABELS,
    STATE_REDUCED_EMBEDDINGS,
    STATE_REDUCED_IDS,
)
from core.vectorizer import Vectorizer
from data_access.vector_db_interface import VectorDBInterface

logger = logging.getLogger(__name__)


def render_cluster_execution_tab(
    vectorizer: Vectorizer, db: VectorDBInterface, truncate_dim: Optional[int]
):
    """Renders the tab for executing clustering algorithms."""
    st.header("⚙️ Ejecutar Algoritmo de Clustering")
    st.markdown(
        "Selecciona un algoritmo y sus parámetros para agrupar **todas** las imágenes de la colección activa. "
        "Si se aplicó una optimización (PCA/UMAP) en la pestaña 'Optimizar', el clustering se ejecutará sobre esos datos reducidos."
    )

    if not db or not db.is_initialized:
        st.warning(
            "⚠️ Base de datos no inicializada. No se puede ejecutar clustering."
        )
        return
    db_count = db.count()
    if db_count <= 0:
        st.warning(
            f"⚠️ Base de datos '{getattr(db, 'collection_name', 'N/A')}' vacía o inaccesible (Count={db_count}). No hay datos para agrupar."
        )

        if STATE_CLUSTER_LABELS in st.session_state:
            del st.session_state[STATE_CLUSTER_LABELS]
        if STATE_CLUSTER_IDS in st.session_state:
            del st.session_state[STATE_CLUSTER_IDS]
        if STATE_REDUCED_EMBEDDINGS in st.session_state:
            del st.session_state[STATE_REDUCED_EMBEDDINGS]
        if STATE_REDUCED_IDS in st.session_state:
            del st.session_state[STATE_REDUCED_IDS]
        return

    embeddings_to_use = None
    ids_to_use = None
    data_source_info = f"datos originales de la colección '{getattr(db, 'collection_name', 'N/A')}'"

    if (
        STATE_REDUCED_EMBEDDINGS in st.session_state
        and st.session_state[STATE_REDUCED_EMBEDDINGS] is not None
    ):
        reduced_embeddings = st.session_state[STATE_REDUCED_EMBEDDINGS]
        reduced_ids = st.session_state.get(STATE_REDUCED_IDS)
        if (
            reduced_embeddings is not None
            and reduced_ids is not None
            and len(reduced_ids) == reduced_embeddings.shape[0]
        ):
            embeddings_to_use = reduced_embeddings
            ids_to_use = reduced_ids
            data_source_info = f"datos reducidos en memoria ({embeddings_to_use.shape[1]} dimensiones)"
            st.info(
                f"ℹ️ Se utilizarán los {data_source_info} para el clustering."
            )
        else:
            st.warning(
                "⚠️ Se encontraron datos reducidos en sesión pero parecen inválidos. Se usarán los datos originales."
            )

            if STATE_REDUCED_EMBEDDINGS in st.session_state:
                del st.session_state[STATE_REDUCED_EMBEDDINGS]
            if STATE_REDUCED_IDS in st.session_state:
                del st.session_state[STATE_REDUCED_IDS]

    cluster_algo = st.selectbox(
        "Algoritmo de Clustering:",
        options=["MiniBatchKMeans", "HDBSCAN"],
        key="cluster_algo_select",
        help="MiniBatchKMeans es rápido pero asume clusters esféricos. HDBSCAN es más flexible pero puede ser más lento.",
    )

    params = {}
    metric_for_eval = "euclidean"

    if cluster_algo == "MiniBatchKMeans":
        params["n_clusters"] = st.number_input(
            "Número de Clusters (k):",
            min_value=2,
            value=5,
            step=1,
            key="kmeans_k",
        )

        if (
            embeddings_to_use is not None
            and params["n_clusters"] > embeddings_to_use.shape[0]
        ):
            st.warning(
                f"k ({params['n_clusters']}) es mayor que el número de puntos ({embeddings_to_use.shape[0]}). Ajustando k."
            )
            params["n_clusters"] = max(2, embeddings_to_use.shape[0])
            st.rerun()

        params["batch_size"] = st.number_input(
            "Tamaño de Lote (KMeans):",
            min_value=32,
            value=min(1024, db_count),
            step=64,
            key="kmeans_batch",
        )
        params["n_init"] = st.number_input(
            "Número de Inicializaciones (n_init):",
            min_value=1,
            value=3,
            step=1,
            key="kmeans_ninit",
            help="Ejecutar K-Means varias veces con diferentes semillas para mejor resultado.",
        )
        st.caption(
            "Nota: MiniBatchKMeans usa distancia Euclidiana. Los datos se normalizan (L2) internamente para simular similitud coseno."
        )
        metric_for_eval = "cosine"

    elif cluster_algo == "HDBSCAN":
        params["min_cluster_size"] = st.number_input(
            "Tamaño Mínimo de Cluster:",
            min_value=2,
            value=5,
            step=1,
            key="hdbscan_min_cluster",
        )

        default_min_samples = params["min_cluster_size"]
        params["min_samples"] = st.number_input(
            "Muestras Mínimas (min_samples):",
            min_value=1,
            value=default_min_samples,
            step=1,
            key="hdbscan_min_samples",
            help="Controla la densidad. A menudo igual a min_cluster_size.",
        )
        params["metric"] = st.selectbox(
            "Métrica de Distancia:",
            options=["cosine", "euclidean", "l2"],
            index=0,
            key="hdbscan_metric",
            help="Métrica usada por HDBSCAN para calcular distancias.",
        )
        metric_for_eval = params["metric"]

    st.divider()
    if st.button(
        "🚀 Ejecutar Clustering", key="run_clustering_btn", type="primary"
    ):
        collection_name = getattr(db, "collection_name", "N/A")
        st.info(
            f"Iniciando clustering ({cluster_algo}) sobre {data_source_info}..."
        )
        with st.spinner(f"Ejecutando {cluster_algo}... Por favor espera."):
            try:

                if embeddings_to_use is None or ids_to_use is None:
                    logger.info(
                        f"Obteniendo embeddings originales de '{collection_name}' para clustering..."
                    )
                    ids_to_use, embeddings_to_use = (
                        db.get_all_embeddings_with_ids(
                            pagination_batch_size=5000
                        )
                    )
                    if (
                        embeddings_to_use is None
                        or embeddings_to_use.shape[0] == 0
                    ):
                        st.error(
                            "No se pudieron obtener embeddings de la base de datos."
                        )
                        return

                logger.info(
                    f"Ejecutando {cluster_algo} sobre {embeddings_to_use.shape[0]} embeddings con dimensión {embeddings_to_use.shape[1]}."
                )

                labels = None
                cluster_stats = {}
                start_cluster_time = time.time()
                if cluster_algo == "MiniBatchKMeans":

                    if params["n_clusters"] > embeddings_to_use.shape[0]:
                        st.warning(
                            f"Ajustando k de {params['n_clusters']} a {embeddings_to_use.shape[0]} porque hay menos puntos."
                        )
                        params["n_clusters"] = embeddings_to_use.shape[0]
                    labels, inertia = clustering.run_minibatch_kmeans(
                        embeddings_to_use, **params
                    )
                    cluster_stats["inertia"] = inertia
                elif cluster_algo == "HDBSCAN":
                    labels, cluster_stats = clustering.run_hdbscan(
                        embeddings_to_use, **params
                    )
                end_cluster_time = time.time()

                if labels is None:
                    st.error(
                        "El algoritmo de clustering no devolvió etiquetas."
                    )
                    return

                st.success(
                    f"¡Clustering ({cluster_algo}) completado en {end_cluster_time - start_cluster_time:.2f}s!"
                )

                logger.info(
                    f"Calculando métricas internas usando métrica: '{metric_for_eval}'"
                )
                metrics = clustering.calculate_internal_metrics(
                    embeddings_to_use, labels, metric=metric_for_eval
                )

                num_clusters_found = cluster_stats.get(
                    "num_clusters",
                    len(set(labels)) - (1 if -1 in labels else 0),
                )
                num_noise_points = cluster_stats.get(
                    "num_noise_points", np.sum(labels == -1)
                )

                st.metric("Clusters Encontrados", num_clusters_found)
                if num_noise_points > 0:
                    st.metric(
                        "Puntos de Ruido (No agrupados)", num_noise_points
                    )

                st.subheader("Métricas de Evaluación (Internas)")
                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric(
                    "Silhouette Score",
                    (
                        f"{metrics['silhouette']:.4f}"
                        if metrics["silhouette"] is not None
                        else "N/A"
                    ),
                    help="[-1, 1], más alto es mejor (buena separación/cohesión).",
                )
                col_m2.metric(
                    "Davies-Bouldin",
                    (
                        f"{metrics['davies_bouldin']:.4f}"
                        if metrics["davies_bouldin"] is not None
                        else "N/A"
                    ),
                    help=">= 0, más bajo es mejor (clusters compactos y bien separados).",
                )
                col_m3.metric(
                    "Calinski-Harabasz",
                    (
                        f"{metrics['calinski_harabasz']:.2f}"
                        if metrics["calinski_harabasz"] is not None
                        else "N/A"
                    ),
                    help="> 0, más alto es mejor (ratio varianza inter/intra cluster).",
                )

                st.session_state[STATE_CLUSTER_IDS] = ids_to_use
                st.session_state[STATE_CLUSTER_LABELS] = labels
                st.info(
                    "Resultados del clustering (IDs y etiquetas) guardados en el estado de sesión para la pestaña 'Visualizar'."
                )

            except (
                PipelineError,
                DatabaseError,
                ValueError,
                ImportError,
            ) as e:
                st.error(f"Error durante el clustering: {e}")
                logger.error(
                    "Error running clustering from Streamlit", exc_info=True
                )
            except Exception as e:
                st.error(f"Error inesperado durante el clustering: {e}")
                logger.error(
                    "Unexpected error running clustering from Streamlit",
                    exc_info=True,
                )
