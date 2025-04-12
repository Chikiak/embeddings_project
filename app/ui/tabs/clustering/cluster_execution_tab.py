# --- app/ui/tabs/clustering/cluster_execution_tab.py ---
import logging
import time
from typing import Optional

import numpy as np
# import pandas as pd # No se usa directamente aquí
import streamlit as st

from app import clustering
from app.exceptions import DatabaseError, PipelineError
from app.ui.state_utils import (
    STATE_CLUSTER_IDS,
    STATE_CLUSTER_LABELS,
    STATE_REDUCED_EMBEDDINGS,
    STATE_REDUCED_IDS,
)
from core.vectorizer import Vectorizer # No se usa directamente, pero podría ser útil
from data_access.vector_db_interface import VectorDBInterface

logger = logging.getLogger(__name__)


def render_cluster_execution_tab(
    vectorizer: Vectorizer, db: VectorDBInterface, truncate_dim: Optional[int]
):
    """
    Renderiza el contenido y maneja la lógica para la pestaña 'Ejecutar Clustering'.

    Args:
        vectorizer: Instancia del Vectorizer (puede ser útil para info).
        db: Instancia de la interfaz de BD vectorial inicializada.
        truncate_dim: Dimensión de truncamiento seleccionada (puede ser None).
    """
    st.subheader("⚙️ Ejecutar Algoritmo de Clustering") # Subheader
    st.markdown(
        """
        Selecciona un algoritmo y sus parámetros para agrupar **todas** las imágenes
        de la colección activa (`{coll_name}`). Si aplicaste una optimización (PCA/UMAP)
        en la pestaña anterior, el clustering se ejecutará sobre esos datos reducidos;
        de lo contrario, usará los embeddings originales.
        """.format(coll_name=getattr(db, 'collection_name', 'N/A'))
    )

    # Validaciones iniciales de la base de datos
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
        # Limpiar estados de clustering si la BD está vacía
        if STATE_CLUSTER_LABELS in st.session_state:
            del st.session_state[STATE_CLUSTER_LABELS]
        if STATE_CLUSTER_IDS in st.session_state:
            del st.session_state[STATE_CLUSTER_IDS]
        # También limpiar datos reducidos si la BD está vacía
        if STATE_REDUCED_EMBEDDINGS in st.session_state:
            del st.session_state[STATE_REDUCED_EMBEDDINGS]
        if STATE_REDUCED_IDS in st.session_state:
            del st.session_state[STATE_REDUCED_IDS]
        return

    st.divider() # Separador

    # Determinar qué datos usar (originales o reducidos)
    embeddings_to_use: Optional[np.ndarray] = None
    ids_to_use: Optional[list] = None
    data_source_info = f"embeddings originales de la colección '{getattr(db, 'collection_name', 'N/A')}'"
    data_dimension = "N/A"

    # Comprobar si hay datos reducidos válidos en el estado de sesión
    if (
        STATE_REDUCED_EMBEDDINGS in st.session_state
        and st.session_state[STATE_REDUCED_EMBEDDINGS] is not None
    ):
        reduced_embeddings = st.session_state[STATE_REDUCED_EMBEDDINGS]
        reduced_ids = st.session_state.get(STATE_REDUCED_IDS)
        # Validar que los datos reducidos sean consistentes
        if (
            isinstance(reduced_embeddings, np.ndarray)
            and reduced_embeddings.ndim == 2
            and isinstance(reduced_ids, list)
            and len(reduced_ids) == reduced_embeddings.shape[0]
            and reduced_embeddings.shape[0] > 0 # Asegurar que no estén vacíos
        ):
            embeddings_to_use = reduced_embeddings
            ids_to_use = reduced_ids
            data_dimension = embeddings_to_use.shape[1]
            data_source_info = f"**datos reducidos** en memoria ({embeddings_to_use.shape[0]} puntos, {data_dimension} dimensiones)"
            st.success( # Usar success para indicar que se usarán datos reducidos
                f"✅ Se utilizarán los {data_source_info} para el clustering."
            )
        else:
            # Si los datos reducidos son inválidos, advertir y limpiar
            st.warning(
                "⚠️ Se encontraron datos reducidos en sesión pero parecen inválidos o corruptos. Se usarán los datos originales."
            )
            logger.warning("Invalid reduced data found in session state. Clearing and using original data.")
            if STATE_REDUCED_EMBEDDINGS in st.session_state:
                del st.session_state[STATE_REDUCED_EMBEDDINGS]
            if STATE_REDUCED_IDS in st.session_state:
                del st.session_state[STATE_REDUCED_IDS]
            # Resetear variables para usar datos originales
            embeddings_to_use = None
            ids_to_use = None
            data_source_info = f"embeddings originales de la colección '{getattr(db, 'collection_name', 'N/A')}'"
            data_dimension = "N/A" # Se determinará al cargar
            st.info(f"ℹ️ Se usarán los {data_source_info}.") # Informar que se usarán originales
    else:
         st.info(f"ℹ️ Se usarán los {data_source_info}.") # Informar si no había datos reducidos

    st.divider() # Separador

    # Selección del algoritmo de clustering
    cluster_algo = st.selectbox(
        "Algoritmo de Clustering:",
        options=["MiniBatchKMeans", "HDBSCAN"],
        key="cluster_algo_select", # Clave única
        help="**MiniBatchKMeans:** Rápido, bueno para clusters esféricos, requiere especificar 'k' (número de clusters). "
             "**HDBSCAN:** Más flexible (formas arbitrarias), no requiere 'k', detecta ruido, puede ser más lento.",
    )

    # Parámetros específicos del algoritmo
    params = {}
    metric_for_eval = "euclidean" # Métrica por defecto para evaluación interna

    if cluster_algo == "MiniBatchKMeans":
        # Parámetro k (número de clusters)
        k_value = st.number_input(
            "Número de Clusters (k):",
            min_value=2,
            max_value=max(2, db_count -1), # k no puede ser mayor que el número de puntos
            value=min(5, max(2, db_count -1)), # Valor inicial razonable
            step=1,
            key="kmeans_k",
            help="Define cuántos grupos se intentarán encontrar."
        )
        params["n_clusters"] = k_value

        # Validar k contra el número de puntos (si ya se conocen los datos reducidos)
        if embeddings_to_use is not None and k_value > embeddings_to_use.shape[0]:
            st.warning(
                f"El número de clusters 'k' ({k_value}) es mayor que el número de puntos disponibles ({embeddings_to_use.shape[0]}). "
                f"Se ajustará k a {embeddings_to_use.shape[0]}."
            )
            params["n_clusters"] = embeddings_to_use.shape[0]
            # No se necesita rerun aquí, el valor se usa directamente en la llamada

        # Otros parámetros de KMeans
        params["batch_size"] = st.number_input(
            "Tamaño de Lote (Batch Size):",
            min_value=32,
            value=min(1024, db_count), # Valor inicial basado en tamaño de BD
            step=64,
            key="kmeans_batch",
            help="Número de muestras usadas en cada iteración de MiniBatch. Más grande puede ser más preciso pero más lento."
        )
        params["n_init"] = st.number_input(
            "Número de Inicializaciones (n_init):",
            min_value=1,
            value=3, # Valor estándar
            step=1,
            key="kmeans_ninit",
            help="Ejecuta K-Means varias veces con diferentes centroides iniciales y elige el mejor resultado (según inercia)."
        )
        st.caption(
            "Nota: MiniBatchKMeans usa distancia Euclidiana internamente. Para simular similitud coseno, los datos se normalizan (L2) antes de ejecutar el algoritmo."
        )
        # La evaluación se hará como si fuera coseno porque es lo más probable para embeddings
        metric_for_eval = "cosine"

    elif cluster_algo == "HDBSCAN":
        # Parámetros de HDBSCAN
        params["min_cluster_size"] = st.number_input(
            "Tamaño Mínimo de Cluster (min_cluster_size):",
            min_value=2,
            value=min(5, max(2, db_count -1)), # Valor inicial razonable
            step=1,
            key="hdbscan_min_cluster",
            help="El número mínimo de puntos necesarios para formar un cluster."
        )
        # min_samples a menudo se deja igual que min_cluster_size o ligeramente menor
        default_min_samples = params["min_cluster_size"]
        params["min_samples"] = st.number_input(
            "Muestras Mínimas (min_samples):",
            min_value=1,
            value=default_min_samples,
            step=1,
            key="hdbscan_min_samples",
            help="Controla cuán conservador es el algoritmo al declarar puntos como ruido (más alto = más ruido). A menudo igual a min_cluster_size.",
        )
        # Métrica de distancia para HDBSCAN
        params["metric"] = st.selectbox(
            "Métrica de Distancia:",
            options=["cosine", "euclidean", "l2", "manhattan"], # Opciones comunes
            index=0, # Cosine por defecto para embeddings
            key="hdbscan_metric",
            help="Métrica usada por HDBSCAN para calcular distancias y densidad.",
        )
        # Usar la misma métrica para la evaluación interna
        metric_for_eval = params["metric"]

    st.divider() # Separador antes del botón

    # Botón para ejecutar el clustering
    if st.button(
        f"🚀 Ejecutar Clustering ({cluster_algo})",
        key="run_clustering_btn",
        type="primary" # Botón primario
    ):
        collection_name = getattr(db, "collection_name", "N/A")
        st.info(
            f"Iniciando clustering ({cluster_algo}) sobre {data_source_info}..."
        )
        # Mostrar spinner durante la ejecución
        with st.spinner(f"⏳ Ejecutando {cluster_algo}... Esto puede tardar unos segundos o minutos."):
            try:
                # Si no se usaron datos reducidos, cargar los originales ahora
                if embeddings_to_use is None or ids_to_use is None:
                    logger.info(
                        f"Obteniendo embeddings originales de '{collection_name}' para clustering..."
                    )
                    load_start_time = time.time()
                    # Obtener todos los embeddings y IDs
                    ids_to_use, embeddings_to_use = (
                        db.get_all_embeddings_with_ids(
                            pagination_batch_size=5000 # Tamaño de página razonable
                        )
                    )
                    load_end_time = time.time()
                    # Validar los datos cargados
                    if (
                        embeddings_to_use is None
                        or not isinstance(embeddings_to_use, np.ndarray)
                        or embeddings_to_use.shape[0] == 0
                        or not isinstance(ids_to_use, list)
                        or len(ids_to_use) != embeddings_to_use.shape[0]
                    ):
                        st.error(
                            f"❌ No se pudieron obtener embeddings válidos de la base de datos '{collection_name}'."
                        )
                        logger.error(f"Failed to fetch valid embeddings from DB collection '{collection_name}'.")
                        return # Detener ejecución

                    data_dimension = embeddings_to_use.shape[1]
                    logger.info(f"Cargados {embeddings_to_use.shape[0]} embeddings originales (Dim: {data_dimension}) en {load_end_time - load_start_time:.2f}s.")

                # Ahora tenemos embeddings_to_use y ids_to_use definidos
                num_points = embeddings_to_use.shape[0]
                current_dim = embeddings_to_use.shape[1]
                logger.info(
                    f"Ejecutando {cluster_algo} sobre {num_points} embeddings con dimensión {current_dim}."
                )

                # Variables para resultados del clustering
                labels: Optional[np.ndarray] = None
                cluster_stats = {} # Para estadísticas adicionales (ej: inercia, ruido)
                start_cluster_time = time.time()

                # Ejecutar el algoritmo seleccionado
                if cluster_algo == "MiniBatchKMeans":
                    # Re-validar k justo antes de llamar
                    if params["n_clusters"] > num_points:
                        st.warning(
                            f"Ajustando k de {params['n_clusters']} a {num_points} porque hay menos puntos."
                        )
                        params["n_clusters"] = num_points
                    # Llamar a la función de clustering
                    labels, inertia = clustering.run_minibatch_kmeans(
                        embeddings_to_use, **params
                    )
                    cluster_stats["inertia"] = inertia # Guardar inercia
                elif cluster_algo == "HDBSCAN":
                    # Llamar a la función de clustering
                    labels, cluster_stats = clustering.run_hdbscan(
                        embeddings_to_use, **params
                    ) # cluster_stats ya viene con info de ruido/clusters

                end_cluster_time = time.time()
                cluster_duration = end_cluster_time - start_cluster_time

                # Validar el resultado (labels)
                if labels is None or not isinstance(labels, np.ndarray) or labels.shape[0] != num_points:
                    st.error(
                        f"❌ El algoritmo de clustering ({cluster_algo}) no devolvió etiquetas válidas."
                    )
                    logger.error(f"Clustering algorithm {cluster_algo} did not return valid labels.")
                    return # Detener ejecución

                st.success(
                    f"¡Clustering ({cluster_algo}) completado en {cluster_duration:.2f}s!"
                )

                # Calcular métricas internas de evaluación
                logger.info(
                    f"Calculando métricas internas usando métrica: '{metric_for_eval}'"
                )
                metrics = clustering.calculate_internal_metrics(
                    embeddings_to_use, labels, metric=metric_for_eval
                )

                # Obtener número de clusters y ruido de las estadísticas o calculándolo
                num_clusters_found = cluster_stats.get(
                    "num_clusters", # Intentar obtener de stats (HDBSCAN)
                    len(set(labels)) - (1 if -1 in labels else 0), # Calcular si no está
                )
                num_noise_points = cluster_stats.get(
                    "num_noise_points", # Intentar obtener de stats (HDBSCAN)
                    np.sum(labels == -1) # Calcular si no está
                )

                # Mostrar resultados principales
                st.subheader("📊 Resumen del Clustering")
                col_res1, col_res2 = st.columns(2)
                col_res1.metric("Clusters Encontrados", num_clusters_found)
                col_res2.metric("Puntos de Ruido (No agrupados)", num_noise_points)
                if cluster_algo == "MiniBatchKMeans" and "inertia" in cluster_stats:
                     st.metric("Inercia (KMeans)", f"{cluster_stats['inertia']:.2f}", help="Suma de distancias cuadradas a los centroides más cercanos (menor es mejor).")

                st.divider() # Separador

                # Mostrar métricas de evaluación interna
                st.subheader("📈 Métricas de Evaluación (Internas)")
                st.caption(f"Calculadas usando la métrica: `{metric_for_eval}`. Excluyen puntos de ruido.")
                col_m1, col_m2, col_m3 = st.columns(3)
                # Silhouette Score
                silhouette_val = metrics.get("silhouette")
                col_m1.metric(
                    "Silhouette Score",
                    f"{silhouette_val:.4f}" if silhouette_val is not None else "N/A",
                    help="Rango [-1, 1]. Más alto es mejor (clusters densos y bien separados). Ideal > 0.5.",
                    # Delta opcional para indicar calidad (ejemplo simple)
                    delta=f"{silhouette_val - 0.5:.2f}" if silhouette_val is not None and silhouette_val > 0 else None,
                    delta_color="normal" # O "inverse" si menor es mejor
                )
                # Davies-Bouldin Index
                db_val = metrics.get("davies_bouldin")
                col_m2.metric(
                    "Davies-Bouldin Index",
                    f"{db_val:.4f}" if db_val is not None else "N/A",
                    help="Rango >= 0. Más bajo es mejor (clusters compactos y bien separados). Ideal < 1.0.",
                     delta=f"{db_val - 1.0:.2f}" if db_val is not None else None,
                     delta_color="inverse" # Menor es mejor
                )
                # Calinski-Harabasz Index
                ch_val = metrics.get("calinski_harabasz")
                col_m3.metric(
                    "Calinski-Harabasz Index",
                    f"{ch_val:.2f}" if ch_val is not None else "N/A",
                    help="Ratio varianza inter-cluster / intra-cluster. Más alto es mejor (mayor separación).",
                )

                # Guardar resultados en estado de sesión para la pestaña de visualización
                st.session_state[STATE_CLUSTER_IDS] = ids_to_use
                st.session_state[STATE_CLUSTER_LABELS] = labels
                st.info(
                    "✅ Resultados del clustering (IDs y etiquetas) guardados. Ve a la pestaña 'Visualizar Resultados' para ver el gráfico."
                )

            # Manejo de errores específicos del pipeline de clustering
            except (PipelineError, DatabaseError, ValueError, ImportError) as e:
                st.error(f"❌ Error durante el clustering ({cluster_algo}): {e}")
                logger.error(
                    f"Error running clustering ({cluster_algo}) from Streamlit", exc_info=True
                )
                # Limpiar resultados parciales si falla
                if STATE_CLUSTER_LABELS in st.session_state: del st.session_state[STATE_CLUSTER_LABELS]
                if STATE_CLUSTER_IDS in st.session_state: del st.session_state[STATE_CLUSTER_IDS]
            # Manejo de errores inesperados
            except Exception as e:
                st.error(f"❌ Error inesperado durante el clustering ({cluster_algo}): {e}")
                logger.error(
                    f"Unexpected error running clustering ({cluster_algo}) from Streamlit",
                    exc_info=True,
                )
                # Limpiar resultados parciales si falla
                if STATE_CLUSTER_LABELS in st.session_state: del st.session_state[STATE_CLUSTER_LABELS]
                if STATE_CLUSTER_IDS in st.session_state: del st.session_state[STATE_CLUSTER_IDS]

