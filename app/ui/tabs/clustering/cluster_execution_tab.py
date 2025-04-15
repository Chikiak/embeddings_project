# --- app/ui/tabs/clustering/cluster_execution_tab.py ---
import logging
import time
from typing import Optional, Dict
import numpy as np
import streamlit as st
import pandas as pd # Ensure pandas is imported

from app.services import clustering # Import the full clustering module
from app.exceptions import DatabaseError, PipelineError
from app.ui.state_utils import (
    STATE_CLUSTER_IDS,
    STATE_CLUSTER_LABELS,
    STATE_REDUCED_EMBEDDINGS,
    STATE_REDUCED_IDS,
    STATE_GENERATED_LABELS, # Key for generated labels
)
from core.vectorizer import Vectorizer
from data_access.vector_db_interface import VectorDBInterface

logger = logging.getLogger(__name__)

def render_cluster_execution_tab(
    vectorizer: Vectorizer,
    db: VectorDBInterface, # This 'db' MUST be the instance corresponding to truncate_dim
    truncate_dim: Optional[int]
):
    """
    Renderiza el contenido y maneja la lógica para la pestaña 'Ejecutar Clustering'.
    Ahora también incluye la generación de etiquetas textuales.

    Args:
        vectorizer: Instancia del Vectorizer inicializado.
        db: Instancia de la interfaz de BD vectorial inicializada
            (IMPORTANTE: debe ser la que corresponde a la dimensión `truncate_dim`
             seleccionada en la sidebar, gestionada por streamlit_app.py).
        truncate_dim: Dimensión de truncamiento seleccionada (puede ser None).
    """
    st.subheader("⚙️ Ejecutar Algoritmo de Clustering y Etiquetado")
    st.markdown(
        """
        Selecciona un algoritmo y sus parámetros para agrupar las imágenes
        de la colección activa (`{coll_name}`). Si aplicaste una optimización (PCA/UMAP)
        en la pestaña anterior, el clustering se ejecutará sobre esos datos reducidos;
        de lo contrario, usará los embeddings originales/truncados de la base de datos.
        **Después del clustering, se intentará generar etiquetas textuales automáticamente.**
        """.format(coll_name=getattr(db, 'collection_name', 'N/A'))
    )

    # --- Initial validations (unchanged) ---
    if not db or not db.is_initialized:
        st.warning("⚠️ Base de datos no inicializada.")
        return
    db_count = db.count()
    if db_count <= 0:
        st.warning(f"⚠️ Base de datos '{getattr(db, 'collection_name', 'N/A')}' vacía.")
        # Clear potentially inconsistent state if DB is empty
        keys_to_clear = [STATE_CLUSTER_LABELS, STATE_CLUSTER_IDS, STATE_REDUCED_EMBEDDINGS, STATE_REDUCED_IDS, STATE_GENERATED_LABELS]
        for key in keys_to_clear:
            if key in st.session_state: del st.session_state[key]
        return

    st.divider()

    # --- Determine which data to use (original/truncated or reduced by PCA/UMAP) ---
    embeddings_to_cluster: Optional[np.ndarray] = None # Embeddings used for clustering (N-Dim or M-Dim)
    ids_for_clustering: Optional[list] = None          # IDs corresponding to embeddings_to_cluster
    data_source_info = f"embeddings originales/truncados de '{getattr(db, 'collection_name', 'N/A')}'"
    data_dimension_clustering = "N/A" # Dimension of data used for clustering
    using_reduced_data = False

    # Check if valid reduced data exists in session state
    if STATE_REDUCED_EMBEDDINGS in st.session_state and st.session_state[STATE_REDUCED_EMBEDDINGS] is not None:
        reduced_embeddings = st.session_state[STATE_REDUCED_EMBEDDINGS]
        reduced_ids = st.session_state.get(STATE_REDUCED_IDS)
        # Validate consistency
        if (isinstance(reduced_embeddings, np.ndarray) and reduced_embeddings.ndim == 2 and
            isinstance(reduced_ids, list) and len(reduced_ids) == reduced_embeddings.shape[0] and
            reduced_embeddings.shape[0] > 0):
            embeddings_to_cluster = reduced_embeddings
            ids_for_clustering = reduced_ids
            data_dimension_clustering = embeddings_to_cluster.shape[1]
            data_source_info = f"**datos reducidos** en memoria ({embeddings_to_cluster.shape[0]} puntos, {data_dimension_clustering} dimensiones)"
            st.success(f"✅ Se utilizarán los {data_source_info} para el clustering.")
            using_reduced_data = True
        else:
            # If reduced data is invalid, clear it and use original
            st.warning("⚠️ Datos reducidos en sesión inválidos. Se usarán los datos originales/truncados de la BD.")
            logger.warning("Invalid reduced data found in session state. Clearing and using original/truncated data.")
            keys_to_clear = [STATE_REDUCED_EMBEDDINGS, STATE_REDUCED_IDS]
            for key in keys_to_clear:
                if key in st.session_state: del st.session_state[key]
            # Fallback info message
            data_source_info = f"embeddings originales/truncados de '{getattr(db, 'collection_name', 'N/A')}'"
            st.info(f"ℹ️ Se usarán los {data_source_info}.")
    else:
         # No reduced data found, use original
         st.info(f"ℹ️ Se usarán los {data_source_info}.")

    st.divider()

    # --- Algorithm selection and parameters (unchanged) ---
    cluster_algo = st.selectbox("Algoritmo de Clustering:", ["MiniBatchKMeans", "HDBSCAN"], key="cluster_algo_select")
    params = {}
    metric_for_eval = "euclidean" # Default metric for evaluation
    if cluster_algo == "MiniBatchKMeans":
        # Ensure k is valid based on potential data size
        max_k_value = max(2, db_count - 1 if db_count > 1 else 2)
        default_k_value = min(5, max_k_value)
        k_value = st.number_input("Número de Clusters (k):", min_value=2, max_value=max_k_value, value=default_k_value, step=1, key="kmeans_k")
        params["n_clusters"] = k_value
        # Check against actual data size if available (only matters if reduced data is smaller than db_count)
        if embeddings_to_cluster is not None and k_value > embeddings_to_cluster.shape[0]:
             st.warning(f"Ajustando k a {embeddings_to_cluster.shape[0]} (número de puntos disponibles).")
             params["n_clusters"] = embeddings_to_cluster.shape[0]
        params["batch_size"] = st.number_input("Tamaño de Lote (Batch Size):", min_value=32, value=min(1024, db_count), step=64, key="kmeans_batch")
        params["n_init"] = st.number_input("Número de Inicializaciones (n_init):", min_value=1, value=3, step=1, key="kmeans_ninit")
        st.caption("Nota: MiniBatchKMeans usa datos normalizados (L2) internamente para clustering.")
        metric_for_eval = "cosine" # Use cosine for internal metrics evaluation as KMeans optimizes for it on normalized data
    elif cluster_algo == "HDBSCAN":
        max_min_cluster_size = max(2, db_count - 1 if db_count > 1 else 2)
        default_min_cluster_size = min(5, max_min_cluster_size)
        params["min_cluster_size"] = st.number_input("Tamaño Mínimo de Cluster:", min_value=2, value=default_min_cluster_size, step=1, key="hdbscan_min_cluster")
        # Default min_samples to min_cluster_size
        default_min_samples = params["min_cluster_size"]
        params["min_samples"] = st.number_input("Muestras Mínimas:", min_value=1, value=default_min_samples, step=1, key="hdbscan_min_samples")
        params["metric"] = st.selectbox("Métrica de Distancia:", ["cosine", "euclidean", "l2", "manhattan"], index=0, key="hdbscan_metric")
        metric_for_eval = params["metric"] # Use the same metric for evaluation

    st.divider()

    # --- Button to run clustering and labeling ---
    if st.button(
        f"🚀 Ejecutar Clustering ({cluster_algo}) y Generar Etiquetas",
        key="run_clustering_and_labeling_btn",
        type="primary"
    ):
        collection_name = getattr(db, "collection_name", "N/A")
        st.info(f"Iniciando clustering ({cluster_algo}) sobre {data_source_info}...")

        # Clear previous labeling results
        if STATE_GENERATED_LABELS in st.session_state:
            del st.session_state[STATE_GENERATED_LABELS]

        with st.spinner(f"⏳ Ejecutando {cluster_algo}..."):
            try:
                # 1. Get/Load Embeddings for Clustering (if not using reduced)
                load_start_time = time.time()
                if not using_reduced_data:
                    logger.info(f"Obteniendo embeddings originales/truncados de '{collection_name}' para clustering...")
                    # The 'db' instance already corresponds to the truncate_dim
                    ids_for_clustering, embeddings_to_cluster = db.get_all_embeddings_with_ids(pagination_batch_size=5000) # Adjust batch size if needed
                    if (embeddings_to_cluster is None or not isinstance(embeddings_to_cluster, np.ndarray) or embeddings_to_cluster.shape[0] == 0 or
                        not isinstance(ids_for_clustering, list) or len(ids_for_clustering) != embeddings_to_cluster.shape[0]):
                        st.error(f"❌ No se pudieron obtener embeddings válidos de la base de datos '{collection_name}'.")
                        logger.error(f"Failed to fetch valid embeddings from DB collection '{collection_name}'.")
                        return # Exit if loading fails
                    data_dimension_clustering = embeddings_to_cluster.shape[1]
                    logger.info(f"Cargados {embeddings_to_cluster.shape[0]} embeddings (Dim: {data_dimension_clustering}) en {time.time() - load_start_time:.2f}s.")
                else:
                    # If using reduced, they are already loaded
                    data_dimension_clustering = embeddings_to_cluster.shape[1]
                    logger.info(f"Usando {embeddings_to_cluster.shape[0]} embeddings reducidos (Dim: {data_dimension_clustering}).")

                # Validate that we have data to cluster
                if embeddings_to_cluster is None or ids_for_clustering is None or embeddings_to_cluster.shape[0] == 0:
                     st.error("❌ No hay datos válidos disponibles para ejecutar el clustering.")
                     return

                # 2. Execute Clustering
                num_points = embeddings_to_cluster.shape[0]
                logger.info(f"Ejecutando {cluster_algo} sobre {num_points} embeddings con dimensión {data_dimension_clustering}.")

                labels: Optional[np.ndarray] = None
                cluster_stats = {} # To store algorithm-specific stats like inertia
                start_cluster_time = time.time()

                if cluster_algo == "MiniBatchKMeans":
                    # Adjust k again just before running, in case data size changed unexpectedly
                    if params["n_clusters"] > num_points:
                        st.warning(f"Ajustando k de {params['n_clusters']} a {num_points} antes de ejecutar.")
                        params["n_clusters"] = num_points
                    if params["n_clusters"] <= 0: # Final check
                         st.error("Número de clusters inválido (<=0). No se puede ejecutar KMeans.")
                         return
                    labels, inertia = clustering.run_minibatch_kmeans(embeddings_to_cluster, **params)
                    cluster_stats["inertia"] = inertia
                elif cluster_algo == "HDBSCAN":
                    labels, cluster_stats = clustering.run_hdbscan(embeddings_to_cluster, **params)

                end_cluster_time = time.time()
                cluster_duration = end_cluster_time - start_cluster_time

                # Validate clustering results
                if labels is None or not isinstance(labels, np.ndarray) or labels.shape[0] != num_points:
                    st.error(f"❌ El algoritmo de clustering ({cluster_algo}) no devolvió etiquetas válidas.")
                    logger.error(f"Clustering algorithm {cluster_algo} did not return valid labels.")
                    return # Exit if clustering fails

                st.success(f"¡Clustering ({cluster_algo}) completado en {cluster_duration:.2f}s!")

                # 3. Generate Text Labels
                generated_labels_dict: Dict[int, str] = {}
                st.info("Intentando generar etiquetas textuales para los clusters...")
                with st.spinner("🏷️ Generando etiquetas..."):
                    try:
                        # --- START: CORRECTED FUNCTION CALL ---
                        # Call the labeling function passing the correct arguments
                        # Note: Parameter names in the function definition were updated in clustering.py
                        generated_labels_dict = clustering.label_clusters_with_text(
                            clustering_embeddings=embeddings_to_cluster, # Embeddings used for clustering (N-dim or M-dim)
                            cluster_labels=labels,                       # Resulting labels
                            cluster_member_ids=ids_for_clustering,       # Corresponding IDs
                            vectorizer=vectorizer,                       # Vectorizer instance
                            image_db_target_dim=db,                      # Image DB instance for target dimension (M-dim)
                            target_dimension=truncate_dim,               # Target semantic dimension (M-dim)
                            k_text_neighbors=1                           # Get the top 1 text label
                        )
                        # --- END: CORRECTED FUNCTION CALL ---

                        if generated_labels_dict:
                            st.success(f"Etiquetas generadas para {len(generated_labels_dict)} clusters.")
                        else:
                            st.warning("No se pudieron generar etiquetas textuales (¿archivo de etiquetas vacío o error?).")

                    except Exception as label_e:
                        st.error(f"❌ Error durante la generación de etiquetas: {label_e}")
                        logger.error("Error calling label_clusters_with_text", exc_info=True)
                        # Continue without labels, but log the error

                # 4. Calculate Internal Metrics (unchanged)
                logger.info(f"Calculando métricas internas usando métrica: '{metric_for_eval}'")
                # Use the embeddings that clustering was performed on for evaluation
                metrics = clustering.calculate_internal_metrics(embeddings_to_cluster, labels, metric=metric_for_eval)

                # 5. Get Statistics (unchanged)
                num_clusters_found = cluster_stats.get("num_clusters", len(set(labels)) - (1 if -1 in labels else 0))
                num_noise_points = cluster_stats.get("num_noise_points", np.sum(labels == -1))

                # 6. Display Summary (including labels)
                st.subheader("📊 Resumen del Clustering")
                col_res1, col_res2 = st.columns(2)
                col_res1.metric("Clusters Encontrados", num_clusters_found)
                col_res2.metric("Puntos de Ruido", num_noise_points)
                if cluster_algo == "MiniBatchKMeans" and "inertia" in cluster_stats:
                     st.metric("Inercia (KMeans)", f"{cluster_stats['inertia']:.2f}")

                # Display generated labels in a table if available
                if generated_labels_dict:
                    st.subheader("🏷️ Etiquetas Sugeridas")
                    # Create DataFrame from the dictionary
                    labels_df = pd.DataFrame(
                        generated_labels_dict.items(),
                        columns=["Cluster ID", "Etiqueta Sugerida"]
                    ).sort_values(by="Cluster ID") # Sort by cluster ID
                    # Display the DataFrame
                    st.dataframe(labels_df, use_container_width=True, height=min(300, (len(labels_df)+1)*35 + 3)) # Adjust height dynamically
                else:
                    st.caption("No se generaron etiquetas textuales.")

                st.divider()
                st.subheader("📈 Métricas de Evaluación (Internas)")
                # Display metrics: Silhouette, Davies-Bouldin, etc. (unchanged)
                col_m1, col_m2, col_m3 = st.columns(3)
                silhouette_val = metrics.get("silhouette")
                # Display Silhouette score with a simple delta indication (higher is better, target ~1)
                col_m1.metric("Silhouette Score", f"{silhouette_val:.4f}" if silhouette_val is not None else "N/A",
                              delta=f"{silhouette_val - 0.5:.2f}" if silhouette_val is not None else None,
                              help="Más cercano a 1 es mejor. Mide cuán similar es un objeto a su propio cluster comparado con otros clusters.")
                db_val = metrics.get("davies_bouldin")
                # Display Davies-Bouldin score with delta (lower is better, target 0)
                col_m2.metric("Davies-Bouldin Index", f"{db_val:.4f}" if db_val is not None else "N/A",
                              delta=f"{db_val:.2f}" if db_val is not None else None, delta_color="inverse",
                              help="Más cercano a 0 es mejor. Mide la similitud promedio entre clusters.")
                ch_val = metrics.get("calinski_harabasz")
                # Display Calinski-Harabasz score (higher is better)
                col_m3.metric("Calinski-Harabasz Index", f"{ch_val:.2f}" if ch_val is not None else "N/A",
                              help="Ratio de dispersión entre clusters vs intra-cluster. Más alto es mejor.")


                # 7. Save results to session state
                st.session_state[STATE_CLUSTER_IDS] = ids_for_clustering # Save IDs used for clustering
                st.session_state[STATE_CLUSTER_LABELS] = labels
                st.session_state[STATE_GENERATED_LABELS] = generated_labels_dict # Save generated labels
                st.success("✅ Resultados del clustering y etiquetas guardados. Ve a 'Visualizar Resultados'.")

            # --- Error Handling (unchanged, already clears generated labels state) ---
            except (PipelineError, DatabaseError, ValueError, ImportError) as e:
                st.error(f"❌ Error durante el pipeline de clustering/etiquetado ({cluster_algo}): {e}")
                logger.error(f"Error running clustering/labeling pipeline ({cluster_algo})", exc_info=True)
                # Clear potentially inconsistent results
                keys_to_clear = [STATE_CLUSTER_LABELS, STATE_CLUSTER_IDS, STATE_GENERATED_LABELS]
                for key in keys_to_clear:
                    if key in st.session_state: del st.session_state[key]
            except Exception as e:
                st.error(f"❌ Error inesperado durante el pipeline de clustering/etiquetado ({cluster_algo}): {e}")
                logger.error(f"Unexpected error running clustering/labeling pipeline ({cluster_algo})", exc_info=True)
                # Clear potentially inconsistent results
                keys_to_clear = [STATE_CLUSTER_LABELS, STATE_CLUSTER_IDS, STATE_GENERATED_LABELS]
                for key in keys_to_clear:
                    if key in st.session_state: del st.session_state[key]

