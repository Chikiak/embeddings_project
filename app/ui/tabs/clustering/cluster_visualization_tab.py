# app/ui/tabs/clustering/cluster_visualization_tab.py
import streamlit as st
import logging
import numpy as np
import pandas as pd
import time
from typing import Optional

# Plotting library
try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:
    px = None
    go = None
    logging.error("Plotly library not found. Visualization tab will not work. Install with 'pip install plotly'.")

# Project imports
from core.vectorizer import Vectorizer
from data_access.vector_db_interface import VectorDBInterface
from app import clustering # Import the clustering logic module
from app.exceptions import PipelineError, DatabaseError, InitializationError
from app.ui.state_utils import STATE_CLUSTER_LABELS, STATE_CLUSTER_IDS # Import state keys

logger = logging.getLogger(__name__)


def render_cluster_visualization_tab(vectorizer: Vectorizer, db: VectorDBInterface, truncate_dim: Optional[int]):
    """Renders the tab for visualizing clustering results."""
    st.header("🎨 Visualizar Clusters (Post-Clustering)")
    st.markdown(
        "Aplica t-SNE o UMAP a los embeddings **originales** y colorea los puntos según las etiquetas de cluster obtenidas "
        "en la pestaña 'Ejecutar Clustering'. Esto ayuda a entender cómo se agrupan los datos en un espacio 2D o 3D."
        "\n\n**Nota:** Requiere que se haya ejecutado un clustering previamente en esta sesión."
    )

    if px is None or go is None:
        st.error("La biblioteca 'plotly' no está instalada. Por favor, instálala (`pip install plotly`) para usar la visualización.")
        return

    # --- Check for Clustering Results in Session State ---
    if STATE_CLUSTER_LABELS not in st.session_state or STATE_CLUSTER_IDS not in st.session_state:
        st.info("ℹ️ Ejecuta primero un algoritmo de clustering en la pestaña 'Ejecutar Clustering' para poder visualizar los resultados.")
        return

    cluster_labels = st.session_state.get(STATE_CLUSTER_LABELS)
    cluster_ids = st.session_state.get(STATE_CLUSTER_IDS)

    # Validate the data from session state
    if cluster_labels is None or cluster_ids is None or not isinstance(cluster_labels, np.ndarray) or not isinstance(cluster_ids, list) or len(cluster_ids) != len(cluster_labels):
        st.error("Error: Los datos de clustering guardados en la sesión son inválidos o están corruptos.")
        # Clean up invalid state
        if STATE_CLUSTER_LABELS in st.session_state: del st.session_state[STATE_CLUSTER_LABELS]
        if STATE_CLUSTER_IDS in st.session_state: del st.session_state[STATE_CLUSTER_IDS]
        st.rerun()
        return

    st.success(f"Resultados de clustering encontrados para {len(cluster_ids)} imágenes.")

    # --- Visualization Technique Selection ---
    vis_method = st.selectbox(
        "Técnica de Reducción para Visualización:",
        options=["UMAP", "t-SNE", "PCA (Rápido, Lineal)"],
        key="vis_method_select",
        help="UMAP es generalmente un buen balance rendimiento/calidad. t-SNE es bueno para estructura local pero puede ser lento. PCA es rápido pero solo captura varianza lineal."
    )

    # --- Visualization Parameters ---
    vis_params = {}
    n_components_vis = st.radio(
        "Dimensiones de Visualización:",
        options=[2, 3],
        index=0,
        horizontal=True,
        key="vis_n_components",
        help="Visualizar en 2D (más común) o 3D."
    )
    vis_params['n_components'] = n_components_vis

    # Parameters specific to the chosen method
    if vis_method == "UMAP":
         if clustering.umap is None: # Check if UMAP library is available
             st.error("La biblioteca 'umap-learn' no está instalada. No se puede usar UMAP. Instálala con 'pip install umap-learn'.")
             return
         # Get number of data points for sensible defaults/limits
         num_points = len(cluster_ids)
         vis_params['n_neighbors'] = st.slider("Vecinos Cercanos (UMAP-Vis):", min_value=2, max_value=min(200, num_points-1 if num_points > 1 else 2), value=min(15, num_points-1 if num_points > 1 else 2), step=1, key="umap_vis_n_neighbors")
         vis_params['min_dist'] = st.slider("Distancia Mínima (UMAP-Vis):", min_value=0.0, max_value=0.99, value=0.1, step=0.05, key="umap_vis_min_dist")
         vis_params['metric'] = st.selectbox("Métrica (UMAP-Vis):", options=['cosine', 'euclidean', 'l2'], index=0, key="umap_vis_metric")
    elif vis_method == "t-SNE":
        vis_params['perplexity'] = st.slider("Perplejidad (t-SNE):", min_value=5.0, max_value=50.0, value=30.0, step=1.0, key="tsne_perplexity", help="Relacionado con el número de vecinos cercanos a considerar.")
        vis_params['n_iter'] = st.select_slider("Iteraciones (t-SNE):", options=[250, 500, 1000, 2000], value=1000, key="tsne_n_iter", help="Número de optimizaciones. Más iteraciones pueden dar mejores resultados pero tardan más.")
        vis_params['metric'] = st.selectbox("Métrica (t-SNE):", options=['cosine', 'euclidean', 'l2'], index=0, key="tsne_vis_metric")
    # PCA parameters are just n_components, already handled

    # --- Generate Visualization Button ---
    st.divider()
    if st.button(f"📊 Generar Visualización {n_components_vis}D con {vis_method}", key="run_visualization_btn", type="primary"):
        collection_name = getattr(db, 'collection_name', 'N/A')
        st.info(f"Generando visualización {vis_method} ({n_components_vis}D) para los datos de '{collection_name}'...")
        with st.spinner(f"Aplicando {vis_method} y generando gráfico... Esto puede tardar."):
            try:
                # 1. Get the ORIGINAL embeddings corresponding to the clustered IDs
                start_fetch_time = time.time()
                logger.info(f"Fetching original embeddings for {len(cluster_ids)} clustered IDs from '{collection_name}'...")

                # --- Efficiently fetch embeddings by ID (Requires DB implementation) ---
                # Placeholder: Using get_all as fallback (Inefficient!)
                # YOU SHOULD IMPLEMENT get_embeddings_by_ids in chroma_db.py
                all_ids_orig, all_embeddings_orig = db.get_all_embeddings_with_ids(pagination_batch_size=5000)
                if all_embeddings_orig is None or all_embeddings_orig.shape[0] == 0:
                    raise DatabaseError("Failed to fetch original embeddings.")

                # Create a mapping from ID to its index in the full list
                id_to_idx_orig = {id_val: idx for idx, id_val in enumerate(all_ids_orig)}
                # Find the indices corresponding to our clustered IDs
                indices_to_fetch = [id_to_idx_orig.get(cid) for cid in cluster_ids]

                # Filter out any IDs that weren't found in the full fetch (shouldn't happen ideally)
                valid_indices_mask = [idx is not None for idx in indices_to_fetch]
                if not all(valid_indices_mask):
                    logger.warning("Some clustered IDs were not found in the full embedding fetch. Visualization might be incomplete.")
                    # Filter labels and IDs to match the found embeddings
                    embeddings_to_visualize = all_embeddings_orig[[idx for idx in indices_to_fetch if idx is not None]]
                    labels_to_visualize = cluster_labels[valid_indices_mask]
                    ids_to_visualize = [cid for cid, mask in zip(cluster_ids, valid_indices_mask) if mask]
                else:
                    embeddings_to_visualize = all_embeddings_orig[[idx for idx in indices_to_fetch if idx is not None]] # Use list of valid indices
                    labels_to_visualize = cluster_labels
                    ids_to_visualize = cluster_ids

                end_fetch_time = time.time()
                logger.info(f"Fetched {embeddings_to_visualize.shape[0]} original embeddings in {end_fetch_time - start_fetch_time:.2f}s.")

                # 2. Apply selected dimensionality reduction for visualization
                start_reduce_time = time.time()
                reduced_embeddings_vis = None
                if vis_method == "PCA":
                    reduced_embeddings_vis = clustering.apply_pca(embeddings_to_visualize, **vis_params)
                elif vis_method == "UMAP":
                    reduced_embeddings_vis = clustering.apply_umap(embeddings_to_visualize, **vis_params)
                elif vis_method == "t-SNE":
                    reduced_embeddings_vis = clustering.apply_tsne(embeddings_to_visualize, **vis_params)
                end_reduce_time = time.time()

                if reduced_embeddings_vis is None:
                    st.error(f"Failed to apply {vis_method} for visualization.")
                    return # Stop execution

                logger.info(f"{vis_method} applied in {end_reduce_time - start_reduce_time:.2f}s. Reduced shape: {reduced_embeddings_vis.shape}")

                # 3. Prepare Data for Plotly
                # Convert numerical labels to strings for discrete colors, handle noise (-1)
                labels_str = [f"Cluster {l}" if l != -1 else "Ruido (-1)" for l in labels_to_visualize]
                # Create a Pandas DataFrame
                df_vis = pd.DataFrame(reduced_embeddings_vis, columns=[f"Dim_{i+1}" for i in range(n_components_vis)])
                df_vis['label'] = labels_str
                df_vis['id'] = ids_to_visualize # Add IDs for hover info

                # --- Generate Plotly Chart ---
                st.subheader(f"Visualización {vis_method} ({n_components_vis}D)")
                fig = None
                # Define hover data
                hover_data_config = {'label': True, 'id': True} # Show label and ID on hover
                # Add original dimensions to hover data if needed (can be slow)
                # hover_data_config.update({f'OrigDim_{i}': True for i in range(min(5, embeddings_to_visualize.shape[1]))}) # Show first 5 original dims

                plot_title = f"Clusters Visualizados con {vis_method} ({n_components_vis}D) en '{collection_name}'"

                if n_components_vis == 2:
                    fig = px.scatter(
                        df_vis, x='Dim_1', y='Dim_2', color='label',
                        title=plot_title,
                        hover_data=hover_data_config,
                        color_discrete_sequence=px.colors.qualitative.Vivid, # Use a vivid color sequence
                        category_orders={"label": sorted(df_vis['label'].unique())} # Ensure consistent legend order
                    )
                    fig.update_traces(marker=dict(size=5, opacity=0.7))
                elif n_components_vis == 3:
                     fig = px.scatter_3d(
                         df_vis, x='Dim_1', y='Dim_2', z='Dim_3', color='label',
                         title=plot_title,
                         hover_data=hover_data_config,
                         color_discrete_sequence=px.colors.qualitative.Vivid,
                         category_orders={"label": sorted(df_vis['label'].unique())}
                     )
                     fig.update_traces(marker=dict(size=3, opacity=0.7))

                if fig:
                     fig.update_layout(
                         legend_title_text='Etiqueta Cluster',
                         xaxis_title=f"{vis_method} Dim 1",
                         yaxis_title=f"{vis_method} Dim 2",
                         # zaxis_title is automatic for 3D in Plotly
                         margin=dict(l=0, r=0, b=0, t=40), # Adjust margins
                         height=700, # Increase height for better view
                         legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1) # Legend on top
                    )
                     st.plotly_chart(fig, use_container_width=True)
                else:
                     st.error("No se pudo generar el gráfico.")

            except (PipelineError, DatabaseError, ValueError, ImportError) as e:
                st.error(f"Error generando visualización ({vis_method}): {e}")
                logger.error(f"Error generating visualization ({vis_method})", exc_info=True)
            except Exception as e:
                st.error(f"Error inesperado generando visualización ({vis_method}): {e}")
                logger.error(f"Unexpected error generating visualization ({vis_method})", exc_info=True)

