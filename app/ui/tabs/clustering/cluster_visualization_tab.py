import logging
import time
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

# Attempt to import Plotly
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    px = None
    go = None
    PLOTLY_AVAILABLE = False
    # Log error only once if Plotly is missing
    if not hasattr(st, '_plotly_error_logged'):
        logging.error("Plotly library not found. Visualization tab will not work. Install with 'pip install plotly'.")
        st._plotly_error_logged = True # Use Streamlit's session state implicitly if needed, or a simple flag

from app.services import clustering # Import the clustering module
from app.exceptions import DatabaseError, InitializationError, PipelineError
from app.ui.state_utils import (
    STATE_CLUSTER_IDS,
    STATE_CLUSTER_LABELS,
    STATE_GENERATED_LABELS, # Import key for generated labels
)
from core.vectorizer import Vectorizer
from data_access.vector_db_interface import VectorDBInterface

logger = logging.getLogger(__name__)

# Define a color palette (adjust as needed)
# Using Plotly's default qualitative palette is often a good start
# CLUSTER_COLOR_PALETTE = px.colors.qualitative.Plotly
CLUSTER_COLOR_PALETTE = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3",
    "#FF6692", "#B6E880", "#FF97FF", "#FECB52", # Plotly default + extras
    "#0d6efd", "#6f42c1", "#d63384", "#dc3545", "#fd7e14", "#ffc107",
    "#198754", "#20c997", "#0dcaf0", "#adb5bd"
]


def render_cluster_visualization_tab(
    vectorizer: Vectorizer, db: VectorDBInterface, truncate_dim: Optional[int]
):
    """
    Renderiza el contenido y maneja la lógica para la pestaña 'Visualizar Resultados'.
    Ahora incluye las etiquetas textuales generadas en el hover.

    Args:
        vectorizer: Instancia del Vectorizer.
        db: Instancia de la interfaz de BD vectorial inicializada (para imágenes).
        truncate_dim: Dimensión de truncamiento seleccionada (puede ser None).
    """
    st.subheader("🎨 Visualizar Resultados (Post-Clustering)")
    st.markdown(
        """
        Visualiza los resultados del último clustering ejecutado. Se aplica una técnica
        de reducción de dimensionalidad (**t-SNE**, **UMAP** o **PCA**) a los embeddings
        **originales** y se colorea cada punto según su cluster.
        Pasa el ratón sobre un punto para ver su ID y la etiqueta textual generada (si existe).
        """
    )

    # Check if Plotly is installed
    if not PLOTLY_AVAILABLE:
        st.error("❌ La biblioteca 'plotly' no está instalada. Instálala (`pip install plotly`) para usar la visualización.")
        return

    # Check if clustering results exist in session state
    if (STATE_CLUSTER_LABELS not in st.session_state or
        STATE_CLUSTER_IDS not in st.session_state):
        st.info("ℹ️ No se encontraron resultados de clustering. Ejecuta primero un algoritmo en la pestaña 'Ejecutar Clustering'.")
        return

    # Get and validate clustering data and generated labels
    cluster_labels = st.session_state.get(STATE_CLUSTER_LABELS)
    cluster_ids = st.session_state.get(STATE_CLUSTER_IDS)
    # Get generated labels (can be None or an empty dict)
    generated_labels_map = st.session_state.get(STATE_GENERATED_LABELS, {})

    # Validate data integrity
    if (cluster_labels is None or cluster_ids is None or
        not isinstance(cluster_labels, np.ndarray) or not isinstance(cluster_ids, list) or
        len(cluster_ids) != len(cluster_labels)):
        st.error("❌ Error: Los datos de clustering guardados son inválidos o inconsistentes.")
        logger.error("Invalid or corrupt clustering data found in session state.")
        # Clear corrupted state
        keys_to_clear = [STATE_CLUSTER_LABELS, STATE_CLUSTER_IDS, STATE_GENERATED_LABELS]
        for key in keys_to_clear:
            if key in st.session_state: del st.session_state[key]
        st.rerun() # Rerun to reflect cleared state
        return

    num_clustered_points = len(cluster_ids)
    st.success(f"✅ Resultados de clustering encontrados para {num_clustered_points} puntos.")
    if generated_labels_map:
         st.caption(f"ℹ️ Etiquetas textuales generadas disponibles para {len(generated_labels_map)} clusters.")
    else:
         st.caption("ℹ️ No se encontraron etiquetas textuales generadas en esta sesión.")
    st.divider()

    # --- Visualization technique selection and parameters ---
    st.markdown("**Configuración de la Visualización:**")

    # --- START: CORRECTED SELECTBOX ---
    # Removed the incorrect ellipsis (...)
    vis_method = st.selectbox(
        "Técnica de Reducción para Visualización:",
        ["UMAP", "t-SNE", "PCA"],
        index=0, # Default to UMAP
        key="vis_method_select",
        help="Elige cómo reducir los datos para visualizarlos en 2D o 3D."
    )
    # --- END: CORRECTED SELECTBOX ---

    vis_params = {}
    n_components_vis = st.radio("Dimensiones:", [2, 3], index=0, horizontal=True, key="vis_n_components")
    vis_params["n_components"] = n_components_vis

    # Parameters specific to the chosen method
    if vis_method == "UMAP":
         if clustering.umap is None:
              st.error("❌ 'umap-learn' no instalado. No se puede usar UMAP.")
              return
         # Sliders for UMAP parameters
         vis_params["n_neighbors"] = st.slider("Vecinos (UMAP n_neighbors):", 2, 100, 15, 1, key="vis_umap_neighbors")
         vis_params["min_dist"] = st.slider("Dist. Mínima (UMAP min_dist):", 0.0, 0.99, 0.1, 0.05, key="vis_umap_min_dist")
         vis_params["metric"] = st.selectbox("Métrica (UMAP metric):", ["cosine", "euclidean", "manhattan"], index=0, key="vis_umap_metric")
    elif vis_method == "t-SNE":
         # Sliders for t-SNE parameters
         # Adjust perplexity max based on number of points
         max_perplexity = max(5.0, min(50.0, float(num_clustered_points - 1))) if num_clustered_points > 1 else 5.0
         default_perplexity = min(30.0, max_perplexity)
         vis_params["perplexity"] = st.slider("Perplejidad (t-SNE perplexity):", 5.0, max_perplexity, default_perplexity, 1.0, key="vis_tsne_perplexity")
         vis_params["n_iter"] = st.slider("Iteraciones (t-SNE n_iter):", 250, 2000, 1000, 50, key="vis_tsne_n_iter")
         vis_params["metric"] = st.selectbox("Métrica (t-SNE metric):", ["cosine", "euclidean"], index=0, key="vis_tsne_metric")
         if num_clustered_points > 5000: st.warning("⚠️ t-SNE puede ser muy lento con muchos puntos.")
    # PCA has no extra parameters here other than n_components

    st.divider()

    # --- Button to generate visualization ---
    if st.button(f"📊 Generar Visualización {n_components_vis}D con {vis_method}", key="generate_vis_btn", type="primary"):
        collection_name = getattr(db, "collection_name", "N/A") # Get current collection name for title
        st.info(f"Generando visualización {vis_method} ({n_components_vis}D) para '{collection_name}'...")
        with st.spinner(f"⏳ Aplicando {vis_method} y generando gráfico..."):
            try:
                # 1. Fetch ORIGINAL Embeddings (corresponding to the target dimension)
                start_fetch_time = time.time()
                logger.info(f"Fetching original embeddings for {len(cluster_ids)} clustered IDs from '{collection_name}'...")
                # Use the 'db' instance passed to the function, which corresponds to the selected truncate_dim
                embeddings_dict = db.get_embeddings_by_ids(ids=cluster_ids)
                if embeddings_dict is None:
                    raise DatabaseError("Fallo al obtener embeddings originales por ID para visualización.")

                # Filter IDs found and map labels, handle missing embeddings
                valid_ids_found = []
                valid_embeddings_list = []
                missing_ids = []
                id_to_label_map = dict(zip(cluster_ids, cluster_labels)) # Map ID -> Numeric Label

                for cid in cluster_ids:
                    embedding = embeddings_dict.get(cid)
                    # Check if embedding is valid (list of floats)
                    if embedding is not None and isinstance(embedding, list) and embedding:
                        valid_ids_found.append(cid)
                        valid_embeddings_list.append(embedding)
                    else:
                        missing_ids.append(cid)
                        logger.warning(f"Could not find valid original embedding for clustered ID: {cid}")

                if not valid_embeddings_list:
                    raise PipelineError("No se encontraron embeddings originales válidos para los IDs clusterizados.")

                # Get the numeric labels corresponding to the successfully fetched embeddings
                labels_to_visualize_numeric = np.array([id_to_label_map[vid] for vid in valid_ids_found])
                embeddings_to_visualize = np.array(valid_embeddings_list, dtype=np.float32)
                ids_to_visualize = valid_ids_found # Use only the IDs for which embeddings were found

                end_fetch_time = time.time()
                num_embeddings_fetched = embeddings_to_visualize.shape[0]
                logger.info(f"Fetched {num_embeddings_fetched} original embeddings in {end_fetch_time - start_fetch_time:.2f}s.")
                if missing_ids:
                    st.warning(f"⚠️ No se encontraron embeddings originales para {len(missing_ids)} IDs. Se visualizarán {num_embeddings_fetched} puntos.")

                # 2. Apply Dimensionality Reduction for Visualization (unchanged)
                start_reduce_time = time.time()
                reduced_embeddings_vis = None
                if vis_method == "PCA":
                    reduced_embeddings_vis = clustering.apply_pca(embeddings_to_visualize, **vis_params)
                elif vis_method == "UMAP":
                    reduced_embeddings_vis = clustering.apply_umap(embeddings_to_visualize, **vis_params)
                elif vis_method == "t-SNE":
                    reduced_embeddings_vis = clustering.apply_tsne(embeddings_to_visualize, **vis_params)
                end_reduce_time = time.time()

                if reduced_embeddings_vis is None or reduced_embeddings_vis.shape[0] != num_embeddings_fetched:
                    st.error(f"❌ Falló la aplicación de {vis_method} para visualización.")
                    logger.error(f"Reduction method {vis_method} failed or returned inconsistent shape for visualization.")
                    return

                logger.info(f"{vis_method} aplicado para visualización en {end_reduce_time - start_reduce_time:.2f}s. Shape: {reduced_embeddings_vis.shape}")

                # 3. Prepare data for Plotly (MODIFIED to include generated label)
                # Create combined text labels (Cluster Num: Generated Label) for legend/color
                labels_str_combined = []
                for l_num in labels_to_visualize_numeric:
                    base_label = f"Cluster {l_num}" if l_num != -1 else "Ruido (-1)"
                    # Get generated label from the map (can be None)
                    generated_label = generated_labels_map.get(l_num)
                    if generated_label:
                        # Combine number and text for legend/color
                        labels_str_combined.append(f"{base_label}: {generated_label}")
                    else:
                        labels_str_combined.append(base_label) # Use only number if no label

                # Create DataFrame
                df_vis = pd.DataFrame(
                    reduced_embeddings_vis,
                    columns=[f"Dim_{i+1}" for i in range(n_components_vis)],
                )
                df_vis["label_display"] = labels_str_combined # Column for color/legend
                df_vis["id"] = ids_to_visualize # ID for hover
                df_vis["numeric_label"] = labels_to_visualize_numeric # Original numeric label
                # Add a separate column just for the generated label text for hover (optional but useful)
                df_vis['generated_label_text'] = df_vis['numeric_label'].map(generated_labels_map).fillna("N/A")

                # Sort by numeric label for consistent coloring/legend order
                df_vis = df_vis.sort_values(by="numeric_label")

                # 4. Generate Plotly chart (MODIFIED)
                st.subheader(f"Visualización {vis_method} ({n_components_vis}D)")
                fig = None

                # Configure hover: Show ID, Numeric Label, and Generated Label
                hover_data_config = {
                    "id": True, # Show ID
                    "numeric_label": True, # Show cluster number
                    "generated_label_text": True, # Show generated label text
                    "label_display": False, # Hide the combined one from hover (already in color/legend)
                 }
                # Hide dimension axes from hover
                for i in range(n_components_vis):
                    hover_data_config[f"Dim_{i+1}"] = False

                plot_title = f"Clusters Visualizados con {vis_method} ({n_components_vis}D) en '{collection_name}'"

                # Create chart using 'label_display' for color
                common_scatter_args = dict(
                    data_frame=df_vis,
                    color="label_display", # Color by the combined label
                    title=plot_title,
                    hover_name="id", # Show ID prominently on hover
                    hover_data=hover_data_config, # Use custom hover config
                    color_discrete_sequence=CLUSTER_COLOR_PALETTE, # Use defined palette
                    # Category orders can be complex with text, maybe omit or simplify
                    # category_orders={"label_display": sorted(df_vis["label_display"].unique(), ...)},
                    labels={"label_display": "Cluster / Etiqueta", # Name for legend
                            "numeric_label": "Cluster ID", # Name for hover
                            "generated_label_text": "Etiqueta Sugerida"} # Name for hover
                )

                if n_components_vis == 2:
                    fig = px.scatter(**common_scatter_args, x="Dim_1", y="Dim_2")
                    # Customize markers
                    fig.update_traces(marker=dict(size=6, opacity=0.8, line=dict(width=0.5, color='DarkSlateGrey')))
                elif n_components_vis == 3:
                    fig = px.scatter_3d(**common_scatter_args, x="Dim_1", y="Dim_2", z="Dim_3")
                    # Customize markers for 3D
                    fig.update_traces(marker=dict(size=4, opacity=0.7))

                # 5. Display the chart (Layout mostly unchanged)
                if fig:
                    # Customize layout
                    fig.update_layout(
                        xaxis_title=f"{vis_method} Dim 1",
                        yaxis_title=f"{vis_method} Dim 2",
                        # zaxis_title=f"{vis_method} Dim 3" # Only for 3D, Plotly handles this
                        margin=dict(l=0, r=0, b=0, t=40), # Tight margins
                        height=700, # Adjust height as needed
                        legend_title_text='Clusters', # Title for the legend
                        legend=dict(
                            orientation="v", # Vertical legend
                            yanchor="top", y=0.99,
                            xanchor="left", x=0.01,
                            bgcolor='rgba(255,255,255,0.7)', # Semi-transparent background
                            bordercolor='rgba(200,200,200,0.5)',
                            borderwidth=1
                        ),
                        font=dict(family="Lato, sans-serif", size=12, color="#212529"),
                        title_font=dict(family="Inter, sans-serif", size=18, color="#0056b3"),
                        paper_bgcolor='rgba(0,0,0,0)', # Transparent background
                        plot_bgcolor='rgba(240,240,240,0.5)', # Light gray plot background
                    )
                    # Adjust tooltips (hover labels) appearance
                    fig.update_layout(hoverlabel=dict(bgcolor="white", font_size=12, font_family="Lato, sans-serif"))

                    # Display the chart in Streamlit
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("❌ No se pudo generar el gráfico.")

            # --- Error Handling (unchanged) ---
            except (PipelineError, DatabaseError, ValueError, ImportError) as e:
                st.error(f"❌ Error generando visualización ({vis_method}): {e}")
                logger.error(f"Error generating visualization ({vis_method})", exc_info=True)
            except Exception as e:
                st.error(f"❌ Error inesperado generando visualización ({vis_method}): {e}")
                logger.error(f"Unexpected error generating visualization ({vis_method})", exc_info=True)

