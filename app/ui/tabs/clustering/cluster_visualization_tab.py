# --- app/ui/tabs/clustering/cluster_visualization_tab.py ---
import logging
import time
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

# Intentar importar Plotly, manejar error si no está
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    px = None
    go = None
    PLOTLY_AVAILABLE = False
    # Loggear el error una vez al inicio
    logging.error(
        "Plotly library not found. Visualization tab will not work. Install with 'pip install plotly'."
    )

from app import clustering
from app.exceptions import DatabaseError, InitializationError, PipelineError
from app.ui.state_utils import STATE_CLUSTER_IDS, STATE_CLUSTER_LABELS
from core.vectorizer import Vectorizer # No se usa directamente, pero podría ser útil
from data_access.vector_db_interface import VectorDBInterface

logger = logging.getLogger(__name__)

# Definir una paleta de colores para los clusters (consistente con style.css si es posible)
# Ejemplo: Usando colores primarios/secundarios y otros definidos
CLUSTER_COLOR_PALETTE = [
    "#007bff", "#28a745", "#ffc107", "#dc3545", "#17a2b8", "#6c757d",
    "#fd7e14", "#6610f2", "#e83e8c", "#20c997", "#0d6efd", "#adb5bd"
] # Añade más colores si esperas muchos clusters


def render_cluster_visualization_tab(
    vectorizer: Vectorizer, db: VectorDBInterface, truncate_dim: Optional[int]
):
    """
    Renderiza el contenido y maneja la lógica para la pestaña 'Visualizar Resultados (Post-Clustering)'.

    Args:
        vectorizer: Instancia del Vectorizer (puede ser útil para info).
        db: Instancia de la interfaz de BD vectorial inicializada.
        truncate_dim: Dimensión de truncamiento seleccionada (puede ser None).
    """
    st.subheader("🎨 Visualizar Resultados (Post-Clustering)") # Subheader
    st.markdown(
        """
        Visualiza los resultados del último clustering ejecutado en esta sesión.
        Se aplica una técnica de reducción de dimensionalidad (**t-SNE**, **UMAP** o **PCA**)
        a los embeddings **originales** de la base de datos, y luego se colorea cada punto
        según la etiqueta de cluster asignada en la pestaña 'Ejecutar Clustering'.

        Esto ayuda a entender cómo se agrupan los datos en un espacio de baja dimensión (2D o 3D).
        """
    )

    # Verificar si Plotly está instalado
    if not PLOTLY_AVAILABLE:
        st.error(
            "❌ La biblioteca 'plotly' no está instalada. "
            "Por favor, instálala (`pip install plotly`) para usar la visualización."
        )
        return # No continuar sin Plotly

    # Verificar si hay resultados de clustering en el estado de sesión
    if (
        STATE_CLUSTER_LABELS not in st.session_state
        or STATE_CLUSTER_IDS not in st.session_state
    ):
        st.info(
            "ℹ️ No se encontraron resultados de clustering en esta sesión. "
            "Ejecuta primero un algoritmo en la pestaña 'Ejecutar Clustering'."
        )
        return

    # Obtener y validar los datos de clustering del estado
    cluster_labels = st.session_state.get(STATE_CLUSTER_LABELS)
    cluster_ids = st.session_state.get(STATE_CLUSTER_IDS)

    if (
        cluster_labels is None
        or cluster_ids is None
        or not isinstance(cluster_labels, np.ndarray)
        or not isinstance(cluster_ids, list)
        or len(cluster_ids) != len(cluster_labels)
    ):
        st.error(
            "❌ Error: Los datos de clustering guardados en la sesión son inválidos o están corruptos."
        )
        logger.error("Invalid or corrupt clustering data found in session state.")
        # Limpiar estado corrupto
        if STATE_CLUSTER_LABELS in st.session_state:
            del st.session_state[STATE_CLUSTER_LABELS]
        if STATE_CLUSTER_IDS in st.session_state:
            del st.session_state[STATE_CLUSTER_IDS]
        st.rerun() # Re-ejecutar para reflejar el estado limpio
        return

    num_clustered_points = len(cluster_ids)
    st.success(
        f"✅ Resultados de clustering encontrados para {num_clustered_points} puntos."
    )
    st.divider() # Separador

    # Selección de la técnica de visualización
    st.markdown("**Configuración de la Visualización:**")
    vis_method = st.selectbox(
        "Técnica de Reducción para Visualización:",
        options=["UMAP", "t-SNE", "PCA (Rápido, Lineal)"],
        key="vis_method_select", # Clave única
        index=0, # UMAP por defecto
        help="Elige cómo reducir los embeddings originales a 2D/3D para graficar. "
             "UMAP suele ser un buen balance. t-SNE es bueno para estructura local pero lento. PCA es rápido pero solo captura varianza lineal."
    )

    # Parámetros comunes de visualización
    vis_params = {}
    n_components_vis = st.radio(
        "Dimensiones de Visualización:",
        options=[2, 3],
        index=0, # 2D por defecto
        horizontal=True,
        key="vis_n_components", # Clave única
        help="Visualizar en 2D (más común y fácil de interpretar) o 3D (interactivo).",
    )
    vis_params["n_components"] = n_components_vis

    # Parámetros específicos de la técnica de visualización
    if vis_method == "UMAP":
        # Verificar si UMAP está disponible
        if clustering.umap is None:
            st.error(
                "❌ La biblioteca 'umap-learn' no está instalada. No se puede usar UMAP para visualización. "
                "Instálala con: `pip install umap-learn`"
            )
            return # No continuar si UMAP no está

        # Parámetros de UMAP para visualización (pueden ser diferentes a los de optimización)
        num_points_for_vis = num_clustered_points # Usar el número de puntos clusterizados
        vis_params["n_neighbors"] = st.slider(
            "Vecinos Cercanos (UMAP-Vis):",
            min_value=2,
            max_value=min(200, num_points_for_vis - 1 if num_points_for_vis > 1 else 2),
            value=min(15, num_points_for_vis - 1 if num_points_for_vis > 1 else 2), # Default 15
            step=1,
            key="umap_vis_n_neighbors", # Clave única
            help="Controla el balance local/global para la visualización."
        )
        vis_params["min_dist"] = st.slider(
            "Distancia Mínima (UMAP-Vis):",
            min_value=0.0,
            max_value=0.99,
            value=0.1, # Default 0.1
            step=0.05,
            key="umap_vis_min_dist", # Clave única
            help="Controla la separación de los puntos en el gráfico."
        )
        vis_params["metric"] = st.selectbox(
            "Métrica (UMAP-Vis):",
            options=["cosine", "euclidean", "l2"], # Opciones comunes
            index=0, # Cosine por defecto
            key="umap_vis_metric", # Clave única
            help="Métrica usada por UMAP para la reducción de visualización."
        )
    elif vis_method == "t-SNE":
        # Parámetros de t-SNE
        vis_params["perplexity"] = st.slider(
            "Perplejidad (t-SNE):",
            min_value=5.0,
            max_value=min(50.0, num_clustered_points -1 if num_clustered_points > 1 else 5.0), # Ajustar max según puntos
            value=min(30.0, num_clustered_points -1 if num_clustered_points > 1 else 5.0), # Default 30
            step=1.0,
            key="tsne_perplexity", # Clave única
            help="Relacionado con el número de vecinos cercanos a considerar (típicamente 5-50)."
        )
        vis_params["n_iter"] = st.select_slider(
            "Iteraciones (t-SNE):",
            options=[250, 500, 1000, 2000, 5000], # Más opciones
            value=1000, # Default 1000
            key="tsne_n_iter", # Clave única
            help="Número de optimizaciones. Más iteraciones pueden dar mejores resultados pero tardan más.",
        )
        vis_params["metric"] = st.selectbox(
            "Métrica (t-SNE):",
            options=["cosine", "euclidean", "l2"], # Opciones comunes
            index=0, # Cosine por defecto
            key="tsne_vis_metric", # Clave única
            help="Métrica usada por t-SNE para calcular distancias."
        )
        if num_clustered_points > 5000:
             st.warning("⚠️ t-SNE puede ser muy lento con más de 5000 puntos.")

    # PCA no necesita parámetros adicionales aquí (solo n_components)

    st.divider() # Separador

    # Botón para generar la visualización
    if st.button(
        f"📊 Generar Visualización {n_components_vis}D con {vis_method}",
        key="run_visualization_btn", # Clave única
        type="primary" # Botón primario
    ):
        collection_name = getattr(db, "collection_name", "N/A")
        st.info(
            f"Generando visualización {vis_method} ({n_components_vis}D) para los datos de '{collection_name}'..."
        )
        # Mostrar spinner durante el proceso
        with st.spinner(
            f"⏳ Aplicando {vis_method} y generando gráfico... Esto puede tardar, especialmente con t-SNE."
        ):
            try:
                # 1. Obtener los embeddings ORIGINALES correspondientes a los IDs clusterizados
                start_fetch_time = time.time()
                logger.info(
                    f"Fetching original embeddings for {len(cluster_ids)} clustered IDs from '{collection_name}'..."
                )

                # Intenta obtener solo los embeddings necesarios por ID
                embeddings_dict = db.get_embeddings_by_ids(ids=cluster_ids)

                if embeddings_dict is None:
                     raise DatabaseError("Fallo al obtener embeddings por ID.")

                # Filtrar IDs que se encontraron y tienen embedding válido
                valid_ids_found = []
                valid_embeddings_list = []
                missing_ids = []
                for cid in cluster_ids:
                    embedding = embeddings_dict.get(cid)
                    if embedding is not None and isinstance(embedding, list):
                        valid_ids_found.append(cid)
                        valid_embeddings_list.append(embedding)
                    else:
                        missing_ids.append(cid)
                        logger.warning(f"Could not find embedding for clustered ID: {cid}")

                if not valid_embeddings_list:
                     raise PipelineError("No se encontraron embeddings válidos para los IDs clusterizados.")

                # Mapear los IDs encontrados a sus labels originales
                id_to_label = {cid: label for cid, label in zip(cluster_ids, cluster_labels)}
                labels_to_visualize_list = [id_to_label[vid] for vid in valid_ids_found]

                # Convertir a NumPy arrays
                embeddings_to_visualize = np.array(valid_embeddings_list, dtype=np.float32)
                labels_to_visualize = np.array(labels_to_visualize_list)
                ids_to_visualize = valid_ids_found # Usar solo los IDs encontrados

                end_fetch_time = time.time()
                num_embeddings_fetched = embeddings_to_visualize.shape[0]
                logger.info(
                    f"Fetched {num_embeddings_fetched} original embeddings in {end_fetch_time - start_fetch_time:.2f}s."
                )
                if missing_ids:
                     st.warning(f"⚠️ No se encontraron embeddings para {len(missing_ids)} IDs clusterizados. Se visualizarán {num_embeddings_fetched} puntos.")


                # 2. Aplicar la técnica de reducción seleccionada
                start_reduce_time = time.time()
                reduced_embeddings_vis = None
                if vis_method == "PCA":
                    reduced_embeddings_vis = clustering.apply_pca(
                        embeddings_to_visualize, **vis_params
                    )
                elif vis_method == "UMAP":
                    reduced_embeddings_vis = clustering.apply_umap(
                        embeddings_to_visualize, **vis_params
                    )
                elif vis_method == "t-SNE":
                    reduced_embeddings_vis = clustering.apply_tsne(
                        embeddings_to_visualize, **vis_params
                    )
                end_reduce_time = time.time()
                reduction_duration = end_reduce_time - start_reduce_time

                # Validar resultado de la reducción
                if reduced_embeddings_vis is None or reduced_embeddings_vis.shape[0] != num_embeddings_fetched:
                    st.error(
                        f"❌ Falló la aplicación de {vis_method} para visualización."
                    )
                    logger.error(f"Reduction method {vis_method} failed or returned inconsistent shape.")
                    return # Detener

                logger.info(
                    f"{vis_method} aplicado en {reduction_duration:.2f}s. Reduced shape: {reduced_embeddings_vis.shape}"
                )

                # 3. Preparar datos para Plotly
                # Crear etiquetas de texto más descriptivas
                labels_str = [
                    f"Cluster {l}" if l != -1 else "Ruido (-1)"
                    for l in labels_to_visualize
                ]
                # Crear DataFrame de Pandas
                df_vis = pd.DataFrame(
                    reduced_embeddings_vis,
                    columns=[f"Dim_{i+1}" for i in range(n_components_vis)],
                )
                df_vis["label"] = labels_str # Usar etiquetas de texto
                df_vis["id"] = ids_to_visualize # Añadir IDs para hover info
                # Añadir la etiqueta numérica original también para posible ordenamiento
                df_vis["numeric_label"] = labels_to_visualize

                # Ordenar por etiqueta numérica para consistencia en colores/leyenda
                df_vis = df_vis.sort_values(by="numeric_label")

                # 4. Generar el gráfico Plotly
                st.subheader(
                    f"Visualización {vis_method} ({n_components_vis}D)"
                )
                fig = None # Inicializar figura

                # Configuración del hover (mostrar etiqueta e ID)
                hover_data_config = {"label": True, "id": True}
                # Quitar dimensiones del hover ya que son los ejes
                for i in range(n_components_vis):
                    hover_data_config[f"Dim_{i+1}"] = False

                plot_title = f"Clusters Visualizados con {vis_method} ({n_components_vis}D) en '{collection_name}'"

                # Crear gráfico 2D o 3D
                common_scatter_args = dict(
                    data_frame=df_vis,
                    color="label", # Colorear por etiqueta de texto
                    title=plot_title,
                    hover_data=hover_data_config,
                    color_discrete_sequence=CLUSTER_COLOR_PALETTE, # Usar paleta definida
                    # Ordenar categorías para leyenda consistente
                    category_orders={
                        "label": sorted(df_vis["label"].unique(), key=lambda x: int(x.split()[-1]) if x != "Ruido (-1)" else -1)
                    },
                    labels={"label": "Etiqueta Cluster"} # Nombre para leyenda
                )

                if n_components_vis == 2:
                    fig = px.scatter(
                        **common_scatter_args,
                        x="Dim_1",
                        y="Dim_2",
                    )
                    # Ajustar tamaño y opacidad de marcadores 2D
                    fig.update_traces(marker=dict(size=5, opacity=0.75, line=dict(width=0.5, color='DarkSlateGrey')))
                elif n_components_vis == 3:
                    fig = px.scatter_3d(
                         **common_scatter_args,
                         x="Dim_1",
                         y="Dim_2",
                         z="Dim_3",
                    )
                    # Ajustar tamaño y opacidad de marcadores 3D
                    fig.update_traces(marker=dict(size=3, opacity=0.7))

                # 5. Mostrar el gráfico si se generó
                if fig:
                    # Personalizar layout del gráfico
                    fig.update_layout(
                        # legend_title_text="Etiqueta Cluster", # Ya se define en labels
                        xaxis_title=f"{vis_method} Dim 1",
                        yaxis_title=f"{vis_method} Dim 2",
                        # zaxis_title=f"{vis_method} Dim 3" # Solo para 3D, Plotly lo maneja
                        margin=dict(l=0, r=0, b=0, t=40), # Márgenes ajustados
                        height=700, # Altura del gráfico
                        legend=dict( # Posición de la leyenda
                            orientation="h", # Horizontal
                            yanchor="bottom",
                            y=1.02, # Encima del gráfico
                            xanchor="right",
                            x=1
                        ),
                        # Usar fuentes definidas en CSS si es posible (Plotly a veces las ignora)
                        font=dict(
                            family="Lato, sans-serif",
                            size=12,
                            color="#212529" # Usar --text-color
                        ),
                        title_font=dict(
                            family="Inter, sans-serif",
                            size=18,
                             color="#0056b3" # Usar --primary-darker
                        ),
                        # Fondo transparente para que tome el de la app
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                    )
                    # Mostrar en Streamlit
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("❌ No se pudo generar el gráfico.")

            # Manejo de errores específicos del pipeline
            except (PipelineError, DatabaseError, ValueError, ImportError) as e:
                st.error(f"❌ Error generando visualización ({vis_method}): {e}")
                logger.error(
                    f"Error generating visualization ({vis_method})",
                    exc_info=True,
                )
            # Manejo de errores inesperados
            except Exception as e:
                st.error(
                    f"❌ Error inesperado generando visualización ({vis_method}): {e}"
                )
                logger.error(
                    f"Unexpected error generating visualization ({vis_method})",
                    exc_info=True,
                )

