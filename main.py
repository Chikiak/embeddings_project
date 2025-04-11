import os
import sys
import time
import argparse  # Importar argparse
import logging  # Importar logging

# --- Configuración Inicial y Rutas ---
# Asegurarse de que el directorio raíz del proyecto esté en sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Importar componentes necesarios DESPUÉS de potencialmente añadir a sys.path
import config  # Cargar configuración
from core.vectorizer import Vectorizer
from data_access.vector_db import VectorDatabase
from app import pipeline  # Importar funciones del pipeline

# --- Configuración de Logging ---
# Configurar el logging básico aquí, en el punto de entrada
# El nivel y formato pueden ser ajustados según necesidad
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,  # Asegurar salida a consola
)
# Obtener un logger para este módulo específico
logger = logging.getLogger(__name__)


def main():
    """
    Función principal de ejecución para la herramienta CLI.
    """
    # --- Argument Parser ---
    # Definir los argumentos que la herramienta aceptará desde la línea de comandos
    parser = argparse.ArgumentParser(
        description="Herramienta CLI para Búsqueda Vectorial de Imágenes"
    )
    parser.add_argument(
        "--image-dir",
        default="images",
        help="Directorio que contiene las imágenes a procesar.",
    )
    parser.add_argument(
        "--db-path",
        default=config.CHROMA_DB_PATH,
        help="Ruta para la base de datos vectorial persistente.",
    )
    parser.add_argument(
        "--collection-name",
        default=config.CHROMA_COLLECTION_NAME,
        help="Nombre de la colección en la BD.",
    )
    parser.add_argument(
        "--model",
        default=config.MODEL_NAME,
        help="Nombre del modelo de Hugging Face a usar.",
    )
    parser.add_argument(
        "--device",
        default=config.DEVICE,
        choices=["cuda", "cpu"],
        help="Dispositivo para la inferencia ('cuda' o 'cpu').",
    )
    parser.add_argument(
        "--n-results",
        type=int,
        default=config.DEFAULT_N_RESULTS,
        help="Número de resultados a mostrar en búsquedas.",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=config.DEFAULT_N_CLUSTERS,
        help="Número de clústeres para el análisis.",
    )
    # Permitir que truncate_dim sea None si no se proporciona un entero válido
    parser.add_argument(
        "--truncate-dim",
        type=lambda x: int(x) if x.isdigit() else None,
        default=config.VECTOR_DIMENSION,
        help="Dimensión para truncar vectores (dejar vacío o no numérico para no truncar).",
    )

    # Acciones que puede realizar la herramienta
    parser.add_argument(
        "--index",
        action="store_true",
        help="Realizar la indexación de imágenes en --image-dir.",
    )
    parser.add_argument(
        "--search-text",
        help="Realizar búsqueda por texto con la consulta proporcionada.",
    )
    parser.add_argument(
        "--search-image",
        help="Realizar búsqueda por imagen usando la ruta de imagen proporcionada.",
    )
    parser.add_argument(
        "--cluster",
        action="store_true",
        help="Realizar clustering de las imágenes indexadas.",
    )
    parser.add_argument(
        "--clear-collection",
        action="store_true",
        help="¡PELIGRO! Elimina todos los elementos de la colección especificada.",
    )
    parser.add_argument(
        "--delete-collection",
        action="store_true",
        help="¡PELIGRO! Elimina toda la colección especificada.",
    )

    # Parsear los argumentos proporcionados por el usuario
    args = parser.parse_args()

    logger.info("--- Vector Image Search CLI ---")
    start_time = time.time()

    # --- Validación de Acción Requerida ---
    # Asegurar que el usuario haya solicitado al menos una acción
    if not any(
        [
            args.index,
            args.search_text,
            args.search_image,
            args.cluster,
            args.clear_collection,
            args.delete_collection,
        ]
    ):
        parser.error(
            "No action requested. Please specify an action like --index, --search-text, etc."
        )
        return  # Salir si no hay acción

    # --- Inicialización de Componentes ---
    # Inicializar Vectorizer y VectorDatabase usando los argumentos parseados o valores de config
    logger.info("--- Initializing Components ---")
    try:
        vectorizer = Vectorizer(
            model_name=args.model,
            device=args.device,
            trust_remote_code=config.TRUST_REMOTE_CODE,  # TRUST_REMOTE_CODE usualmente viene de config por seguridad
        )
        db = VectorDatabase(path=args.db_path, collection_name=args.collection_name)
    except Exception as e:
        logger.error(f"FATAL ERROR during initialization: {e}", exc_info=True)
        logger.error(
            "Please check model name, dependencies, DB path permissions, and network access."
        )
        return  # Salir si la inicialización falla

    # --- Ejecutar Acciones Basadas en Argumentos ---

    # Acción: Limpiar Colección
    if args.clear_collection:
        logger.warning(
            f"ACTION: Clearing all items from collection '{args.collection_name}' at path '{args.db_path}'..."
        )
        # Solicitar confirmación explícita al usuario
        confirm = input(
            f"Are you absolutely sure you want to clear ALL items from '{args.collection_name}'? [yes/NO]: "
        )
        if confirm.lower() == "yes":
            try:
                db.clear_collection()
                logger.info(
                    f"Collection '{args.collection_name}' cleared successfully."
                )
            except Exception as e:
                logger.error(f"Error clearing collection: {e}", exc_info=True)
        else:
            logger.info("Clear operation cancelled by user.")

    # Acción: Eliminar Colección
    if args.delete_collection:
        logger.warning(
            f"ACTION: Deleting the ENTIRE collection '{args.collection_name}' from path '{args.db_path}'..."
        )
        # Solicitar confirmación explícita y doble
        confirm1 = input(
            f"Are you absolutely sure you want to DELETE the collection '{args.collection_name}'? This is IRREVERSIBLE. [yes/NO]: "
        )
        if confirm1.lower() == "yes":
            confirm2 = input("Please type 'DELETE' to confirm: ")
            if confirm2 == "DELETE":
                try:
                    db.delete_collection()
                    logger.info(
                        f"Collection '{args.collection_name}' deleted successfully."
                    )
                except Exception as e:
                    logger.error(f"Error deleting collection: {e}", exc_info=True)
            else:
                logger.info("Delete operation cancelled (confirmation mismatch).")
        else:
            logger.info("Delete operation cancelled by user.")

    # Acción: Indexar Imágenes
    if args.index:
        logger.info(f"--- ACTION: Processing Image Directory: {args.image_dir} ---")
        # Validar que el directorio de imágenes exista
        if not os.path.isdir(args.image_dir):
            logger.error(
                f"Image directory not found: '{args.image_dir}'. Please check the path."
            )
            # Intentar crear el directorio si no existe (opcional)
            try:
                logger.info(f"Attempting to create directory: '{args.image_dir}'")
                os.makedirs(args.image_dir)
                logger.info(
                    f"Directory '{args.image_dir}' created. Please add images to it and re-run --index."
                )
            except OSError as e:
                logger.error(f"Could not create directory '{args.image_dir}': {e}")
        else:
            # Llamar a la función del pipeline para procesar el directorio
            processed_count = pipeline.process_directory(
                directory_path=args.image_dir,
                vectorizer=vectorizer,
                db=db,
                batch_size=config.BATCH_SIZE_IMAGES,  # Se podría añadir un argumento para esto
                truncate_dim=args.truncate_dim,
                # No se necesitan callbacks para la versión CLI
            )
            logger.info(
                f"Finished processing directory. Stored {processed_count} new/updated embeddings."
            )

    # Acción: Búsqueda por Texto
    if args.search_text:
        logger.info(
            f"--- ACTION: Text-to-Image Search for query: '{args.search_text}' ---"
        )
        # Verificar si la base de datos tiene elementos antes de buscar
        if db.collection is None or db.collection.count() == 0:
            logger.warning(
                "Cannot perform search: The database collection is empty or not initialized."
            )
        else:
            # Llamar a la función del pipeline para buscar por texto
            text_search_results = pipeline.search_by_text(
                query_text=args.search_text,
                vectorizer=vectorizer,
                db=db,
                n_results=args.n_results,
                truncate_dim=args.truncate_dim,
            )
            # Mostrar resultados o mensaje si no hay
            if text_search_results and text_search_results.get("ids"):
                logger.info(
                    f"\nTop {len(text_search_results['ids'])} matching images for query:"
                )
                for i, (img_id, dist) in enumerate(
                    zip(text_search_results["ids"], text_search_results["distances"])
                ):
                    metadata = (
                        text_search_results.get("metadatas", [])[i]
                        if text_search_results.get("metadatas")
                        and i < len(text_search_results.get("metadatas", []))
                        else {}
                    )
                    logger.info(
                        f"  {i+1}. ID: {img_id} (Distance: {dist:.4f}) Metadata: {metadata}"
                    )
            elif text_search_results:  # La búsqueda se ejecutó pero no encontró nada
                logger.info(
                    "Search successful, but no similar images found in the database for this query."
                )
            else:  # La búsqueda falló (ya se logueó el error en el pipeline)
                logger.error("Text search failed to return results.")

    # Acción: Búsqueda por Imagen
    if args.search_image:
        logger.info(
            f"--- ACTION: Image-to-Image Search using image: '{args.search_image}' ---"
        )
        # Validar que el archivo de imagen de consulta exista
        if not os.path.isfile(args.search_image):
            logger.error(
                f"Query image file not found: '{args.search_image}'. Please check the path."
            )
        # Verificar si la base de datos tiene elementos
        elif db.collection is None or db.collection.count() == 0:
            logger.warning(
                "Cannot perform search: The database collection is empty or not initialized."
            )
        else:
            # Llamar a la función del pipeline para buscar por imagen
            image_search_results = pipeline.search_by_image(
                query_image_path=args.search_image,
                vectorizer=vectorizer,
                db=db,
                n_results=args.n_results,
                truncate_dim=args.truncate_dim,
            )
            # Mostrar resultados o mensaje si no hay
            if image_search_results and image_search_results.get("ids"):
                logger.info(
                    f"\nTop {len(image_search_results['ids'])} similar images found:"
                )
                for i, (img_id, dist) in enumerate(
                    zip(image_search_results["ids"], image_search_results["distances"])
                ):
                    metadata = (
                        image_search_results.get("metadatas", [])[i]
                        if image_search_results.get("metadatas")
                        and i < len(image_search_results.get("metadatas", []))
                        else {}
                    )
                    logger.info(
                        f"  {i+1}. ID: {img_id} (Distance: {dist:.4f}) Metadata: {metadata}"
                    )
            elif image_search_results:  # Búsqueda OK, sin resultados
                logger.info(
                    "Search successful, but no similar images found (after potential self-filtering)."
                )
            else:  # Búsqueda falló
                logger.error("Image search failed to return results.")

    # Acción: Clustering
    if args.cluster:
        logger.info(
            f"--- ACTION: Clustering indexed images into {args.n_clusters} clusters ---"
        )
        # Verificar si hay suficientes elementos para los clústeres solicitados
        item_count = db.collection.count() if db.collection else 0
        if item_count < args.n_clusters:
            logger.warning(
                f"Cannot perform clustering: Need at least {args.n_clusters} indexed items, but found only {item_count}. Skipping."
            )
        elif item_count == 0:
            logger.warning(
                "Cannot perform clustering: The database collection is empty."
            )
        else:
            # Llamar a la función del pipeline para realizar clustering
            cluster_assignments = pipeline.perform_clustering(
                db=db, n_clusters=args.n_clusters
            )
            # Mostrar resultados (ejemplo)
            if cluster_assignments:
                logger.info("\nCluster assignments (showing sample):")
                count = 0
                for img_id, cluster_id in cluster_assignments.items():
                    logger.info(
                        f"  Image ID: {os.path.basename(img_id)} -> Cluster ID: {cluster_id}"
                    )  # Mostrar solo nombre de archivo
                    count += 1
                    if count >= 15:  # Mostrar hasta 15 asignaciones
                        logger.info("  ... (more assignments exist)")
                        break
                # Aquí se podrían añadir análisis adicionales por clúster
            else:
                logger.error("Clustering failed or did not return assignments.")

    # --- Finalización ---
    end_time = time.time()
    logger.info(f"\n--- Total Execution Time: {end_time - start_time:.2f} seconds ---")


if __name__ == "__main__":
    # Comprobación opcional de versión de Python
    if sys.version_info < (3, 8):
        logger.warning(
            "Warning: This script is best run with Python 3.8+ for full compatibility with dependencies."
        )
    # Llamar a la función principal
    main()
