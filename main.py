# main.py
import os
import sys
import time
import argparse
import logging

# --- Initial Setup and Path Configuration ---
# Ensure the project root directory is in sys.path for imports
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import necessary components AFTER potentially modifying sys.path
import config  # Load application configuration
from app.factory import get_initialized_components  # Use the factory
from app import cli_handlers  # Import the handlers for CLI actions

# --- Logging Configuration ---
# Configure basic logging for the application entry point
logging.basicConfig(
    level=logging.INFO,  # Set default logging level
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,  # Log to standard output
)
logger = logging.getLogger(__name__)  # Get logger for this module


# --- Argument Parsing ---
def setup_arg_parser() -> argparse.ArgumentParser:
    """Sets up and returns the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        description="Herramienta CLI para Búsqueda Vectorial de Imágenes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter  # Show defaults in help
    )

    # Configuration Arguments
    parser.add_argument("--image-dir", default=config.DEFAULT_IMAGE_DIR,
                        help="Directorio que contiene las imágenes a procesar.")
    parser.add_argument("--db-path", default=config.CHROMA_DB_PATH,
                        help="Ruta para la base de datos vectorial persistente.")
    parser.add_argument("--collection-name", default=config.CHROMA_COLLECTION_NAME,
                        help="Nombre de la colección en la BD.")
    parser.add_argument("--model", default=config.MODEL_NAME, help="Nombre del modelo de Hugging Face a usar.")
    parser.add_argument("--device", default=config.DEVICE, choices=["cuda", "cpu"],
                        help="Dispositivo para la inferencia ('cuda' o 'cpu').")
    parser.add_argument("--n-results", type=int, default=config.DEFAULT_N_RESULTS,
                        help="Número de resultados a mostrar en búsquedas.")
    parser.add_argument("--n-clusters", type=int, default=config.DEFAULT_N_CLUSTERS,
                        help="Número de clústeres para el análisis.")
    parser.add_argument(
        "--truncate-dim",
        type=lambda x: int(x) if x and x.isdigit() else None,  # Allow empty string or non-digit for None
        default=config.VECTOR_DIMENSION,
        help="Dimensión para truncar vectores (dejar vacío o no numérico para no truncar)."
    )

    # Action Arguments (Mutually Exclusive Group might be better if only one action allowed at a time)
    # For now, allow multiple actions (e.g., index then search)
    parser.add_argument("--index", action="store_true", help="Realizar la indexación de imágenes en --image-dir.")
    parser.add_argument("--search-text", metavar='QUERY',
                        help="Realizar búsqueda por texto con la consulta proporcionada.")
    parser.add_argument("--search-image", metavar='IMAGE_PATH',
                        help="Realizar búsqueda por imagen usando la ruta de imagen proporcionada.")
    parser.add_argument("--cluster", action="store_true", help="Realizar clustering de las imágenes indexadas.")
    parser.add_argument("--clear-collection", action="store_true",
                        help="¡PELIGRO! Elimina todos los elementos de la colección especificada.")
    parser.add_argument("--delete-collection", action="store_true",
                        help="¡PELIGRO! Elimina toda la colección especificada.")

    return parser


# --- Main Execution Logic ---
def main():
    """Main execution function for the CLI tool."""
    parser = setup_arg_parser()
    args = parser.parse_args()

    logger.info("--- Vector Image Search CLI ---")
    start_time = time.time()

    # --- Check if any action was requested ---
    action_requested = any([
        args.index, args.search_text, args.search_image,
        args.cluster, args.clear_collection, args.delete_collection
    ])
    if not action_requested:
        parser.error(
            "No action requested. Please specify an action (e.g., --index, --search-text QUERY). Use -h for help.")
        return  # Exit if no action

    # --- Initialize Components using Factory ---
    logger.info("--- Initializing Components ---")
    try:
        # Get vectorizer and database instances from the factory
        vectorizer, db = get_initialized_components()
        # Pass necessary config values from args to handlers if needed,
        # otherwise handlers can import config directly if appropriate.
    except RuntimeError as e:
        logger.critical(f"Initialization failed: {e}. Exiting.")
        # Optionally log more details if needed from e.__cause__
        return  # Exit if components fail to initialize

    # --- Execute Actions based on Arguments by calling handlers ---
    if args.clear_collection:
        cli_handlers.handle_clear_collection(args, db)
        # Decide if other actions should proceed after clear/delete
        # For safety, maybe exit after destructive actions unless combined explicitly?
        # For now, we continue.

    if args.delete_collection:
        cli_handlers.handle_delete_collection(args, db)
        # Exit after delete as the DB object might be invalid
        logger.info("Exiting after delete operation.")
        end_time = time.time()
        logger.info(f"\n--- Total Execution Time: {end_time - start_time:.2f} seconds ---")
        return

    if args.index:
        cli_handlers.handle_indexing(args, vectorizer, db)

    if args.search_text:
        cli_handlers.handle_text_search(args, vectorizer, db)

    if args.search_image:
        cli_handlers.handle_image_search(args, vectorizer, db)

    if args.cluster:
        cli_handlers.handle_clustering(args, db)  # Vectorizer not needed for clustering

    # --- Finalization ---
    end_time = time.time()
    logger.info(f"\n--- Total Execution Time: {end_time - start_time:.2f} seconds ---")


# --- Script Entry Point ---
if __name__ == "__main__":
    # Optional Python version check
    if sys.version_info < (3, 8):
        logger.warning("Warning: This script is recommended to run with Python 3.8+ for full compatibility.")
    main()
