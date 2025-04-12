import argparse
import logging
import os
import sys
import time

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


import config
from app import cli_handlers
from app.exceptions import InitializationError
from app.factory import get_initialized_components
from app.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def setup_arg_parser() -> argparse.ArgumentParser:
    """
    Configura y devuelve el analizador de argumentos para la CLI.

    Returns:
        El objeto ArgumentParser configurado.
    """
    parser = argparse.ArgumentParser(
        description="Herramienta CLI para Búsqueda Vectorial de Imágenes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--image-dir",
        default=config.DEFAULT_IMAGE_DIR,
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
        "--truncate-dim",
        type=lambda x: int(x) if x and x.isdigit() else None,
        default=config.VECTOR_DIMENSION,
        help="Dimensión para truncar vectores (dejar vacío o no numérico para no truncar).",
    )

    parser.add_argument(
        "--index",
        action="store_true",
        help="Realizar la indexación de imágenes en --image-dir.",
    )
    parser.add_argument(
        "--search-text",
        metavar="QUERY",
        help="Realizar búsqueda por texto con la consulta proporcionada.",
    )
    parser.add_argument(
        "--search-image",
        metavar="IMAGE_PATH",
        help="Realizar búsqueda por imagen usando la ruta de imagen proporcionada.",
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

    return parser


def main():
    """Función principal de ejecución para la herramienta CLI."""
    parser = setup_arg_parser()
    args = parser.parse_args()

    logger.info("--- Vector Image Search CLI ---")
    start_time = time.time()

    action_requested = any(
        [
            args.index,
            args.search_text,
            args.search_image,
            args.clear_collection,
            args.delete_collection,
        ]
    )
    if not action_requested:
        parser.error(
            "No action requested. Please specify an action (e.g., --index, --search-text QUERY). Use -h for help."
        )
        return
    logger.info("--- Initializing Components ---")
    try:
        vectorizer, db = get_initialized_components()
    except InitializationError as e:
        logger.critical(f"Initialization failed: {e}. Exiting.")
        return
    except Exception as e:
        logger.critical(
            f"Unexpected error during initialization: {e}. Exiting.",
            exc_info=True,
        )
        return

    if args.clear_collection:
        cli_handlers.handle_clear_collection(args, db)

    if args.delete_collection:
        cli_handlers.handle_delete_collection(args, db)
        logger.info("Exiting after delete operation.")
        end_time = time.time()
        logger.info(
            f"\n--- Total Execution Time: {end_time - start_time:.2f} seconds ---"
        )
        return

    if args.index:
        cli_handlers.handle_indexing(args, vectorizer, db)

    if args.search_text:
        cli_handlers.handle_text_search(args, vectorizer, db)

    if args.search_image:
        cli_handlers.handle_image_search(args, vectorizer, db)

    end_time = time.time()
    logger.info(
        f"\n--- Total Execution Time: {end_time - start_time:.2f} seconds ---"
    )


if __name__ == "__main__":
    if sys.version_info < (3, 8):
        logger.warning(
            "Warning: This script is recommended to run with Python 3.8+ for full compatibility."
        )
    main()
