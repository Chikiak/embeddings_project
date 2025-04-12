# app/cli_handlers.py
import os
import logging
import time
from argparse import Namespace  # Use Namespace for type hint

# Import core components and pipeline
from core.vectorizer import Vectorizer
from data_access.vector_db_interface import VectorDBInterface
from app import pipeline
import config  # To access default batch size etc.

logger = logging.getLogger(__name__)


def handle_indexing(args: Namespace, vectorizer: Vectorizer, db: VectorDBInterface):
    """Handles the --index action."""
    logger.info(f"--- ACTION: Processing Image Directory: {args.image_dir} ---")
    if not os.path.isdir(args.image_dir):
        logger.error(f"Image directory not found: '{args.image_dir}'.")
        try:
            logger.info(f"Attempting to create directory: '{args.image_dir}'")
            os.makedirs(args.image_dir)
            logger.info(f"Directory '{args.image_dir}' created. Please add images and re-run --index.")
        except OSError as e:
            logger.error(f"Could not create directory '{args.image_dir}': {e}")
        return  # Stop processing if directory was initially missing

    # Call the pipeline function
    processed_count = pipeline.process_directory(
        directory_path=args.image_dir,
        vectorizer=vectorizer,
        db=db,
        batch_size=config.BATCH_SIZE_IMAGES,  # Use config or add CLI arg
        truncate_dim=args.truncate_dim,
        # No progress/status callbacks needed for CLI
    )
    logger.info(f"Finished processing directory. Stored/updated {processed_count} embeddings.")


def handle_text_search(args: Namespace, vectorizer: Vectorizer, db: VectorDBInterface):
    """Handles the --search-text action."""
    logger.info(f"--- ACTION: Text-to-Image Search for query: '{args.search_text}' ---")
    if not db.is_initialized or db.count() == 0:
        logger.warning("Cannot perform search: The database is empty or not initialized.")
        return

    start_search_time = time.time()
    search_results = pipeline.search_by_text(
        query_text=args.search_text,
        vectorizer=vectorizer,
        db=db,
        n_results=args.n_results,
        truncate_dim=args.truncate_dim,
    )
    end_search_time = time.time()
    logger.info(f"Search completed in {end_search_time - start_search_time:.2f}s.")

    if search_results is None:
        logger.error("Text search failed to return results (check previous logs).")
    elif search_results.is_empty:
        logger.info("Search successful, but no similar images found in the database.")
    else:
        logger.info(f"\nTop {search_results.count} matching images for query:")
        for i, item in enumerate(search_results.items):
            similarity_str = f"{item.similarity:.4f}" if item.similarity is not None else "N/A"
            logger.info(f"  {i + 1}. ID: {item.id} (Similarity: {similarity_str}) Metadata: {item.metadata}")


def handle_image_search(args: Namespace, vectorizer: Vectorizer, db: VectorDBInterface):
    """Handles the --search-image action."""
    logger.info(f"--- ACTION: Image-to-Image Search using image: '{args.search_image}' ---")
    if not os.path.isfile(args.search_image):
        logger.error(f"Query image file not found: '{args.search_image}'. Please check the path.")
        return
    if not db.is_initialized or db.count() == 0:
        logger.warning("Cannot perform search: The database is empty or not initialized.")
        return

    start_search_time = time.time()
    search_results = pipeline.search_by_image(
        query_image_path=args.search_image,
        vectorizer=vectorizer,
        db=db,
        n_results=args.n_results,
        truncate_dim=args.truncate_dim,
    )
    end_search_time = time.time()
    logger.info(f"Search completed in {end_search_time - start_search_time:.2f}s.")

    if search_results is None:
        logger.error("Image search failed to return results (check previous logs).")
    elif search_results.is_empty:
        logger.info("Search successful, but no similar images found (after potential self-filtering).")
    else:
        logger.info(f"\nTop {search_results.count} similar images found:")
        for i, item in enumerate(search_results.items):
            similarity_str = f"{item.similarity:.4f}" if item.similarity is not None else "N/A"
            logger.info(f"  {i + 1}. ID: {item.id} (Similarity: {similarity_str}) Metadata: {item.metadata}")


def handle_clustering(args: Namespace, db: VectorDBInterface):
    """Handles the --cluster action."""
    logger.info(f"--- ACTION: Clustering indexed images into {args.n_clusters} clusters ---")
    item_count = db.count()
    if item_count < args.n_clusters:
        logger.warning(
            f"Cannot perform clustering: Need at least {args.n_clusters} indexed items, but found only {item_count}. Skipping.")
        return
    if item_count == 0:
        logger.warning("Cannot perform clustering: The database collection is empty.")
        return

    start_cluster_time = time.time()
    cluster_assignments = pipeline.perform_clustering(
        db=db, n_clusters=args.n_clusters
    )
    end_cluster_time = time.time()
    logger.info(f"Clustering completed in {end_cluster_time - start_cluster_time:.2f}s.")

    if cluster_assignments:
        logger.info("\nCluster assignments (showing sample):")
        count = 0
        for img_id, cluster_id in cluster_assignments.items():
            # Show only basename for brevity in CLI
            logger.info(f"  Image: {os.path.basename(img_id)} -> Cluster ID: {cluster_id}")
            count += 1
            if count >= 15:  # Limit output sample size
                logger.info("  ... (more assignments exist)")
                break
    else:
        logger.error("Clustering failed or did not return assignments (check previous logs).")


def _confirm_action(prompt_message: str, confirmation_keyword: str | None = None) -> bool:
    """Helper function to get user confirmation."""
    confirm = input(f"{prompt_message} [yes/NO]: ")
    if confirm.lower() != "yes":
        return False
    if confirmation_keyword:
        confirm2 = input(f"Please type '{confirmation_keyword}' to confirm: ")
        if confirm2 != confirmation_keyword:
            return False
    return True


def handle_clear_collection(args: Namespace, db: VectorDBInterface):
    """Handles the --clear-collection action."""
    logger.warning(f"ACTION: Clearing all items from collection '{args.collection_name}' at path '{args.db_path}'...")
    prompt = f"Are you absolutely sure you want to clear ALL items from '{args.collection_name}'?"
    if _confirm_action(prompt):
        try:
            success = db.clear_collection()
            if success:
                logger.info(f"Collection '{args.collection_name}' cleared successfully.")
            else:
                logger.error("Failed to clear collection (check previous logs).")
        except Exception as e:
            logger.error(f"Error during clear_collection: {e}", exc_info=True)
    else:
        logger.info("Clear operation cancelled by user.")


def handle_delete_collection(args: Namespace, db: VectorDBInterface):
    """Handles the --delete-collection action."""
    logger.warning(f"ACTION: Deleting the ENTIRE collection '{args.collection_name}' from path '{args.db_path}'...")
    prompt = f"Are you absolutely sure you want to DELETE the collection '{args.collection_name}'? This is IRREVERSIBLE."
    if _confirm_action(prompt, confirmation_keyword="DELETE"):
        try:
            success = db.delete_collection()
            if success:
                logger.info(f"Collection '{args.collection_name}' deleted successfully.")
            else:
                logger.error("Failed to delete collection (check previous logs).")
        except Exception as e:
            logger.error(f"Error during delete_collection: {e}", exc_info=True)
    else:
        logger.info("Delete operation cancelled by user.")
