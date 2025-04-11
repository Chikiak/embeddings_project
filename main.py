import os
import sys
import time

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import necessary components
import config  # Load configuration first
from core.vectorizer import Vectorizer
from data_access.vector_db import VectorDatabase
from app import pipeline  # Import functions from the pipeline module


def main():
    """
    Main execution function to run the MVP pipeline.
    """
    print("--- Vector Image Search MVP ---")
    start_time = time.time()

    # --- Configuration ---
    # These are loaded from config.py, but you can override them here if needed
    MODEL_NAME = config.MODEL_NAME
    DEVICE = config.DEVICE
    TRUST_REMOTE_CODE = config.TRUST_REMOTE_CODE
    DB_PATH = config.CHROMA_DB_PATH
    COLLECTION_NAME = config.CHROMA_COLLECTION_NAME
    IMAGE_DIR = "images"
    BATCH_SIZE = config.BATCH_SIZE_IMAGES
    TRUNCATE_DIM = config.VECTOR_DIMENSION
    N_CLUSTERS = config.DEFAULT_N_CLUSTERS
    N_RESULTS = config.DEFAULT_N_RESULTS

    # --- Sanity Check: Image Directory ---
    if not os.path.isdir(IMAGE_DIR):
        print(f"\nERROR: Image directory not found at '{IMAGE_DIR}'")
        print(
            "Please create this directory and place images inside it, or update IMAGE_DIR in main.py."
        )
        # Optionally, create the directory if it doesn't exist
        try:
            print(f"Attempting to create directory: '{IMAGE_DIR}'")
            os.makedirs(IMAGE_DIR)
            print(
                f"Directory '{IMAGE_DIR}' created. Please add images to it and re-run."
            )
        except OSError as e:
            print(f"Could not create directory '{IMAGE_DIR}': {e}")
        return  # Exit if directory is not valid

    # --- Initialization ---
    print("\n--- Initializing Components ---")
    try:
        vectorizer = Vectorizer(
            model_name=MODEL_NAME, device=DEVICE, trust_remote_code=TRUST_REMOTE_CODE
        )
        db = VectorDatabase(path=DB_PATH, collection_name=COLLECTION_NAME)
    except Exception as e:
        print(f"\nFATAL ERROR during initialization: {e}")
        print("Please check model name, dependencies, and permissions.")
        return  # Exit if initialization fails

    # --- Step 1: Process Image Directory (Indexing) ---
    # This will find images, vectorize them, and store in ChromaDB
    print("\n--- Step 1: Processing Image Directory ---")
    processed_count = pipeline.process_directory(
        directory_path=IMAGE_DIR,
        vectorizer=vectorizer,
        db=db,
        batch_size=BATCH_SIZE,
        truncate_dim=TRUNCATE_DIM,
    )
    print(f"\nFinished processing directory. Stored {processed_count} embeddings.")

    # --- Step 2: Text-to-Image Search ---
    if processed_count > 0:  # Only search if something was indexed
        print("\n--- Step 2: Text-to-Image Search Example ---")
        # <<< IMPORTANT: Change this query to something relevant to your images
        text_query = "una foto de un perro jugando"  # Example query in Spanish
        text_search_results = pipeline.search_by_text(
            query_text=text_query,
            vectorizer=vectorizer,
            db=db,
            n_results=N_RESULTS,
            truncate_dim=TRUNCATE_DIM,
        )
        if text_search_results:
            print("\nTop matching images for query:")
            for i, (img_id, dist) in enumerate(
                zip(text_search_results["ids"], text_search_results["distances"])
            ):
                metadata = (
                    text_search_results["metadatas"][i]
                    if text_search_results["metadatas"]
                    else {}
                )
                print(
                    f"  {i+1}. ID: {img_id} (Distance: {dist:.4f}) Metadata: {metadata}"
                )
        else:
            print("Text search did not return results.")
    else:
        print("\nSkipping Step 2 (Text Search) as no images were processed.")

    # --- Step 3: Image-to-Image Search ---
    if processed_count > 0:
        print("\n--- Step 3: Image-to-Image Search Example ---")
        # <<< IMPORTANT: Set this to the path of an image file (ideally one from your IMAGE_DIR)
        # Find a sample image to use for the query
        all_files_in_dir = [
            os.path.join(IMAGE_DIR, f)
            for f in os.listdir(IMAGE_DIR)
            if os.path.isfile(os.path.join(IMAGE_DIR, f))
        ]
        image_files_in_dir = [
            f for f in all_files_in_dir if f.lower().endswith(config.IMAGE_EXTENSIONS)
        ]

        if image_files_in_dir:
            query_image_path = image_files_in_dir[0]  # Use the first found image
            print(f"Using image '{query_image_path}' as query...")

            image_search_results = pipeline.search_by_image(
                query_image_path=query_image_path,
                vectorizer=vectorizer,
                db=db,
                n_results=N_RESULTS,
                truncate_dim=TRUNCATE_DIM,
            )
            if image_search_results:
                print("\nTop similar images found:")
                for i, (img_id, dist) in enumerate(
                    zip(image_search_results["ids"], image_search_results["distances"])
                ):
                    metadata = (
                        image_search_results["metadatas"][i]
                        if image_search_results["metadatas"]
                        else {}
                    )
                    print(
                        f"  {i+1}. ID: {img_id} (Distance: {dist:.4f}) Metadata: {metadata}"
                    )
            else:
                print("Image search did not return results.")
        else:
            print("No image file found in the directory to use as a query image.")
    else:
        print("\nSkipping Step 3 (Image Search) as no images were processed.")

    # --- Step 4: Clustering ---
    if (
        processed_count >= N_CLUSTERS and processed_count > 0
    ):  # Only cluster if enough images were processed
        print("\n--- Step 4: Clustering Example ---")
        cluster_assignments = pipeline.perform_clustering(db=db, n_clusters=N_CLUSTERS)
        if cluster_assignments:
            print("\nCluster assignments (sample):")
            count = 0
            for img_id, cluster_id in cluster_assignments.items():
                print(f"  Image ID: {img_id} -> Cluster ID: {cluster_id}")
                count += 1
                if count >= 10:  # Print first 10 assignments
                    print("  ...")
                    break
            # You could further analyze cluster contents here
        else:
            print("Clustering did not return assignments.")
    elif processed_count > 0:
        print(
            f"\nSkipping Step 4 (Clustering) as the number of processed images ({processed_count}) is less than the desired number of clusters ({N_CLUSTERS})."
        )
    else:
        print("\nSkipping Step 4 (Clustering) as no images were processed.")

    # --- Cleanup / End ---
    end_time = time.time()
    print(f"\n--- Total Execution Time: {end_time - start_time:.2f} seconds ---")


if __name__ == "__main__":
    if sys.version_info < (3, 7):  # Example check, adjust as needed
        print(
            "Warning: This script might require Python 3.7+ for optimal compatibility."
        )

    main()
