# config.py
import os
from typing import Tuple, Optional

# Optional: Load variables from .env file for local development
from dotenv import load_dotenv

load_dotenv()

# --- Model Configuration ---
MODEL_NAME: str = os.getenv("APP_MODEL_NAME", "jinaai/jina-clip-v2")  # Example: Use standard CLIP
DEVICE: str = os.getenv("APP_DEVICE", "cpu")  # Default to CPU for broader compatibility
TRUST_REMOTE_CODE: bool = os.getenv("APP_TRUST_REMOTE_CODE", "True").lower() == "true"  # Default to False for security

# --- Image Processing Configuration ---
IMAGE_EXTENSIONS_STR: str = os.getenv("APP_IMAGE_EXTENSIONS", ".jpg,.jpeg,.png,.webp,.bmp")  # Common extensions
IMAGE_EXTENSIONS: Tuple[str, ...] = tuple(ext.strip().lower() for ext in IMAGE_EXTENSIONS_STR.split(","))
BATCH_SIZE_IMAGES: int = int(os.getenv("APP_BATCH_SIZE_IMAGES", "16"))  # Batch size for indexing images
DEFAULT_IMAGE_DIR: str = os.getenv("APP_DEFAULT_IMAGE_DIR", "images")  # Default directory for images

# --- Text Processing Configuration ---
BATCH_SIZE_TEXT: int = int(os.getenv("APP_BATCH_SIZE_TEXT", "32"))  # Batch size for vectorizing text queries

# --- Vector Database Configuration ---
CHROMA_DB_PATH: str = os.getenv("APP_CHROMA_DB_PATH", "vector_db_store/")  # Path for ChromaDB persistent storage
CHROMA_COLLECTION_NAME: str = os.getenv("APP_CHROMA_COLLECTION_NAME",
                                        "image_embeddings_clip_base")  # Make name specific to model/settings
VECTOR_DIMENSION_STR: str = os.getenv("APP_VECTOR_DIMENSION", "")  # Default to empty string, meaning no truncation
VECTOR_DIMENSION: Optional[int] = (
    int(VECTOR_DIMENSION_STR) if VECTOR_DIMENSION_STR.isdigit() else None  # Convert to int or None
)
# Note: CLIP ViT-B/32 has dimension 512. Set APP_VECTOR_DIMENSION=512 or leave empty.
# If using Jina CLIP v2 (768 dim), update MODEL_NAME and potentially VECTOR_DIMENSION.

# --- Clustering Configuration ---
DEFAULT_N_CLUSTERS: int = int(os.getenv("APP_DEFAULT_N_CLUSTERS", "10"))

# --- Search Configuration ---
DEFAULT_N_RESULTS: int = int(os.getenv("APP_DEFAULT_N_RESULTS", "10"))  # Default number of search results

# --- Logging Configuration (Example - can be set in main/streamlit entry points) ---
# LOG_LEVEL: str = os.getenv("APP_LOG_LEVEL", "INFO").upper()

# --- END OF CONFIGURATION ---
