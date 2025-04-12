# config.py
import os
from typing import Tuple, Optional

from dotenv import load_dotenv

load_dotenv()

# --- Model Configuration ---
MODEL_NAME: str = os.getenv("APP_MODEL_NAME", "jinaai/jina-clip-v2")
DEVICE: str = os.getenv("APP_DEVICE", "cpu")
TRUST_REMOTE_CODE: bool = os.getenv("APP_TRUST_REMOTE_CODE", "True").lower() == "true"

# --- Image Processing Configuration ---
IMAGE_EXTENSIONS_STR: str = os.getenv(
    "APP_IMAGE_EXTENSIONS", ".jpg,.jpeg,.png,.webp,.bmp"
)
IMAGE_EXTENSIONS: Tuple[str, ...] = tuple(
    ext.strip().lower() for ext in IMAGE_EXTENSIONS_STR.split(",")
)
BATCH_SIZE_IMAGES: int = int(os.getenv("APP_BATCH_SIZE_IMAGES", "16"))
DEFAULT_IMAGE_DIR: str = os.getenv("APP_DEFAULT_IMAGE_DIR", "images")

# --- Text Processing Configuration ---
BATCH_SIZE_TEXT: int = int(os.getenv("APP_BATCH_SIZE_TEXT", "32"))

# --- Vector Database Configuration ---
CHROMA_DB_PATH: str = os.getenv("APP_CHROMA_DB_PATH", "vector_db_store/")
CHROMA_COLLECTION_NAME: str = os.getenv(
    "APP_CHROMA_COLLECTION_NAME", "image_embeddings_clip_base"
)
VECTOR_DIMENSION_STR: str = os.getenv("APP_VECTOR_DIMENSION", "")
VECTOR_DIMENSION: Optional[int] = (
    int(VECTOR_DIMENSION_STR) if VECTOR_DIMENSION_STR.isdigit() else None
)

# --- Search Configuration ---
DEFAULT_N_RESULTS: int = int(os.getenv("APP_DEFAULT_N_RESULTS", "10"))

# --- Logging Configuration ---
LOG_LEVEL: str = os.getenv("APP_LOG_LEVEL", "INFO").upper()
