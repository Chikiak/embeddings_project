# config.py
import os
from typing import Tuple, Optional, List # Added List

from dotenv import load_dotenv

load_dotenv()

# --- Model Configuration ---
# Lista de modelos de embedding disponibles para el usuario
AVAILABLE_MODELS: List[str] = [
    # Agregar aquí otros modelos compatibles
    "jinaai/jina-clip-v2",
    # "openai/clip-vit-base-patch32",
    # "google/siglip-base-patch16-224",
]

# Modelo por defecto si el usuario no elige (o para la CLI)
DEFAULT_MODEL_NAME: str = os.getenv("APP_MODEL_NAME", AVAILABLE_MODELS[0]) # Usa el primero como default
MODEL_NAME: str = DEFAULT_MODEL_NAME # Mantén MODEL_NAME para compatibilidad CLI/default

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
    "APP_CHROMA_COLLECTION_NAME", "image_embeddings_default" # Changed default name slightly
)
VECTOR_DIMENSION_STR: str = os.getenv("APP_VECTOR_DIMENSION", "")
VECTOR_DIMENSION: Optional[int] = (
    int(VECTOR_DIMENSION_STR) if VECTOR_DIMENSION_STR.isdigit() else None
)

# --- Search Configuration ---
DEFAULT_N_RESULTS: int = int(os.getenv("APP_DEFAULT_N_RESULTS", "10"))

# --- Logging Configuration ---
LOG_LEVEL: str = os.getenv("APP_LOG_LEVEL", "INFO").upper()
