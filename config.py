import os
from typing import List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()


AVAILABLE_MODELS: List[str] = [
    "jinaai/jina-clip-v2",
]


DEFAULT_MODEL_NAME: str = os.getenv("APP_MODEL_NAME", AVAILABLE_MODELS[0])
MODEL_NAME: str = DEFAULT_MODEL_NAME

DEVICE: str = os.getenv("APP_DEVICE", "cpu")
TRUST_REMOTE_CODE: bool = (
    os.getenv("APP_TRUST_REMOTE_CODE", "True").lower() == "true"
)


IMAGE_EXTENSIONS_STR: str = os.getenv(
    "APP_IMAGE_EXTENSIONS", ".jpg,.jpeg,.png,.webp,.bmp"
)
IMAGE_EXTENSIONS: Tuple[str, ...] = tuple(
    ext.strip().lower() for ext in IMAGE_EXTENSIONS_STR.split(",")
)
BATCH_SIZE_IMAGES: int = int(os.getenv("APP_BATCH_SIZE_IMAGES", "16"))
DEFAULT_IMAGE_DIR: str = os.getenv("APP_DEFAULT_IMAGE_DIR", "images")


BATCH_SIZE_TEXT: int = int(os.getenv("APP_BATCH_SIZE_TEXT", "32"))


CHROMA_DB_PATH: str = os.getenv("APP_CHROMA_DB_PATH", "vector_db_store/")
CHROMA_COLLECTION_NAME: str = os.getenv(
    "APP_CHROMA_COLLECTION_NAME", "image_embeddings_default"
)
VECTOR_DIMENSION_STR: str = os.getenv("APP_VECTOR_DIMENSION", "")
VECTOR_DIMENSION: Optional[int] = (
    int(VECTOR_DIMENSION_STR) if VECTOR_DIMENSION_STR.isdigit() else None
)


DEFAULT_N_RESULTS: int = int(os.getenv("APP_DEFAULT_N_RESULTS", "10"))


LOG_LEVEL: str = os.getenv("APP_LOG_LEVEL", "INFO").upper()
