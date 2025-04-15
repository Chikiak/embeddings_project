# --- config.py ---
import os
from typing import List, Optional, Tuple

from dotenv import load_dotenv

# Carga variables de entorno desde .env
load_dotenv()

# --- Model Configuration ---
# Lista de modelos disponibles (puedes añadir más aquí)
AVAILABLE_MODELS: List[str] = [
    "jinaai/jina-clip-v2"
    # "openai/clip-vit-base-patch32", # Ejemplo si añades más
    # "openai/clip-vit-large-patch14"
]

# Modelo por defecto a usar (puede ser sobreescrito por .env)
DEFAULT_MODEL_NAME: str = os.getenv("APP_MODEL_NAME", AVAILABLE_MODELS[0])
# Asegura que el modelo por defecto esté disponible
MODEL_NAME: str = DEFAULT_MODEL_NAME if DEFAULT_MODEL_NAME in AVAILABLE_MODELS else AVAILABLE_MODELS[0]

# Dispositivo para inferencia ('cuda' si hay GPU disponible y deseado, sino 'cpu')
DEVICE: str = os.getenv("APP_DEVICE", "cpu")
TRUST_REMOTE_CODE: bool = (
    os.getenv("APP_TRUST_REMOTE_CODE", "True").lower() == "true"
)

# --- Image Processing Configuration ---
# Extensiones de imagen soportadas (definidas directamente)
IMAGE_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif")
# Tamaño de lote para procesar imágenes durante la indexación
BATCH_SIZE_IMAGES: int = int(os.getenv("APP_BATCH_SIZE_IMAGES", "16"))
# Directorio por defecto para buscar imágenes
DEFAULT_IMAGE_DIR: str = os.getenv("APP_DEFAULT_IMAGE_DIR", "images")

# --- Text Processing Configuration ---
# Tamaño de lote para procesar texto
BATCH_SIZE_TEXT: int = int(os.getenv("APP_BATCH_SIZE_TEXT", "32"))

# --- Database Configuration ---
# Ruta para almacenar la base de datos vectorial persistente
CHROMA_DB_PATH: str = os.getenv("APP_CHROMA_DB_PATH", "vector_db_store/")
# Nombre base para las colecciones de BD de IMAGEN (se añadirán sufijos _dimXXX o _full)
CHROMA_COLLECTION_NAME_BASE: str = os.getenv(
    "APP_CHROMA_COLLECTION_NAME_BASE", "image_embeddings"
)
# Nombre base para las colecciones de BD de TEXTO (se añadirán sufijos) - NUEVO
TEXT_DB_COLLECTION_NAME_BASE: str = os.getenv(
    "APP_TEXT_DB_COLLECTION_NAME_BASE", "text_labels"
)

# Dimensión objetivo para los vectores (None o 0 para usar la nativa del modelo)
VECTOR_DIMENSION_STR: str = os.getenv("APP_VECTOR_DIMENSION", "0") # Default 0 para nativa
VECTOR_DIMENSION: Optional[int] = (
    int(VECTOR_DIMENSION_STR) if VECTOR_DIMENSION_STR.isdigit() and int(VECTOR_DIMENSION_STR) > 0 else None
)

# --- Search Configuration ---
# Número por defecto de resultados a devolver en búsquedas
DEFAULT_N_RESULTS: int = int(os.getenv("APP_DEFAULT_N_RESULTS", "12"))

# --- Logging Configuration ---
# Nivel de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_LEVEL: str = os.getenv("APP_LOG_LEVEL", "INFO").upper()

# --- Validation (Optional but Recommended) ---
if MODEL_NAME not in AVAILABLE_MODELS:
    # Usar print o logger aquí podría ser problemático si el logger aún no está configurado
    print(f"ERROR: Model '{MODEL_NAME}' not found in AVAILABLE_MODELS list in config.py. Available: {AVAILABLE_MODELS}")
    MODEL_NAME = AVAILABLE_MODELS[0] # Fallback al primero
    print(f"Warning: Falling back to default model: {MODEL_NAME}")

if LOG_LEVEL not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
     print(f"Warning: Invalid LOG_LEVEL '{LOG_LEVEL}' in config/env. Defaulting to INFO.")
     LOG_LEVEL = "INFO"
if DEVICE not in ["cpu", "cuda"]:
    print(f"Warning: Invalid DEVICE '{DEVICE}' in config/env. Defaulting to 'cpu'.")
    DEVICE = "cpu"
