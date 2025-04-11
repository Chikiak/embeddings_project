import os
from typing import Tuple, Optional

# Opcional: para cargar desde archivo .env
from dotenv import load_dotenv
load_dotenv() # Carga variables de un archivo .env si existe

# --- Model Configuration ---
# Nombre del modelo CLIP a usar desde Hugging Face Hub.
MODEL_NAME: str = os.getenv('APP_MODEL_NAME', 'jinaai/jina-clip-v2')

# Dispositivo para la inferencia: 'cuda' si hay GPU compatible y PyTorch con soporte CUDA, sino 'cpu'.
DEVICE: str = os.getenv('APP_DEVICE', 'cpu')

# Permite ejecutar código personalizado definido en el repositorio del modelo.
# Necesario para algunos modelos (como Jina CLIP v2), pero implica un riesgo de seguridad.
# Establecer a False si usas un modelo que no lo requiere o por políticas de seguridad estrictas.
TRUST_REMOTE_CODE: bool = os.getenv('APP_TRUST_REMOTE_CODE', 'True').lower() == 'true'

# --- Image Processing Configuration ---
# Extensiones de archivo de imagen aceptadas (insensible a mayúsculas/minúsculas).
# Leído desde variable de entorno o usa valor por defecto.
IMAGE_EXTENSIONS_STR: str = os.getenv('APP_IMAGE_EXTENSIONS', ".jpg,.jpeg,.png,.webp,.bmp,.gif")
IMAGE_EXTENSIONS: Tuple[str, ...] = tuple(ext.strip().lower() for ext in IMAGE_EXTENSIONS_STR.split(',')) # Normalizado a minúsculas

# Tamaño del lote para procesar imágenes durante la vectorización. Ajustar según memoria disponible.
BATCH_SIZE_IMAGES: int = int(os.getenv('APP_BATCH_SIZE_IMAGES', '8'))

# --- Text Processing Configuration ---
# Tamaño del lote para procesar consultas de texto durante la vectorización.
BATCH_SIZE_TEXT: int = int(os.getenv('APP_BATCH_SIZE_TEXT', '16'))

# --- Vector Database Configuration ---
# Ruta del directorio para almacenar los archivos persistentes de la base de datos ChromaDB.
CHROMA_DB_PATH: str = os.getenv('APP_CHROMA_DB_PATH', "vector_db_store/")

# Nombre de la colección dentro de ChromaDB para almacenar los embeddings de imágenes.
CHROMA_COLLECTION_NAME: str = os.getenv('APP_CHROMA_COLLECTION_NAME', "image_embeddings")

# Dimensión a la que se truncarán los vectores de embedding antes de guardarlos.
# Puede reducir el uso de memoria y acelerar búsquedas a costa de posible pérdida de precisión.
# Establecer a None para usar la dimensión completa original del modelo (Jina v2 es 768).
# ¡IMPORTANTE! Cambiar esto requiere re-indexar todas las imágenes.
VECTOR_DIMENSION_STR: str = os.getenv('APP_VECTOR_DIMENSION', '64') # Leer como string
VECTOR_DIMENSION: Optional[int] = int(VECTOR_DIMENSION_STR) if VECTOR_DIMENSION_STR.isdigit() else None # Convertir a int o None

# --- Clustering Configuration ---
# Número predeterminado de clústeres para el algoritmo KMeans.
DEFAULT_N_CLUSTERS: int = int(os.getenv('APP_DEFAULT_N_CLUSTERS', '5'))

# --- Search Configuration ---
# Número predeterminado de resultados similares a recuperar en las búsquedas.
DEFAULT_N_RESULTS: int = int(os.getenv('APP_DEFAULT_N_RESULTS', '5'))

# --- FIN DE CONFIGURACIÓN ---