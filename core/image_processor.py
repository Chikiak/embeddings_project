import os
from PIL import Image, UnidentifiedImageError
from typing import List, Optional, Tuple
import logging

# Import configuration constants
from config import IMAGE_EXTENSIONS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def find_image_files(directory_path: str) -> List[str]:
    """
    Recursively finds all image files in the specified directory.

    Args:
        directory_path: The path to the directory to search.

    Returns:
        A list of full paths to image files matching the extensions
        defined in config.IMAGE_EXTENSIONS. Returns empty list if dir not found.
    """
    image_files = []
    if not os.path.isdir(directory_path):
        logging.error(f"Directory not found at {directory_path}")
        return image_files # Devuelve lista vacía si el directorio no existe

    logging.info(f"Searching for images in: {directory_path}")
    try:
        for root, _, files in os.walk(directory_path):
            for filename in files:
                # Comparación insensible a mayúsculas/minúsculas
                if filename.lower().endswith(IMAGE_EXTENSIONS):
                    full_path = os.path.join(root, filename)
                    image_files.append(full_path)
                    # logging.debug(f"  Found: {full_path}") # Usar debug para verbosidad
    except OSError as e:
         logging.error(f"Error walking directory {directory_path}: {e}")
         return []

    logging.info(f"Found {len(image_files)} potential image files.")
    return image_files


def load_image(image_path: str) -> Optional[Image.Image]:
    """
    Loads an image from the given file path using Pillow.

    Args:
        image_path: The full path to the image file.

    Returns:
        A PIL Image object in RGB format, or None if loading fails.
    """
    if not isinstance(image_path, str) or not image_path:
         logging.warning("Invalid image path provided (empty or not a string).")
         return None

    try:
        img = Image.open(image_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    except FileNotFoundError:
        # Este es un error esperado si la ruta es incorrecta
        logging.warning(f"Image file not found: {image_path}")
        return None
    except UnidentifiedImageError:
        # Indica archivo corrupto o no es una imagen válida
        logging.warning(f"Cannot identify image file (possibly corrupt or invalid format): {image_path}")
        return None
    except (IOError, OSError) as e: # Captura errores de lectura/sistema de archivos
         logging.warning(f"I/O or OS error loading image {image_path}: {e}")
         return None
    except Exception as e: # Captura para otros errores inesperados
        logging.error(f"Unexpected error loading image {image_path}: {e}", exc_info=True)
        return None


def batch_load_images(image_paths: List[str]) -> Tuple[List[Image.Image], List[str]]:
    """
    Loads a batch of images from a list of paths, skipping failed ones.

    Args:
        image_paths: A list of image file paths.

    Returns:
        A tuple containing:
        - A list of successfully loaded PIL Image objects (in RGB).
        - A list of the corresponding paths for the successfully loaded images.
    """
    loaded_images = []
    corresponding_paths = []
    if not isinstance(image_paths, list):
         logging.error("batch_load_images expects a list of paths.")
         return [], []

    for path in image_paths:
        img = load_image(path)
        if img:
            loaded_images.append(img)
            corresponding_paths.append(path)

    return loaded_images, corresponding_paths