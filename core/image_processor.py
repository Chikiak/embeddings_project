import logging
import os
from typing import List, Optional, Tuple

from PIL import ExifTags, Image, ImageOps, UnidentifiedImageError

from app.exceptions import ImageProcessingError
from config import IMAGE_EXTENSIONS

logger = logging.getLogger(__name__)

try:
    ORIENTATION_TAG_ID = next(
        k for k, v in ExifTags.TAGS.items() if v == "Orientation"
    )
except StopIteration:
    ORIENTATION_TAG_ID = None
    logger.debug("EXIF Orientation Tag ID not found in Pillow's ExifTags.")


def correct_image_orientation(img: Image.Image) -> Image.Image:
    """
    Corrige la orientación de una imagen PIL basándose en sus datos EXIF.

    Utiliza ImageOps.exif_transpose para un enfoque robusto.

    Args:
        img: El objeto PIL Image de entrada.

    Returns:
        El objeto PIL Image con la orientación corregida. Devuelve la imagen
        original si no se encuentran datos de orientación EXIF o si ocurre un error.
    """
    if ORIENTATION_TAG_ID is None:
        logger.debug(
            "Skipping orientation correction: Orientation tag ID not available."
        )
        return img

    try:
        corrected_img = ImageOps.exif_transpose(img)
        if corrected_img is not img:
            logger.debug("Applied EXIF orientation correction.")
        return corrected_img
    except Exception as e:
        logger.warning(
            f"Could not apply EXIF orientation correction: {e}", exc_info=True
        )
        return img


def find_image_files(directory_path: str) -> List[str]:
    """
    Encuentra recursivamente todos los archivos de imagen en el directorio especificado.

    Args:
        directory_path: La ruta al directorio a buscar.

    Returns:
        Una lista de rutas completas a archivos de imagen que coinciden con las
        extensiones definidas en config.IMAGE_EXTENSIONS (insensible a mayúsculas).
        Devuelve una lista vacía si el directorio no se encuentra o si ocurren errores.

    Raises:
        ValueError: Si directory_path no es una cadena válida.
        FileNotFoundError: Si directory_path no es un directorio válido.
        ImageProcessingError: Si ocurre un error inesperado durante la búsqueda.
    """
    image_files: List[str] = []
    if not isinstance(directory_path, str) or not directory_path:
        raise ValueError(
            "Invalid directory path provided (empty or not a string)."
        )
    if not os.path.isdir(directory_path):
        raise FileNotFoundError(
            f"Directory not found or not accessible: {directory_path}"
        )

    logger.info(
        f"Searching for images with extensions {IMAGE_EXTENSIONS} in: {directory_path}"
    )
    files_scanned = 0
    try:
        for root, _, files in os.walk(directory_path, topdown=True):
            for filename in files:
                files_scanned += 1
                if filename.lower().endswith(IMAGE_EXTENSIONS):
                    full_path = os.path.join(root, filename)
                    try:
                        if os.path.isfile(full_path):
                            image_files.append(full_path)
                            logger.debug(
                                f"  Found potential image file: {full_path}"
                            )
                        else:
                            logger.debug(
                                f"  Skipping non-file entry: {full_path}"
                            )
                    except OSError as stat_error:
                        logger.warning(
                            f"  Could not stat file {full_path}, skipping: {stat_error}"
                        )

    except OSError as e:
        msg = f"OS error while walking directory {directory_path}: {e}"
        logger.error(msg, exc_info=True)
        raise ImageProcessingError(msg) from e
    except Exception as e:
        msg = f"Unexpected error finding image files in {directory_path}: {e}"
        logger.error(msg, exc_info=True)
        raise ImageProcessingError(msg) from e

    if not image_files:
        logger.warning(
            f"No image files with supported extensions found in {directory_path} (Scanned {files_scanned} files)."
        )
    else:
        logger.info(
            f"Found {len(image_files)} potential image files (Scanned {files_scanned} total files)."
        )

    return image_files


def load_image(
    image_path: str, apply_orientation_correction: bool = True
) -> Optional[Image.Image]:
    """
    Carga una imagen desde la ruta de archivo dada usando Pillow.

    Maneja errores comunes, convierte a RGB y opcionalmente corrige la orientación.

    Args:
        image_path: La ruta completa al archivo de imagen.
        apply_orientation_correction: Si es True, intenta corregir la orientación basada en EXIF.

    Returns:
        Un objeto PIL Image en formato RGB, o None si la carga falla.
        Logs de advertencia o error si la carga falla.
    """
    if not isinstance(image_path, str) or not image_path:
        logger.warning(
            "Invalid image path provided to load_image (empty or not a string)."
        )
        return None

    if not os.path.isfile(image_path):
        logger.warning(
            f"Image file does not exist or is not a file: {image_path}"
        )
        return None

    try:
        img = Image.open(image_path)

        if apply_orientation_correction:
            img = correct_image_orientation(img)

        if img.mode != "RGB":
            logger.debug(
                f"Converting image '{os.path.basename(image_path)}' from mode {img.mode} to RGB."
            )
            if img.mode == "RGBA" or (
                img.mode == "P" and "transparency" in img.info
            ):
                try:
                    background = Image.new("RGB", img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[-1])
                    img = background
                    logger.debug(
                        f"  Successfully converted transparent image {os.path.basename(image_path)} to RGB with white background."
                    )
                except Exception as conversion_err:
                    logger.warning(
                        f"  Could not handle transparency for {os.path.basename(image_path)} during RGB conversion, using direct convert: {conversion_err}"
                    )
                    img = img.convert("RGB")
            else:
                img = img.convert("RGB")

        return img

    except FileNotFoundError:
        logger.error(
            f"FileNotFoundError for image path (should have been caught earlier): {image_path}"
        )
        return None
    except UnidentifiedImageError:
        logger.warning(
            f"Cannot identify image file (possibly corrupt or unsupported format): {image_path}"
        )
        return None
    except (IOError, OSError) as e:
        logger.warning(f"I/O or OS error loading image {image_path}: {e}")
        return None
    except Exception as e:
        logger.error(
            f"Unexpected error loading or processing image {image_path}: {e}",
            exc_info=True,
        )
        return None


def batch_load_images(
    image_paths: List[str],
) -> Tuple[List[Image.Image], List[str]]:
    """
    Carga un lote de imágenes desde una lista de rutas, omitiendo las que fallan al cargar.

    Args:
        image_paths: Una lista de rutas a archivos de imagen.

    Returns:
        Una tupla que contiene:
        - Una lista de objetos PIL Image cargados con éxito (RGB, orientación corregida).
        - Una lista de las rutas correspondientes para las imágenes cargadas con éxito.

    Raises:
        TypeError: Si image_paths no es una lista.
    """
    loaded_images: List[Image.Image] = []
    corresponding_paths: List[str] = []

    if not isinstance(image_paths, list):
        raise TypeError(
            "batch_load_images expects a list of paths, received non-list input."
        )
    if not image_paths:
        logger.debug("batch_load_images received an empty list of paths.")
        return [], []

    num_requested = len(image_paths)
    logger.info(f"Attempting to load a batch of {num_requested} images...")

    for path in image_paths:
        img = load_image(path)
        if img is not None:
            loaded_images.append(img)
            corresponding_paths.append(path)

    num_loaded = len(loaded_images)
    num_failed = num_requested - num_loaded

    if num_failed == 0:
        logger.info(
            f"Batch loading finished. Successfully loaded all {num_loaded}/{num_requested} images."
        )
    else:
        logger.warning(
            f"Batch loading finished. Successfully loaded: {num_loaded}/{num_requested}. Failed: {num_failed}. Check previous logs for specific file errors."
        )

    return loaded_images, corresponding_paths
