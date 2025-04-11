import os
from PIL import Image, UnidentifiedImageError, ExifTags
from typing import List, Optional, Tuple
import logging  # Importar logging

# Importar configuración
from config import IMAGE_EXTENSIONS

# Obtener un logger para este módulo (la configuración se hace en main.py/streamlit_app.py)
logger = logging.getLogger(__name__)


# --- Helper para corregir orientación EXIF ---
# Buscar el tag de Orientación en los datos EXIF
def get_orientation_tag_id():
    for k, v in ExifTags.TAGS.items():
        if v == "Orientation":
            return k
    return None


ORIENTATION_TAG_ID = get_orientation_tag_id()


def correct_image_orientation(img: Image.Image) -> Image.Image:
    """
    Corrige la orientación de una imagen PIL basada en sus datos EXIF.
    """
    try:
        exif = img._getexif()  # Obtener datos EXIF (puede ser None)
        if exif is None or ORIENTATION_TAG_ID is None:
            return img  # No hay EXIF o tag de orientación, devolver original

        orientation = exif.get(ORIENTATION_TAG_ID)

        # Rotar/voltear según el valor de orientación EXIF
        if orientation == 2:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 3:
            img = img.rotate(180)
        elif orientation == 4:
            img = img.rotate(180).transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 5:
            img = img.rotate(-90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 6:
            img = img.rotate(-90, expand=True)
        elif orientation == 7:
            img = img.rotate(90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 8:
            img = img.rotate(90, expand=True)

        # Después de aplicar la transformación, eliminar el tag de orientación
        # para evitar doble corrección si se guarda y se vuelve a cargar.
        # Esto requiere modificar el diccionario exif si existe.
        # Nota: Modificar exif directamente puede ser complejo.
        # Una alternativa es simplemente devolver la imagen corregida sin modificar exif.
        # if ORIENTATION_TAG_ID in exif:
        #     del exif[ORIENTATION_TAG_ID] # Esto puede no funcionar directamente con el objeto exif de PIL

    except (AttributeError, KeyError, IndexError, TypeError):
        # Ignorar errores al leer/procesar EXIF, devolver imagen tal cual
        # logger.debug(f"Could not process EXIF orientation for image.", exc_info=True) # Loguear en debug si es necesario
        pass
    except Exception as e:
        logger.warning(
            f"Unexpected error during EXIF orientation correction: {e}", exc_info=True
        )

    return img


# --- Funciones Principales ---


def find_image_files(directory_path: str) -> List[str]:
    """
    Encuentra recursivamente todos los archivos de imagen en el directorio especificado.

    Args:
        directory_path: La ruta al directorio a buscar.

    Returns:
        Una lista de rutas completas a archivos de imagen que coinciden con las
        extensiones definidas en config.IMAGE_EXTENSIONS (insensible a mayúsculas/minúsculas).
        Devuelve una lista vacía si el directorio no se encuentra o hay errores.
    """
    image_files: List[str] = []
    # Validar que la ruta sea un directorio existente
    if not os.path.isdir(directory_path):
        logger.error(f"Directory not found or not accessible: {directory_path}")
        return image_files  # Devuelve lista vacía

    logger.info(
        f"Searching for images with extensions {IMAGE_EXTENSIONS} in: {directory_path}"
    )
    try:
        # os.walk recorre el árbol de directorios
        for root, _, files in os.walk(directory_path):
            for filename in files:
                # Comprobar si la extensión (en minúsculas) está en las permitidas
                if filename.lower().endswith(IMAGE_EXTENSIONS):
                    full_path = os.path.join(root, filename)
                    # Verificar si es realmente un archivo (y no un enlace roto, etc.)
                    if os.path.isfile(full_path):
                        image_files.append(full_path)
                        logger.debug(f"  Found potential image file: {full_path}")
                    else:
                        logger.debug(f"  Skipping non-file entry: {full_path}")

    except OSError as e:
        # Error al acceder a directorios/archivos durante el recorrido
        logger.error(
            f"OS error while walking directory {directory_path}: {e}", exc_info=True
        )
        return []  # Devolver lista vacía en caso de error
    except Exception as e:
        # Capturar otros errores inesperados
        logger.error(
            f"Unexpected error finding image files in {directory_path}: {e}",
            exc_info=True,
        )
        return []

    if not image_files:
        logger.warning(
            f"No image files with supported extensions found in {directory_path}"
        )
    else:
        logger.info(f"Found {len(image_files)} potential image files.")

    return image_files


def load_image(
    image_path: str, apply_orientation_correction: bool = True
) -> Optional[Image.Image]:
    """
    Carga una imagen desde la ruta de archivo dada usando Pillow.

    Args:
        image_path: La ruta completa al archivo de imagen.
        apply_orientation_correction: Si es True, intenta corregir la orientación
                                       basándose en los metadatos EXIF.

    Returns:
        Un objeto PIL Image en formato RGB, o None si la carga falla.
    """
    # Validar entrada
    if not isinstance(image_path, str) or not image_path:
        logger.warning("Invalid image path provided (empty or not a string).")
        return None
    if not os.path.isfile(image_path):
        logger.warning(f"Image file does not exist or is not a file: {image_path}")
        return None

    try:
        # Abrir la imagen con Pillow
        img = Image.open(image_path)

        # 1. Corregir orientación (opcional)
        if apply_orientation_correction:
            img = correct_image_orientation(img)

        # 2. Convertir a RGB si es necesario (formatos como RGBA, P, L)
        # La mayoría de modelos CLIP esperan RGB
        if img.mode != "RGB":
            logger.debug(
                f"Converting image {os.path.basename(image_path)} from mode {img.mode} to RGB."
            )
            # Usar un fondo blanco para la transparencia si se convierte desde RGBA o P
            if img.mode in ["RGBA", "P"]:
                # Crear una imagen de fondo blanco del mismo tamaño
                background = Image.new("RGB", img.size, (255, 255, 255))
                # Pegar la imagen original sobre el fondo blanco usando su máscara alfa si existe
                try:
                    # img.split()[3] es la máscara alfa para RGBA
                    mask = img.split()[3] if img.mode == "RGBA" else None
                    # Para modo 'P' (paleta), convertir primero a RGBA puede ayudar
                    if img.mode == "P":
                        img = img.convert("RGBA")
                        mask = img.split()[3]
                    background.paste(img, mask=mask)
                    img = background
                except IndexError:  # Si no hay canal alfa
                    img = img.convert("RGB")
                except Exception as paste_err:  # Otros errores al pegar
                    logger.warning(
                        f"Could not properly handle transparency for {os.path.basename(image_path)}, converting directly to RGB: {paste_err}"
                    )
                    img = img.convert("RGB")

            else:  # Para otros modos (ej. L, CMYK), la conversión directa suele ser suficiente
                img = img.convert("RGB")

        # logger.debug(f"Successfully loaded and processed image: {image_path}")
        return img

    # Manejar errores específicos de Pillow y del sistema de archivos
    except FileNotFoundError:
        # Esto no debería ocurrir debido a la verificación os.path.isfile, pero por si acaso
        logger.error(f"Image file disappeared after check?: {image_path}")
        return None
    except UnidentifiedImageError:
        # Indica archivo corrupto o formato no soportado por Pillow
        logger.warning(
            f"Cannot identify image file (possibly corrupt or invalid format): {image_path}"
        )
        return None
    except (IOError, OSError) as e:
        # Captura errores de lectura/sistema de archivos (permisos, disco lleno, etc.)
        logger.warning(f"I/O or OS error loading image {image_path}: {e}")
        return None
    except Exception as e:
        # Captura para otros errores inesperados durante la carga/conversión
        logger.error(
            f"Unexpected error loading or processing image {image_path}: {e}",
            exc_info=True,
        )
        return None


def batch_load_images(image_paths: List[str]) -> Tuple[List[Image.Image], List[str]]:
    """
    Carga un lote de imágenes desde una lista de rutas, omitiendo las que fallan al cargar.

    Args:
        image_paths: Una lista de rutas a archivos de imagen.

    Returns:
        Una tupla que contiene:
        - Una lista de objetos PIL Image cargados exitosamente (en RGB y con orientación corregida).
        - Una lista de las rutas correspondientes a las imágenes cargadas exitosamente.
    """
    loaded_images: List[Image.Image] = []
    corresponding_paths: List[str] = []

    # Validar entrada
    if not isinstance(image_paths, list):
        logger.error(
            "batch_load_images expects a list of paths, received non-list input."
        )
        return [], []
    if not image_paths:
        logger.debug("batch_load_images received an empty list of paths.")
        return [], []

    num_requested = len(image_paths)
    logger.info(f"Attempting to load a batch of {num_requested} images...")

    # Iterar sobre cada ruta e intentar cargar la imagen
    for path in image_paths:
        img = load_image(
            path
        )  # load_image ya maneja errores y loguea advertencias/errores
        if img:
            # Si la carga fue exitosa, añadir la imagen y su ruta a las listas
            loaded_images.append(img)
            corresponding_paths.append(path)
        # Si img es None, load_image ya registró el problema

    num_loaded = len(loaded_images)
    num_failed = num_requested - num_loaded
    logger.info(
        f"Batch loading finished. Successfully loaded: {num_loaded}/{num_requested}. Failed: {num_failed}"
    )
    if num_failed > 0:
        logger.warning(
            f"{num_failed} images could not be loaded in this batch. Check previous logs for specific file errors."
        )

    # Devolver las listas de imágenes cargadas y sus rutas
    return loaded_images, corresponding_paths
