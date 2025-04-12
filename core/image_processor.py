# core/image_processor.py
import os
from PIL import Image, UnidentifiedImageError, ExifTags, ImageOps
from typing import List, Optional, Tuple
import logging

# Importar configuración
from config import IMAGE_EXTENSIONS

logger = logging.getLogger(__name__)

# --- EXIF Orientation Helper ---
# Cache the Orientation TAG ID
try:
    ORIENTATION_TAG_ID = next(k for k, v in ExifTags.TAGS.items() if v == "Orientation")
except StopIteration:
    ORIENTATION_TAG_ID = None
    logger.debug("EXIF Orientation Tag ID not found in Pillow's ExifTags.")


def correct_image_orientation(img: Image.Image) -> Image.Image:
    """
    Corrects the orientation of a PIL image based on its EXIF data.
    Uses ImageOps.exif_transpose for a more robust approach.

    Args:
        img: The input PIL Image object.

    Returns:
        The orientation-corrected PIL Image object. Returns the original image
        if no EXIF orientation data is found or an error occurs.
    """
    # Check if the necessary tag ID was found
    if ORIENTATION_TAG_ID is None:
        logger.debug("Skipping orientation correction: Orientation tag ID not available.")
        return img

    try:
        # Use Pillow's built-in function for robust handling
        # This function handles various orientation values correctly.
        corrected_img = ImageOps.exif_transpose(img)
        # Check if the image object actually changed (i.e., if transpose was applied)
        if corrected_img is not img:
            logger.debug("Applied EXIF orientation correction.")
        return corrected_img
    except Exception as e:
        # Catch potential errors during EXIF processing or transposition
        logger.warning(f"Could not apply EXIF orientation correction: {e}", exc_info=True)
        return img  # Return original image on error


# --- Main Functions ---

def find_image_files(directory_path: str) -> List[str]:
    """
    Recursively finds all image files in the specified directory.

    Args:
        directory_path: The path to the directory to search.

    Returns:
        A list of full paths to image files matching extensions defined in
        config.IMAGE_EXTENSIONS (case-insensitive). Returns an empty list
        if the directory is not found or errors occur.
    """
    image_files: List[str] = []
    if not isinstance(directory_path, str) or not directory_path:
        logger.error("Invalid directory path provided (empty or not a string).")
        return image_files
    if not os.path.isdir(directory_path):
        logger.error(f"Directory not found or not accessible: {directory_path}")
        return image_files

    logger.info(f"Searching for images with extensions {IMAGE_EXTENSIONS} in: {directory_path}")
    files_scanned = 0
    try:
        for root, _, files in os.walk(directory_path, topdown=True):  # Use topdown=True (default)
            for filename in files:
                files_scanned += 1
                # Check if the file extension (lowercase) is in the allowed tuple
                if filename.lower().endswith(IMAGE_EXTENSIONS):
                    full_path = os.path.join(root, filename)
                    # Final check: Ensure it's actually a file (not a broken link, etc.)
                    # Use try-except for os.path.isfile in case of permission errors during check
                    try:
                        if os.path.isfile(full_path):
                            image_files.append(full_path)
                            logger.debug(f"  Found potential image file: {full_path}")
                        else:
                            logger.debug(f"  Skipping non-file entry: {full_path}")
                    except OSError as stat_error:
                        logger.warning(f"  Could not stat file {full_path}, skipping: {stat_error}")

    except OSError as e:
        logger.error(f"OS error while walking directory {directory_path}: {e}", exc_info=True)
        return []  # Return empty list on error
    except Exception as e:
        logger.error(f"Unexpected error finding image files in {directory_path}: {e}", exc_info=True)
        return []

    if not image_files:
        logger.warning(
            f"No image files with supported extensions found in {directory_path} (Scanned {files_scanned} files).")
    else:
        logger.info(f"Found {len(image_files)} potential image files (Scanned {files_scanned} total files).")

    return image_files


def load_image(
        image_path: str, apply_orientation_correction: bool = True
) -> Optional[Image.Image]:
    """
    Loads an image from the given file path using Pillow. Handles common errors,
    converts to RGB, and optionally corrects orientation.

    Args:
        image_path: The full path to the image file.
        apply_orientation_correction: If True, attempts to correct orientation based on EXIF.

    Returns:
        A PIL Image object in RGB format, or None if loading fails.
    """
    if not isinstance(image_path, str) or not image_path:
        logger.warning("Invalid image path provided to load_image (empty or not a string).")
        return None

    # Check if file exists *before* trying to open, providing a clearer error message
    if not os.path.isfile(image_path):
        logger.warning(f"Image file does not exist or is not a file: {image_path}")
        return None

    try:
        # Open the image
        img = Image.open(image_path)

        # 1. Apply orientation correction (optional)
        if apply_orientation_correction:
            img = correct_image_orientation(img)  # Uses the improved helper

        # 2. Convert to RGB if necessary (common requirement for models)
        if img.mode != "RGB":
            # Log the conversion attempt
            logger.debug(f"Converting image '{os.path.basename(image_path)}' from mode {img.mode} to RGB.")
            # Use ImageOps.pad to handle transparency with a white background if needed
            # This is generally safer than manual pasting
            if img.mode == 'RGBA' or (img.mode == 'P' and 'transparency' in img.info):
                try:
                    # Create an RGB image with white background
                    background = Image.new("RGB", img.size, (255, 255, 255))
                    # Paste the image onto the background using its alpha channel or transparency info
                    background.paste(img, mask=img.split()[-1])  # Assumes alpha is the last channel
                    img = background
                    logger.debug(
                        f"  Successfully converted transparent image {os.path.basename(image_path)} to RGB with white background.")
                except Exception as conversion_err:
                    logger.warning(
                        f"  Could not handle transparency for {os.path.basename(image_path)} during RGB conversion, using direct convert: {conversion_err}")
                    # Fallback to direct conversion if pasting fails
                    img = img.convert("RGB")
            else:
                # For other modes (L, CMYK, etc.), direct conversion is usually sufficient
                img = img.convert("RGB")

        # If we got here, loading and conversion were successful
        # logger.debug(f"Successfully loaded and processed image: {image_path}")
        return img

    # --- Error Handling ---
    except FileNotFoundError:
        # Should be caught by os.path.isfile, but include for robustness
        logger.error(f"FileNotFoundError for image path (should have been caught earlier): {image_path}")
        return None
    except UnidentifiedImageError:
        # Pillow couldn't identify the image format (corrupt or unsupported)
        logger.warning(f"Cannot identify image file (possibly corrupt or unsupported format): {image_path}")
        return None
    except (IOError, OSError) as e:
        # Catch file read errors, permission issues, etc.
        logger.warning(f"I/O or OS error loading image {image_path}: {e}")
        return None
    except Exception as e:
        # Catch any other unexpected errors during loading or processing
        logger.error(f"Unexpected error loading or processing image {image_path}: {e}", exc_info=True)
        return None


def batch_load_images(image_paths: List[str]) -> Tuple[List[Image.Image], List[str]]:
    """
    Loads a batch of images from a list of paths, skipping those that fail to load.

    Args:
        image_paths: A list of paths to image files.

    Returns:
        A tuple containing:
        - A list of successfully loaded PIL Image objects (RGB, orientation corrected).
        - A list of the corresponding paths for the successfully loaded images.
    """
    loaded_images: List[Image.Image] = []
    corresponding_paths: List[str] = []

    if not isinstance(image_paths, list):
        logger.error("batch_load_images expects a list of paths, received non-list input.")
        return [], []
    if not image_paths:
        logger.debug("batch_load_images received an empty list of paths.")
        return [], []

    num_requested = len(image_paths)
    logger.info(f"Attempting to load a batch of {num_requested} images...")

    # Iterate and attempt to load each image
    for path in image_paths:
        # load_image handles its own logging for failures
        img = load_image(path)
        if img is not None:
            # Append successful loads and their paths
            loaded_images.append(img)
            corresponding_paths.append(path)
        # No else needed, load_image already logged the failure

    num_loaded = len(loaded_images)
    num_failed = num_requested - num_loaded

    if num_failed == 0:
        logger.info(f"Batch loading finished. Successfully loaded all {num_loaded}/{num_requested} images.")
    else:
        logger.warning(
            f"Batch loading finished. Successfully loaded: {num_loaded}/{num_requested}. Failed: {num_failed}. Check previous logs for specific file errors.")

    return loaded_images, corresponding_paths
