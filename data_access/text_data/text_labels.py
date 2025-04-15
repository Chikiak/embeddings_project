# app/text_labels.py
import os
import logging
from typing import List

logger = logging.getLogger(__name__)

# Nombre del archivo que contendrá las etiquetas predefinidas
# Este archivo debe estar en la misma carpeta que text_labels.py (es decir, en 'app/')
_PREDEFINED_LABELS_FILE = "predefined_labels.txt"

def load_predefined_labels(file_path: str = _PREDEFINED_LABELS_FILE) -> List[str]:
    """
    Carga etiquetas textuales desde un archivo de texto simple.
    Espera una etiqueta por línea. Líneas vacías o con solo espacios se ignoran.

    Args:
        file_path (str): Nombre del archivo (relativo a este script) de donde cargar las etiquetas.

    Returns:
        List[str]: Una lista de las etiquetas cargadas desde el archivo.
                   Devuelve una lista vacía si el archivo no existe o hay un error.
    """
    labels = []
    try:
        # Construye la ruta completa al archivo, asumiendo que está en el mismo directorio que este script
        script_dir = os.path.dirname(__file__)
        full_path = os.path.join(script_dir, file_path)

        if os.path.exists(full_path):
            with open(full_path, 'r', encoding='utf-8') as f:
                # Lee cada línea, quita espacios al inicio/final, y añade si no está vacía
                labels = [line.strip() for line in f if line.strip()]
            if labels:
                logger.info(f"Se cargaron {len(labels)} etiquetas desde '{full_path}'. Ejemplo: '{labels[0]}'")
            else:
                logger.warning(f"El archivo de etiquetas '{full_path}' existe pero está vacío o no contiene etiquetas válidas.")
        else:
             # Si el archivo no existe, loguea una advertencia
             logger.warning(f"Archivo de etiquetas predefinidas no encontrado: '{full_path}'. "
                            f"Se usará una lista vacía. Por favor, crea este archivo y añade etiquetas descriptivas.")
             # Puedes añadir unas pocas etiquetas por defecto aquí si lo deseas como fallback:
             # labels = ["paisaje", "persona", "animal", "objeto", "abstracto"]
             # logger.info("Usando lista de etiquetas por defecto como fallback.")

    except Exception as e:
        # Loguea cualquier otro error que ocurra durante la carga
        logger.error(f"Error al cargar etiquetas predefinidas desde '{file_path}': {e}", exc_info=True)

    return labels

# Carga las etiquetas cuando se importa este módulo.
# Esta lista será utilizada por el servicio de clustering.
PREDEFINED_LABELS: List[str] = load_predefined_labels()