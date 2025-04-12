import logging
import sys

import config


def setup_logging():
    """
    Configura el logging básico para la aplicación.

    Utiliza el nivel de log definido en la configuración (config.LOG_LEVEL)
    y dirige la salida a stdout.
    """
    log_level = getattr(logging, config.LOG_LEVEL, logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.info(
        f"Logging configurado a nivel: {logging.getLevelName(log_level)}"
    )
