class VectorImageSearchError(Exception):
    """Clase base para excepciones en esta aplicación."""


class InitializationError(VectorImageSearchError):
    """Error durante la inicialización de componentes (modelo, BD)."""


class DatabaseError(VectorImageSearchError):
    """Error relacionado con operaciones de la base de datos vectorial."""


class VectorizerError(VectorImageSearchError):
    """Error durante el proceso de vectorización (carga de modelo, inferencia)."""


class PipelineError(VectorImageSearchError):
    """Error durante la ejecución de un pipeline (indexación, búsqueda)."""


class ImageProcessingError(VectorImageSearchError):
    """Error durante la carga o procesamiento de imágenes."""
