class VectorImageSearchError(Exception):
    """Clase base para excepciones en esta aplicación."""

    pass


class InitializationError(VectorImageSearchError):
    """Error durante la inicialización de componentes (modelo, BD)."""

    pass


class DatabaseError(VectorImageSearchError):
    """Error relacionado con operaciones de la base de datos vectorial."""

    pass


class VectorizerError(VectorImageSearchError):
    """Error durante el proceso de vectorización (carga de modelo, inferencia)."""

    pass


class PipelineError(VectorImageSearchError):
    """Error durante la ejecución de un pipeline (indexación, búsqueda)."""

    pass


class ImageProcessingError(VectorImageSearchError):
    """Error durante la carga o procesamiento de imágenes."""

    pass
