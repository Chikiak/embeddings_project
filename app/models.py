from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class SearchResultItem:
    """
    Representa un único elemento encontrado durante una búsqueda de similitud.

    Attributes:
        id: Identificador único, típicamente la ruta del archivo de imagen.
        distance: Puntuación de distancia desde la consulta (menor es más similar
                  para distancia coseno). None si no está disponible.
        metadata: Diccionario con metadatos asociados al elemento.
    """

    id: str
    distance: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def similarity(self) -> Optional[float]:
        """
        Calcula la similitud (asumiendo que la distancia es distancia coseno).

        Similarity = 1 - distance. Devuelve None si la distancia es None.
        El valor se limita entre 0.0 y 1.0.

        Returns:
            La puntuación de similitud o None.
        """
        if self.distance is None:
            return None
        return max(0.0, 1.0 - self.distance)


@dataclass(frozen=True)
class SearchResults:
    """
    Representa los resultados completos de una consulta de búsqueda de similitud.

    Attributes:
        items: Lista de objetos SearchResultItem encontrados.
        query_vector: Opcional, el vector utilizado para la consulta.
    """

    items: List[SearchResultItem] = field(default_factory=list)
    query_vector: Optional[List[float]] = None

    @property
    def count(self) -> int:
        """Devuelve el número de elementos en los resultados."""
        return len(self.items)

    @property
    def is_empty(self) -> bool:
        """Comprueba si el conjunto de resultados no contiene elementos."""
        return not self.items
