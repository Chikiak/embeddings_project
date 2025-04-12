# data_access/vector_db_interface.py
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any, Tuple
import numpy as np

from app.models import SearchResults


class VectorDBInterface(ABC):
    """
    Clase Base Abstracta que define la interfaz para operaciones de base de datos vectorial.

    Permite intercambiar diferentes implementaciones de bases de datos vectoriales
    (ej: ChromaDB, FAISS, Qdrant) sin cambiar la lógica central de la aplicación
    que depende de esta interfaz.
    """

    @abstractmethod
    def add_embeddings(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """
        Añade o actualiza embeddings en la base de datos.

        Args:
            ids: Una lista de identificadores únicos para cada embedding.
            embeddings: Una lista de embeddings (listas de floats).
            metadatas: Una lista opcional de diccionarios de metadatos
                       correspondientes a cada embedding.

        Returns:
            True si la operación fue exitosa, False en caso contrario.

        Raises:
            DatabaseError: Si ocurre un error durante la operación.
        """
        pass

    @abstractmethod
    def query_similar(
        self, query_embedding: List[float], n_results: int
    ) -> Optional[SearchResults]:
        """
        Consulta la base de datos por embeddings similares al embedding de consulta.

        Args:
            query_embedding: El vector de embedding a buscar.
            n_results: El número máximo de resultados similares a devolver.

        Returns:
            Un objeto SearchResults que contiene los elementos encontrados,
            o None si la consulta falló. Devuelve un SearchResults vacío si la
            consulta fue exitosa pero no encontró coincidencias.

        Raises:
            DatabaseError: Si ocurre un error durante la consulta.
        """
        pass

    @abstractmethod
    def get_all_embeddings_with_ids(
        self, pagination_batch_size: int = 1000
    ) -> Optional[Tuple[List[str], np.ndarray]]:
        """
        Recupera todos los embeddings y sus IDs correspondientes de la base de datos.

        Utiliza paginación para eficiencia de memoria con grandes conjuntos de datos.

        Args:
            pagination_batch_size: El número de elementos a recuperar en cada
                                   lote durante la paginación.

        Returns:
            Una tupla que contiene:
            - Una lista de todos los IDs.
            - Un array NumPy de todos los embeddings.
            Devuelve None si la recuperación falla. Devuelve ([], np.empty(0, ...))
            si la colección está vacía.

        Raises:
            DatabaseError: Si ocurre un error durante la recuperación.
        """
        pass

    @abstractmethod
    def clear_collection(self) -> bool:
        """
        Elimina todos los elementos de la colección actual.

        Returns:
            True si la colección se limpió con éxito, False en caso contrario.

        Raises:
            DatabaseError: Si ocurre un error durante la limpieza.
        """
        pass

    @abstractmethod
    def delete_collection(self) -> bool:
        """
        Elimina toda la colección de la base de datos.

        Returns:
            True si la colección se eliminó con éxito, False en caso contrario.

        Raises:
            DatabaseError: Si ocurre un error durante la eliminación.
        """
        pass

    @abstractmethod
    def count(self) -> int:
        """
        Devuelve el número total de elementos actualmente en la colección.

        Returns:
            El número de elementos, o -1 si ocurre un error durante el conteo.
        """
        pass

    @property
    @abstractmethod
    def is_initialized(self) -> bool:
        """
        Comprueba si la conexión a la base de datos y la colección están
        correctamente inicializadas.

        Returns:
            True si está inicializado, False en caso contrario.
        """
        pass
