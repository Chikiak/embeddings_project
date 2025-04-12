# data_access/vector_db_interface.py
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any, Tuple
import numpy as np

# Importar modelos de datos
from app.models import SearchResults  # Usar SearchResults directamente


class VectorDBInterface(ABC):
    """
    Abstract Base Class defining the interface for vector database operations.
    This allows swapping different vector database implementations (e.g., ChromaDB, FAISS, Qdrant)
    without changing the core application logic that depends on this interface.
    """

    @abstractmethod
    def add_embeddings(
            self,
            ids: List[str],
            embeddings: List[List[float]],
            metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """
        Adds or updates embeddings in the database.

        Args:
            ids: A list of unique identifiers for each embedding.
            embeddings: A list of embeddings (lists of floats).
            metadatas: An optional list of metadata dictionaries corresponding to each embedding.

        Returns:
            True if the operation was successful, False otherwise.
        """
        pass

    @abstractmethod
    def query_similar(
            self, query_embedding: List[float], n_results: int
    ) -> Optional[SearchResults]:
        """
        Queries the database for embeddings similar to the query embedding.

        Args:
            query_embedding: The embedding vector to search for.
            n_results: The maximum number of similar results to return.

        Returns:
            A SearchResults object containing the found items, or None if the query failed.
            Returns an empty SearchResults object if the query was successful but found no matches.
        """
        pass

    @abstractmethod
    def get_all_embeddings_with_ids(
            self, pagination_batch_size: int = 1000
    ) -> Optional[Tuple[List[str], np.ndarray]]:
        """
        Retrieves all embeddings and their corresponding IDs from the database.
        Uses pagination for memory efficiency with large datasets.

        Args:
            pagination_batch_size: The number of items to retrieve in each batch during pagination.

        Returns:
            A tuple containing:
            - A list of all IDs.
            - A NumPy array of all embeddings.
            Returns None if retrieval fails. Returns ([], np.empty(0, ...)) if the collection is empty.
        """
        pass

    @abstractmethod
    def clear_collection(self) -> bool:
        """
        Removes all items from the current collection.

        Returns:
            True if the collection was successfully cleared, False otherwise.
        """
        pass

    @abstractmethod
    def delete_collection(self) -> bool:
        """
        Deletes the entire collection from the database.

        Returns:
            True if the collection was successfully deleted, False otherwise.
        """
        pass

    @abstractmethod
    def count(self) -> int:
        """
        Returns the total number of items currently in the collection.

        Returns:
            The number of items, or -1 if an error occurs during counting.
        """
        pass

    @property
    @abstractmethod
    def is_initialized(self) -> bool:
        """
        Checks if the database connection and collection are properly initialized.

        Returns:
            True if initialized, False otherwise.
        """
        pass
