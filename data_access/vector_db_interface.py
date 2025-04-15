from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from core.models import SearchResults
except ImportError:

    class SearchResults:
        def __init__(self, items=None, query_vector=None):
            self.items = items if items is not None else []
            self.query_vector = query_vector

        @property
        def count(self):
            return len(self.items)

        @property
        def is_empty(self):
            return not self.items


class VectorDBInterface(ABC):
    """
    Abstract Base Class defining the interface for vector database operations.

    Allows swapping different vector database implementations (e.g., ChromaDB, FAISS, Qdrant)
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
            metadatas: An optional list of metadata dictionaries
                       corresponding to each embedding.

        Returns:
            True if the operation was successful, False otherwise.

        Raises:
            DatabaseError: If an error occurs during the operation.
            ValueError: If input lists have mismatched lengths or invalid types.
        """

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
            A SearchResults object containing the found items,
            or None if the query failed catastrophically. Returns an empty
            SearchResults if the query was successful but found no matches.

        Raises:
            DatabaseError: If an error occurs during the query.
            ValueError: If the query embedding is invalid.
        """

    @abstractmethod
    def get_all_embeddings_with_ids(
        self, pagination_batch_size: int = 1000
    ) -> Optional[Tuple[List[str], np.ndarray]]:
        """
        Retrieves all embeddings and their corresponding IDs from the database.

        Uses pagination for memory efficiency with large datasets.

        Args:
            pagination_batch_size: The number of items to retrieve in each
                                   batch during pagination.

        Returns:
            A tuple containing:
            - A list of all IDs.
            - A NumPy array of all embeddings.
            Returns None if the retrieval fails. Returns ([], np.empty(0, ...))
            if the collection is empty. The dimension of the empty array might
            be a fallback or determined from metadata if possible.

        Raises:
            DatabaseError: If an error occurs during retrieval.
        """

    @abstractmethod
    def get_embeddings_by_ids(
        self, ids: List[str]
    ) -> Optional[Dict[str, Optional[List[float]]]]:
        """
        Retrieves specific embeddings based on their IDs.

        Args:
            ids: A list of IDs to retrieve embeddings for.

        Returns:
            A dictionary mapping each requested ID to its embedding (as a list of floats).
            If an ID is not found or has no embedding, its value might be None
            or the ID might be omitted from the dictionary (implementation specific).
            Returns None if the entire operation fails.

        Raises:
            DatabaseError: If a database error occurs during retrieval.
            ValueError: If the input ID list is invalid.
        """

    @abstractmethod
    def update_metadata_batch(
        self, ids: List[str], metadatas: List[Dict[str, Any]]
    ) -> bool:
        """
        Updates the metadata for multiple existing items identified by their IDs.

        Note: This typically replaces the entire metadata dictionary for each ID.
              Use caution if you only intend to add/modify specific keys.

        Args:
            ids: A list of unique identifiers for the items to update.
            metadatas: A list of new metadata dictionaries, corresponding
                       to the order of IDs.

        Returns:
            True if the update operation was successful for all items, False otherwise.
            Implementations might vary on atomicity (all succeed or all fail).

        Raises:
            DatabaseError: If a database error occurs during the update.
            ValueError: If input lists have mismatched lengths or invalid types.
        """

    @abstractmethod
    def clear_collection(self) -> bool:
        """
        Deletes all items from the current collection.

        Returns:
            True if the collection was cleared successfully, False otherwise.

        Raises:
            DatabaseError: If an error occurs during the operation.
        """

    @abstractmethod
    def delete_collection(self) -> bool:
        """
        Deletes the entire collection from the database.

        Returns:
            True if the collection was deleted successfully, False otherwise.

        Raises:
            DatabaseError: If an error occurs during deletion.
        """

    @abstractmethod
    def count(self) -> int:
        """
        Returns the total number of items currently in the collection.

        Returns:
            The number of items, or -1 if an error occurs during counting.
        """

    @property
    @abstractmethod
    def is_initialized(self) -> bool:
        """
        Checks if the database connection and collection are properly initialized.

        Returns:
            True if initialized, False otherwise.
        """

    @abstractmethod
    def get_dimension_from_metadata(self) -> Optional[Union[str, int]]:
        """
        Attempts to retrieve the embedding dimension stored in the collection's metadata.

        Returns:
            The dimension (int or 'full') or None if not found or an error occurs.
        """
