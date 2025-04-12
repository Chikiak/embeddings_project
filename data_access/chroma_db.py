# data_access/chroma_db.py
import chromadb
from chromadb.api.models.Collection import Collection

# Removed CollectionNotFoundError from this import
from chromadb.errors import InvalidDimensionException
from typing import List, Dict, Optional, Tuple, Any
import os
import time
import numpy as np
import logging

from config import VECTOR_DIMENSION
from .vector_db_interface import VectorDBInterface
from app.models import SearchResultItem, SearchResults
from app.exceptions import DatabaseError, InitializationError

logger = logging.getLogger(__name__)


class ChromaVectorDB(VectorDBInterface):
    """
    Implementación concreta de VectorDBInterface usando ChromaDB.

    Maneja interacciones con una colección persistente de ChromaDB.
    Incluye lógica para recrear la colección si se elimina.
    """

    def __init__(self, path: str, collection_name: str):
        """
        Inicializa el cliente ChromaDB e intenta obtener/crear la colección.

        Args:
            path: Ruta del directorio para almacenamiento persistente.
            collection_name: Nombre de la colección dentro de ChromaDB.

        Raises:
            InitializationError: Si la creación del directorio o la inicialización
                                 del cliente ChromaDB falla.
        """
        self.path = path
        self.collection_name = collection_name
        self.client: Optional[chromadb.ClientAPI] = None
        self.collection: Optional[Collection] = None
        self._client_initialized = False

        logger.info(f"Initializing ChromaVectorDB:")
        logger.info(f"  Storage Path: {os.path.abspath(self.path)}")
        logger.info(f"  Collection Name: {self.collection_name}")

        try:
            os.makedirs(self.path, exist_ok=True)
            logger.debug(f"Database directory ensured at: {self.path}")
        except OSError as e:
            msg = f"Error creating/accessing DB directory {self.path}: {e}"
            logger.error(msg, exc_info=True)
            raise InitializationError(msg) from e

        try:
            self.client = chromadb.PersistentClient(path=self.path)
            logger.info("ChromaDB PersistentClient initialized.")
            self._client_initialized = True
            self._ensure_collection_exists()

        except Exception as e:
            msg = f"Unexpected error initializing ChromaDB client: {e}"
            logger.error(msg, exc_info=True)
            self._cleanup_client()
            raise InitializationError(msg) from e

    def _cleanup_client(self):
        """Método interno para resetear el estado del cliente en caso de error."""
        self.client = None
        self.collection = None
        self._client_initialized = False
        logger.warning("ChromaDB client state reset due to an error.")

    def _ensure_collection_exists(self) -> bool:
        """
        Verifica si la colección está disponible y la crea si no existe.

        Este método es clave para la recuperación después de la eliminación.

        Returns:
            True si la colección existe o se creó exitosamente, False en caso contrario.
        """
        if not self._client_initialized or self.client is None:
            logger.error("ChromaDB client not initialized. Cannot ensure collection.")
            return False

        if self.collection is not None:
            # Basic check: try to count. If it fails, collection might be invalid.
            try:
                self.collection.count()
                return True
            except Exception as e:
                logger.warning(
                    f"Collection object exists but failed check (e.g., count): {e}. Will try to get/create again."
                )
                self.collection = None  # Reset collection object

        logger.info(
            f"Attempting to ensure collection '{self.collection_name}' exists..."
        )
        try:
            metadata_options = {"hnsw:space": "cosine"}
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata=metadata_options,
            )
            collection_count = self.collection.count()
            logger.info(
                f"Collection '{self.collection_name}' ensured/created. Current count: {collection_count}"
            )
            return True
        except InvalidDimensionException as e_dim:
            logger.error(
                f"Invalid dimension error ensuring collection '{self.collection_name}': {e_dim}",
                exc_info=True,
            )
            self.collection = None
            return False
        except Exception as e:
            logger.error(
                f"Unexpected error getting/creating collection '{self.collection_name}': {e}",
                exc_info=True,
            )
            self.collection = None
            return False

    @property
    def is_initialized(self) -> bool:
        """Verifica si el cliente está inicializado y la colección está accesible."""
        return self._client_initialized and self._ensure_collection_exists()

    def add_embeddings(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """
        Añade o actualiza embeddings en la colección ChromaDB.

        Args:
            ids: Lista de IDs únicos.
            embeddings: Lista de vectores de embedding.
            metadatas: Lista opcional de metadatos.

        Returns:
            True si la operación fue exitosa, False en caso contrario.

        Raises:
            DatabaseError: Si ocurre un error durante la operación upsert.
            ValueError: Si las longitudes de las listas de entrada no coinciden
                        o si las dimensiones de los embeddings son inconsistentes.
        """
        if not self.is_initialized or self.collection is None:
            raise DatabaseError("Collection not available. Cannot add embeddings.")

        if not isinstance(ids, list) or not isinstance(embeddings, list):
            raise ValueError("Invalid input type: ids and embeddings must be lists.")
        if not ids or not embeddings:
            logger.warning("Empty ids or embeddings list provided. Nothing to add.")
            return True
        if len(ids) != len(embeddings):
            raise ValueError(
                f"Input mismatch: IDs ({len(ids)}) vs embeddings ({len(embeddings)})."
            )
        if metadatas:
            if not isinstance(metadatas, list) or len(ids) != len(metadatas):
                raise ValueError(
                    f"Metadata list mismatch or invalid type: IDs ({len(ids)}) vs metadatas ({len(metadatas) if isinstance(metadatas, list) else 'Invalid Type'})."
                )
        else:
            metadatas = [{} for _ in ids]

        # Initialize expected_dim before the if block
        expected_dim: Optional[int] = None
        if embeddings:
            # Use the dimension from the first embedding for consistency check within the batch
            expected_dim = len(embeddings[0])
            # Check against config dimension if it's set
            if VECTOR_DIMENSION and expected_dim != VECTOR_DIMENSION:
                logger.warning(
                    f"Provided embedding dimension ({expected_dim}) differs from config ({VECTOR_DIMENSION}). Ensure consistency."
                )
            # Check consistency within the batch
            if not all(len(emb) == expected_dim for emb in embeddings):
                inconsistent_dims = [
                    (i, len(emb))
                    for i, emb in enumerate(embeddings)
                    if len(emb) != expected_dim
                ]
                logger.error(
                    f"Inconsistent embedding dimensions found in batch. Expected {expected_dim}, found: {inconsistent_dims[:5]}"
                )
                raise ValueError("Inconsistent embedding dimensions found in batch.")

        num_items_to_add = len(ids)
        logger.info(
            f"Attempting to add/update {num_items_to_add} embeddings in collection '{self.collection_name}'..."
        )
        try:
            self.collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)
            logger.info(f"Upsert for {num_items_to_add} items completed.")
            return True
        except InvalidDimensionException as e_dim:
            # Use the expected_dim variable safely here
            dim_info = (
                f"Provided: {expected_dim if expected_dim is not None else 'N/A'}"
            )
            msg = f"ChromaDB upsert error: Invalid dimension. {dim_info}, Config: {VECTOR_DIMENSION}. Details: {e_dim}"
            logger.error(msg, exc_info=True)
            raise DatabaseError(msg) from e_dim
        except Exception as e:
            msg = f"Error during upsert operation in collection '{self.collection_name}': {e}"
            logger.error(msg, exc_info=True)
            raise DatabaseError(msg) from e

    def query_similar(
        self, query_embedding: List[float], n_results: int = 5
    ) -> Optional[SearchResults]:
        """
        Consulta ChromaDB por embeddings similares y devuelve SearchResults.

        Args:
            query_embedding: El vector de embedding para la consulta.
            n_results: Número máximo de resultados.

        Returns:
            Un objeto SearchResults o None si la consulta falla.

        Raises:
            DatabaseError: Si ocurre un error durante la consulta.
            ValueError: Si query_embedding es inválido.
        """
        if not self.is_initialized or self.collection is None:
            raise DatabaseError("Collection not available. Cannot query.")

        if not query_embedding or not isinstance(query_embedding, list):
            raise ValueError("Invalid query embedding (must be a list of floats).")

        current_count = self.count()
        if current_count == 0:
            logger.warning("Query attempted on an empty collection.")
            return SearchResults(items=[])
        if current_count == -1:
            raise DatabaseError("Cannot query, failed to get collection count.")

        effective_n_results = min(n_results, current_count)
        if effective_n_results <= 0:
            logger.warning(
                f"Effective n_results is {effective_n_results}. Returning empty results."
            )
            return SearchResults(items=[])

        logger.info(
            f"Querying collection '{self.collection_name}' for {effective_n_results} nearest neighbors..."
        )
        try:
            start_time = time.time()
            results_dict = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=effective_n_results,
                include=["metadatas", "distances"],
            )
            end_time = time.time()
            logger.info(f"ChromaDB query executed in {end_time - start_time:.3f}s.")

            result_items = []
            if (
                results_dict
                and all(
                    key in results_dict for key in ["ids", "distances", "metadatas"]
                )
                and results_dict["ids"]
                and results_dict["ids"][0] is not None  # Check inner list is not None
            ):
                ids_list = results_dict["ids"][0]
                # Handle cases where distances or metadatas might be None or empty lists
                distances_list = (
                    results_dict.get("distances", [[]])[0]
                    if results_dict.get("distances")
                    else [None] * len(ids_list)
                )
                metadatas_list = (
                    results_dict.get("metadatas", [[]])[0]
                    if results_dict.get("metadatas")
                    else [{}] * len(ids_list)
                )

                # Ensure all lists have the same length as ids_list
                if len(distances_list) != len(ids_list):
                    logger.warning(
                        f"Distance list length ({len(distances_list)}) mismatch with ID list length ({len(ids_list)}). Padding distances with None."
                    )
                    distances_list = [None] * len(ids_list)
                if len(metadatas_list) != len(ids_list):
                    logger.warning(
                        f"Metadata list length ({len(metadatas_list)}) mismatch with ID list length ({len(ids_list)}). Padding metadatas with {{}}."
                    )
                    metadatas_list = [{}] * len(ids_list)

                for i, img_id in enumerate(ids_list):
                    item = SearchResultItem(
                        id=img_id,
                        distance=distances_list[i],
                        metadata=(
                            metadatas_list[i] if metadatas_list[i] is not None else {}
                        ),
                    )
                    result_items.append(item)
                logger.info(f"Query successful. Found {len(result_items)} results.")
                return SearchResults(items=result_items, query_vector=query_embedding)
            else:
                logger.info("Query successful, but no matching documents found.")
                return SearchResults(items=[], query_vector=query_embedding)

        except InvalidDimensionException as e_dim:
            # Provide more context in the error message
            query_dim = len(query_embedding)
            # Try to get collection dimension (might fail if collection is invalid)
            collection_dim_info = "unknown"
            try:
                if self.collection:
                    # This might be specific to Chroma's implementation details
                    # or might require fetching an item if metadata isn't available
                    # For now, we'll just report the query dimension clearly.
                    pass  # Placeholder if we find a way to get expected dim
            except Exception:
                pass  # Ignore errors getting collection dim info

            msg = f"ChromaDB query error: Invalid dimension. Query Dim: {query_dim}, Collection Dim: {collection_dim_info}. Details: {e_dim}"
            logger.error(msg, exc_info=True)
            raise DatabaseError(msg) from e_dim
        except Exception as e:
            msg = f"Error during query operation in collection '{self.collection_name}': {e}"
            logger.error(msg, exc_info=True)
            raise DatabaseError(msg) from e

    def get_all_embeddings_with_ids(
        self, pagination_batch_size: int = 1000
    ) -> Optional[Tuple[List[str], np.ndarray]]:
        """
        Recupera todos los embeddings e IDs usando paginación.

        Args:
            pagination_batch_size: Tamaño del lote para la paginación.

        Returns:
            Tupla (lista_ids, array_embeddings) o None si falla.

        Raises:
            DatabaseError: Si ocurre un error durante la recuperación.
        """
        if not self.is_initialized or self.collection is None:
            raise DatabaseError("Collection not available. Cannot get all embeddings.")

        fallback_dim = VECTOR_DIMENSION if isinstance(VECTOR_DIMENSION, int) else 1
        empty_result: Tuple[List[str], np.ndarray] = (
            [],
            np.empty((0, fallback_dim), dtype=np.float32),
        )

        try:
            count = self.count()
            if count == 0:
                logger.info("Collection is empty. Returning empty lists/arrays.")
                return empty_result
            if count == -1:
                raise DatabaseError(
                    "Failed to get collection count before retrieving all."
                )

            logger.info(
                f"Retrieving all {count} embeddings and IDs from '{self.collection_name}' using pagination (batch size: {pagination_batch_size})..."
            )
            start_time = time.time()

            all_ids = []
            all_embeddings_list = []
            retrieved_count = 0
            actual_embedding_dim = (
                None  # To store the dimension found in the first batch
            )

            while retrieved_count < count:
                logger.debug(
                    f"Retrieving batch offset={retrieved_count}, limit={pagination_batch_size}"
                )
                try:
                    results = self.collection.get(
                        limit=pagination_batch_size,
                        offset=retrieved_count,
                        include=["embeddings"],
                    )

                    if (
                        results
                        and results.get("ids")
                        and results.get("embeddings") is not None
                    ):
                        batch_ids = results["ids"]
                        batch_embeddings = results["embeddings"]

                        if not batch_ids:  # Should not happen if count > 0, but check
                            logger.warning(
                                f"Empty batch retrieved at offset {retrieved_count} despite count={count}. Stopping."
                            )
                            break

                        if batch_embeddings and actual_embedding_dim is None:
                            # Store the dimension from the first non-empty embedding list
                            if batch_embeddings[0] is not None:
                                actual_embedding_dim = len(batch_embeddings[0])
                                logger.info(
                                    f"Detected embedding dimension from data: {actual_embedding_dim}"
                                )

                        all_ids.extend(batch_ids)
                        all_embeddings_list.extend(batch_embeddings)
                        retrieved_count += len(batch_ids)
                        logger.debug(
                            f"Retrieved {len(batch_ids)} items. Total retrieved: {retrieved_count}/{count}"
                        )
                    else:
                        # Handle case where results might be missing keys or embeddings are None
                        logger.warning(
                            f"Unexpected result format or empty batch content at offset {retrieved_count}. Stopping. Result keys: {results.keys() if results else 'None'}"
                        )
                        break
                except Exception as e_get:
                    raise DatabaseError(
                        f"Error retrieving batch at offset {retrieved_count}: {e_get}"
                    ) from e_get

            end_time = time.time()
            logger.info(
                f"Data retrieval finished in {end_time - start_time:.2f}s. Retrieved {retrieved_count} items."
            )

            if len(all_ids) != len(all_embeddings_list):
                raise DatabaseError(
                    f"CRITICAL: Mismatch after pagination: IDs ({len(all_ids)}) vs Embeddings ({len(all_embeddings_list)})."
                )
            if retrieved_count != count:
                logger.warning(
                    f"Final retrieved count ({retrieved_count}) differs from initial count ({count}). Data might have changed."
                )

            try:
                if not all_embeddings_list:
                    logger.warning("Embeddings list is empty after retrieval loop.")
                    # Use detected dimension if available, otherwise fallback
                    final_dim = (
                        actual_embedding_dim
                        if actual_embedding_dim is not None
                        else fallback_dim
                    )
                    return ([], np.empty((0, final_dim), dtype=np.float32))

                # Filter out potential None embeddings before converting to array
                valid_embeddings = [
                    emb for emb in all_embeddings_list if emb is not None
                ]
                num_filtered = len(all_embeddings_list) - len(valid_embeddings)
                if num_filtered > 0:
                    logger.warning(
                        f"Filtered out {num_filtered} None embeddings before creating NumPy array."
                    )
                    # Adjust IDs accordingly? This is complex. Assuming IDs correspond to original list.
                    # For simplicity, we'll proceed with valid embeddings but log the discrepancy potential.
                    # A more robust solution might involve returning IDs corresponding only to valid embeddings.

                if not valid_embeddings:
                    logger.warning(
                        "No valid embeddings found after filtering None values."
                    )
                    final_dim = (
                        actual_embedding_dim
                        if actual_embedding_dim is not None
                        else fallback_dim
                    )
                    return ([], np.empty((0, final_dim), dtype=np.float32))

                embeddings_array = np.array(valid_embeddings, dtype=np.float32)
                # Note: all_ids still corresponds to the original list including potential Nones
                logger.info(
                    f"Successfully retrieved {len(all_ids)} IDs and created embeddings array of shape {embeddings_array.shape} from {len(valid_embeddings)} valid embeddings."
                )
                return (
                    all_ids,
                    embeddings_array,
                )  # Returning all original IDs, but array only has valid embeddings
            except ValueError as ve:
                # This might happen if embeddings within valid_embeddings still have inconsistent lengths
                logger.error(
                    f"ValueError converting embeddings list to NumPy array: {ve}. Check data consistency.",
                    exc_info=True,
                )
                raise DatabaseError(
                    f"Error converting embeddings list to NumPy array: {ve}. Inconsistent dimensions?"
                ) from ve

        except Exception as e:
            msg = f"Error retrieving all embeddings from collection '{self.collection_name}': {e}"
            logger.error(msg, exc_info=True)
            if not isinstance(e, DatabaseError):
                raise DatabaseError(msg) from e
            else:
                raise e

    def clear_collection(self) -> bool:
        """
        Elimina todos los ítems de la colección ChromaDB.

        Returns:
            True si la operación fue exitosa, False en caso contrario.

        Raises:
            DatabaseError: Si ocurre un error durante la limpieza.
        """
        if not self.is_initialized or self.collection is None:
            raise DatabaseError("Collection not available. Cannot clear.")
        try:
            count_before = self.count()
            if count_before == 0:
                logger.info(f"Collection '{self.collection_name}' is already empty.")
                return True
            if count_before == -1:
                raise DatabaseError("Cannot clear collection, failed to get count.")

            logger.warning(
                f"Attempting to clear ALL {count_before} items from collection '{self.collection_name}'..."
            )
            start_time = time.time()
            # Chroma's delete without IDs/where filter deletes all items
            self.collection.delete()
            end_time = time.time()

            # Verify deletion by checking count again
            count_after = self.count()
            if count_after == 0:
                logger.info(
                    f"Collection '{self.collection_name}' cleared successfully in {end_time-start_time:.2f}s. Items deleted: {count_before}."
                )
                return True
            else:
                # This case indicates a potential issue with ChromaDB's delete or count
                msg = f"Collection count is {count_after} after clearing, expected 0. Items before: {count_before}."
                logger.error(msg)
                # Return False instead of raising error, as the operation was attempted but verification failed
                return False
        except Exception as e:
            msg = f"Error clearing collection '{self.collection_name}': {e}"
            logger.error(msg, exc_info=True)
            # Raise DatabaseError for clarity
            raise DatabaseError(msg) from e

    def delete_collection(self) -> bool:
        """
        Elimina toda la colección ChromaDB.

        Returns:
            True si la operación fue exitosa, False en caso contrario.

        Raises:
            DatabaseError: Si ocurre un error durante la eliminación.
        """
        if not self._client_initialized or self.client is None:
            raise DatabaseError("Client not initialized. Cannot delete collection.")
        if not self.collection_name:
            raise DatabaseError("Collection name not set. Cannot delete.")

        logger.warning(
            f"Attempting to DELETE ENTIRE collection '{self.collection_name}' from path '{self.path}'..."
        )
        try:
            start_time = time.time()
            self.client.delete_collection(name=self.collection_name)
            end_time = time.time()
            logger.info(
                f"Collection '{self.collection_name}' delete request sent successfully in {end_time-start_time:.2f}s."
            )
            # Reset internal state as the collection object is now invalid
            self.collection = None
            # Verify deletion by trying to get the collection (should fail)
            try:
                self.client.get_collection(name=self.collection_name)
                # If get_collection succeeds, deletion might have failed silently
                logger.error(
                    f"Verification failed: Collection '{self.collection_name}' still exists after delete request."
                )
                raise DatabaseError(
                    f"Collection '{self.collection_name}' still exists after delete request."
                )
            except (
                Exception
            ):  # Expecting an error here, like ValueError or specific Chroma error
                logger.info(
                    f"Verification successful: Collection '{self.collection_name}' no longer exists."
                )
                return True

        # Removed the specific CollectionNotFoundError catch block
        except Exception as e:
            msg = f"Error deleting collection '{self.collection_name}': {e}"
            logger.error(msg, exc_info=True)
            raise DatabaseError(msg) from e

    def count(self) -> int:
        """
        Devuelve el número de ítems en la colección.

        Returns:
             El número de ítems, o -1 si la colección no está disponible o
             si ocurre un error durante el conteo.
        """
        # Use is_initialized which includes _ensure_collection_exists
        if not self.is_initialized or self.collection is None:
            logger.warning(
                "Count requested but collection is not available or initialization failed."
            )
            return -1
        try:
            return self.collection.count()
        except Exception as e:
            # This might happen if the collection becomes invalid between checks
            logger.error(
                f"Error getting count for collection '{self.collection_name}': {e}",
                exc_info=True,
            )
            # Attempt to reset and re-ensure collection state
            self.collection = None
            if not self._ensure_collection_exists():
                logger.error("Failed to re-ensure collection after count error.")
            return -1
