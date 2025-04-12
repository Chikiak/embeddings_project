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
        self._last_init_error = None # Store last init error

        logger.info(f"Initializing ChromaVectorDB:")
        logger.info(f"  Storage Path: {os.path.abspath(self.path)}")
        logger.info(f"  Collection Name: {self.collection_name}")

        try:
            os.makedirs(self.path, exist_ok=True)
            logger.debug(f"Database directory ensured at: {self.path}")
        except OSError as e:
            msg = f"Error creating/accessing DB directory {self.path}: {e}"
            logger.error(msg, exc_info=True)
            self._last_init_error = msg
            raise InitializationError(msg) from e

        try:
            # Consider adding settings for timeout or other client parameters if needed
            self.client = chromadb.PersistentClient(path=self.path)
            logger.info("ChromaDB PersistentClient initialized.")
            self._client_initialized = True
            # Attempt to ensure collection exists immediately after client init
            if not self._ensure_collection_exists():
                 # If collection ensuring fails right away, it's an initialization error
                 msg = f"Failed to get or create collection '{self.collection_name}' during initial setup."
                 logger.error(msg)
                 self._last_init_error = msg
                 # Don't raise immediately, let is_initialized handle it, but log error
                 # raise InitializationError(msg) # Optional: raise here if preferred

        except Exception as e:
            msg = f"Unexpected error initializing ChromaDB client: {e}"
            logger.error(msg, exc_info=True)
            self._last_init_error = msg
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

        # If collection object exists, do a quick check (e.g., count) to see if it's still valid
        if self.collection is not None:
            try:
                self.collection.count()
                logger.debug(f"Collection '{self.collection_name}' object exists and count check passed.")
                return True
            except Exception as e:
                logger.warning(
                    f"Collection object for '{self.collection_name}' exists but failed check (e.g., count): {e}. Will try to get/create again."
                )
                self.collection = None  # Reset collection object as it seems invalid

        # If collection object is None or was reset, try to get/create it
        logger.info(
            f"Attempting to ensure collection '{self.collection_name}' exists..."
        )
        try:
            # Define metadata for cosine distance
            # Note: Default is L2, explicitly set to cosine if needed.
            # Jina models often use cosine similarity.
            metadata_options = {"hnsw:space": "cosine"}
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata=metadata_options,
                # embedding_function=None # Explicitly disable default embedding function if providing own embeddings
            )
            collection_count = self.collection.count() # Get count after ensuring
            logger.info(
                f"Collection '{self.collection_name}' ensured/created. Current count: {collection_count}"
            )
            return True
        except InvalidDimensionException as e_dim:
            # This might happen if trying to create with inconsistent dimensions over time
            logger.error(
                f"Invalid dimension error ensuring collection '{self.collection_name}': {e_dim}",
                exc_info=True,
            )
            self.collection = None # Ensure collection is None on error
            return False
        except Exception as e:
            # Catch other potential errors during get_or_create_collection
            logger.error(
                f"Unexpected error getting/creating collection '{self.collection_name}': {e}",
                exc_info=True,
            )
            self.collection = None # Ensure collection is None on error
            return False

    @property
    def is_initialized(self) -> bool:
        """Verifica si el cliente está inicializado y la colección está accesible."""
        # Check client first, then try to ensure collection exists
        if not self._client_initialized:
             return False
        # Try to ensure collection exists, this will attempt creation if needed
        return self._ensure_collection_exists()

    def add_embeddings(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """
        Añade o actualiza embeddings en la colección ChromaDB.
        """
        if not self.is_initialized or self.collection is None:
            # Attempt to re-ensure collection one last time before failing
            if not self._ensure_collection_exists() or self.collection is None:
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
            # Ensure metadatas is a list of dicts even if None provided
            metadatas = [{} for _ in ids] # Create default empty dicts

        # Dimension check (optional but recommended)
        expected_dim: Optional[int] = None
        if embeddings and embeddings[0]:
            expected_dim = len(embeddings[0])
            # Check consistency within the batch
            if not all(len(emb) == expected_dim for emb in embeddings if emb is not None):
                inconsistent_dims = [
                    (i, len(emb))
                    for i, emb in enumerate(embeddings)
                    if emb is not None and len(emb) != expected_dim
                ]
                logger.error(
                    f"Inconsistent embedding dimensions found in batch. Expected {expected_dim}, found examples: {inconsistent_dims[:5]}"
                )
                raise ValueError("Inconsistent embedding dimensions found in batch.")

        num_items_to_add = len(ids)
        logger.info(
            f"Attempting to add/update {num_items_to_add} embeddings in collection '{self.collection_name}'..."
        )
        try:
            # Upsert handles both add and update based on ID
            self.collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)
            logger.info(f"Upsert for {num_items_to_add} items completed.")
            return True
        except InvalidDimensionException as e_dim:
            dim_info = f"Provided batch dim: {expected_dim if expected_dim is not None else 'N/A'}"
            msg = f"ChromaDB upsert error: Invalid dimension. {dim_info}. Details: {e_dim}"
            logger.error(msg, exc_info=True)
            raise DatabaseError(msg) from e_dim
        except Exception as e:
            # Catch other potential ChromaDB errors during upsert
            msg = f"Error during upsert operation in collection '{self.collection_name}': {e}"
            logger.error(msg, exc_info=True)
            raise DatabaseError(msg) from e

    def query_similar(
        self, query_embedding: List[float], n_results: int = 5
    ) -> Optional[SearchResults]:
        """
        Consulta ChromaDB por embeddings similares y devuelve SearchResults.
        """
        if not self.is_initialized or self.collection is None:
             # Attempt to re-ensure collection one last time before failing
            if not self._ensure_collection_exists() or self.collection is None:
                raise DatabaseError("Collection not available. Cannot query.")

        if not query_embedding or not isinstance(query_embedding, list):
            raise ValueError("Invalid query embedding (must be a list of floats).")

        current_count = self.count()
        if current_count == 0:
            logger.warning("Query attempted on an empty collection.")
            return SearchResults(items=[])
        if current_count == -1:
            # Count failed, implies DB issue
            raise DatabaseError("Cannot query, failed to get collection count.")

        # Ensure n_results is not greater than the number of items in the collection
        effective_n_results = min(n_results, current_count)
        if effective_n_results <= 0:
            logger.warning(
                f"Effective n_results is {effective_n_results} (requested: {n_results}, count: {current_count}). Returning empty results."
            )
            return SearchResults(items=[])

        logger.info(
            f"Querying collection '{self.collection_name}' for {effective_n_results} nearest neighbors..."
        )
        try:
            start_time = time.time()
            # Query the collection
            results_dict = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=effective_n_results,
                include=["metadatas", "distances"], # Request distances and metadatas
            )
            end_time = time.time()
            logger.info(f"ChromaDB query executed in {end_time - start_time:.3f}s.")

            # Process results
            result_items = []
            # Check if the dictionary and required keys/sublists exist and are not None
            if (
                results_dict
                and results_dict.get("ids") is not None
                and results_dict.get("distances") is not None
                and results_dict.get("metadatas") is not None
                and results_dict["ids"] # Check if the outer list exists
                and results_dict["ids"][0] is not None # Check if the inner list for the first query exists
            ):
                ids_list = results_dict["ids"][0]
                distances_list = results_dict["distances"][0]
                metadatas_list = results_dict["metadatas"][0]

                # Basic sanity check for length consistency
                if not (len(ids_list) == len(distances_list) == len(metadatas_list)):
                     logger.warning(f"Query result lists length mismatch! IDs:{len(ids_list)}, Dist:{len(distances_list)}, Meta:{len(metadatas_list)}. Trying to proceed cautiously.")
                     # Attempt to proceed using the length of IDs as reference, might lead to errors if lists are truly mismatched
                     min_len = len(ids_list)
                     distances_list = distances_list[:min_len]
                     metadatas_list = metadatas_list[:min_len]


                for i, img_id in enumerate(ids_list):
                    # Ensure metadata is a dict, default to empty if None
                    metadata = metadatas_list[i] if metadatas_list[i] is not None else {}
                    item = SearchResultItem(
                        id=img_id,
                        distance=distances_list[i], # Distance can be None if not calculated
                        metadata=metadata,
                    )
                    result_items.append(item)

                logger.info(f"Query successful. Found {len(result_items)} results.")
                return SearchResults(items=result_items, query_vector=query_embedding)
            else:
                # Handle cases where results are empty or structure is unexpected
                logger.info("Query successful, but no matching documents found or result format unexpected.")
                logger.debug(f"Raw query result from ChromaDB: {results_dict}")
                return SearchResults(items=[], query_vector=query_embedding)

        except InvalidDimensionException as e_dim:
            query_dim = len(query_embedding)
            msg = f"ChromaDB query error: Invalid dimension. Query Dim: {query_dim}. Details: {e_dim}"
            logger.error(msg, exc_info=True)
            raise DatabaseError(msg) from e_dim
        except Exception as e:
            # Catch other potential errors during query
            msg = f"Error during query operation in collection '{self.collection_name}': {e}"
            logger.error(msg, exc_info=True)
            raise DatabaseError(msg) from e

    def get_all_embeddings_with_ids(
        self, pagination_batch_size: int = 1000
    ) -> Optional[Tuple[List[str], np.ndarray]]:
        """
        Recupera todos los embeddings e IDs usando paginación.
        Maneja embeddings devueltos como ndarray o list.

        Args:
            pagination_batch_size: Tamaño del lote para la paginación.

        Returns:
            Tupla (lista_ids, array_embeddings) o None si falla.
            Devuelve ([], np.empty(0, ...)) si la colección está vacía.

        Raises:
            DatabaseError: Si ocurre un error durante la recuperación.
        """
        if not self.is_initialized or self.collection is None:
             # Attempt to re-ensure collection one last time before failing
            if not self._ensure_collection_exists() or self.collection is None:
                raise DatabaseError("Collection not available. Cannot get all embeddings.")

        # Determine fallback dimension for empty array
        fallback_dim = 1 # Default fallback
        try:
            if VECTOR_DIMENSION:
                fallback_dim = VECTOR_DIMENSION
        except Exception:
            logger.warning("Could not determine precise fallback dimension, using 1.")


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
                raise DatabaseError("Failed to get collection count before retrieving all.")

            logger.info(
                f"Retrieving all {count} embeddings and IDs from '{self.collection_name}' using pagination (batch size: {pagination_batch_size})..."
            )
            start_time = time.time()

            all_ids = []
            all_embeddings_list = [] # This will store lists of floats
            retrieved_count = 0
            actual_embedding_dim = None # To store the dimension found in the first batch

            while retrieved_count < count:
                offset = retrieved_count
                limit = min(pagination_batch_size, count - retrieved_count)
                logger.debug(f"Retrieving batch offset={offset}, limit={limit}")

                try:
                    # Include embeddings and metadatas (optional, but helps debug)
                    results = self.collection.get(
                        limit=limit,
                        offset=offset,
                        include=["embeddings", "metadatas"],
                    )
                    if offset == 0:
                        logger.debug(f"Raw result from collection.get (offset 0): {results}")

                    if results is None:
                        logger.error(f"collection.get returned None at offset {offset}. Stopping retrieval.")
                        raise DatabaseError(f"Received None from collection.get at offset {offset}.")

                    batch_ids = results.get("ids")
                    batch_embeddings_raw = results.get("embeddings") # Might be list or ndarray

                    # --- Check IDs ---
                    if not isinstance(batch_ids, list):
                        logger.error(f"Invalid data type for 'ids' at offset {offset}. Type: {type(batch_ids)}. Stopping.")
                        logger.debug(f"Problematic batch result: {results}")
                        raise DatabaseError(f"Invalid 'ids' data type in batch at offset {offset}.")
                    if not batch_ids:
                        if offset == 0 and count > 0:
                             logger.error(f"CRITICAL: Received empty 'ids' list at offset 0 despite collection count={count}. Stopping.")
                             raise DatabaseError(f"Empty 'ids' list at offset 0 with count={count}.")
                        else:
                             logger.info(f"Received empty 'ids' list at offset {offset}. Assuming end of collection.")
                             break

                    # --- Check and Process Embeddings ---
                    batch_embeddings_list: Optional[List[List[float]]] = None # To store the processed list

                    if isinstance(batch_embeddings_raw, list):
                        logger.debug(f"Embeddings received as list at offset {offset}.")
                        # Assume it's List[List[float]] or List[None], etc.
                        batch_embeddings_list = batch_embeddings_raw
                    elif isinstance(batch_embeddings_raw, np.ndarray):
                        logger.debug(f"Embeddings received as numpy.ndarray at offset {offset}. Converting to list.")
                        try:
                            batch_embeddings_list = batch_embeddings_raw.tolist()
                        except Exception as e_tolist:
                            logger.error(f"Failed to convert numpy array to list at offset {offset}: {e_tolist}", exc_info=True)
                            raise DatabaseError(f"Failed converting ndarray embeddings at offset {offset}") from e_tolist
                    else:
                        # Handle unexpected type for embeddings
                        logger.error(f"Invalid data type for 'embeddings' at offset {offset}. Type: {type(batch_embeddings_raw)}. Stopping.")
                        logger.debug(f"Problematic batch result: {results}")
                        raise DatabaseError(f"Invalid 'embeddings' data type in batch at offset {offset}.")

                    # --- Check Consistency and Append ---
                    if batch_embeddings_list is None:
                         # This case should be caught by the type checks above, but as a safeguard
                         logger.error(f"Embeddings list is None after processing at offset {offset}. Stopping.")
                         raise DatabaseError(f"Embeddings became None unexpectedly at offset {offset}.")

                    if len(batch_ids) != len(batch_embeddings_list):
                         logger.error(f"CRITICAL BATCH MISMATCH at offset {offset}: IDs ({len(batch_ids)}) vs Processed Embeddings ({len(batch_embeddings_list)}). Stopping retrieval.")
                         logger.debug(f"Problematic batch result: {results}")
                         raise DatabaseError(f"Batch length mismatch after processing embeddings at offset {offset}.")

                    # Store the dimension from the first valid embedding found
                    if actual_embedding_dim is None:
                        for emb in batch_embeddings_list:
                            if emb is not None and isinstance(emb, list):
                                actual_embedding_dim = len(emb)
                                logger.info(f"Detected embedding dimension from data: {actual_embedding_dim}")
                                break # Found the dimension

                    # Add batch data to overall lists
                    all_ids.extend(batch_ids)
                    # Extend with the processed list (always List[List[float]] or List[None])
                    all_embeddings_list.extend(batch_embeddings_list)
                    retrieved_count += len(batch_ids)
                    logger.debug(f"Retrieved {len(batch_ids)} items. Total retrieved: {retrieved_count}/{count}")

                except Exception as e_get:
                    # Catch errors during the .get() call itself or during processing
                    logger.error(f"Error processing batch at offset {offset}: {e_get}", exc_info=True)
                    raise DatabaseError(f"Error retrieving/processing batch at offset {offset}: {e_get}") from e_get

            end_time = time.time()
            logger.info(
                f"Data retrieval finished in {end_time - start_time:.2f}s. Retrieved {retrieved_count} items."
            )

            # Final consistency checks
            if len(all_ids) != len(all_embeddings_list):
                raise DatabaseError(
                    f"CRITICAL: Final mismatch after pagination: IDs ({len(all_ids)}) vs Embeddings ({len(all_embeddings_list)})."
                )
            if retrieved_count != count and retrieved_count < count:
                logger.warning(
                    f"Final retrieved count ({retrieved_count}) is less than initial count ({count}). Data might have changed or retrieval stopped early."
                )

            # --- Convert to NumPy array ---
            try:
                if not all_embeddings_list:
                    logger.warning("Embeddings list is empty after retrieval loop.")
                    final_dim = actual_embedding_dim if actual_embedding_dim is not None else fallback_dim
                    return ([], np.empty((0, final_dim), dtype=np.float32))

                # Filter out potential None embeddings before converting to array
                valid_ids = []
                valid_embeddings = []
                none_count = 0
                for i, emb in enumerate(all_embeddings_list):
                    if emb is not None and isinstance(emb, list):
                         # Determine dimension if not already set
                         if actual_embedding_dim is None:
                             actual_embedding_dim = len(emb)
                             logger.info(f"Detected embedding dimension during filtering: {actual_embedding_dim}")
                         # Optional: Check dimension consistency again here if needed
                         elif len(emb) != actual_embedding_dim:
                             logger.warning(f"Skipping embedding at index {i} for ID '{all_ids[i]}' due to dimension mismatch (Expected {actual_embedding_dim}, Got {len(emb)}).")
                             none_count += 1 # Treat as invalid
                             continue
                         valid_embeddings.append(emb)
                         valid_ids.append(all_ids[i])
                    else:
                         none_count += 1
                         logger.debug(f"Found None or invalid embedding for ID: {all_ids[i]} at index {i}")

                if none_count > 0:
                    logger.warning(
                        f"Filtered out {none_count} None or invalid embeddings before creating NumPy array. Returning {len(valid_ids)} valid items."
                    )

                if not valid_embeddings:
                    logger.warning("No valid embeddings found after filtering None values.")
                    final_dim = actual_embedding_dim if actual_embedding_dim is not None else fallback_dim
                    return ([], np.empty((0, final_dim), dtype=np.float32))

                # Convert the list of valid embeddings to a NumPy array
                embeddings_array = np.array(valid_embeddings, dtype=np.float32)

                logger.info(
                    f"Successfully retrieved {len(valid_ids)} valid ID/embedding pairs. Embeddings array shape: {embeddings_array.shape}."
                )
                # Return only the IDs corresponding to the valid embeddings
                return (valid_ids, embeddings_array)

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
            # Ensure DatabaseError is raised for consistency
            if not isinstance(e, DatabaseError):
                raise DatabaseError(msg) from e
            else:
                raise e # Re-raise if it's already a DatabaseError

    def clear_collection(self) -> bool:
        """
        Elimina todos los ítems de la colección ChromaDB.
        """
        if not self.is_initialized or self.collection is None:
             # Attempt to re-ensure collection one last time before failing
            if not self._ensure_collection_exists() or self.collection is None:
                raise DatabaseError("Collection not available. Cannot clear.")
        try:
            count_before = self.count()
            if count_before == 0:
                logger.info(f"Collection '{self.collection_name}' is already empty.")
                return True
            if count_before == -1:
                # Count failed, implies DB issue
                raise DatabaseError("Cannot clear collection, failed to get count.")

            logger.warning(
                f"Attempting to clear ALL {count_before} items from collection '{self.collection_name}'..."
            )
            start_time = time.time()

            # --- Updated Clear Logic ---
            ids_to_delete = []
            try:
                # Fetch only IDs, potentially in batches if count is huge
                # For simplicity, get all at once if count is manageable
                if count_before < 50000: # Arbitrary threshold
                     get_result = self.collection.get(include=[]) # Get only IDs
                     if get_result and get_result.get('ids'):
                         ids_to_delete = get_result['ids']
                else:
                    # Implement pagination for get() if needed for very large collections
                    logger.warning("Large collection detected, fetching IDs for clear might be slow (implement pagination if needed).")
                    # Placeholder for pagination logic if required
                    get_result = self.collection.get(include=[]) # Simple get for now
                    if get_result and get_result.get('ids'):
                        ids_to_delete = get_result['ids']

            except Exception as get_err:
                logger.error(f"Error fetching IDs for clearing collection: {get_err}", exc_info=True)
                raise DatabaseError("Failed to fetch IDs before clearing.") from get_err

            if not ids_to_delete:
                 logger.info("Collection appears empty or failed to retrieve IDs for clearing.")
                 if self.count() == 0:
                     return True
                 else:
                     logger.error("Failed to get IDs for non-empty collection during clear.")
                     return False

            logger.info(f"Deleting {len(ids_to_delete)} items by ID...")
            self.collection.delete(ids=ids_to_delete)
            # -------------------------

            end_time = time.time()

            # Verify deletion by checking count again
            count_after = self.count()
            if count_after == 0:
                logger.info(
                    f"Collection '{self.collection_name}' cleared successfully in {end_time-start_time:.2f}s. Items deleted: {count_before}."
                )
                return True
            else:
                msg = f"Collection count is {count_after} after clearing, expected 0. Items before: {count_before}."
                logger.error(msg)
                return False
        except Exception as e:
            msg = f"Error clearing collection '{self.collection_name}': {e}"
            logger.error(msg, exc_info=True)
            raise DatabaseError(msg) from e

    def delete_collection(self) -> bool:
        """
        Elimina toda la colección ChromaDB.
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
            self.collection = None
            try:
                self.client.get_collection(name=self.collection_name)
                logger.error(
                    f"Verification failed: Collection '{self.collection_name}' still exists after delete request."
                )
                return False
            except Exception as get_err:
                logger.info(f"Verification successful: Collection '{self.collection_name}' no longer exists (get failed: {get_err}).")
                return True

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
        if not self.is_initialized or self.collection is None:
            logger.warning(
                "Count requested but collection is not available or initialization failed."
            )
            return -1
        try:
            count = self.collection.count()
            logger.debug(f"Collection '{self.collection_name}' count: {count}")
            return count
        except Exception as e:
            logger.error(
                f"Error getting count for collection '{self.collection_name}': {e}",
                exc_info=True,
            )
            self.collection = None
            return -1
