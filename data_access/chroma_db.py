import logging
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import chromadb
import numpy as np
from chromadb.api.models.Collection import Collection
from chromadb.errors import IDAlreadyExistsError, InvalidDimensionException

try:
    from config import VECTOR_DIMENSION
except ImportError:
    VECTOR_DIMENSION = None


from .vector_db_interface import VectorDBInterface

try:
    from app.exceptions import DatabaseError, InitializationError
    from app.models import SearchResultItem, SearchResults
except ImportError:

    logging.warning(
        "Could not import app.models or app.exceptions. Using placeholders."
    )

    class SearchResultItem:
        def __init__(self, id, distance=None, metadata=None):
            self.id = id
            self.distance = distance
            self.metadata = metadata or {}

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

    class DatabaseError(Exception):
        pass

    class InitializationError(Exception):
        pass


logger = logging.getLogger(__name__)


class ChromaVectorDB(VectorDBInterface):
    """
    Concrete implementation of VectorDBInterface using ChromaDB.

    Handles interactions with a persistent ChromaDB collection.
    Includes logic for recreating the collection if deleted and
    managing dimension metadata.
    """

    def __init__(
        self,
        path: str,
        collection_name: str,
        expected_dimension_metadata: Optional[Union[int, str]] = None,
    ):
        """
        Initializes the ChromaDB client and attempts to get/create the collection.

        Args:
            path: Directory path for persistent storage.
            collection_name: Name of the collection within ChromaDB.
            expected_dimension_metadata: Expected dimension (int or 'full') for this collection.
                                         Used to add metadata if the collection is created.
        """
        self.path = path
        self.collection_name = collection_name
        self._expected_dimension_metadata = expected_dimension_metadata
        self.client: Optional[chromadb.ClientAPI] = None
        self.collection: Optional[Collection] = None
        self._client_initialized = False
        self._last_init_error: Optional[str] = None

        logger.info(f"Initializing ChromaVectorDB:")
        logger.info(f"  Storage Path: {os.path.abspath(self.path)}")
        logger.info(f"  Target Collection Name: {self.collection_name}")
        if self._expected_dimension_metadata:
            logger.info(
                f"  Expected dimension metadata for new collection: {self._expected_dimension_metadata}"
            )

        try:
            os.makedirs(self.path, exist_ok=True)
            logger.debug(f"Database directory ensured at: {self.path}")
        except OSError as e:
            msg = f"Error creating/accessing DB directory {self.path}: {e}"
            logger.error(msg, exc_info=True)
            self._last_init_error = msg
            raise InitializationError(msg) from e

        try:

            self.client = chromadb.PersistentClient(path=self.path)
            logger.info("ChromaDB PersistentClient initialized.")
            self._client_initialized = True

            if not self._ensure_collection_exists():
                msg = f"Failed to get or create collection '{self.collection_name}' during initial setup."
                logger.error(msg)
                self._last_init_error = msg

        except Exception as e:
            msg = f"Unexpected error initializing ChromaDB client: {e}"
            logger.error(msg, exc_info=True)
            self._last_init_error = msg
            self._cleanup_client()
            raise InitializationError(msg) from e

    def _cleanup_client(self):
        """Internal method to reset client state on error."""
        self.client = None
        self.collection = None
        self._client_initialized = False
        logger.warning("ChromaDB client state reset due to an error.")

    def _ensure_collection_exists(self) -> bool:
        """
        Verifies collection availability, creating it if necessary.
        Adds dimension metadata upon creation if provided during init.

        Returns:
            True if the collection exists or was created successfully, False otherwise.
        """
        if not self._client_initialized or self.client is None:
            logger.error(
                "ChromaDB client not initialized. Cannot ensure collection."
            )
            return False

        if self.collection is not None:
            try:
                self.collection.peek(limit=1)
                logger.debug(
                    f"Collection object for '{self.collection_name}' exists and seems valid."
                )
                return True
            except Exception as e:
                logger.warning(
                    f"Collection object for '{self.collection_name}' exists but failed check: {e}. Will try to get/create again."
                )
                self.collection = None

        logger.info(
            f"Attempting to ensure collection '{self.collection_name}' exists..."
        )
        try:

            creation_metadata = None
            if self._expected_dimension_metadata is not None:
                creation_metadata = {
                    "embedding_dimension": self._expected_dimension_metadata
                }
                logger.debug(
                    f"Will set metadata {creation_metadata} if collection '{self.collection_name}' is created."
                )

            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata=creation_metadata,
            )
            collection_count = self.collection.count()
            logger.info(
                f"Collection '{self.collection_name}' ensured/created. Current count: {collection_count}"
            )

            current_meta = self.collection.metadata
            has_meta_dim = (
                current_meta is not None
                and "embedding_dimension" in current_meta
            )

            if (
                self._expected_dimension_metadata is not None
                and not has_meta_dim
            ):
                try:
                    logger.info(
                        f"Attempting to add/update 'embedding_dimension' metadata for '{self.collection_name}' to '{self._expected_dimension_metadata}'."
                    )
                    new_metadata = current_meta.copy() if current_meta else {}
                    new_metadata["embedding_dimension"] = (
                        self._expected_dimension_metadata
                    )
                    self.collection.modify(metadata=new_metadata)
                    logger.info(
                        f"Metadata for '{self.collection_name}' modified successfully."
                    )

                    self.collection = self.client.get_collection(
                        name=self.collection_name
                    )
                except Exception as mod_err:
                    logger.warning(
                        f"Could not modify metadata for collection '{self.collection_name}': {mod_err}",
                        exc_info=False,
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
        """Checks if the client is initialized and the collection is accessible."""
        if not self._client_initialized:
            return False

        return self._ensure_collection_exists()

    def get_dimension_from_metadata(self) -> Optional[Union[str, int]]:
        """
        Attempts to retrieve the dimension stored in the collection's metadata.

        Returns:
            The dimension (int or 'full') or None if not found or an error occurs.
        """
        if not self.is_initialized or self.collection is None:

            if not self._ensure_collection_exists() or self.collection is None:
                logger.warning(
                    "Collection not available, cannot get metadata."
                )
                return None

        try:

            metadata = self.collection.metadata
            if metadata and "embedding_dimension" in metadata:
                dim_value = metadata["embedding_dimension"]

                if (
                    isinstance(dim_value, int) and dim_value > 0
                ) or dim_value == "full":
                    logger.debug(
                        f"Found dimension '{dim_value}' in metadata for collection '{self.collection_name}'."
                    )
                    return dim_value
                else:
                    logger.warning(
                        f"Invalid value found for 'embedding_dimension' metadata in '{self.collection_name}': {dim_value}"
                    )
                    return None
            else:
                logger.debug(
                    f"No 'embedding_dimension' key found in metadata for '{self.collection_name}'."
                )
                return None
        except Exception as e:
            logger.warning(
                f"Error accessing metadata for collection '{self.collection_name}': {e}",
                exc_info=False,
            )
            return None

    def add_embeddings(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """
        Adds or updates embeddings in the ChromaDB collection.
        Verifies dimension against metadata if available.
        """
        if not self.is_initialized or self.collection is None:
            if not self._ensure_collection_exists() or self.collection is None:
                raise DatabaseError(
                    "Collection not available. Cannot add embeddings."
                )

        if not isinstance(ids, list) or not isinstance(embeddings, list):
            raise ValueError("ids and embeddings must be lists.")
        if not ids or not embeddings:
            logger.info(
                "Empty ids or embeddings list provided to add_embeddings. Nothing to do."
            )
            return True
        if len(ids) != len(embeddings):
            raise ValueError(
                f"Input mismatch: IDs ({len(ids)}) vs embeddings ({len(embeddings)})."
            )
        if metadatas and (
            not isinstance(metadatas, list) or len(ids) != len(metadatas)
        ):
            raise ValueError("Metadata list mismatch or invalid type.")

        if metadatas is None:
            metadatas = [{} for _ in ids]
        elif len(metadatas) != len(ids):
            raise ValueError(
                f"Input mismatch: IDs ({len(ids)}) vs metadatas ({len(metadatas)})."
            )

        expected_dim_meta = self.get_dimension_from_metadata()
        actual_batch_dim: Optional[int] = None
        if embeddings and embeddings[0] and isinstance(embeddings[0], list):
            actual_batch_dim = len(embeddings[0])

            if not all(
                isinstance(e, list) and len(e) == actual_batch_dim
                for e in embeddings
            ):
                raise ValueError(
                    "Inconsistent embedding dimensions within the provided batch."
                )

        collection_count = self.count()
        if expected_dim_meta is not None and collection_count > 0:
            if (
                isinstance(expected_dim_meta, int)
                and actual_batch_dim != expected_dim_meta
            ):
                msg = (
                    f"Dimension mismatch! Trying to add embeddings with dimension {actual_batch_dim}, "
                    f"but collection '{self.collection_name}' expects dimension {expected_dim_meta} based on metadata."
                )
                logger.error(msg)
                raise DatabaseError(msg)
            elif expected_dim_meta == "full":

                pass
        elif expected_dim_meta is None and collection_count > 0:
            logger.warning(
                f"Adding embeddings to collection '{self.collection_name}' which has data but lacks dimension metadata. "
                f"ChromaDB might enforce dimension based on existing data."
            )

        num_items_to_add = len(ids)
        logger.info(
            f"Attempting to add/update {num_items_to_add} embeddings (dim: {actual_batch_dim or 'N/A'}) in collection '{self.collection_name}'..."
        )
        try:

            self.collection.upsert(
                ids=ids, embeddings=embeddings, metadatas=metadatas
            )
            logger.info(f"Upsert for {num_items_to_add} items completed.")

            if (
                collection_count == 0
                and self._expected_dimension_metadata is not None
            ):
                current_meta_after_add = self.collection.metadata
                if (
                    current_meta_after_add is None
                    or "embedding_dimension" not in current_meta_after_add
                ):
                    try:
                        logger.info(
                            f"Attempting to set 'embedding_dimension' metadata after first add for '{self.collection_name}'."
                        )
                        new_metadata = (
                            current_meta_after_add.copy()
                            if current_meta_after_add
                            else {}
                        )
                        new_metadata["embedding_dimension"] = (
                            self._expected_dimension_metadata
                        )
                        self.collection.modify(metadata=new_metadata)
                        logger.info(
                            f"Metadata set successfully for '{self.collection_name}' after first add."
                        )

                        self.collection = self.client.get_collection(
                            name=self.collection_name
                        )
                    except Exception as mod_err:
                        logger.warning(
                            f"Could not set metadata after first add for '{self.collection_name}': {mod_err}",
                            exc_info=False,
                        )

            return True
        except InvalidDimensionException as e_dim:

            raise DatabaseError(
                f"ChromaDB upsert error: Invalid dimension. Batch dim: {actual_batch_dim}. Details: {e_dim}"
            ) from e_dim
        except IDAlreadyExistsError as e_id:

            logger.warning(
                f"IDAlreadyExistsError during upsert (unexpected): {e_id}. Check ChromaDB version/behavior."
            )

            return False
        except Exception as e:

            logger.error(
                f"Unexpected error during upsert operation in collection '{self.collection_name}': {e}",
                exc_info=True,
            )
            raise DatabaseError(
                f"Error during upsert operation in collection '{self.collection_name}': {e}"
            ) from e

    def query_similar(
        self, query_embedding: List[float], n_results: int = 5
    ) -> Optional[SearchResults]:
        """
        Queries ChromaDB for similar embeddings and returns SearchResults.
        Verifies query dimension against metadata if available.
        """
        if not self.is_initialized or self.collection is None:
            if not self._ensure_collection_exists() or self.collection is None:
                raise DatabaseError("Collection not available. Cannot query.")

        if not query_embedding or not isinstance(query_embedding, list):
            raise ValueError(
                "Invalid query embedding (must be a list of floats)."
            )

        query_dim = len(query_embedding)
        expected_dim_meta = self.get_dimension_from_metadata()

        current_count = self.count()
        if expected_dim_meta is not None and current_count > 0:
            if (
                isinstance(expected_dim_meta, int)
                and query_dim != expected_dim_meta
            ):
                msg = (
                    f"Query dimension mismatch! Query has dimension {query_dim}, "
                    f"but collection '{self.collection_name}' expects dimension {expected_dim_meta} based on metadata."
                )
                logger.error(msg)
                raise DatabaseError(msg)

        elif expected_dim_meta is None and current_count > 0:
            logger.warning(
                f"Querying collection '{self.collection_name}' which lacks dimension metadata. "
                f"Ensure query vector dimension ({query_dim}) matches data."
            )

        if current_count <= 0:
            logger.warning(
                f"Query attempted on an empty or inaccessible collection '{self.collection_name}' (count={current_count})."
            )
            return SearchResults(items=[])

        effective_n_results = min(n_results, current_count)
        if effective_n_results <= 0:
            logger.warning(
                f"Effective n_results is {effective_n_results}. Returning empty results."
            )
            return SearchResults(items=[])

        logger.info(
            f"Querying collection '{self.collection_name}' (Expected Dim: {expected_dim_meta or 'Unknown'}) "
            f"for {effective_n_results} neighbors (Query Dim: {query_dim})..."
        )
        try:
            start_time = time.time()

            results_dict = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=effective_n_results,
                include=["metadatas", "distances"],
            )
            end_time = time.time()
            logger.info(
                f"ChromaDB query executed in {end_time - start_time:.3f}s."
            )

            result_items = []

            if (
                results_dict
                and results_dict.get("ids")
                and results_dict["ids"]
                and results_dict["ids"][0] is not None
            ):

                ids_list = results_dict["ids"][0]

                distances_list = (
                    results_dict.get("distances", [[]])[0]
                    if results_dict.get("distances")
                    else []
                )
                metadatas_list = (
                    results_dict.get("metadatas", [[]])[0]
                    if results_dict.get("metadatas")
                    else []
                )

                num_ids = len(ids_list)
                if len(distances_list) != num_ids:
                    logger.warning(
                        f"Query result distance list length mismatch ({len(distances_list)} vs {num_ids} IDs). Padding with None."
                    )
                    distances_list = (distances_list + [None] * num_ids)[
                        :num_ids
                    ]
                if len(metadatas_list) != num_ids:
                    logger.warning(
                        f"Query result metadata list length mismatch ({len(metadatas_list)} vs {num_ids} IDs). Padding with {{}}."
                    )
                    metadatas_list = (metadatas_list + [{}] * num_ids)[
                        :num_ids
                    ]

                for i, img_id in enumerate(ids_list):
                    metadata = (
                        metadatas_list[i]
                        if metadatas_list[i] is not None
                        else {}
                    )
                    distance = distances_list[i]
                    item = SearchResultItem(
                        id=img_id, distance=distance, metadata=metadata
                    )
                    result_items.append(item)

                logger.info(
                    f"Query successful. Found {len(result_items)} results."
                )
                return SearchResults(
                    items=result_items, query_vector=query_embedding
                )
            else:

                logger.info(
                    "Query successful, but no matching documents found or result format unexpected."
                )
                return SearchResults(items=[], query_vector=query_embedding)

        except InvalidDimensionException as e_dim:

            raise DatabaseError(
                f"ChromaDB query error: Invalid dimension. Query Dim: {query_dim}. Details: {e_dim}"
            ) from e_dim
        except Exception as e:

            logger.error(
                f"Error during query operation in collection '{self.collection_name}': {e}",
                exc_info=True,
            )
            raise DatabaseError(
                f"Error during query operation in collection '{self.collection_name}': {e}"
            ) from e

    def get_all_embeddings_with_ids(
        self, pagination_batch_size: int = 1000
    ) -> Optional[Tuple[List[str], np.ndarray]]:
        """
        Retrieves all embeddings and IDs using pagination.
        """
        if not self.is_initialized or self.collection is None:
            if not self._ensure_collection_exists() or self.collection is None:
                raise DatabaseError(
                    "Collection not available. Cannot get all embeddings."
                )

        fallback_dim = 1
        try:
            meta_dim = self.get_dimension_from_metadata()
            if isinstance(meta_dim, int):
                fallback_dim = meta_dim
            elif VECTOR_DIMENSION:
                fallback_dim = VECTOR_DIMENSION
        except Exception:
            pass

        empty_result: Tuple[List[str], np.ndarray] = (
            [],
            np.empty((0, fallback_dim), dtype=np.float32),
        )

        try:
            count = self.count()
            if count <= 0:
                logger.info(
                    f"Collection '{self.collection_name}' is empty or inaccessible (count={count}). Returning empty result."
                )
                return empty_result

            logger.info(
                f"Retrieving all {count} embeddings and IDs from '{self.collection_name}' (batch size: {pagination_batch_size})..."
            )
            start_time = time.time()
            all_ids = []
            all_embeddings_list = []
            retrieved_count = 0
            actual_embedding_dim = None

            while retrieved_count < count:
                offset = retrieved_count

                limit = min(pagination_batch_size, count - retrieved_count)
                if limit <= 0:
                    break

                try:
                    logger.debug(
                        f"Fetching batch: offset={offset}, limit={limit}"
                    )

                    results = self.collection.get(
                        limit=limit, offset=offset, include=["embeddings"]
                    )

                    if (
                        results is None
                        or results.get("ids") is None
                        or results.get("embeddings") is None
                    ):
                        logger.error(
                            f"collection.get returned invalid data at offset {offset}. Stopping retrieval."
                        )
                        break

                    batch_ids = results["ids"]
                    batch_embeddings_raw = results["embeddings"]

                    if not batch_ids:
                        logger.debug(
                            "Received empty batch_ids, assuming end of data."
                        )
                        break

                    if isinstance(batch_embeddings_raw, np.ndarray):
                        batch_embeddings_list = batch_embeddings_raw.tolist()
                    elif isinstance(batch_embeddings_raw, list):
                        batch_embeddings_list = batch_embeddings_raw
                    else:
                        raise DatabaseError(
                            f"Invalid embedding data type received at offset {offset}: {type(batch_embeddings_raw)}"
                        )

                    if len(batch_ids) != len(batch_embeddings_list):
                        raise DatabaseError(
                            f"Batch length mismatch at offset {offset}: IDs ({len(batch_ids)}) vs Embeddings ({len(batch_embeddings_list)})"
                        )

                    if actual_embedding_dim is None:
                        for emb in batch_embeddings_list:
                            if emb is not None and isinstance(emb, list):
                                actual_embedding_dim = len(emb)
                                logger.debug(
                                    f"Determined embedding dimension from data: {actual_embedding_dim}"
                                )
                                break

                        if actual_embedding_dim is None:
                            actual_embedding_dim = fallback_dim
                            logger.warning(
                                f"Could not determine dimension from first batch, using fallback: {fallback_dim}"
                            )

                    all_ids.extend(batch_ids)
                    all_embeddings_list.extend(batch_embeddings_list)
                    retrieved_count += len(batch_ids)
                    logger.debug(
                        f"Retrieved {len(batch_ids)} items in this batch. Total retrieved: {retrieved_count}/{count}"
                    )

                except Exception as e_get:
                    raise DatabaseError(
                        f"Error retrieving batch at offset {offset}: {e_get}"
                    ) from e_get

            end_time = time.time()
            logger.info(
                f"Data retrieval finished in {end_time - start_time:.2f}s. Retrieved {retrieved_count} items."
            )

            final_dim = (
                actual_embedding_dim
                if actual_embedding_dim is not None
                else fallback_dim
            )
            valid_ids = []
            valid_embeddings = []

            for i, emb in enumerate(all_embeddings_list):
                if emb is not None and isinstance(emb, list):
                    if len(emb) == final_dim:
                        valid_embeddings.append(emb)
                        valid_ids.append(all_ids[i])
                    else:
                        logger.warning(
                            f"Skipping item ID '{all_ids[i]}' due to dimension mismatch (Expected {final_dim}, Got {len(emb)})."
                        )
                else:
                    logger.warning(
                        f"Skipping item ID '{all_ids[i]}' due to invalid embedding (None or not a list)."
                    )

            if not valid_embeddings:
                logger.warning(
                    "No valid embeddings found after filtering. Returning empty result."
                )
                return ([], np.empty((0, final_dim), dtype=np.float32))

            embeddings_array = np.array(valid_embeddings, dtype=np.float32)
            logger.info(
                f"Returning {len(valid_ids)} valid ID/embedding pairs. Array shape: {embeddings_array.shape}."
            )
            return (valid_ids, embeddings_array)

        except Exception as e:
            logger.error(
                f"Error retrieving all embeddings from collection '{self.collection_name}': {e}",
                exc_info=True,
            )
            raise DatabaseError(f"Error retrieving all embeddings: {e}") from e

    def get_embeddings_by_ids(
        self, ids: List[str]
    ) -> Optional[Dict[str, Optional[List[float]]]]:
        """
        Retrieves specific embeddings from ChromaDB based on their IDs.
        """
        if not self.is_initialized or self.collection is None:
            if not self._ensure_collection_exists() or self.collection is None:
                raise DatabaseError(
                    "Collection not available. Cannot get embeddings by IDs."
                )
        if not isinstance(ids, list):
            raise ValueError("Input 'ids' must be a list of strings.")
        if not ids:
            logger.debug("Received empty ID list for get_embeddings_by_ids.")
            return {}

        logger.info(
            f"Attempting to retrieve embeddings for {len(ids)} IDs from collection '{self.collection_name}'..."
        )
        results_dict: Dict[str, Optional[List[float]]] = {
            id_val: None for id_val in ids
        }

        try:

            retrieved_data = self.collection.get(
                ids=ids, include=["embeddings"]
            )

            if (
                retrieved_data
                and retrieved_data.get("ids")
                and retrieved_data.get("embeddings")
            ):
                retrieved_ids = retrieved_data["ids"]
                retrieved_embeddings = retrieved_data["embeddings"]

                if len(retrieved_ids) != len(retrieved_embeddings):
                    logger.error(
                        f"Mismatch between retrieved IDs ({len(retrieved_ids)}) and embeddings ({len(retrieved_embeddings)}) count."
                    )

                found_count = 0
                for i, r_id in enumerate(retrieved_ids):
                    if r_id in results_dict:

                        embedding = retrieved_embeddings[i]
                        if embedding is not None and isinstance(
                            embedding, list
                        ):
                            results_dict[r_id] = embedding
                            found_count += 1
                        else:
                            logger.warning(
                                f"Invalid embedding format found for ID '{r_id}'. Setting to None."
                            )
                            results_dict[r_id] = None

                logger.info(
                    f"Successfully retrieved embeddings for {found_count}/{len(ids)} requested IDs."
                )

                not_found_ids = [
                    id_val for id_val in ids if id_val not in retrieved_ids
                ]
                if not_found_ids:
                    logger.warning(
                        f"Could not find embeddings for {len(not_found_ids)} IDs: {not_found_ids[:10]}..."
                    )

            else:
                logger.warning(
                    f"ChromaDB get operation for IDs returned no data or unexpected format."
                )

            return results_dict

        except Exception as e:
            logger.error(
                f"Error retrieving embeddings by IDs from collection '{self.collection_name}': {e}",
                exc_info=True,
            )
            raise DatabaseError(
                f"Error retrieving embeddings by IDs: {e}"
            ) from e

    def update_metadata_batch(
        self, ids: List[str], metadatas: List[Dict[str, Any]]
    ) -> bool:
        """
        Updates metadata for multiple items in ChromaDB using a batch operation.
        """
        if not self.is_initialized or self.collection is None:
            if not self._ensure_collection_exists() or self.collection is None:
                raise DatabaseError(
                    "Collection not available. Cannot update metadata."
                )

        if not isinstance(ids, list) or not isinstance(metadatas, list):
            raise ValueError("ids and metadatas must be lists.")
        if not ids or not metadatas:
            logger.info(
                "Empty ids or metadatas list provided to update_metadata_batch. Nothing to do."
            )
            return True
        if len(ids) != len(metadatas):
            raise ValueError(
                f"Input mismatch: IDs ({len(ids)}) vs metadatas ({len(metadatas)})."
            )
        if not all(isinstance(m, dict) for m in metadatas):
            raise ValueError(
                "All items in the metadatas list must be dictionaries."
            )

        num_items_to_update = len(ids)
        logger.info(
            f"Attempting to update metadata for {num_items_to_update} items in collection '{self.collection_name}'..."
        )

        try:

            start_time = time.time()
            self.collection.update(ids=ids, metadatas=metadatas)
            end_time = time.time()
            logger.info(
                f"Metadata update request for {num_items_to_update} items sent successfully in {end_time - start_time:.2f}s."
            )

            return True
        except Exception as e:
            logger.error(
                f"Error during batch metadata update in collection '{self.collection_name}': {e}",
                exc_info=True,
            )

            return False

    def clear_collection(self) -> bool:
        """
        Deletes all items from the ChromaDB collection.
        Uses batch deletion for potentially large collections.
        """
        if not self.is_initialized or self.collection is None:
            if not self._ensure_collection_exists() or self.collection is None:
                raise DatabaseError("Collection not available. Cannot clear.")
        try:
            count_before = self.count()
            if count_before <= 0:
                logger.info(
                    f"Collection '{self.collection_name}' is already empty or inaccessible (count={count_before})."
                )
                return True

            logger.warning(
                f"Attempting to clear ALL {count_before} items from collection '{self.collection_name}'..."
            )
            start_time = time.time()

            ids_to_delete = []
            batch_size_get = 5000
            retrieved_count = 0
            while retrieved_count < count_before:
                offset = retrieved_count
                limit = min(batch_size_get, count_before - retrieved_count)
                if limit <= 0:
                    break
                results = self.collection.get(
                    limit=limit, offset=offset, include=[]
                )
                if not results or not results.get("ids"):
                    logger.warning(
                        f"Could not retrieve IDs at offset {offset} during clear operation. Stopping."
                    )
                    break
                batch_ids = results["ids"]
                if not batch_ids:
                    break
                ids_to_delete.extend(batch_ids)
                retrieved_count += len(batch_ids)

            if not ids_to_delete:
                logger.info(
                    "No IDs found to delete, collection might have been cleared concurrently."
                )
                return self.count() == 0

            logger.info(f"Deleting {len(ids_to_delete)} items by ID...")

            batch_size_delete = 500
            num_delete_batches = math.ceil(
                len(ids_to_delete) / batch_size_delete
            )
            for i in range(num_delete_batches):
                batch_start_index = i * batch_size_delete
                batch_end_index = batch_start_index + batch_size_delete
                batch_del_ids = ids_to_delete[
                    batch_start_index:batch_end_index
                ]
                if batch_del_ids:
                    logger.debug(
                        f"Deleting batch {i+1}/{num_delete_batches} ({len(batch_del_ids)} items)..."
                    )
                    self.collection.delete(ids=batch_del_ids)

            end_time = time.time()
            count_after = self.count()
            if count_after == 0:
                logger.info(
                    f"Collection '{self.collection_name}' cleared successfully in {end_time-start_time:.2f}s."
                )
                return True
            else:

                logger.error(
                    f"Collection count is {count_after} after clearing, expected 0. Potential concurrent additions?"
                )
                return False
        except Exception as e:
            logger.error(
                f"Error clearing collection '{self.collection_name}': {e}",
                exc_info=True,
            )
            raise DatabaseError(
                f"Error clearing collection '{self.collection_name}': {e}"
            ) from e

    def delete_collection(self) -> bool:
        """
        Deletes the entire ChromaDB collection.
        """
        if not self._client_initialized or self.client is None:
            raise DatabaseError(
                "Client not initialized. Cannot delete collection."
            )
        if not self.collection_name:
            raise DatabaseError("Collection name not set.")

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
            except Exception:

                logger.info(
                    f"Verification successful: Collection '{self.collection_name}' no longer exists."
                )
                return True
        except Exception as e:
            logger.error(
                f"Error deleting collection '{self.collection_name}': {e}",
                exc_info=True,
            )
            raise DatabaseError(
                f"Error deleting collection '{self.collection_name}': {e}"
            ) from e

    def count(self) -> int:
        """
        Returns the number of items in the collection.
        """
        if not self.is_initialized or self.collection is None:

            if not self._ensure_collection_exists() or self.collection is None:
                logger.warning(
                    f"Count requested but collection '{self.collection_name}' is not available."
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
