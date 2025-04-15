import logging
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import chromadb
import numpy as np
from chromadb.api.models.Collection import Collection
from chromadb.errors import InvalidDimensionException

from .vector_db_interface import VectorDBInterface
from app.exceptions import DatabaseError
from core.models import SearchResultItem, SearchResults

logger = logging.getLogger(__name__)

# Constante para la clave de metadatos de dimensión
DIMENSION_METADATA_KEY = "embedding_dimension"

class ChromaVectorDB(VectorDBInterface):
    """
    Implementación concreta de VectorDBInterface usando ChromaDB.
    Maneja interacciones con una colección persistente de ChromaDB.
    """

    def __init__(
        self,
        path: str,
        collection_name: str,
        expected_dimension_metadata: Optional[Union[int, str]] = None,
        creation_metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Inicializa el cliente ChromaDB y obtiene/crea la colección.

        Args:
            path: Ruta del directorio para almacenamiento persistente.
            collection_name: Nombre de la colección.
            expected_dimension_metadata: Dimensión esperada ('full' o int) para añadir a metadatos si se crea.
            creation_metadata: Metadatos adicionales a usar al *crear* la colección (ej: {"hnsw:space": "cosine"}).
        """
        self.path = path
        self.collection_name = collection_name
        self._expected_dimension_metadata = expected_dimension_metadata
        self._creation_metadata = creation_metadata or {} # Asegura que sea un dict
        self.client: Optional[chromadb.ClientAPI] = None
        self.collection: Optional[Collection] = None
        self._is_initialized_flag = False
        self._last_init_error: Optional[str] = None

        logger.info(f"Initializing ChromaVectorDB for collection: '{self.collection_name}' at path: '{os.path.abspath(self.path)}'")
        if self._expected_dimension_metadata:
            logger.info(f"  Expected dimension metadata for new collection: {self._expected_dimension_metadata}")
        if self._creation_metadata:
             logger.info(f"  Will use creation metadata if collection is new: {self._creation_metadata}")

        try:
            os.makedirs(self.path, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.path)
            logger.info("ChromaDB PersistentClient initialized.")
            self._ensure_collection_exists()
            self._is_initialized_flag = self.collection is not None
        except Exception as e:
            msg = f"Failed to initialize ChromaDB client or collection '{self.collection_name}': {e}"
            logger.error(msg, exc_info=True)
            self._last_init_error = msg
            self._cleanup_client()

    def _cleanup_client(self):
        """Resetea el estado del cliente en caso de error."""
        self.client = None
        self.collection = None
        self._is_initialized_flag = False
        logger.warning("ChromaDB client state reset due to an error.")

    def _ensure_collection_exists(self) -> bool:
        """
        Verifica la disponibilidad de la colección, creándola si es necesario.
        Añade metadatos de dimensión y otros metadatos de creación si se proporcionaron.

        Returns:
            True si la colección existe o se creó con éxito, False en caso contrario.
        """
        if not self.client:
            logger.error("ChromaDB client not initialized.")
            return False

        if self.collection is not None:
            try:
                # Check if the collection object is still valid by performing a cheap operation
                self.collection.peek(limit=1)
                logger.debug(f"Collection '{self.collection_name}' object seems valid.")
                return True
            except Exception as e:
                logger.warning(f"Collection object check failed for '{self.collection_name}': {e}. Re-fetching...")
                self.collection = None # Force re-fetch

        logger.debug(f"Attempting to get or create collection '{self.collection_name}'...")
        try:
            # Prepare metadata for creation, including the expected dimension
            final_creation_metadata = self._creation_metadata.copy()
            if self._expected_dimension_metadata is not None:
                final_creation_metadata[DIMENSION_METADATA_KEY] = self._expected_dimension_metadata

            # Get or create the collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata=final_creation_metadata if final_creation_metadata else None, # Pass metadata only if it exists
            )
            logger.info(f"Collection '{self.collection_name}' obtained/created successfully.")

            # --- Verification and Potential Update of Metadata ---
            # Check if existing metadata matches the expected/creation metadata
            current_meta = self.collection.metadata or {}
            needs_update = False
            expected_meta_to_check = final_creation_metadata # Metadata we expect or just set

            for key, value in expected_meta_to_check.items():
                 if current_meta.get(key) != value:
                      logger.warning(f"Metadata key '{key}' mismatch/missing for '{self.collection_name}'. Expected '{value}', got '{current_meta.get(key)}'. Attempting update.")
                      needs_update = True
                      break # Found a mismatch, no need to check further

            # If metadata needs update, try to modify it
            if needs_update:
                 try:
                      logger.info(f"Updating metadata for '{self.collection_name}' to ensure consistency: {expected_meta_to_check}")
                      # Merge current metadata with expected/new metadata, prioritizing the latter
                      merged_meta = current_meta.copy()
                      merged_meta.update(expected_meta_to_check)
                      self.collection.modify(metadata=merged_meta)
                      # Re-fetch the collection object to ensure the metadata change is reflected locally
                      self.collection = self.client.get_collection(name=self.collection_name)
                      logger.info(f"Metadata for '{self.collection_name}' updated.")
                 except Exception as mod_err:
                      # Log error but don't necessarily fail initialization if modification fails
                      logger.error(f"Could not modify metadata for '{self.collection_name}': {mod_err}")

            return True # Collection exists or was created
        except Exception as e:
            logger.error(f"Failed to get/create collection '{self.collection_name}': {e}", exc_info=True)
            self.collection = None
            self._last_init_error = str(e)
            return False

    @property
    def is_initialized(self) -> bool:
        """Comprueba si el cliente está inicializado y la colección es accesible."""
        # First check the flag set during init
        if not self._is_initialized_flag:
             return False
        # Then, perform a check to ensure the collection is still accessible
        return self._ensure_collection_exists()

    def get_dimension_from_metadata(self) -> Optional[Union[str, int]]:
        """Obtiene la dimensión de los metadatos de la colección."""
        if not self.is_initialized or self.collection is None:
            logger.warning(f"Cannot get dimension metadata, collection '{self.collection_name}' not available.")
            return None
        try:
            metadata = self.collection.metadata # Fetch current metadata
            if metadata and DIMENSION_METADATA_KEY in metadata:
                dim_value = metadata[DIMENSION_METADATA_KEY]
                logger.debug(f"Found dimension '{dim_value}' in metadata for '{self.collection_name}'.")
                # Attempt to convert to int if possible, otherwise return as is (e.g., 'full')
                try:
                    return int(dim_value)
                except (ValueError, TypeError):
                    return str(dim_value) # Return as string if not an integer
            else:
                logger.debug(f"Dimension metadata key '{DIMENSION_METADATA_KEY}' not found in metadata for '{self.collection_name}'.")
                return None
        except Exception as e:
            logger.warning(f"Error accessing metadata for '{self.collection_name}': {e}")
            return None

    def _check_dimension(self, embeddings: List[List[float]]) -> Optional[int]:
        """Verifica la consistencia de dimensiones en un lote y devuelve la dimensión."""
        if not embeddings:
            return None
        try:
            first_dim = len(embeddings[0])
            if first_dim <= 0:
                raise ValueError("Embeddings cannot have zero dimension.")
            # Check if all embeddings have the same dimension as the first one
            if not all(len(e) == first_dim for e in embeddings[1:]):
                 dims = {len(e) for e in embeddings}
                 raise ValueError(f"Inconsistent embedding dimensions within the batch. Found dimensions: {dims}")
            return first_dim
        except IndexError:
             # This happens if embeddings = [[]]
             raise ValueError("Invalid embedding format: found empty list inside the main list.")
        except TypeError:
             # This happens if elements are not lists or have no len()
             raise ValueError("Invalid embedding format: elements are not lists or cannot determine length.")


    def _validate_collection_dimension(self, batch_dim: Optional[int]):
        """Valida la dimensión del lote contra los metadatos de la colección."""
        if batch_dim is None:
            logger.debug("No batch dimension provided, skipping collection dimension validation.")
            return # Nothing to validate if batch dim is unknown

        expected_dim_meta = self.get_dimension_from_metadata()

        if expected_dim_meta is not None:
            # Case 1: Metadata has a specific integer dimension
            if isinstance(expected_dim_meta, int):
                if batch_dim != expected_dim_meta:
                    raise InvalidDimensionException(
                        f"Dimension mismatch! Batch dim: {batch_dim}, Collection '{self.collection_name}' expects: {expected_dim_meta} based on metadata."
                    )
                else:
                    # Dimensions match, all good
                    logger.debug(f"Batch dimension {batch_dim} matches collection metadata dimension for '{self.collection_name}'.")
            # Case 2: Metadata indicates 'full' dimension
            elif expected_dim_meta == "full":
                 logger.debug(f"Collection '{self.collection_name}' expects 'full' dimension. Skipping specific dimension check for batch dim {batch_dim}.")
            # Case 3: Metadata has an unexpected string value
            elif isinstance(expected_dim_meta, str):
                 logger.warning(f"Unexpected dimension metadata value '{expected_dim_meta}' for collection '{self.collection_name}'. Cannot validate batch dimension {batch_dim}.")
        # Case 4: No dimension metadata found, but collection has items
        elif self.count() > 0:
             # Attempt to infer dimension from existing data as a fallback check
             try:
                  sample = self.collection.peek(limit=1)
                  if sample and sample.get('embeddings') and sample['embeddings']:
                       inferred_dim = len(sample['embeddings'][0])
                       if batch_dim != inferred_dim:
                            # Raise error if batch dim doesn't match inferred dim
                            raise InvalidDimensionException(
                                f"Dimension mismatch! Batch dim: {batch_dim}, Collection '{self.collection_name}' seems to have dimension {inferred_dim} based on existing data (no metadata found)."
                            )
                       else:
                            # Batch dim matches inferred dim
                            logger.info(f"Validated batch dimension {batch_dim} against inferred dimension for '{self.collection_name}'. Consider adding dimension to collection metadata.")
                  else:
                       logger.warning(f"Could not retrieve sample embedding to infer dimension for '{self.collection_name}' (no metadata found).")
             except Exception as e:
                  logger.warning(f"Could not infer dimension from existing data in '{self.collection_name}': {e}")
        # Case 5: No metadata and collection is empty
        else:
             logger.info(f"Collection '{self.collection_name}' is empty and has no dimension metadata. Allowing add operation with batch dimension {batch_dim}. Consider setting metadata upon creation.")


    def add_embeddings(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """Añade o actualiza embeddings."""
        if not self.is_initialized or self.collection is None:
            raise DatabaseError(f"Collection '{self.collection_name}' not available for add_embeddings.")
        if not isinstance(ids, list) or not isinstance(embeddings, list):
             raise ValueError("Inputs 'ids' and 'embeddings' must be lists.")
        if metadatas is not None and not isinstance(metadatas, list):
             raise ValueError("Input 'metadatas' must be a list if provided.")

        if len(ids) != len(embeddings) or (metadatas and len(ids) != len(metadatas)):
            raise ValueError(f"Input list lengths mismatch: ids({len(ids)}), embeddings({len(embeddings)}), metadatas({len(metadatas) if metadatas else 'N/A'}).")
        if not ids:
            logger.debug("add_embeddings called with empty lists. Nothing to add.")
            return True

        try:
            # Check dimensions within the batch first
            batch_dim = self._check_dimension(embeddings)
            # Validate batch dimension against collection metadata/data
            self._validate_collection_dimension(batch_dim)

            logger.info(f"Upserting {len(ids)} items into '{self.collection_name}' (Batch Dim: {batch_dim or 'N/A'})...")
            # Ensure metadatas is a list of dicts, even if empty
            processed_metadatas = metadatas if metadatas is not None else [{}] * len(ids)
            # Filter out None metadatas just in case, replace with empty dict
            processed_metadatas = [md if isinstance(md, dict) else {} for md in processed_metadatas]

            self.collection.upsert(ids=ids, embeddings=embeddings, metadatas=processed_metadatas)
            logger.debug(f"Upsert completed for {len(ids)} items in '{self.collection_name}'.")
            return True
        except (InvalidDimensionException, ValueError) as e:
            logger.error(f"Data validation error during upsert into '{self.collection_name}': {e}")
            # Do not raise DatabaseError here, let the specific error propagate if needed
            raise # Re-raise the specific validation error
        except Exception as e:
            logger.error(f"Unexpected error during upsert into '{self.collection_name}': {e}", exc_info=True)
            self._is_initialized_flag = False # Mark as potentially broken
            raise DatabaseError(f"Unexpected upsert error: {e}") from e

    def query_similar(
        self, query_embedding: List[float], n_results: int = 5
    ) -> Optional[SearchResults]:
        """Consulta embeddings similares."""
        if not self.is_initialized or self.collection is None:
            raise DatabaseError(f"Collection '{self.collection_name}' not available for query.")
        if not query_embedding or not isinstance(query_embedding, list):
             raise ValueError("Query embedding must be a non-empty list.")
        if n_results <= 0: raise ValueError("n_results must be positive.")

        query_dim = len(query_embedding)
        try:
             # Validate the query dimension against the collection
             self._validate_collection_dimension(query_dim)
        except InvalidDimensionException as e:
             logger.error(f"Query dimension validation failed for '{self.collection_name}': {e}")
             raise DatabaseError(f"Query dimension mismatch: {e}") from e
        except Exception as e:
             logger.error(f"Unexpected error during query dimension validation for '{self.collection_name}': {e}", exc_info=True)
             raise DatabaseError(f"Query dimension validation failed unexpectedly: {e}") from e


        collection_count = self.count()
        if collection_count < 0: # Check for error from count()
             logger.error(f"Could not get count for collection '{self.collection_name}'. Aborting query.")
             raise DatabaseError(f"Failed to get collection count for '{self.collection_name}'.")
        if collection_count == 0:
            logger.warning(f"Querying empty collection '{self.collection_name}'. Returning empty results.")
            return SearchResults(items=[])

        # Adjust n_results if it exceeds the number of items in the collection
        effective_n_results = min(n_results, collection_count)
        if effective_n_results <= 0: # Should not happen if count > 0, but safety check
             logger.warning(f"Effective n_results is {effective_n_results} (requested: {n_results}, count: {collection_count}). Returning empty results.")
             return SearchResults(items=[])

        logger.info(f"Querying '{self.collection_name}' for {effective_n_results} neighbors (Query Dim: {query_dim})...")
        try:
            start_time = time.time()
            # Perform the query
            results_dict = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=effective_n_results,
                include=["metadatas", "distances"], # Request metadata and distances
            )
            query_duration = time.time() - start_time
            logger.debug(f"ChromaDB query on '{self.collection_name}' took {query_duration:.3f}s.")

            result_items = []
            # Process results if the dictionary structure is valid
            if (results_dict and
                results_dict.get("ids") and isinstance(results_dict["ids"], list) and results_dict["ids"] and
                results_dict.get("distances") and isinstance(results_dict["distances"], list) and results_dict["distances"] and
                results_dict.get("metadatas") and isinstance(results_dict["metadatas"], list) and results_dict["metadatas"]):

                ids_list = results_dict["ids"][0] # Results are nested in a list
                distances_list = results_dict["distances"][0]
                metadatas_list = results_dict["metadatas"][0]

                # Basic check for consistent lengths
                if not (len(ids_list) == len(distances_list) == len(metadatas_list)):
                     logger.warning(f"Query result lists length mismatch in '{self.collection_name}'! "
                                    f"IDs: {len(ids_list)}, Distances: {len(distances_list)}, Metadatas: {len(metadatas_list)}. "
                                    f"Processing minimum common length.")
                     min_len = min(len(ids_list), len(distances_list), len(metadatas_list))
                     ids_list, distances_list, metadatas_list = ids_list[:min_len], distances_list[:min_len], metadatas_list[:min_len]

                # Create SearchResultItem objects
                for i, img_id in enumerate(ids_list):
                    # Ensure metadata is a dict, fallback to empty dict
                    metadata = metadatas_list[i] if isinstance(metadatas_list[i], dict) else {}
                    # Ensure distance is float or None
                    distance = float(distances_list[i]) if distances_list[i] is not None else None

                    item = SearchResultItem(
                        id=str(img_id), # Ensure ID is string
                        distance=distance,
                        metadata=metadata
                    )
                    result_items.append(item)
            else:
                 logger.info(f"Query on '{self.collection_name}' returned no results or unexpected format: {results_dict}")


            logger.info(f"Query on '{self.collection_name}' processed {len(result_items)} results.")
            return SearchResults(items=result_items, query_vector=query_embedding)

        except InvalidDimensionException as e_dim:
            # Catch dimension errors specifically during query
            logger.error(f"ChromaDB query error on '{self.collection_name}': Invalid dimension. Query Dim: {query_dim}. Details: {e_dim}")
            raise DatabaseError(f"Query failed due to dimension mismatch: {e_dim}") from e_dim
        except Exception as e:
            # Catch other unexpected errors during query
            logger.error(f"Unexpected error during query on '{self.collection_name}': {e}", exc_info=True)
            self._is_initialized_flag = False # Mark as potentially broken
            raise DatabaseError(f"Query failed unexpectedly: {e}") from e

    def get_all_embeddings_with_ids(
        self, pagination_batch_size: int = 1000 # Note: Chroma's get() might not truly paginate this way
    ) -> Optional[Tuple[List[str], np.ndarray]]:
        """Obtiene todos los embeddings y IDs. CORREGIDO para manejar tipos."""
        if not self.is_initialized or self.collection is None:
            raise DatabaseError(f"Collection '{self.collection_name}' not available for get_all_embeddings.")

        try:
            count = self.count()
            # Determine a fallback dimension for the empty array if needed
            fallback_dim = 1 # Default fallback
            meta_dim = self.get_dimension_from_metadata()
            if isinstance(meta_dim, int) and meta_dim > 0:
                fallback_dim = meta_dim
            elif meta_dim == "full":
                 logger.warning(f"Collection '{self.collection_name}' metadata indicates 'full' dimension. Cannot determine exact dimension for empty array fallback. Using {fallback_dim}.")

            if count < 0: # Error getting count
                 logger.error(f"Failed to get count for '{self.collection_name}'. Cannot retrieve all embeddings.")
                 raise DatabaseError(f"Failed to get count for '{self.collection_name}'.")
            if count == 0:
                logger.info(f"Collection '{self.collection_name}' is empty. Returning empty arrays with fallback dim {fallback_dim}.")
                # Return empty list and empty numpy array with the determined/fallback dimension
                return ([], np.empty((0, fallback_dim), dtype=np.float32))

            logger.info(f"Retrieving all {count} embeddings/IDs from '{self.collection_name}' using collection.get()...")
            start_time = time.time()
            # Use collection.get() to retrieve all data. Warning: potentially memory-intensive!
            # Chroma's `get` might load everything into memory, pagination_batch_size is ignored here.
            results = self.collection.get(include=["embeddings"]) # Only need embeddings

            if results is None or results.get("ids") is None or results.get("embeddings") is None:
                 logger.error(f"Failed to retrieve valid data with collection.get() from '{self.collection_name}'. Result was None or missing keys.")
                 raise DatabaseError("Failed to retrieve data with collection.get()")

            all_ids = results["ids"]
            all_embeddings_raw = results["embeddings"]

            # Validate consistency between retrieved IDs and embeddings
            if len(all_ids) != len(all_embeddings_raw):
                 logger.error(f"Mismatch in retrieved data from '{self.collection_name}': {len(all_ids)} IDs vs {len(all_embeddings_raw)} embeddings.")
                 raise DatabaseError(f"Mismatch in retrieved data: {len(all_ids)} IDs vs {len(all_embeddings_raw)} embeddings.")

            valid_ids = []
            valid_embeddings_list = []
            final_dim = None # Dimension determined from the first valid embedding

            logger.debug(f"Starting validation and conversion for {len(all_embeddings_raw)} raw embeddings...")
            for i, emb_raw in enumerate(all_embeddings_raw):
                current_id = all_ids[i]
                embedding_list: Optional[List[float]] = None # Variable to hold the list version

                # --- START: FIX FOR NUMPY ARRAY / LIST HANDLING ---
                if isinstance(emb_raw, list):
                    embedding_list = emb_raw # Already a list
                elif isinstance(emb_raw, np.ndarray):
                    try:
                        embedding_list = emb_raw.tolist() # Convert numpy array to list
                        # logger.debug(f"Converted numpy array to list for ID '{current_id}'") # Optional: too verbose?
                    except Exception as np_conv_err:
                        logger.warning(f"Skipping ID '{current_id}' in '{self.collection_name}' - failed to convert numpy array to list: {np_conv_err}")
                        continue # Skip this item if conversion fails
                else:
                    # If it's neither list nor ndarray (or None)
                    logger.warning(f"Skipping ID '{current_id}' in '{self.collection_name}' due to unexpected embedding type ({type(emb_raw)}). Expected list or numpy array.")
                    continue # Skip this item
                # --- END: FIX FOR NUMPY ARRAY / LIST HANDLING ---

                # Now validate the resulting embedding_list
                if embedding_list is None or not embedding_list: # Check if None or empty
                    logger.warning(f"Skipping ID '{current_id}' in '{self.collection_name}' due to None or empty embedding list after conversion checks.")
                    continue

                current_len = len(embedding_list)
                if current_len <= 0:
                    logger.warning(f"Skipping ID '{current_id}' in '{self.collection_name}' due to zero length embedding.")
                    continue

                # Determine the expected dimension from the first valid item
                if final_dim is None:
                    final_dim = current_len
                    logger.info(f"Determined embedding dimension from first valid item in '{self.collection_name}': {final_dim}")
                    # Optional: Cross-check with metadata dimension if available
                    meta_dim_check = self.get_dimension_from_metadata()
                    if isinstance(meta_dim_check, int) and meta_dim_check != final_dim:
                         logger.error(f"CRITICAL MISMATCH in '{self.collection_name}': Dimension from data ({final_dim}) differs from metadata ({meta_dim_check})!")
                         # Decide how to handle: raise error, trust data, trust metadata? Here we log error and trust data.

                # Check if the current embedding matches the determined dimension
                if current_len == final_dim:
                    try:
                        # Ensure all elements are floats
                        valid_embeddings_list.append(list(map(float, embedding_list)))
                        valid_ids.append(str(current_id)) # Ensure ID is string
                    except (ValueError, TypeError) as convert_err:
                         logger.warning(f"Skipping ID '{current_id}' in '{self.collection_name}' due to float conversion error: {convert_err}")
                else:
                    # Log dimension mismatch for this specific item
                    logger.warning(f"Skipping ID '{current_id}' in '{self.collection_name}' due to dimension mismatch (Expected {final_dim}, Got {current_len}).")

            # After processing all items
            if not valid_embeddings_list:
                # If no valid embeddings were found after filtering
                logger.error(f"No valid embeddings found after filtering in '{self.collection_name}'! Check data integrity or filtering logic.")
                # Return empty list and empty numpy array with the determined/fallback dimension
                return ([], np.empty((0, final_dim if final_dim else fallback_dim), dtype=np.float32))

            # Convert the list of valid embeddings to a NumPy array
            embeddings_array = np.array(valid_embeddings_list, dtype=np.float32)
            retrieval_duration = time.time() - start_time
            logger.info(f"Retrieved and validated {len(valid_ids)} items from '{self.collection_name}' in {retrieval_duration:.2f}s. Final array shape: {embeddings_array.shape}")
            return (valid_ids, embeddings_array)

        except Exception as e:
            logger.error(f"Error retrieving all embeddings from '{self.collection_name}': {e}", exc_info=True)
            self._is_initialized_flag = False # Mark as potentially broken
            raise DatabaseError(f"Failed to get all embeddings: {e}") from e

    def get_embeddings_by_ids(
        self, ids: List[str]
    ) -> Optional[Dict[str, Optional[List[float]]]]:
        """Obtiene embeddings específicos por ID."""
        if not self.is_initialized or self.collection is None:
            raise DatabaseError(f"Collection '{self.collection_name}' not available for get_embeddings_by_ids.")
        if not ids:
            logger.debug("get_embeddings_by_ids called with empty ID list.")
            return {}
        if not isinstance(ids, list):
             raise ValueError("Input 'ids' must be a list.")

        # Ensure all IDs are strings
        str_ids = [str(id_val) for id_val in ids]

        logger.info(f"Retrieving embeddings for {len(str_ids)} IDs from '{self.collection_name}'...")
        try:
            # Retrieve data for the specified IDs
            retrieved_data = self.collection.get(ids=str_ids, include=["embeddings"])
            # Initialize results dictionary with None for all requested IDs
            results_dict: Dict[str, Optional[List[float]]] = {id_val: None for id_val in str_ids}

            # Process retrieved data if valid
            if retrieved_data is not None and retrieved_data.get("ids") is not None and retrieved_data.get("embeddings") is not None:
                retrieved_ids = retrieved_data["ids"]
                retrieved_embeddings = retrieved_data["embeddings"]
                # Create a map for quick lookup
                retrieved_map = {r_id: emb for r_id, emb in zip(retrieved_ids, retrieved_embeddings)}

                # Populate the results_dict with found embeddings
                for req_id in str_ids:
                    if req_id in retrieved_map:
                        embedding_raw = retrieved_map[req_id]
                        embedding_list: Optional[List[float]] = None

                        # --- START: FIX FOR NUMPY ARRAY / LIST HANDLING ---
                        if isinstance(embedding_raw, list):
                            embedding_list = embedding_raw
                        elif isinstance(embedding_raw, np.ndarray):
                             try:
                                 embedding_list = embedding_raw.tolist()
                             except Exception: pass # Ignore conversion error here, will be caught below
                        # --- END: FIX FOR NUMPY ARRAY / LIST HANDLING ---

                        # Validate and convert to float list
                        if isinstance(embedding_list, list) and embedding_list:
                            try:
                                results_dict[req_id] = list(map(float, embedding_list))
                            except (ValueError, TypeError):
                                logger.warning(f"Could not convert embedding for ID '{req_id}' in '{self.collection_name}'. Setting to None.")
                                # results_dict[req_id] remains None (default)
                        else:
                            logger.warning(f"Invalid, empty, or non-convertible embedding found for ID '{req_id}' in '{self.collection_name}'. Setting to None.")
                            # results_dict[req_id] remains None (default)

            else:
                 logger.warning(f"ChromaDB get by IDs returned no data or unexpected format for '{self.collection_name}'. All requested IDs will have None embeddings.")

            found_count = sum(1 for v in results_dict.values() if v is not None)
            logger.info(f"Retrieved {found_count}/{len(str_ids)} requested embeddings from '{self.collection_name}'.")
            return results_dict

        except Exception as e:
            logger.error(f"Error retrieving embeddings by IDs from '{self.collection_name}': {e}", exc_info=True)
            self._is_initialized_flag = False # Mark as potentially broken
            raise DatabaseError(f"Failed to get embeddings by IDs: {e}") from e

    def update_metadata_batch(
        self, ids: List[str], metadatas: List[Dict[str, Any]]
    ) -> bool:
        """Actualiza metadatos por lotes."""
        if not self.is_initialized or self.collection is None:
            raise DatabaseError(f"Collection '{self.collection_name}' not available for update_metadata.")
        if not isinstance(ids, list) or not isinstance(metadatas, list):
             raise ValueError("Inputs 'ids' and 'metadatas' must be lists.")
        if len(ids) != len(metadatas):
             raise ValueError(f"Input list lengths mismatch for update_metadata: ids({len(ids)}), metadatas({len(metadatas)}).")
        if not ids:
            logger.debug("update_metadata_batch called with empty lists. Nothing to update.")
            return True

        # Ensure IDs are strings and metadatas are dicts
        str_ids = [str(id_val) for id_val in ids]
        valid_metadatas = [md if isinstance(md, dict) else {} for md in metadatas]

        logger.info(f"Updating metadata for {len(str_ids)} items in '{self.collection_name}'...")
        try:
            # Use collection.update for metadata changes
            self.collection.update(ids=str_ids, metadatas=valid_metadatas)
            logger.debug(f"Metadata update request sent for {len(str_ids)} items in '{self.collection_name}'.")
            return True
        except Exception as e:
            logger.error(f"Error during batch metadata update in '{self.collection_name}': {e}", exc_info=True)
            # Consider if this should mark the DB as uninitialized
            # self._is_initialized_flag = False
            # Return False to indicate failure
            return False

    def clear_collection(self) -> bool:
        """Elimina todos los items de la colección."""
        # Ensure collection exists before trying to clear
        if not self.is_initialized:
            # Attempt to re-initialize if needed
            if not self._ensure_collection_exists():
                 raise DatabaseError(f"Collection '{self.collection_name}' not available for clear_collection.")
            if self.collection is None: # Should not happen if _ensure_collection_exists passed
                 raise DatabaseError(f"Collection object is None after ensure_collection_exists for '{self.collection_name}'. Cannot clear.")

        # Now self.collection should be valid if we reached here
        if self.collection is None: # Final safety check
             raise DatabaseError(f"Collection object is None for '{self.collection_name}'. Cannot clear.")

        try:
            count_before = self.count()
            if count_before < 0: # Error getting count
                logger.error(f"Cannot clear collection '{self.collection_name}' because count failed.")
                return False
            if count_before == 0:
                logger.info(f"Collection '{self.collection_name}' is already empty.")
                return True

            logger.warning(f"Clearing ALL {count_before} items from collection '{self.collection_name}'...")
            start_time = time.time()

            # ChromaDB's delete method can accept a list of IDs.
            # Fetching all IDs first might be memory intensive for huge collections.
            # A more robust approach for very large collections might involve batch deletion,
            # but ChromaDB's API for that isn't as straightforward as `clear()`.
            # Let's try deleting all retrieved IDs.
            try:
                # Get all IDs (potentially memory intensive)
                get_result = self.collection.get(limit=count_before, include=[]) # Don't need embeddings
                all_ids_to_delete = get_result.get('ids')

                if all_ids_to_delete:
                    logger.info(f"Deleting {len(all_ids_to_delete)} items by ID from '{self.collection_name}'...")
                    # Delete in batches if the list is very large? Chroma might handle this internally.
                    # For simplicity, delete all at once for now.
                    self.collection.delete(ids=all_ids_to_delete)
                else:
                    # This case should ideally not happen if count_before > 0
                    logger.warning(f"Count was {count_before} but no IDs retrieved via get() during clear operation for '{self.collection_name}'. Collection might be inconsistent.")
                    # Attempt delete without IDs (may not work, depends on Chroma version/backend)
                    # self.collection.delete() # Potentially dangerous or ineffective, keep commented unless necessary
                    logger.warning("Attempting delete without specific IDs might fail or is disabled.")
                    # Since we couldn't get IDs, we can't confirm deletion this way.

            except Exception as del_e:
                 logger.error(f"Error during delete operation within clear_collection for '{self.collection_name}': {del_e}", exc_info=True)
                 # Don't immediately raise, check count afterwards
                 # return False # Indicate failure

            # Verify by counting again
            count_after = self.count()
            success = count_after == 0
            clear_duration = time.time() - start_time

            if success:
                logger.info(f"Collection '{self.collection_name}' cleared successfully in {clear_duration:.2f}s.")
            else:
                logger.error(f"Failed to fully clear collection '{self.collection_name}'. Count before: {count_before}, Count after: {count_after}. Duration: {clear_duration:.2f}s.")
            return success
        except Exception as e:
            logger.error(f"Error clearing collection '{self.collection_name}': {e}", exc_info=True)
            self._is_initialized_flag = False # Mark as potentially broken
            raise DatabaseError(f"Failed to clear collection: {e}") from e

    def delete_collection(self) -> bool:
        """Elimina toda la colección."""
        if not self.client: raise DatabaseError("Client not initialized, cannot delete collection.")
        if not self.collection_name: raise DatabaseError("Collection name not set, cannot delete.")

        logger.warning(f"Deleting ENTIRE collection '{self.collection_name}' from path '{self.path}'...")
        collection_name_to_delete = self.collection_name
        try:
            self.client.delete_collection(name=collection_name_to_delete)
            # Reset local state
            self.collection = None
            self._is_initialized_flag = False
            # Note: We don't handle external caches (like in factory.py) here.
            # That should be managed by the code using this class.
            logger.info(f"Collection '{collection_name_to_delete}' deleted successfully via client.")
            return True
        except chromadb.errors.NotEnoughElementsException:
             # This can happen if the collection didn't exist
             logger.warning(f"Collection '{collection_name_to_delete}' did not exist or was already deleted. Considering delete successful.")
             self.collection = None
             self._is_initialized_flag = False
             return True
        except Exception as e:
            # Check if the error message indicates the collection doesn't exist
            # This error handling might need adjustment based on exact ChromaDB exceptions
            if "does not exist" in str(e).lower() or "CollectionNotFound" in str(type(e)):
                 logger.warning(f"Collection '{collection_name_to_delete}' did not exist. Considering delete successful.")
                 self.collection = None
                 self._is_initialized_flag = False
                 return True
            # Log other errors and raise DatabaseError
            logger.error(f"Error deleting collection '{collection_name_to_delete}': {e}", exc_info=True)
            raise DatabaseError(f"Failed to delete collection: {e}") from e

    def count(self) -> int:
        """Devuelve el número de items."""
        # Ensure collection exists before trying to count
        if not self.is_initialized:
            # Attempt to re-initialize if needed
            if not self._ensure_collection_exists():
                 logger.warning(f"Count requested but collection '{self.collection_name}' is not available.")
                 return -1 # Return -1 to indicate error
            if self.collection is None: # Should not happen if _ensure_collection_exists passed
                 logger.error(f"Collection object is None after ensure_collection_exists for '{self.collection_name}'. Cannot count.")
                 return -1

        # Now self.collection should be valid
        if self.collection is None: # Final safety check
             logger.error(f"Collection object is None for '{self.collection_name}'. Cannot count.")
             return -1
        try:
            item_count = self.collection.count()
            logger.debug(f"Collection '{self.collection_name}' count: {item_count}")
            return item_count
        except Exception as e:
            logger.error(f"Error getting count for '{self.collection_name}': {e}")
            # If count fails, assume the collection might be broken
            self.collection = None
            self._is_initialized_flag = False
            return -1 # Return -1 to indicate error

# --- Fin data_access/chroma_db.py ---

