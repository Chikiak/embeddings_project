# --- data_access/chroma_db.py ---
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
from app.models import SearchResultItem, SearchResults

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
    ):
        """
        Inicializa el cliente ChromaDB y obtiene/crea la colección.

        Args:
            path: Ruta del directorio para almacenamiento persistente.
            collection_name: Nombre de la colección.
            expected_dimension_metadata: Dimensión esperada ('full' o int) para añadir a metadatos si se crea.
        """
        self.path = path
        self.collection_name = collection_name
        self._expected_dimension_metadata = expected_dimension_metadata
        self.client: Optional[chromadb.ClientAPI] = None
        self.collection: Optional[Collection] = None
        self._is_initialized_flag = False
        self._last_init_error: Optional[str] = None

        logger.info(f"Initializing ChromaVectorDB for collection: '{self.collection_name}' at path: '{os.path.abspath(self.path)}'")
        if self._expected_dimension_metadata:
            logger.info(f"  Expected dimension metadata for new collection: {self._expected_dimension_metadata}")

        try:
            os.makedirs(self.path, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.path)
            logger.info("ChromaDB PersistentClient initialized.")
            # Intenta obtener/crear la colección inmediatamente
            self._ensure_collection_exists()
            self._is_initialized_flag = self.collection is not None
        except Exception as e:
            msg = f"Failed to initialize ChromaDB client or collection '{self.collection_name}': {e}"
            logger.error(msg, exc_info=True)
            self._last_init_error = msg
            self._cleanup_client()
            # No relanzar aquí, is_initialized devolverá False

    def _cleanup_client(self):
        """Resetea el estado del cliente en caso de error."""
        self.client = None
        self.collection = None
        self._is_initialized_flag = False
        logger.warning("ChromaDB client state reset due to an error.")

    def _ensure_collection_exists(self) -> bool:
        """
        Verifica la disponibilidad de la colección, creándola si es necesario.
        Añade metadatos de dimensión al crearla si se proporcionaron.

        Returns:
            True si la colección existe o se creó con éxito, False en caso contrario.
        """
        if not self.client:
            logger.error("ChromaDB client not initialized.")
            return False

        if self.collection is not None:
            # Verifica si el objeto colección sigue siendo válido (simple check)
            try:
                self.collection.peek(limit=1)
                return True
            except Exception as e:
                logger.warning(f"Collection object check failed for '{self.collection_name}': {e}. Re-fetching...")
                self.collection = None # Fuerza re-fetch

        logger.debug(f"Attempting to get or create collection '{self.collection_name}'...")
        try:
            creation_metadata = {}
            if self._expected_dimension_metadata is not None:
                creation_metadata[DIMENSION_METADATA_KEY] = self._expected_dimension_metadata

            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata=creation_metadata if creation_metadata else None, # Solo pasar si no está vacío
            )
            logger.info(f"Collection '{self.collection_name}' obtained/created successfully.")

            # Verifica y actualiza metadatos si es necesario (simplificado)
            current_meta = self.collection.metadata
            if self._expected_dimension_metadata is not None and \
               (current_meta is None or current_meta.get(DIMENSION_METADATA_KEY) != self._expected_dimension_metadata):
                logger.info(f"Updating dimension metadata for '{self.collection_name}' to '{self._expected_dimension_metadata}'.")
                try:
                    new_metadata = current_meta.copy() if current_meta else {}
                    new_metadata[DIMENSION_METADATA_KEY] = self._expected_dimension_metadata
                    self.collection.modify(metadata=new_metadata)
                    # Re-fetch collection object after modification might be needed depending on Chroma version behavior
                    self.collection = self.client.get_collection(name=self.collection_name)
                except Exception as mod_err:
                    logger.warning(f"Could not modify metadata for '{self.collection_name}': {mod_err}")
                    # Continue, but metadata might be inconsistent

            return True
        except Exception as e:
            logger.error(f"Failed to get/create collection '{self.collection_name}': {e}", exc_info=True)
            self.collection = None
            self._last_init_error = str(e)
            return False

    @property
    def is_initialized(self) -> bool:
        """Comprueba si el cliente está inicializado y la colección es accesible."""
        # Re-check collection status each time for robustness
        return self._is_initialized_flag and self._ensure_collection_exists()

    def get_dimension_from_metadata(self) -> Optional[Union[str, int]]:
        """Obtiene la dimensión de los metadatos de la colección."""
        if not self.is_initialized or self.collection is None:
            return None
        try:
            metadata = self.collection.metadata
            if metadata and DIMENSION_METADATA_KEY in metadata:
                dim_value = metadata[DIMENSION_METADATA_KEY]
                # Simplificado: Asume que el valor almacenado es correcto si existe
                logger.debug(f"Found dimension '{dim_value}' in metadata for '{self.collection_name}'.")
                return dim_value
            return None
        except Exception as e:
            logger.warning(f"Error accessing metadata for '{self.collection_name}': {e}")
            return None

    def _check_dimension(self, embeddings: List[List[float]]) -> Optional[int]:
        """Verifica la consistencia de dimensiones en un lote y devuelve la dimensión."""
        if not embeddings:
            return None
        first_dim = len(embeddings[0])
        if first_dim <= 0:
            raise ValueError("Embeddings cannot have zero dimension.")
        if not all(len(e) == first_dim for e in embeddings):
            raise ValueError("Inconsistent embedding dimensions within the batch.")
        return first_dim

    def _validate_collection_dimension(self, batch_dim: Optional[int]):
        """Valida la dimensión del lote contra los metadatos de la colección."""
        if batch_dim is None: return # No hay datos para validar

        expected_dim_meta = self.get_dimension_from_metadata()
        # Solo valida si la colección ya tiene datos y metadatos definidos
        if expected_dim_meta is not None and self.count() > 0:
            if isinstance(expected_dim_meta, int) and batch_dim != expected_dim_meta:
                raise DatabaseError(
                    f"Dimension mismatch! Batch dim: {batch_dim}, Collection '{self.collection_name}' expects: {expected_dim_meta}."
                )
            # Si expected_dim_meta es 'full', no hay validación explícita aquí

    def add_embeddings(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """Añade o actualiza embeddings."""
        if not self.is_initialized or self.collection is None:
            raise DatabaseError("Collection not available.")
        if len(ids) != len(embeddings) or (metadatas and len(ids) != len(metadatas)):
            raise ValueError("Input list lengths mismatch.")
        if not ids: return True # Nada que añadir

        try:
            batch_dim = self._check_dimension(embeddings)
            self._validate_collection_dimension(batch_dim) # Valida antes de upsert

            logger.info(f"Upserting {len(ids)} items into '{self.collection_name}' (Batch Dim: {batch_dim or 'N/A'})...")
            self.collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)
            logger.debug(f"Upsert completed for {len(ids)} items.")
            return True
        except (InvalidDimensionException, ValueError, DatabaseError) as e:
            logger.error(f"Error upserting into '{self.collection_name}': {e}")
            raise DatabaseError(f"Upsert failed: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error during upsert into '{self.collection_name}': {e}", exc_info=True)
            raise DatabaseError(f"Unexpected upsert error: {e}") from e

    def query_similar(
        self, query_embedding: List[float], n_results: int = 5
    ) -> Optional[SearchResults]:
        """Consulta embeddings similares."""
        if not self.is_initialized or self.collection is None:
            raise DatabaseError("Collection not available.")
        if not query_embedding: raise ValueError("Query embedding cannot be empty.")

        query_dim = len(query_embedding)
        # Valida dimensión de la consulta (simplificado)
        try:
             self._validate_collection_dimension(query_dim)
        except DatabaseError as e:
             logger.error(f"Query dimension validation failed: {e}")
             raise

        collection_count = self.count()
        if collection_count <= 0:
            logger.warning(f"Querying empty collection '{self.collection_name}'.")
            return SearchResults(items=[])

        effective_n_results = min(n_results, collection_count)
        if effective_n_results <= 0: return SearchResults(items=[])

        logger.info(f"Querying '{self.collection_name}' for {effective_n_results} neighbors (Query Dim: {query_dim})...")
        try:
            start_time = time.time()
            results_dict = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=effective_n_results,
                include=["metadatas", "distances"],
            )
            logger.debug(f"ChromaDB query took {time.time() - start_time:.3f}s.")

            result_items = []
            # Simplificado: asume que el formato es correcto si no es None
            if results_dict and results_dict.get("ids") and results_dict["ids"][0]:
                ids_list = results_dict["ids"][0]
                distances_list = results_dict.get("distances", [[]])[0]
                metadatas_list = results_dict.get("metadatas", [[]])[0]

                # Basic length check
                if len(ids_list) == len(distances_list) == len(metadatas_list):
                    for i, img_id in enumerate(ids_list):
                        item = SearchResultItem(
                            id=img_id,
                            distance=distances_list[i],
                            metadata=metadatas_list[i] or {}
                        )
                        result_items.append(item)
                else:
                     logger.warning("Query result lists length mismatch, returning potentially partial results.")
                     # Intenta construir con lo que se pueda
                     min_len = min(len(ids_list), len(distances_list), len(metadatas_list))
                     for i in range(min_len):
                         item = SearchResultItem(id=ids_list[i], distance=distances_list[i], metadata=metadatas_list[i] or {})
                         result_items.append(item)


            logger.info(f"Query found {len(result_items)} results.")
            return SearchResults(items=result_items, query_vector=query_embedding)

        except InvalidDimensionException as e_dim:
            raise DatabaseError(f"ChromaDB query error: Invalid dimension. Query Dim: {query_dim}. Details: {e_dim}") from e_dim
        except Exception as e:
            logger.error(f"Error during query on '{self.collection_name}': {e}", exc_info=True)
            raise DatabaseError(f"Query failed: {e}") from e

    def get_all_embeddings_with_ids(
        self, pagination_batch_size: int = 1000 # Argumento no usado si get obtiene todo
    ) -> Optional[Tuple[List[str], np.ndarray]]:
        """Obtiene todos los embeddings y IDs. CORREGIDO para manejar tipos."""
        if not self.is_initialized or self.collection is None:
            raise DatabaseError("Collection not available.")

        try:
            count = self.count()
            # Determina una dimensión de fallback si la colección está vacía
            fallback_dim = 1
            meta_dim = self.get_dimension_from_metadata()
            if isinstance(meta_dim, int):
                fallback_dim = meta_dim

            if count <= 0:
                logger.info(f"Collection '{self.collection_name}' is empty. Returning empty arrays.")
                return ([], np.empty((0, fallback_dim), dtype=np.float32))

            logger.info(f"Retrieving all {count} embeddings/IDs from '{self.collection_name}'...")
            start_time = time.time()
            # Obtiene todos los datos de una vez
            results = self.collection.get(include=["embeddings"])

            if results is None or results.get("ids") is None or results.get("embeddings") is None:
                 raise DatabaseError("Failed to retrieve data with collection.get()")

            all_ids = results["ids"]
            all_embeddings_raw = results["embeddings"] # Esto podría ser List[list] o List[np.ndarray]

            if len(all_ids) != len(all_embeddings_raw):
                 raise DatabaseError(f"Mismatch in retrieved data: {len(all_ids)} IDs vs {len(all_embeddings_raw)} embeddings.")

            valid_ids = []
            valid_embeddings = []
            final_dim = None # Determina la dimensión del primer embedding válido

            logger.debug(f"Starting validation filtering for {len(all_embeddings_raw)} raw embeddings...")
            # --- INICIO: BUCLE DE FILTRADO CORREGIDO ---
            for i, emb_raw in enumerate(all_embeddings_raw):
                is_valid_type = isinstance(emb_raw, list) or isinstance(emb_raw, np.ndarray)
                current_len = -1

                if emb_raw is None or not is_valid_type:
                    logger.warning(f"Skipping ID '{all_ids[i]}' due to invalid type ({type(emb_raw)}).")
                    continue # Salta este embedding

                try:
                    # Obtiene la longitud (funciona para list y ndarray)
                    current_len = len(emb_raw)
                    if current_len <= 0:
                         logger.warning(f"Skipping ID '{all_ids[i]}' due to zero length.")
                         continue # Salta embedding inválido
                except TypeError:
                    logger.warning(f"Skipping ID '{all_ids[i]}' - Could not get length for type {type(emb_raw)}.")
                    continue # Salta si no se puede obtener longitud

                # Determina la dimensión esperada del primer embedding válido
                if final_dim is None:
                    final_dim = current_len
                    logger.info(f"Determined expected embedding dimension from first valid item: {final_dim}")
                    # Compara con metadatos si existen
                    meta_dim = self.get_dimension_from_metadata()
                    if isinstance(meta_dim, int) and meta_dim != final_dim:
                         logger.error(f"CRITICAL MISMATCH: Dimension from data ({final_dim}) differs from metadata ({meta_dim}) for collection '{self.collection_name}'!")
                         # Podrías lanzar un error aquí si prefieres ser estricto
                         # raise DatabaseError("Metadata dimension mismatch detected during full retrieval.")

                # Comprueba si la dimensión actual coincide con la esperada (final_dim)
                if current_len == final_dim:
                    # Asegura que se almacene como lista de floats para consistencia posterior
                    valid_embeddings.append(list(map(float, emb_raw))) # Convierte a lista de floats
                    valid_ids.append(all_ids[i])
                else:
                    logger.warning(f"Skipping ID '{all_ids[i]}' due to dimension mismatch (Expected {final_dim}, Got {current_len}).")
            # --- FIN: BUCLE DE FILTRADO CORREGIDO ---

            if not valid_embeddings:
                logger.error("No valid embeddings found after filtering! Check data integrity or filtering logic.")
                # Devuelve array vacío con la dimensión determinada (o fallback si no se determinó)
                return ([], np.empty((0, final_dim if final_dim else fallback_dim), dtype=np.float32))

            embeddings_array = np.array(valid_embeddings, dtype=np.float32)
            logger.info(f"Retrieved and validated {len(valid_ids)} items in {time.time() - start_time:.2f}s. Final array shape: {embeddings_array.shape}")
            return (valid_ids, embeddings_array)

        except Exception as e:
            logger.error(f"Error retrieving all embeddings from '{self.collection_name}': {e}", exc_info=True)
            raise DatabaseError(f"Failed to get all embeddings: {e}") from e

    def get_embeddings_by_ids(
        self, ids: List[str]
    ) -> Optional[Dict[str, Optional[List[float]]]]:
        """Obtiene embeddings específicos por ID."""
        if not self.is_initialized or self.collection is None:
            raise DatabaseError("Collection not available.")
        if not ids: return {}

        logger.info(f"Retrieving embeddings for {len(ids)} IDs from '{self.collection_name}'...")
        try:
            retrieved_data = self.collection.get(ids=ids, include=["embeddings"])
            results_dict: Dict[str, Optional[List[float]]] = {id_val: None for id_val in ids}

            if retrieved_data is not None and retrieved_data.get("ids") is not None and retrieved_data.get("embeddings") is not None:
                retrieved_ids = retrieved_data["ids"]
                retrieved_embeddings = retrieved_data["embeddings"]
                for i, r_id in enumerate(retrieved_ids):
                    if r_id in results_dict:
                        embedding = retrieved_embeddings[i]
                        # Acepta lista o ndarray, convierte a lista
                        if isinstance(embedding, list):
                             results_dict[r_id] = list(map(float, embedding))
                        elif isinstance(embedding, np.ndarray):
                             results_dict[r_id] = list(map(float, embedding.tolist()))
                        else:
                             results_dict[r_id] = None # Marca como None si no es tipo esperado
            else:
                 logger.warning("ChromaDB get by IDs returned no data or unexpected format.")

            found_count = sum(1 for v in results_dict.values() if v is not None)
            logger.info(f"Retrieved {found_count}/{len(ids)} requested embeddings.")
            return results_dict

        except Exception as e:
            logger.error(f"Error retrieving embeddings by IDs from '{self.collection_name}': {e}", exc_info=True)
            raise DatabaseError(f"Failed to get embeddings by IDs: {e}") from e

    def update_metadata_batch(
        self, ids: List[str], metadatas: List[Dict[str, Any]]
    ) -> bool:
        """Actualiza metadatos por lotes."""
        if not self.is_initialized or self.collection is None:
            raise DatabaseError("Collection not available.")
        if len(ids) != len(metadatas): raise ValueError("Input list lengths mismatch.")
        if not ids: return True

        logger.info(f"Updating metadata for {len(ids)} items in '{self.collection_name}'...")
        try:
            self.collection.update(ids=ids, metadatas=metadatas)
            logger.debug(f"Metadata update request sent for {len(ids)} items.")
            return True
        except Exception as e:
            logger.error(f"Error during batch metadata update in '{self.collection_name}': {e}", exc_info=True)
            return False # Simplificado: devuelve False en lugar de relanzar

    def clear_collection(self) -> bool:
        """Elimina todos los items de la colección."""
        if not self.is_initialized or self.collection is None:
            raise DatabaseError("Collection not available.")
        try:
            count_before = self.count()
            if count_before <= 0:
                logger.info(f"Collection '{self.collection_name}' is already empty.")
                return True

            logger.warning(f"Clearing ALL {count_before} items from collection '{self.collection_name}'...")
            start_time = time.time()
            # Intenta obtener todos los IDs para asegurar que se borran
            all_ids = self.collection.get(include=[])['ids']
            if all_ids:
                logger.info(f"Deleting {len(all_ids)} items by ID...")
                # Borra en lotes si son muchos (Opcional, delete podría manejarlo)
                batch_size_delete = 5000 # Ajusta según sea necesario
                for i in range(0, len(all_ids), batch_size_delete):
                     batch_ids = all_ids[i:i + batch_size_delete]
                     self.collection.delete(ids=batch_ids)
                     logger.debug(f"Deleted batch of {len(batch_ids)} IDs.")
            else:
                 logger.info("No IDs found to delete during clear.")

            count_after = self.count()
            success = count_after == 0
            if success:
                logger.info(f"Collection '{self.collection_name}' cleared in {time.time()-start_time:.2f}s.")
            else:
                logger.error(f"Failed to clear collection '{self.collection_name}'. Count after: {count_after}.")
            return success
        except Exception as e:
            logger.error(f"Error clearing collection '{self.collection_name}': {e}", exc_info=True)
            raise DatabaseError(f"Failed to clear collection: {e}") from e

    def delete_collection(self) -> bool:
        """Elimina toda la colección."""
        if not self.client: raise DatabaseError("Client not initialized.")
        if not self.collection_name: raise DatabaseError("Collection name not set.")

        logger.warning(f"Deleting ENTIRE collection '{self.collection_name}'...")
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = None
            self._is_initialized_flag = False
            logger.info(f"Collection '{self.collection_name}' deleted successfully.")
            return True
        except Exception as e:
            if "does not exist" in str(e).lower():
                 logger.warning(f"Collection '{self.collection_name}' did not exist, consider delete successful.")
                 self.collection = None
                 self._is_initialized_flag = False
                 return True
            logger.error(f"Error deleting collection '{self.collection_name}': {e}", exc_info=True)
            raise DatabaseError(f"Failed to delete collection: {e}") from e

    def count(self) -> int:
        """Devuelve el número de items."""
        if not self.is_initialized:
            if not self._ensure_collection_exists():
                 logger.warning(f"Count requested but collection '{self.collection_name}' is not available.")
                 return -1
        # Asegura que self.collection esté asignado si is_initialized es True
        if self.collection is None:
             logger.error(f"Collection object is None despite is_initialized being True for '{self.collection_name}'.")
             return -1
        try:
            return self.collection.count()
        except Exception as e:
            logger.error(f"Error getting count for '{self.collection_name}': {e}")
            # Invalida la colección si falla el conteo
            self.collection = None
            self._is_initialized_flag = False
            return -1
