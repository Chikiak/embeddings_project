# data_access/chroma_db.py
import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.errors import InvalidDimensionException
from typing import List, Dict, Optional, Tuple, Any, Union # Añadir Union
import os
import time
import numpy as np
import logging

# Asumiendo que config.py está accesible
# from config import VECTOR_DIMENSION
# Valor por defecto por si config no está disponible
VECTOR_DIMENSION = None

from .vector_db_interface import VectorDBInterface
from app.models import SearchResultItem, SearchResults # Asegúrate que models.py está en la ruta
from app.exceptions import DatabaseError, InitializationError # Asegúrate que exceptions.py está en la ruta

logger = logging.getLogger(__name__)


class ChromaVectorDB(VectorDBInterface):
    """
    Implementación concreta de VectorDBInterface usando ChromaDB.

    Maneja interacciones con una colección persistente de ChromaDB.
    Incluye lógica para recrear la colección si se elimina y para
    gestionar metadatos de dimensión.
    """

    # --- MODIFICADO __init__ ---
    def __init__(self,
                 path: str,
                 collection_name: str,
                 expected_dimension_metadata: Optional[Union[int, str]] = None # NUEVO
                ):
        """
        Inicializa el cliente ChromaDB e intenta obtener/crear la colección.

        Args:
            path: Ruta del directorio para almacenamiento persistente.
            collection_name: Nombre de la colección dentro de ChromaDB.
            expected_dimension_metadata: Dimensión (int o 'full') esperada para esta colección.
                                         Se usará para añadir metadatos si la colección se crea.
        """
        self.path = path
        self.collection_name = collection_name
        # --- NUEVO ATRIBUTO ---
        # Almacena el metadato esperado, se usará en _ensure_collection_exists si se crea la colección
        self._expected_dimension_metadata = expected_dimension_metadata
        # --------------------
        self.client: Optional[chromadb.ClientAPI] = None
        self.collection: Optional[Collection] = None
        self._client_initialized = False
        self._last_init_error: Optional[str] = None # Almacena último error de inicialización

        logger.info(f"Initializing ChromaVectorDB:")
        logger.info(f"  Storage Path: {os.path.abspath(self.path)}")
        logger.info(f"  Target Collection Name: {self.collection_name}")
        if self._expected_dimension_metadata:
             logger.info(f"  Expected dimension metadata for new collection: {self._expected_dimension_metadata}")

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
            # Intenta asegurar que la colección exista inmediatamente
            if not self._ensure_collection_exists():
                 msg = f"Failed to get or create collection '{self.collection_name}' during initial setup."
                 logger.error(msg)
                 self._last_init_error = msg
                 # No lanzar error aquí, is_initialized lo manejará
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

    # --- MODIFICADO _ensure_collection_exists ---
    def _ensure_collection_exists(self) -> bool:
        """
        Verifica si la colección está disponible y la crea si no existe.
        Añade metadatos de dimensión si se crea y se proporcionaron.

        Returns:
            True si la colección existe o se creó exitosamente, False en caso contrario.
        """
        if not self._client_initialized or self.client is None:
            logger.error("ChromaDB client not initialized. Cannot ensure collection.")
            return False

        # Si el objeto collection existe, hacer una comprobación rápida
        if self.collection is not None:
            try:
                self.collection.count() # Operación ligera para verificar validez
                logger.debug(f"Collection object for '{self.collection_name}' exists and seems valid.")
                return True
            except Exception as e:
                logger.warning(
                    f"Collection object for '{self.collection_name}' exists but failed check: {e}. Will try to get/create again."
                )
                self.collection = None # Resetear objeto inválido

        # Intentar obtener o crear la colección
        logger.info(
            f"Attempting to ensure collection '{self.collection_name}' exists..."
        )
        try:
            # --- Preparar metadatos para la creación ---
            # Estos metadatos solo se usan si la colección NO existe y se va a CREAR.
            creation_metadata = None
            if self._expected_dimension_metadata is not None:
                creation_metadata = {"embedding_dimension": self._expected_dimension_metadata}
                logger.debug(f"Will set metadata {creation_metadata} if collection '{self.collection_name}' is created.")

            # Obtener o crear la colección
            # NOTA: ChromaDB < 0.5.0 usa los metadatos solo al crear.
            # Si la colección ya existe, los metadatos pasados aquí se ignoran.
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata=creation_metadata, # Pasa metadatos para la creación
                # embedding_function=None # Deshabilitar si provees tus propios embeddings
            )
            collection_count = self.collection.count()
            logger.info(
                f"Collection '{self.collection_name}' ensured/created. Current count: {collection_count}"
            )

            # --- Verificación/Actualización de metadatos post-creación/obtención ---
            # Si la colección se acaba de crear (count==0) O si no tiene metadatos de dimensión,
            # y nosotros SÍ esperábamos unos, intentamos añadirlos/actualizarlos.
            current_meta = self.collection.metadata
            has_meta_dim = current_meta is not None and 'embedding_dimension' in current_meta

            if self._expected_dimension_metadata is not None and not has_meta_dim:
                 try:
                     logger.info(f"Attempting to add/update 'embedding_dimension' metadata for '{self.collection_name}' to '{self._expected_dimension_metadata}'.")
                     # Crear el diccionario de metadatos a añadir/actualizar
                     new_metadata = current_meta.copy() if current_meta else {}
                     new_metadata['embedding_dimension'] = self._expected_dimension_metadata
                     # Usar modify para actualizar metadatos
                     self.collection.modify(metadata=new_metadata)
                     logger.info(f"Metadata for '{self.collection_name}' modified successfully.")
                     # Actualizar el objeto de colección local con los nuevos metadatos (opcional)
                     self.collection = self.client.get_collection(name=self.collection_name)
                 except Exception as mod_err:
                     logger.warning(f"Could not modify metadata for collection '{self.collection_name}': {mod_err}", exc_info=False)
            # ---------------------------------------------------------------------
            return True
        except InvalidDimensionException as e_dim:
            logger.error(f"Invalid dimension error ensuring collection '{self.collection_name}': {e_dim}", exc_info=True)
            self.collection = None
            return False
        except Exception as e:
            logger.error(f"Unexpected error getting/creating collection '{self.collection_name}': {e}", exc_info=True)
            self.collection = None
            return False

    @property
    def is_initialized(self) -> bool:
        """Verifica si el cliente está inicializado y la colección está accesible."""
        if not self._client_initialized:
             return False
        # Intenta asegurar que la colección exista (esto la creará si es necesario)
        return self._ensure_collection_exists()

    # --- NUEVO MÉTODO PÚBLICO ---
    def get_dimension_from_metadata(self) -> Optional[Union[str, int]]:
        """
        Intenta obtener la dimensión almacenada en los metadatos de la colección.

        Returns:
            La dimensión (int o 'full') o None si no se encuentra o hay error.
        """
        if not self.is_initialized or self.collection is None:
             # Intenta reasegurar la colección una última vez
             if not self._ensure_collection_exists() or self.collection is None:
                 logger.warning("Collection not available, cannot get metadata.")
                 return None

        try:
            # Accede a los metadatos del objeto de colección existente
            metadata = self.collection.metadata # Más eficiente que client.get_collection
            if metadata and 'embedding_dimension' in metadata:
                dim_value = metadata['embedding_dimension']
                # Validación simple
                if (isinstance(dim_value, int) and dim_value > 0) or dim_value == 'full':
                    logger.debug(f"Found dimension '{dim_value}' in metadata for collection '{self.collection_name}'.")
                    return dim_value
                else:
                    logger.warning(f"Invalid value found for 'embedding_dimension' metadata in '{self.collection_name}': {dim_value}")
                    return None
            else:
                logger.debug(f"No 'embedding_dimension' key found in metadata for '{self.collection_name}'.")
                return None # No encontrado
        except Exception as e:
            logger.warning(f"Error accessing metadata for collection '{self.collection_name}': {e}", exc_info=False)
            return None # Error
    # --- FIN NUEVO MÉTODO ---

    def add_embeddings(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """
        Añade o actualiza embeddings en la colección ChromaDB.
        Verifica la dimensión contra los metadatos si existen.
        """
        if not self.is_initialized or self.collection is None:
            if not self._ensure_collection_exists() or self.collection is None:
                 raise DatabaseError("Collection not available. Cannot add embeddings.")

        # ... (validaciones de entrada existentes: listas, longitud, metadatos) ...
        if not isinstance(ids, list) or not isinstance(embeddings, list): raise ValueError("ids and embeddings must be lists.")
        if not ids or not embeddings: return True # Nada que añadir
        if len(ids) != len(embeddings): raise ValueError(f"Input mismatch: IDs ({len(ids)}) vs embeddings ({len(embeddings)}).")
        if metadatas and (not isinstance(metadatas, list) or len(ids) != len(metadatas)): raise ValueError("Metadata list mismatch or invalid type.")
        if not metadatas: metadatas = [{} for _ in ids]

        # --- Verificación de Dimensión (Mejorada) ---
        expected_dim_meta = self.get_dimension_from_metadata()
        actual_batch_dim: Optional[int] = None
        if embeddings and embeddings[0] and isinstance(embeddings[0], list):
            actual_batch_dim = len(embeddings[0])
            # Validar consistencia dentro del lote
            if not all(isinstance(e, list) and len(e) == actual_batch_dim for e in embeddings):
                 raise ValueError("Inconsistent embedding dimensions within the provided batch.")

        # Comprobar contra metadatos si existen y la colección NO está vacía
        collection_count = self.count()
        if expected_dim_meta is not None and collection_count > 0:
            if isinstance(expected_dim_meta, int) and actual_batch_dim != expected_dim_meta:
                msg = f"Dimension mismatch! Trying to add embeddings with dimension {actual_batch_dim}, but collection '{self.collection_name}' expects dimension {expected_dim_meta} based on metadata."
                logger.error(msg)
                raise DatabaseError(msg)
            elif expected_dim_meta == 'full':
                 # Si los metadatos dicen 'full', no hacemos una comprobación estricta aquí,
                 # asumimos que el vectorizador ya produjo la dimensión correcta.
                 # Podríamos opcionalmente comparar con vectorizer.native_dimension si estuviera disponible aquí.
                 pass
        elif expected_dim_meta is None and collection_count > 0:
             logger.warning(f"Adding embeddings to collection '{self.collection_name}' which has data but lacks dimension metadata. ChromaDB might enforce dimension based on existing data.")
             # Podríamos intentar obtener un item existente y comprobar su dimensión, pero es costoso.
        # -----------------------------------------

        num_items_to_add = len(ids)
        logger.info(f"Attempting to add/update {num_items_to_add} embeddings (dim: {actual_batch_dim or 'N/A'}) in collection '{self.collection_name}'...")
        try:
            self.collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)
            logger.info(f"Upsert for {num_items_to_add} items completed.")

            # --- Intento de añadir metadatos si era la primera vez ---
            # Si la colección ESTABA vacía antes de este upsert y esperábamos metadatos,
            # intentamos añadirlos ahora que hay datos.
            if collection_count == 0 and self._expected_dimension_metadata is not None:
                 current_meta_after_add = self.collection.metadata
                 if current_meta_after_add is None or 'embedding_dimension' not in current_meta_after_add:
                     try:
                         logger.info(f"Attempting to set 'embedding_dimension' metadata after first add for '{self.collection_name}'.")
                         new_metadata = current_meta_after_add.copy() if current_meta_after_add else {}
                         new_metadata['embedding_dimension'] = self._expected_dimension_metadata
                         self.collection.modify(metadata=new_metadata)
                         logger.info(f"Metadata set successfully for '{self.collection_name}' after first add.")
                         # Actualizar el objeto local
                         self.collection = self.client.get_collection(name=self.collection_name)
                     except Exception as mod_err:
                         logger.warning(f"Could not set metadata after first add for '{self.collection_name}': {mod_err}", exc_info=False)
            # ------------------------------------------------------
            return True
        except InvalidDimensionException as e_dim:
            # ... (manejo de error existente) ...
            raise DatabaseError(f"ChromaDB upsert error: Invalid dimension. Batch dim: {actual_batch_dim}. Details: {e_dim}") from e_dim
        except Exception as e:
            # ... (manejo de error existente) ...
            raise DatabaseError(f"Error during upsert operation in collection '{self.collection_name}': {e}") from e

    def query_similar(
        self, query_embedding: List[float], n_results: int = 5
    ) -> Optional[SearchResults]:
        """
        Consulta ChromaDB por embeddings similares y devuelve SearchResults.
        Verifica la dimensión de la consulta contra los metadatos si existen.
        """
        if not self.is_initialized or self.collection is None:
            if not self._ensure_collection_exists() or self.collection is None:
                raise DatabaseError("Collection not available. Cannot query.")

        if not query_embedding or not isinstance(query_embedding, list):
            raise ValueError("Invalid query embedding (must be a list of floats).")

        # --- Verificación de Dimensión de Consulta ---
        query_dim = len(query_embedding)
        expected_dim_meta = self.get_dimension_from_metadata()

        if expected_dim_meta is not None and self.count() > 0:
            if isinstance(expected_dim_meta, int) and query_dim != expected_dim_meta:
                msg = f"Query dimension mismatch! Query has dimension {query_dim}, but collection '{self.collection_name}' expects dimension {expected_dim_meta} based on metadata."
                logger.error(msg)
                raise DatabaseError(msg)
            # Si es 'full', asumimos que la consulta ya tiene la dimensión correcta.
        # -----------------------------------------

        current_count = self.count()
        if current_count <= 0: # Maneja 0 y -1 (error)
            logger.warning(f"Query attempted on an empty or inaccessible collection (count={current_count}).")
            return SearchResults(items=[])

        effective_n_results = min(n_results, current_count)
        if effective_n_results <= 0: return SearchResults(items=[])

        logger.info(f"Querying collection '{self.collection_name}' (dim: {expected_dim_meta or 'Unknown'}) for {effective_n_results} neighbors (query dim: {query_dim})...")
        try:
            start_time = time.time()
            results_dict = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=effective_n_results,
                include=["metadatas", "distances"],
            )
            end_time = time.time()
            logger.info(f"ChromaDB query executed in {end_time - start_time:.3f}s.")

            # ... (procesamiento de resultados existente) ...
            result_items = []
            if (results_dict and results_dict.get("ids") and results_dict.get("ids")[0] is not None):
                ids_list = results_dict["ids"][0]
                distances_list = results_dict.get("distances", [[]])[0] # Default a lista vacía si falta
                metadatas_list = results_dict.get("metadatas", [[]])[0] # Default a lista vacía si falta

                # Asegurar consistencia de longitud (mejorado)
                min_len = len(ids_list)
                if len(distances_list) != min_len or len(metadatas_list) != min_len:
                     logger.warning(f"Query result lists length mismatch! IDs:{len(ids_list)}, Dist:{len(distances_list)}, Meta:{len(metadatas_list)}. Truncating to ID length.")
                     distances_list = distances_list[:min_len] if distances_list else [None] * min_len # Pad with None if missing
                     metadatas_list = metadatas_list[:min_len] if metadatas_list else [{}] * min_len # Pad with {} if missing

                for i, img_id in enumerate(ids_list):
                    metadata = metadatas_list[i] if metadatas_list[i] is not None else {}
                    distance = distances_list[i] if distances_list and i < len(distances_list) else None
                    item = SearchResultItem(id=img_id, distance=distance, metadata=metadata)
                    result_items.append(item)

                logger.info(f"Query successful. Found {len(result_items)} results.")
                return SearchResults(items=result_items, query_vector=query_embedding)
            else:
                logger.info("Query successful, but no matching documents found or result format unexpected.")
                return SearchResults(items=[], query_vector=query_embedding)

        except InvalidDimensionException as e_dim:
            # ... (manejo de error existente) ...
            raise DatabaseError(f"ChromaDB query error: Invalid dimension. Query Dim: {query_dim}. Details: {e_dim}") from e_dim
        except Exception as e:
            # ... (manejo de error existente) ...
            raise DatabaseError(f"Error during query operation in collection '{self.collection_name}': {e}") from e

    def get_all_embeddings_with_ids(
        self, pagination_batch_size: int = 1000
    ) -> Optional[Tuple[List[str], np.ndarray]]:
        """
        Recupera todos los embeddings e IDs usando paginación.
        (Sin cambios funcionales necesarios aquí para la gestión de dimensiones,
         pero se beneficia de la verificación de dimensión en add_embeddings)
        """
        # ... (código existente) ...
        # El código existente ya maneja la paginación y conversión a NumPy.
        # La dimensión del array NumPy resultante dependerá de lo que esté
        # realmente almacenado en la colección.
        if not self.is_initialized or self.collection is None:
            if not self._ensure_collection_exists() or self.collection is None:
                raise DatabaseError("Collection not available. Cannot get all embeddings.")

        # Determinar dimensión de fallback
        fallback_dim = 1
        try:
            meta_dim = self.get_dimension_from_metadata()
            if isinstance(meta_dim, int): fallback_dim = meta_dim
            elif VECTOR_DIMENSION: fallback_dim = VECTOR_DIMENSION
        except Exception: pass # Usa 1 si todo falla

        empty_result: Tuple[List[str], np.ndarray] = ([], np.empty((0, fallback_dim), dtype=np.float32))

        try:
            count = self.count()
            if count <= 0: return empty_result # Maneja 0 y -1

            logger.info(f"Retrieving all {count} embeddings and IDs from '{self.collection_name}' (batch: {pagination_batch_size})...")
            start_time = time.time()
            all_ids = []
            all_embeddings_list = []
            retrieved_count = 0
            actual_embedding_dim = None

            while retrieved_count < count:
                 offset = retrieved_count
                 limit = min(pagination_batch_size, count - retrieved_count)
                 try:
                     results = self.collection.get(limit=limit, offset=offset, include=["embeddings"])
                     if results is None or results.get("ids") is None or results.get("embeddings") is None:
                         logger.error(f"collection.get returned invalid data at offset {offset}. Stopping.")
                         break # Salir del bucle si la paginación falla

                     batch_ids = results["ids"]
                     batch_embeddings_raw = results["embeddings"]

                     if not batch_ids: break # Fin de los datos

                     # Convertir a lista si es ndarray
                     if isinstance(batch_embeddings_raw, np.ndarray): batch_embeddings_list = batch_embeddings_raw.tolist()
                     elif isinstance(batch_embeddings_raw, list): batch_embeddings_list = batch_embeddings_raw
                     else: raise DatabaseError(f"Invalid embedding data type at offset {offset}")

                     if len(batch_ids) != len(batch_embeddings_list): raise DatabaseError(f"Batch length mismatch at offset {offset}")

                     # Detectar dimensión del primer lote válido
                     if actual_embedding_dim is None:
                         for emb in batch_embeddings_list:
                             if emb is not None: actual_embedding_dim = len(emb); break

                     all_ids.extend(batch_ids)
                     all_embeddings_list.extend(batch_embeddings_list)
                     retrieved_count += len(batch_ids)
                 except Exception as e_get:
                     raise DatabaseError(f"Error retrieving batch at offset {offset}: {e_get}") from e_get

            end_time = time.time()
            logger.info(f"Data retrieval finished in {end_time - start_time:.2f}s. Retrieved {retrieved_count} items.")

            # Filtrar Nones y convertir a NumPy
            valid_ids = []
            valid_embeddings = []
            final_dim = actual_embedding_dim if actual_embedding_dim is not None else fallback_dim

            for i, emb in enumerate(all_embeddings_list):
                 if emb is not None and isinstance(emb, list):
                     if len(emb) == final_dim: # Comprobar dimensión
                         valid_embeddings.append(emb)
                         valid_ids.append(all_ids[i])
                     else:
                         logger.warning(f"Skipping item ID '{all_ids[i]}' due to dimension mismatch (Expected {final_dim}, Got {len(emb)}).")

            if not valid_embeddings: return ([], np.empty((0, final_dim), dtype=np.float32))

            embeddings_array = np.array(valid_embeddings, dtype=np.float32)
            logger.info(f"Returning {len(valid_ids)} valid ID/embedding pairs. Array shape: {embeddings_array.shape}.")
            return (valid_ids, embeddings_array)

        except Exception as e:
            # ... (manejo de error existente) ...
            raise DatabaseError(f"Error retrieving all embeddings: {e}") from e


    def clear_collection(self) -> bool:
        """
        Elimina todos los ítems de la colección ChromaDB.
        (Sin cambios funcionales necesarios aquí)
        """
        # ... (código existente) ...
        # El código existente que obtiene todos los IDs y luego llama a delete funciona.
        if not self.is_initialized or self.collection is None:
            if not self._ensure_collection_exists() or self.collection is None:
                raise DatabaseError("Collection not available. Cannot clear.")
        try:
            count_before = self.count()
            if count_before <= 0: return True # Ya vacía o error

            logger.warning(f"Attempting to clear ALL {count_before} items from collection '{self.collection_name}'...")
            start_time = time.time()

            # Obtener IDs para eliminar (más robusto que delete_all que no existe)
            ids_to_delete = []
            limit = 10000 # Límite por lote para obtener IDs
            offset = 0
            while True:
                results = self.collection.get(limit=limit, offset=offset, include=[])
                if not results or not results.get('ids'): break
                batch_ids = results['ids']
                if not batch_ids: break
                ids_to_delete.extend(batch_ids)
                offset += len(batch_ids)
                if len(batch_ids) < limit: break # Último lote

            if not ids_to_delete:
                logger.info("No IDs found to delete, collection might be empty.")
                return self.count() == 0 # Verifica de nuevo

            logger.info(f"Deleting {len(ids_to_delete)} items by ID...")
            # Eliminar en lotes si son muchos IDs
            batch_size_delete = 500
            for i in range(0, len(ids_to_delete), batch_size_delete):
                 batch_del_ids = ids_to_delete[i:i + batch_size_delete]
                 self.collection.delete(ids=batch_del_ids)
                 logger.debug(f"Deleted batch {i//batch_size_delete + 1}...")

            end_time = time.time()
            count_after = self.count()
            if count_after == 0:
                logger.info(f"Collection '{self.collection_name}' cleared successfully in {end_time-start_time:.2f}s.")
                return True
            else:
                logger.error(f"Collection count is {count_after} after clearing, expected 0.")
                return False
        except Exception as e:
            # ... (manejo de error existente) ...
             raise DatabaseError(f"Error clearing collection '{self.collection_name}': {e}") from e


    def delete_collection(self) -> bool:
        """
        Elimina toda la colección ChromaDB.
        (Sin cambios funcionales necesarios aquí)
        """
        # ... (código existente) ...
        if not self._client_initialized or self.client is None: raise DatabaseError("Client not initialized.")
        if not self.collection_name: raise DatabaseError("Collection name not set.")

        logger.warning(f"Attempting to DELETE ENTIRE collection '{self.collection_name}' from path '{self.path}'...")
        try:
            start_time = time.time()
            self.client.delete_collection(name=self.collection_name)
            end_time = time.time()
            logger.info(f"Collection '{self.collection_name}' delete request sent successfully in {end_time-start_time:.2f}s.")
            self.collection = None # Resetear el objeto local
            # Verificar que realmente se eliminó
            try:
                self.client.get_collection(name=self.collection_name)
                logger.error(f"Verification failed: Collection '{self.collection_name}' still exists.")
                return False # No se eliminó
            except Exception:
                logger.info(f"Verification successful: Collection '{self.collection_name}' no longer exists.")
                return True # Se eliminó correctamente
        except Exception as e:
            # ... (manejo de error existente) ...
             raise DatabaseError(f"Error deleting collection '{self.collection_name}': {e}") from e


    def count(self) -> int:
        """
        Devuelve el número de ítems en la colección.
        (Sin cambios funcionales necesarios aquí)
        """
        # ... (código existente) ...
        if not self.is_initialized or self.collection is None:
            # Intenta reasegurar antes de fallar
            if not self._ensure_collection_exists() or self.collection is None:
                logger.warning("Count requested but collection is not available.")
                return -1
        try:
            count = self.collection.count()
            logger.debug(f"Collection '{self.collection_name}' count: {count}")
            return count
        except Exception as e:
            logger.error(f"Error getting count for collection '{self.collection_name}': {e}", exc_info=True)
            self.collection = None # Asume que el objeto collection puede ser inválido
            return -1

