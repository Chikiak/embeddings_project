import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.errors import InvalidDimensionException
from typing import List, Dict, Optional, Tuple, Any
import os
import time # Importar time para logging
import numpy as np
import logging

from config import CHROMA_DB_PATH, CHROMA_COLLECTION_NAME, VECTOR_DIMENSION

logger = logging.getLogger(__name__)


class VectorDatabase:
    """
    Maneja las interacciones con la base de datos vectorial ChromaDB.
    Utiliza un cliente persistente para almacenar datos localmente.
    """

    def __init__(
        self, path: str = CHROMA_DB_PATH, collection_name: str = CHROMA_COLLECTION_NAME
    ):
        self.path = path
        self.collection_name = collection_name
        self.client: Optional[chromadb.ClientAPI] = None
        self.collection: Optional[Collection] = None

        logger.info(f"Initializing VectorDatabase:")
        logger.info(f"  Storage Path: {os.path.abspath(self.path)}")
        logger.info(f"  Collection Name: {self.collection_name}")

        try:
            os.makedirs(self.path, exist_ok=True)
            logger.info(f"Database directory ensured at: {self.path}")
        except OSError as e:
            logger.error(
                f"Error creating or accessing database directory {self.path}: {e}",
                exc_info=True,
            )
            raise RuntimeError(
                f"Failed to create/access DB directory: {self.path}"
            ) from e

        try:
            self.client = chromadb.PersistentClient(path=self.path)
            logger.info("ChromaDB PersistentClient initialized.")

            # Usar 'cosine' como métrica de distancia, adecuado para embeddings normalizados (ej. CLIP).
            metadata_options = {"hnsw:space": "cosine"}

            logger.info(
                f"Getting or creating collection '{self.collection_name}' with metadata {metadata_options}..."
            )
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata=metadata_options,
            )
            logger.info(
                f"Collection '{self.collection_name}' ready. Current item count: {self.collection.count()}"
            )

        except InvalidDimensionException as e_dim:
            logger.error(
                f"ChromaDB Error: Invalid dimension specified or inferred? {e_dim}",
                exc_info=True,
            )
            self.client = None
            self.collection = None
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error initializing ChromaDB client/collection: {e}",
                exc_info=True,
            )
            self.client = None
            self.collection = None
            raise

    def add_embeddings(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        if not self.collection:
            logger.error("Collection not initialized. Cannot add embeddings.")
            return False
        if not isinstance(ids, list) or not isinstance(embeddings, list):
            logger.error("Invalid input type: ids and embeddings must be lists.")
            return False
        if not ids or not embeddings:
            logger.warning("Empty ids or embeddings list provided. Nothing to add.")
            return True
        if len(ids) != len(embeddings):
            logger.error(
                f"Input mismatch: IDs ({len(ids)}) vs embeddings ({len(embeddings)})."
            )
            return False
        if metadatas:
            if not isinstance(metadatas, list):
                logger.error("Invalid input type: metadatas must be a list if provided.")
                return False
            if len(ids) != len(metadatas):
                logger.error(
                    f"Input mismatch: IDs ({len(ids)}) vs metadatas ({len(metadatas)})."
                )
                return False
        else:
            metadatas = [{}] * len(ids)

        expected_dim = len(embeddings[0])
        if VECTOR_DIMENSION and expected_dim != VECTOR_DIMENSION:
            logger.warning(
                f"Provided embedding dimension ({expected_dim}) differs from config ({VECTOR_DIMENSION})."
            )
        if not all(len(emb) == expected_dim for emb in embeddings):
            logger.error("Inconsistent embedding dimensions found within the batch.")
            for i, emb in enumerate(embeddings):
                if len(emb) != expected_dim:
                    logger.error(
                        f"  - ID '{ids[i]}' has dimension {len(emb)}, expected {expected_dim}"
                    )
                    break
            return False

        num_items_to_add = len(ids)
        logger.info(
            f"Attempting to add/update {num_items_to_add} embeddings in collection '{self.collection_name}'..."
        )
        try:
            self.collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)
            new_count = self.collection.count()
            logger.info(f"Upsert successful. Collection count is now: {new_count}")
            return True
        except InvalidDimensionException as e_dim:
            logger.error(
                f"ChromaDB Error during upsert: Invalid dimension. {e_dim}",
                exc_info=True,
            )
            return False
        except Exception as e:
            logger.error(
                f"Error during upsert operation in collection '{self.collection_name}': {e}",
                exc_info=True,
            )
            return False

    def query_similar(
        self, query_embedding: List[float], n_results: int = 5
    ) -> Optional[Dict[str, Any]]:
        if not self.collection:
            logger.error("Collection not initialized. Cannot query.")
            return None
        if not query_embedding or not isinstance(query_embedding, list):
            logger.error("Invalid query embedding provided.")
            return None

        empty_results = {"ids": [], "distances": [], "metadatas": []}

        try:
            current_count = self.collection.count()
            if current_count == 0:
                logger.warning("Query attempted on an empty collection.")
                return empty_results

            effective_n_results = min(n_results, current_count)
            if effective_n_results <= 0:
                logger.warning(f"Effective n_results is {effective_n_results}.")
                return empty_results

            logger.info(
                f"Querying collection '{self.collection_name}' for {effective_n_results} nearest neighbors..."
            )

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=effective_n_results,
                include=["metadatas", "distances"],
            )
            logger.info("Query execution successful.")

            if (
                results
                and all(key in results for key in ["ids", "distances", "metadatas"])
                and results["ids"]
            ):
                single_query_results = {
                    "ids": results["ids"][0],
                    "distances": results["distances"][0],
                    "metadatas": (
                        results["metadatas"][0]
                        if results["metadatas"]
                        else [{}] * len(results["ids"][0])
                    ),
                }
                if not single_query_results["ids"]:
                    logger.info("Query successful, but no matching documents found.")
                    return empty_results
                else:
                    logger.info(f"Found {len(single_query_results['ids'])} results.")
                    return single_query_results
            else:
                logger.info("Query successful, but no matching documents found or results format issue.")
                return empty_results

        except InvalidDimensionException as e_dim:
            logger.error(
                f"ChromaDB Error during query: Invalid dimension. {e_dim}",
                exc_info=True,
            )
            return None
        except Exception as e:
            logger.error(
                f"Error during query operation in collection '{self.collection_name}': {e}",
                exc_info=True,
            )
            return None

    def get_all_embeddings_with_ids(self, pagination_batch_size: int = 10000) -> Optional[Tuple[List[str], np.ndarray]]:
        """
        Recupera todos los embeddings y sus IDs usando paginación para eficiencia de memoria.
        """
        if not self.collection:
            logger.error("Collection not initialized. Cannot get all embeddings.")
            return None

        empty_result: Tuple[List[str], np.ndarray] = (
            [],
            np.empty((0, VECTOR_DIMENSION if VECTOR_DIMENSION else 1), dtype=np.float32),
        )

        try:
            count = self.collection.count()
            if count == 0:
                logger.info("Collection is empty. Returning empty lists/arrays.")
                return empty_result

            logger.info(
                f"Retrieving all {count} embeddings and IDs from '{self.collection_name}' using pagination (batch size: {pagination_batch_size})..."
            )
            start_time = time.time()

            all_ids = []
            all_embeddings_list = []
            retrieved_count = 0

            while retrieved_count < count:
                logger.debug(f"Retrieving batch starting from offset {retrieved_count} with limit {pagination_batch_size}")
                try:
                    results = self.collection.get(
                        limit=pagination_batch_size,
                        offset=retrieved_count,
                        include=["embeddings"]
                    )

                    if results and results.get("ids") and results.get("embeddings") is not None:
                        batch_ids = results["ids"]
                        batch_embeddings = results["embeddings"]

                        if not batch_ids:
                            logger.warning(f"Retrieved empty batch at offset {retrieved_count}. Stopping.")
                            break

                        all_ids.extend(batch_ids)
                        all_embeddings_list.extend(batch_embeddings)
                        retrieved_count += len(batch_ids)
                        logger.debug(f"Retrieved {len(batch_ids)} items. Total retrieved: {retrieved_count}/{count}")
                    else:
                        logger.warning(f"Received unexpected empty result or format issue at offset {retrieved_count}. Stopping.")
                        break

                except Exception as e_get:
                    logger.error(f"Error retrieving batch at offset {retrieved_count}: {e_get}", exc_info=True)
                    return None # Fallo crítico

            end_time = time.time()
            logger.info(f"Data retrieval with pagination finished in {end_time - start_time:.2f}s.")

            # Verificación final
            if len(all_ids) != len(all_embeddings_list):
                logger.error(f"CRITICAL: Mismatch after pagination: IDs ({len(all_ids)}) vs Embeddings ({len(all_embeddings_list)}).")
                return None
            if len(all_ids) != count:
                logger.warning(f"Final retrieved count ({len(all_ids)}) differs from initial count ({count}).")

            # Convertir a NumPy
            try:
                embeddings_array = np.array(all_embeddings_list, dtype=np.float32)
                logger.info(
                    f"Successfully retrieved {len(all_ids)} IDs and embeddings array of shape {embeddings_array.shape} via pagination."
                )
                return all_ids, embeddings_array
            except ValueError as ve:
                logger.error(f"Error converting paginated embeddings list to NumPy array: {ve}.", exc_info=True)
                return None

        except Exception as e:
            logger.error(
                f"Error retrieving all embeddings from collection '{self.collection_name}': {e}",
                exc_info=True,
            )
            return None

    def clear_collection(self) -> bool:
        if not self.collection:
            logger.error("Collection not initialized. Cannot clear.")
            return False
        try:
            count_before = self.collection.count()
            if count_before == 0:
                logger.info(f"Collection '{self.collection_name}' is already empty.")
                return True

            logger.warning(
                f"Attempting to clear ALL {count_before} items from collection '{self.collection_name}'..."
            )
            # Usar delete con where={} o ids=all_ids para eliminar todo
            # Obtener todos los IDs puede ser ineficiente para colecciones masivas
            # Considerar self.client.delete_collection() y self.client.create_collection() como alternativa
            all_ids = self.collection.get(include=[])["ids"] # Puede consumir memoria
            if all_ids:
                logger.info(f"Deleting {len(all_ids)} items...")
                self.collection.delete(ids=all_ids)
                count_after = self.collection.count()
                if count_after == 0:
                    logger.info(f"Successfully cleared collection '{self.collection_name}'. Items removed: {count_before}.")
                    return True
                else:
                    logger.error(f"Collection count is {count_after} after clearing, expected 0.")
                    return False
            else:
                logger.warning("Collection reported items but no IDs were retrieved for deletion.")
                return self.collection.count() == 0

        except Exception as e:
            logger.error(f"Error clearing collection '{self.collection_name}': {e}", exc_info=True)
            return False

    def delete_collection(self) -> bool:
        if not self.client:
            logger.error("Client not initialized. Cannot delete collection.")
            return False
        if not self.collection_name:
            logger.error("Collection name is not set. Cannot delete.")
            return False

        logger.warning(
            f"Attempting to DELETE THE ENTIRE collection '{self.collection_name}' from path '{self.path}'..."
        )
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.info(f"Successfully deleted collection '{self.collection_name}'.")
            self.collection = None
            return True
        except Exception as e:
            logger.error(f"Error deleting collection '{self.collection_name}': {e}", exc_info=True)
            return False
