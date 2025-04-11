import chromadb
from typing import List, Dict, Optional, Tuple, Any
import os
import numpy as np
import logging # Usar logging en lugar de print para errores persistentes

# Import configuration constants
from config import CHROMA_DB_PATH, CHROMA_COLLECTION_NAME, VECTOR_DIMENSION

# Configurar logging básico (puedes configurarlo mejor en tu punto de entrada)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VectorDatabase:
    """
    Handles interactions with the ChromaDB vector database.
    Uses a persistent client to store data locally.
    """

    def __init__(
        self, path: str = CHROMA_DB_PATH, collection_name: str = CHROMA_COLLECTION_NAME
    ):
        """
        Initializes the ChromaDB client and gets/creates the collection.

        Args:
            path: The directory path for storing the persistent database.
            collection_name: The name of the collection to use.
        """
        self.path = path
        self.collection_name = collection_name
        self.client = None
        self.collection = None

        logging.info(f"Initializing VectorDatabase:")
        logging.info(f"  Storage Path: {self.path}")
        logging.info(f"  Collection Name: {self.collection_name}")

        # Ensure the database directory exists
        try:
            if not os.path.exists(self.path):
                os.makedirs(self.path)
                logging.info(f"Created database directory: {self.path}")
        except OSError as e: # Captura específica para error de creación de directorio
            logging.error(f"Error creating directory {self.path}: {e}")
            raise # Re-lanzar es apropiado aquí, la app no puede continuar

        try:
            # Initialize the persistent client
            self.client = chromadb.PersistentClient(path=self.path)

            # Get or create the collection
            # Nota: Evita especificar 'embedding_dimension' en metadatos si VECTOR_DIMENSION es None
            # para permitir que Chroma lo infiera, lo que es más flexible.
            metadata_options = {}
            # Si necesitas una métrica específica diferente a la default (cosine):
            # metadata_options = {"hnsw:space": "l2"} # para distancia euclidiana

            logging.info(f"Getting or creating collection '{self.collection_name}'...")
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata=(metadata_options if metadata_options else None),
            )
            logging.info(
                f"Collection '{self.collection_name}' ready. Item count: {self.collection.count()}"
            )

        # Ejemplo de captura de errores específicos (ajusta según errores reales de ChromaDB)
        # except InvalidDimensionException as e_dim:
        #     logging.error(f"ChromaDB Error: Invalid dimension specified or inferred? {e_dim}")
        #     self.client = None
        #     self.collection = None
        #     raise
        except Exception as e: # Captura general para errores inesperados durante la inicialización
            logging.error(f"Unexpected error initializing ChromaDB client/collection: {e}", exc_info=True) # exc_info=True para traceback
            self.client = None
            self.collection = None
            raise # Re-lanzar para indicar fallo crítico

    def add_embeddings(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Adds or updates embeddings in the collection using upsert.

        Args:
            ids: A list of unique identifiers for each embedding.
            embeddings: A list of embeddings (lists of floats).
            metadatas: An optional list of dictionaries containing metadata.
        """
        if not self.collection:
            logging.error("Collection not initialized. Cannot add embeddings.")
            return
        # Validación de entradas (mejorada)
        if not isinstance(ids, list) or not isinstance(embeddings, list):
             logging.error("Invalid input type: ids and embeddings must be lists.")
             # Consider raising TypeError for programmatic errors
             return
        if not ids or not embeddings:
            logging.warning("Empty ids or embeddings list provided. Nothing to add.")
            return
        if len(ids) != len(embeddings):
            logging.error(f"Mismatch between number of IDs ({len(ids)}) and embeddings ({len(embeddings)}).")
            return
        if metadatas:
             if not isinstance(metadatas, list):
                  logging.error("Invalid input type: metadatas must be a list.")
                  return
             if len(ids) != len(metadatas):
                logging.error(f"Mismatch between number of IDs ({len(ids)}) and metadatas ({len(metadatas)}).")
                return

        logging.info(f"Adding/updating {len(ids)} embeddings to collection '{self.collection_name}'...")
        try:
            # Use upsert: adds if ID doesn't exist, updates if it does.
            self.collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)
            logging.info(f"Upsert successful. New collection count: {self.collection.count()}")
        except Exception as e: # Captura general para errores de upsert
            logging.error(f"Error during upsert operation: {e}", exc_info=True)
            # Considerar logging de IDs fallidos o estrategias de reintento si es aplicable

    def query_similar(
        self, query_embedding: List[float], n_results: int = 5
    ) -> Optional[Dict[str, Any]]:
        """
        Performs a similarity search using a query embedding.

        Args:
            query_embedding: The embedding (list of floats) to search for.
            n_results: The maximum number of similar results to return.

        Returns:
            A dictionary containing the search results (ids, distances, metadatas),
            or None if the query fails or the collection is empty/uninitialized.
        """
        if not self.collection:
            logging.error("Collection not initialized. Cannot query.")
            return None
        if not query_embedding or not isinstance(query_embedding, list):
             logging.error("Invalid query embedding provided (must be a non-empty list).")
             return None

        current_count = self.collection.count()
        if current_count == 0:
            logging.warning("Collection is empty. Cannot perform query.")
            # Devolver estructura vacía consistente en lugar de None podría ser preferible
            return {"ids": [], "distances": [], "metadatas": []}

        # Asegurar que n_results no sea mayor que los items disponibles
        effective_n_results = min(n_results, current_count)
        if effective_n_results <= 0:
             logging.warning("n_results is non-positive. No query performed.")
             return {"ids": [], "distances": [], "metadatas": []}


        logging.info(f"Querying collection '{self.collection_name}' for {effective_n_results} nearest neighbors...")
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding], # Query espera una lista de embeddings
                n_results=effective_n_results,
                include=["metadatas", "distances"], # Especificar qué datos incluir
            )
            logging.info("Query successful.")

            # Extraer y simplificar resultados para la única consulta
            if results and all(key in results for key in ["ids", "distances", "metadatas"]):
                single_query_results = {
                    "ids": results["ids"][0] if results["ids"] else [],
                    "distances": results["distances"][0] if results["distances"] else [],
                    "metadatas": results["metadatas"][0] if results["metadatas"] else [],
                }
                return single_query_results
            else:
                logging.warning("Query returned unexpected format or missing keys.")
                return None # O devolver estructura vacía

        except Exception as e: # Captura general para errores de query
            logging.error(f"Error during query operation: {e}", exc_info=True)
            return None

    def get_all_embeddings_with_ids(self) -> Optional[Tuple[List[str], np.ndarray]]:
        """
        Retrieves all embeddings and their corresponding IDs from the collection.

        Returns:
            A tuple containing: (list of all IDs, NumPy array of all embeddings).
            Returns None if retrieval fails or collection uninitialized.
            Returns ([], np.array([])) if collection is empty.
        """
        if not self.collection:
            logging.error("Collection not initialized. Cannot get all embeddings.")
            return None

        count = self.collection.count()
        if count == 0:
            logging.info("Collection is empty. Returning empty lists/arrays.")
            return [], np.array([], dtype=np.float32) # Devolver estructuras vacías

        logging.info(f"Retrieving all {count} embeddings and IDs from '{self.collection_name}'...")
        try:
            # Retrieve all items. Considerar batching para colecciones MUY grandes si 'get' da problemas de memoria.
            results = self.collection.get(
                include=["embeddings"] # Solo pedir embeddings, IDs vienen por defecto
            )

            if results and "ids" in results and "embeddings" in results:
                ids = results["ids"]
                embeddings_list = results["embeddings"]

                if embeddings_list is None:
                    logging.error("Embeddings were not returned by collection.get().")
                    return None
                if len(ids) != len(embeddings_list):
                     logging.error(f"Mismatch between retrieved IDs ({len(ids)}) and embeddings ({len(embeddings_list)}).")
                     return None

                # Convertir a NumPy array para eficiencia (usado en clustering)
                embeddings_array = np.array(embeddings_list, dtype=np.float32)
                logging.info(f"Successfully retrieved {len(ids)} IDs and embeddings array of shape {embeddings_array.shape}.")
                return ids, embeddings_array
            else:
                logging.error("Failed to retrieve valid IDs or embeddings from the collection.")
                return None

        except Exception as e: # Captura general para errores de 'get'
            logging.error(f"Error retrieving all embeddings: {e}", exc_info=True)
            return None

    def clear_collection(self):
        """Deletes all items from the collection."""
        if not self.collection:
            logging.error("Collection not initialized. Cannot clear.")
            return

        count = self.collection.count()
        if count == 0:
            logging.info(f"Collection '{self.collection_name}' is already empty.")
            return

        logging.warning(f"Attempting to clear ALL {count} items from collection '{self.collection_name}'...")
        try:
            # Obtener todos los IDs para eliminarlos (ChromaDB no tiene un 'delete_all' directo)
            # Esto puede ser ineficiente para colecciones masivas.
            all_ids = self.collection.get(include=[])['ids'] # Solo necesitamos IDs
            if all_ids:
                # Eliminar en lotes si la colección es muy grande podría ser más seguro
                # chunk_size = 1000
                # for i in range(0, len(all_ids), chunk_size):
                #    batch_ids = all_ids[i:i + chunk_size]
                #    self.collection.delete(ids=batch_ids)
                self.collection.delete(ids=all_ids) # Eliminar todos de una vez si no es masiva
                final_count = self.collection.count()
                logging.info(f"Successfully cleared collection. Items before: {count}, Items after: {final_count}")
                if final_count != 0:
                     logging.warning(f"Collection count is {final_count} after clearing, expected 0.")
            else:
                logging.info("No IDs found to delete, collection might already be empty or error retrieving IDs.")

        except Exception as e: # Captura general para errores de 'delete'
            logging.error(f"Error clearing collection: {e}", exc_info=True)

    def delete_collection(self):
        """Deletes the entire collection itself."""
        if not self.client:
            logging.error("Client not initialized. Cannot delete collection.")
            return

        logging.warning(f"Attempting to DELETE THE ENTIRE collection '{self.collection_name}'...")
        try:
            self.client.delete_collection(name=self.collection_name)
            logging.info(f"Successfully deleted collection '{self.collection_name}'.")
            self.collection = None # Resetear el atributo de la instancia
        except Exception as e: # Captura general (podría ser 'collection not found', etc.)
            logging.error(f"Error deleting collection '{self.collection_name}': {e}", exc_info=True)
