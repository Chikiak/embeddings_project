# data_access/chroma_db.py
import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.errors import InvalidDimensionException
from typing import List, Dict, Optional, Tuple, Any
import os
import time
import numpy as np
import logging

# Importar configuración y la interfaz
from config import VECTOR_DIMENSION
from .vector_db_interface import VectorDBInterface
from app.models import SearchResultItem, SearchResults # Importar modelos de datos

logger = logging.getLogger(__name__)

class ChromaVectorDB(VectorDBInterface):
    """
    Implementación concreta de VectorDBInterface usando ChromaDB.
    Maneja interacciones con una colección persistente de ChromaDB.
    Incluye lógica para recrear la colección si se elimina.
    """

    def __init__(
        self, path: str, collection_name: str
    ):
        """
        Inicializa el cliente ChromaDB e intenta obtener/crear la colección.

        Args:
            path: Ruta del directorio para almacenamiento persistente.
            collection_name: Nombre de la colección dentro de ChromaDB.

        Raises:
            RuntimeError: Si la creación del directorio o la inicialización del cliente falla.
                          La creación de la colección se reintentará más tarde si falla aquí.
        """
        self.path = path
        self.collection_name = collection_name
        self.client: Optional[chromadb.ClientAPI] = None
        self.collection: Optional[Collection] = None
        self._client_initialized = False # Indica si el cliente se inicializó

        logger.info(f"Inicializando ChromaVectorDB:")
        logger.info(f"  Storage Path: {os.path.abspath(self.path)}")
        logger.info(f"  Collection Name: {self.collection_name}")

        try:
            os.makedirs(self.path, exist_ok=True)
            logger.debug(f"Directorio de la base de datos asegurado en: {self.path}")
        except OSError as e:
            logger.error(f"Error creando/accediendo al directorio DB {self.path}: {e}", exc_info=True)
            # Si no se puede crear el directorio, es un error fatal temprano
            raise RuntimeError(f"Fallo al crear/acceder al directorio DB: {self.path}") from e

        try:
            # Inicializar solo el cliente aquí
            self.client = chromadb.PersistentClient(path=self.path)
            logger.info("Cliente Persistente ChromaDB inicializado.")
            self._client_initialized = True
            # Intentar asegurar la colección una vez al inicio, pero no fallar si no se puede aún
            self._ensure_collection_exists()

        except Exception as e:
            logger.error(f"Error inesperado inicializando el cliente ChromaDB: {e}", exc_info=True)
            self._cleanup_client() # Limpiar estado del cliente
            # No lanzar error aquí, permitir que _ensure_collection_exists lo maneje después
            # raise RuntimeError(f"Inicialización del cliente ChromaDB falló: {e}") from e

    def _cleanup_client(self):
        """Método interno para resetear el estado del cliente."""
        self.client = None
        self.collection = None
        self._client_initialized = False
        logger.warning("Estado del cliente ChromaDB reseteado debido a un error.")

    def _ensure_collection_exists(self) -> bool:
        """
        Verifica si la colección está disponible y la crea si no existe.
        Este es el método clave para la recuperación después de la eliminación.

        Returns:
            True si la colección existe o se creó exitosamente, False en caso contrario.
        """
        # Verificar primero si el cliente está inicializado
        if not self._client_initialized or self.client is None:
            logger.error("El cliente ChromaDB no está inicializado. No se puede asegurar la colección.")
            # Intentar reinicializar el cliente podría ser una opción aquí, pero aumenta la complejidad.
            # Por ahora, asumimos que si el cliente falló, es un problema más grave.
            return False

        # Si ya tenemos un objeto de colección, asumir que está bien (get_or_create es idempotente)
        # Podríamos añadir una verificación más robusta si fuera necesario (ej. self.collection.count())
        if self.collection is not None:
            return True

        # Si no hay objeto de colección, intentar obtenerlo o crearlo
        logger.info(f"Intentando asegurar la existencia de la colección '{self.collection_name}'...")
        try:
            # Usar 'cosine' distance, adecuado para embeddings normalizados
            metadata_options = {"hnsw:space": "cosine"}
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata=metadata_options,
            )
            collection_count = self.collection.count() # Obtener cuenta actual
            logger.info(f"Colección '{self.collection_name}' asegurada/creada. Cuenta actual: {collection_count}")
            return True
        except InvalidDimensionException as e_dim:
            # Este error es más problemático y puede indicar un problema de configuración
            logger.error(f"Error de dimensión inválida al asegurar la colección '{self.collection_name}': {e_dim}", exc_info=True)
            self.collection = None # Asegurar que la colección quede como None
            return False
        except Exception as e:
            logger.error(f"Error inesperado al obtener/crear la colección '{self.collection_name}': {e}", exc_info=True)
            self.collection = None # Asegurar que la colección quede como None
            return False

    @property
    def is_initialized(self) -> bool:
        """Verifica si el cliente está inicializado y la colección está accesible."""
        # Consideramos inicializado si el cliente está ok Y podemos asegurar la colección
        return self._client_initialized and self._ensure_collection_exists()

    def add_embeddings(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """Añade o actualiza embeddings en la colección ChromaDB."""
        # Asegurar que la colección exista antes de la operación
        if not self._ensure_collection_exists() or self.collection is None:
            logger.error("La colección no está disponible. No se pueden añadir embeddings.")
            return False

        # --- (Resto de validaciones de add_embeddings como antes) ---
        if not isinstance(ids, list) or not isinstance(embeddings, list):
             logger.error("Tipo de entrada inválido: ids y embeddings deben ser listas.")
             return False
        if not ids or not embeddings:
             logger.warning("Lista de ids o embeddings vacía. Nada que añadir.")
             return True
        if len(ids) != len(embeddings):
             logger.error(f"Discrepancia de entrada: IDs ({len(ids)}) vs embeddings ({len(embeddings)}).")
             return False
        if metadatas:
             if not isinstance(metadatas, list) or len(ids) != len(metadatas):
                 logger.error(f"Discrepancia en lista de metadatos o tipo inválido: IDs ({len(ids)}) vs metadatos ({len(metadatas) if isinstance(metadatas, list) else 'Tipo Inválido'}).")
                 return False
        else:
             metadatas = [{} for _ in ids]

        if embeddings:
             expected_dim = len(embeddings[0])
             if VECTOR_DIMENSION and expected_dim != VECTOR_DIMENSION:
                 logger.warning(f"Dimensión de embedding proporcionada ({expected_dim}) difiere de config ({VECTOR_DIMENSION}). Asegurar consistencia.")
             if not all(len(emb) == expected_dim for emb in embeddings):
                 logger.error("Dimensiones de embedding inconsistentes encontradas en el lote.")
                 for i, emb in enumerate(embeddings):
                     if len(emb) != expected_dim:
                         logger.error(f"  - ID '{ids[i]}' tiene dimensión {len(emb)}, esperado {expected_dim}")
                         break
                 return False
        # --- Fin de validaciones ---

        num_items_to_add = len(ids)
        logger.info(f"Intentando añadir/actualizar {num_items_to_add} embeddings en la colección '{self.collection_name}'...")
        try:
            self.collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)
            # Opcional: verificar la cuenta después, aunque puede ser costoso
            # new_count = self.collection.count()
            # logger.info(f"Upsert exitoso. La cuenta de la colección es ahora: {new_count}")
            logger.info(f"Upsert para {num_items_to_add} ítems completado.")
            return True
        except InvalidDimensionException as e_dim:
            logger.error(f"Error ChromaDB durante upsert: Dimensión inválida. {e_dim}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"Error durante operación upsert en colección '{self.collection_name}': {e}", exc_info=True)
            return False

    def query_similar(
        self, query_embedding: List[float], n_results: int = 5
    ) -> Optional[SearchResults]:
        """Consulta ChromaDB por embeddings similares y devuelve SearchResults."""
        # Asegurar que la colección exista antes de la operación
        if not self._ensure_collection_exists() or self.collection is None:
            logger.error("La colección no está disponible. No se puede consultar.")
            return None # Indicar fallo

        if not query_embedding or not isinstance(query_embedding, list):
            logger.error("Embedding de consulta inválido (debe ser lista de floats).")
            return None

        current_count = self.count() # Usa el método count() que también asegura la colección
        if current_count == 0:
            logger.warning("Consulta intentada en una colección vacía.")
            return SearchResults(items=[])
        if current_count == -1: # Error al contar
             logger.error("No se puede consultar, fallo al obtener la cuenta de la colección.")
             return None

        effective_n_results = min(n_results, current_count)
        if effective_n_results <= 0:
            logger.warning(f"n_results efectivo es {effective_n_results}. Devolviendo resultados vacíos.")
            return SearchResults(items=[])

        logger.info(f"Consultando colección '{self.collection_name}' por {effective_n_results} vecinos más cercanos...")
        try:
            start_time = time.time()
            results_dict = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=effective_n_results,
                include=["metadatas", "distances"],
            )
            end_time = time.time()
            logger.info(f"Consulta ChromaDB ejecutada en {end_time - start_time:.3f}s.")

            # Procesar resultados en objeto SearchResults
            result_items = []
            if (
                results_dict and
                all(key in results_dict for key in ["ids", "distances", "metadatas"]) and
                results_dict["ids"] and
                results_dict["ids"][0]
            ):
                ids_list = results_dict["ids"][0]
                distances_list = results_dict["distances"][0]
                metadatas_list = results_dict["metadatas"][0] if results_dict["metadatas"] else [{}] * len(ids_list)

                if not (len(ids_list) == len(distances_list) == len(metadatas_list)):
                     logger.error(f"CRÍTICO: Discrepancia en longitudes devueltas por consulta Chroma: ids({len(ids_list)}), dist({len(distances_list)}), meta({len(metadatas_list)})")
                     return None

                for i, img_id in enumerate(ids_list):
                    item = SearchResultItem(
                        id=img_id,
                        distance=distances_list[i],
                        metadata=metadatas_list[i] if metadatas_list[i] is not None else {}
                    )
                    result_items.append(item)
                logger.info(f"Consulta exitosa. Encontrados {len(result_items)} resultados.")
                return SearchResults(items=result_items, query_vector=query_embedding)
            else:
                logger.info("Consulta exitosa, pero no se encontraron documentos coincidentes.")
                return SearchResults(items=[], query_vector=query_embedding)

        except InvalidDimensionException as e_dim:
            logger.error(f"Error ChromaDB durante consulta: Dimensión inválida. Dim Query: {len(query_embedding)}. {e_dim}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Error durante operación de consulta en colección '{self.collection_name}': {e}", exc_info=True)
            return None

    def get_all_embeddings_with_ids(
        self, pagination_batch_size: int = 1000
    ) -> Optional[Tuple[List[str], np.ndarray]]:
        """Recupera todos los embeddings e IDs usando paginación."""
        # Asegurar que la colección exista antes de la operación
        if not self._ensure_collection_exists() or self.collection is None:
            logger.error("La colección no está disponible. No se pueden obtener todos los embeddings.")
            return None

        fallback_dim = VECTOR_DIMENSION if isinstance(VECTOR_DIMENSION, int) else 1
        empty_result: Tuple[List[str], np.ndarray] = ([], np.empty((0, fallback_dim), dtype=np.float32))

        try:
            count = self.count() # Usa el método count()
            if count == 0:
                logger.info("La colección está vacía. Devolviendo listas/arrays vacíos.")
                return empty_result
            if count == -1:
                logger.error("Fallo al obtener la cuenta de la colección antes de recuperar todo.")
                return None

            logger.info(f"Recuperando todos los {count} embeddings e IDs de '{self.collection_name}' usando paginación (tamaño lote: {pagination_batch_size})...")
            start_time = time.time()

            all_ids = []
            all_embeddings_list = []
            retrieved_count = 0

            while retrieved_count < count:
                logger.debug(f"Recuperando lote offset={retrieved_count}, limit={pagination_batch_size}")
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
                            logger.warning(f"Lote vacío recuperado en offset {retrieved_count} a pesar de count={count}. Deteniendo.")
                            break
                        all_ids.extend(batch_ids)
                        all_embeddings_list.extend(batch_embeddings)
                        retrieved_count += len(batch_ids)
                        logger.debug(f"Recuperados {len(batch_ids)} ítems. Total recuperado: {retrieved_count}/{count}")
                    else:
                        logger.warning(f"Formato de resultado inesperado o lote vacío en offset {retrieved_count}. Deteniendo. Resultado: {results}")
                        break
                except Exception as e_get:
                    logger.error(f"Error recuperando lote en offset {retrieved_count}: {e_get}", exc_info=True)
                    return None

            end_time = time.time()
            logger.info(f"Recuperación de datos finalizada en {end_time - start_time:.2f}s. Recuperados {retrieved_count} ítems.")

            if len(all_ids) != len(all_embeddings_list):
                logger.error(f"CRÍTICO: Discrepancia después de paginación: IDs ({len(all_ids)}) vs Embeddings ({len(all_embeddings_list)}).")
                return None
            if retrieved_count != count:
                 logger.warning(f"Cuenta final recuperada ({retrieved_count}) difiere de cuenta inicial ({count}). Datos pueden haber cambiado.")

            try:
                if not all_embeddings_list:
                     logger.warning("Lista de embeddings vacía después del bucle de recuperación.")
                     return empty_result
                embeddings_array = np.array(all_embeddings_list, dtype=np.float32)
                logger.info(f"Recuperados exitosamente {len(all_ids)} IDs y array de embeddings de forma {embeddings_array.shape}.")
                return all_ids, embeddings_array
            except ValueError as ve:
                logger.error(f"Error convirtiendo lista de embeddings a array NumPy: {ve}. Verificar consistencia de datos.", exc_info=True)
                return None

        except Exception as e:
            logger.error(f"Error recuperando todos los embeddings de la colección '{self.collection_name}': {e}", exc_info=True)
            return None

    def clear_collection(self) -> bool:
        """Elimina todos los ítems de la colección ChromaDB."""
        # Asegurar que la colección exista antes de la operación (aunque vaciar una inexistente no da error)
        if not self._ensure_collection_exists() or self.collection is None:
            logger.error("La colección no está disponible. No se puede limpiar.")
            return False
        try:
            count_before = self.count() # Usa el método count()
            if count_before == 0:
                logger.info(f"La colección '{self.collection_name}' ya está vacía.")
                return True
            if count_before == -1:
                 logger.error("No se puede limpiar la colección, fallo al obtener la cuenta.")
                 return False

            logger.warning(f"Intentando limpiar TODOS los {count_before} ítems de la colección '{self.collection_name}'...")
            start_time = time.time()
            # Usar delete sin filtro para eliminar todo
            self.collection.delete()
            end_time = time.time()
            count_after = self.count()

            if count_after == 0:
                logger.info(f"Colección '{self.collection_name}' limpiada exitosamente en {end_time-start_time:.2f}s. Ítems eliminados: {count_before}.")
                return True
            else:
                logger.error(f"La cuenta de la colección es {count_after} después de limpiar, se esperaba 0. Ítems antes: {count_before}.")
                return False
        except Exception as e:
            logger.error(f"Error limpiando la colección '{self.collection_name}': {e}", exc_info=True)
            return False

    def delete_collection(self) -> bool:
        """Elimina toda la colección ChromaDB."""
        # Necesitamos el cliente para esto
        if not self._client_initialized or self.client is None:
            logger.error("Cliente no inicializado. No se puede eliminar la colección.")
            return False
        if not self.collection_name:
            logger.error("Nombre de colección no establecido. No se puede eliminar.")
            return False

        logger.warning(f"Intentando ELIMINAR TODA la colección '{self.collection_name}' de la ruta '{self.path}'...")
        try:
            start_time = time.time()
            # Llamar a delete_collection en el cliente
            self.client.delete_collection(name=self.collection_name)
            end_time = time.time()
            logger.info(f"Colección '{self.collection_name}' eliminada exitosamente en {end_time-start_time:.2f}s.")
            # Resetear el estado interno ya que la colección ya no existe
            self.collection = None
            # No necesariamente reseteamos _client_initialized, el cliente sigue ahí
            return True
        except Exception as e:
            # Capturar errores potenciales (ej. colección no encontrada, problemas de permisos)
            logger.error(f"Error eliminando la colección '{self.collection_name}': {e}", exc_info=True)
            # Posiblemente la colección ya no existía
            # Podemos intentar verificar si existe después del error si es necesario
            return False

    def count(self) -> int:
        """Devuelve el número de ítems en la colección."""
        # Asegurar que la colección exista antes de la operación
        if not self._ensure_collection_exists() or self.collection is None:
            logger.warning("Solicitud de cuenta pero la colección no está disponible.")
            # Devolver 0 podría ser engañoso si la colección no existe por un error
            # Devolver -1 indica un estado indeterminado/error
            return -1
        try:
            return self.collection.count()
        except Exception as e:
            logger.error(f"Error obteniendo la cuenta para la colección '{self.collection_name}': {e}", exc_info=True)
            return -1 # Indicar error
