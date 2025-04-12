import torch
from transformers import AutoModel, AutoProcessor, PreTrainedModel
from PIL import Image
from typing import List, Optional, Union, cast, Any
import math
import logging
import time

# Importar configuración relevante
from config import DEVICE, TRUST_REMOTE_CODE, VECTOR_DIMENSION

logger = logging.getLogger(__name__)

class Vectorizer:
    """
    Maneja la carga del modelo de embedding y la vectorización de imágenes y texto.
    Gestiona la carga del modelo/procesador, la ubicación del dispositivo, el procesamiento por lotes,
    la normalización y el truncamiento opcional.
    """

    def __init__(
        self,
        model_name: str,
        device: str = DEVICE,
        trust_remote_code: bool = TRUST_REMOTE_CODE,
    ):
        """
        Inicializa el Vectorizer, cargando el modelo y procesador especificados.

        Args:
            model_name: Nombre del modelo de Hugging Face (ej: "openai/clip-vit-base-patch32").
            device: Dispositivo para ejecutar la inferencia ('cuda' o 'cpu'). Usa 'cpu' si cuda no está disponible.
            trust_remote_code: Permite ejecutar código personalizado del repositorio del modelo.

        Raises:
            RuntimeError: Si el modelo o procesador no se cargan correctamente.
        """
        self.model_name = model_name
        self._resolved_device = self._resolve_device(device)
        self.trust_remote_code = trust_remote_code
        self.model: Optional[PreTrainedModel] = None
        self.processor: Optional[Any] = None # Usar Any para compatibilidad amplia
        self._is_loaded = False

        logger.info(f"Inicializando Vectorizer con modelo: {self.model_name} en dispositivo: {self._resolved_device}")
        self._load_model_and_processor()

    def _resolve_device(self, requested_device: str) -> str:
        """Determina el dispositivo real a usar, recurriendo a CPU si CUDA no está disponible."""
        if requested_device.lower() == 'cuda':
            if torch.cuda.is_available():
                logger.info("CUDA disponible. Usando GPU.")
                return 'cuda'
            else:
                logger.warning("CUDA especificado pero no disponible. Usando CPU.")
                return 'cpu'
        else:
            logger.info("Usando CPU.")
            return 'cpu'

    def _load_model_and_processor(self):
        """Carga el modelo y procesador de Hugging Face."""
        if self._is_loaded:
            logger.info("Modelo y procesador ya cargados.")
            return

        try:
            # Cargar Modelo
            logger.info(f"Cargando modelo '{self.model_name}'...")
            start_time = time.time()
            model = AutoModel.from_pretrained(
                self.model_name, trust_remote_code=self.trust_remote_code
            )
            # Mover el modelo al dispositivo resuelto
            self.model = model.to(self._resolved_device)
            self.model.eval() # Poner el modelo en modo de evaluación
            end_time = time.time()
            logger.info(f"Modelo cargado a {self._resolved_device} en {end_time - start_time:.2f}s.")

            # Cargar Procesador
            logger.info(f"Cargando procesador para '{self.model_name}'...")
            start_time = time.time()
            try:
                 self.processor = AutoProcessor.from_pretrained(
                     self.model_name, trust_remote_code=self.trust_remote_code
                 )
            except ValueError as e:
                 # Manejar casos donde un modelo podría no tener un procesador dedicado
                 logger.warning(f"No se pudo cargar AutoProcessor para '{self.model_name}': {e}. Asumiendo que no se requiere.")
                 self.processor = None # Establecer explícitamente a None

            end_time = time.time()
            if self.processor:
                logger.info(f"Procesador cargado en {end_time - start_time:.2f}s.")
            else:
                 logger.info(f"Paso de carga del procesador completado (sin procesador cargado/requerido) en {end_time - start_time:.2f}s.")

            self._is_loaded = True # Marcar como cargado exitosamente

        except OSError as e:
            logger.error(f"OSError al cargar modelo/procesador '{self.model_name}': {e}. ¿Nombre correcto? ¿Acceso a internet?", exc_info=True)
            self._is_loaded = False
            raise RuntimeError(f"Fallo al cargar modelo/procesador: {e}") from e
        except ImportError as e:
            logger.error(f"ImportError durante carga: {e}. ¿Faltan dependencias? (ej: 'pip install transformers[vision]')", exc_info=True)
            self._is_loaded = False
            raise RuntimeError(f"Faltan dependencias para cargar modelo: {e}") from e
        except Exception as e:
            logger.error(f"Error inesperado al cargar modelo o procesador '{self.model_name}': {e}", exc_info=True)
            self._is_loaded = False
            raise RuntimeError(f"Error inesperado durante carga de modelo: {e}") from e

    @property
    def is_ready(self) -> bool:
        """Verifica si el modelo (y el procesador, si aplica) están cargados."""
        # El modelo siempre debe estar cargado. El procesador es opcional.
        return self._is_loaded and self.model is not None

    def _process_batch(
        self,
        batch_data: Union[List[Image.Image], List[str]],
        batch_type: str, # "image" o "text"
        batch_size: int,
        truncate_dim: Optional[int],
        is_query: bool = False # Indicador para modelos que distinguen query/doc
    ) -> List[Optional[List[float]]]:
        """Método interno para procesar una lista de imágenes o textos en lotes."""
        # Verificar si el vectorizador está listo antes de procesar
        if not self.is_ready or self.model is None:
            logger.error("Vectorizer no está listo (modelo no cargado). No se puede procesar el lote.")
            return [None] * len(batch_data)

        # Verificar si se necesita un procesador y si está cargado
        if batch_type == "image" and self.processor is None:
             logger.error("Procesamiento de imágenes requiere un procesador, pero no está cargado.")
             return [None] * len(batch_data)
        # Permitir textos sin procesador (con advertencia)
        if batch_type == "text" and self.processor is None:
             logger.warning("Procesamiento de texto usualmente requiere un procesador. Procediendo sin él, pero los resultados pueden ser inesperados.")

        # Inicializar lista para todos los embeddings del lote completo
        all_embeddings: List[Optional[List[float]]] = []
        # Obtener número total de ítems en este lote de datos
        num_items_total = len(batch_data)
        # Calcular número de sub-lotes necesarios
        num_sub_batches = math.ceil(num_items_total / batch_size)
        logger.debug(f"Procesando {num_items_total} ítems de tipo '{batch_type}' en {num_sub_batches} sub-lotes de tamaño {batch_size}.")

        # Asegurar que el modelo no es None (ya verificado por self.is_ready)
        model = cast(PreTrainedModel, self.model)
        processor = self.processor # Puede ser None

        # Iterar sobre los sub-lotes
        for i in range(num_sub_batches):
            sub_batch_start_time = time.time()
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, num_items_total)
            current_sub_batch = batch_data[start_idx:end_idx]
            sub_batch_len = len(current_sub_batch)

            # Saltar si el sub-lote está vacío
            if not current_sub_batch:
                logger.debug(f"Sub-lote {i+1}/{num_sub_batches} está vacío, saltando.")
                continue

            # Pre-rellenar resultados del sub-lote con None
            sub_batch_results: List[Optional[List[float]]] = [None] * sub_batch_len

            try:
                # Ejecutar inferencia sin calcular gradientes
                with torch.no_grad():
                    if batch_type == "image":
                        if not processor: # Doble verificación por si acaso
                             raise RuntimeError("Procesador es None, no se pueden procesar imágenes.")
                        if not all(isinstance(item, Image.Image) for item in current_sub_batch):
                            raise TypeError("Lote de imágenes debe contener solo objetos PIL Image.")
                        # Procesar imágenes
                        inputs = processor(images=current_sub_batch, return_tensors="pt", padding=True).to(self._resolved_device)
                        embeddings_tensor = model.get_image_features(**inputs)

                    elif batch_type == "text":
                        if not all(isinstance(item, str) for item in current_sub_batch):
                            raise TypeError("Lote de texto debe contener solo strings.")
                        # Procesar texto
                        if processor:
                            inputs = processor(
                                text=current_sub_batch, return_tensors="pt", padding=True, truncation=True
                            ).to(self._resolved_device)
                            embeddings_tensor = model.get_text_features(**inputs)
                        else:
                            # Manejar modelos sin procesador estándar (adaptar según API del modelo)
                            embeddings_tensor = model(current_sub_batch) # ¡Adaptar si es necesario!
                            if not isinstance(embeddings_tensor, torch.Tensor):
                                 embeddings_tensor = torch.tensor(embeddings_tensor).to(self._resolved_device)
                    else:
                        raise ValueError(f"Tipo de lote inválido: {batch_type}")

                    # --- Post-procesamiento: Normalización y Truncamiento ---
                    # 1. Normalizar embeddings originales (norma L2)
                    embeddings_tensor = torch.nn.functional.normalize(embeddings_tensor, p=2, dim=-1)

                    # 2. Truncar si se especifica Y la dimensión original es mayor
                    was_truncated = False
                    original_dim = embeddings_tensor.shape[-1]
                    if truncate_dim is not None and original_dim > truncate_dim:
                        embeddings_tensor = embeddings_tensor[:, :truncate_dim]
                        was_truncated = True
                    # (Opcional: Añadir logs si se desea para truncamiento)

                    # 3. Re-normalizar DESPUÉS del truncamiento (si se truncó)
                    if was_truncated:
                        embeddings_tensor = torch.nn.functional.normalize(embeddings_tensor, p=2, dim=-1)

                    # Convertir a lista de listas (floats)
                    valid_embeddings_list = embeddings_tensor.cpu().numpy().tolist()

                    # Verificar si el número de resultados coincide con el tamaño del sub-lote
                    if len(valid_embeddings_list) == sub_batch_len:
                        sub_batch_results = valid_embeddings_list # Asignar resultados exitosos
                    else:
                        logger.error(f"Sub-lote {i+1}/{num_sub_batches}: ¡Discrepancia en tamaño de salida! Esperado {sub_batch_len}, obtenido {len(valid_embeddings_list)}. Rellenando con None.")
                        # Mantener sub_batch_results lleno de None

            except Exception as e:
                # Capturar cualquier error durante la vectorización del sub-lote
                logger.error(f"Error vectorizando sub-lote {i+1}/{num_sub_batches} de tipo '{batch_type}': {e}", exc_info=True)
                # Mantener sub_batch_results lleno de None en caso de error

            # Añadir los resultados (o Nones) de este sub-lote a la lista general
            all_embeddings.extend(sub_batch_results)
            sub_batch_end_time = time.time()
            logger.debug(f"Sub-lote {i+1}/{num_sub_batches} ({batch_type}) procesado en {sub_batch_end_time - sub_batch_start_time:.3f}s.")

        # --- Verificación Final (DENTRO de _process_batch) ---
        # Asegurar que la longitud de la lista de salida coincida con la de entrada
        if len(all_embeddings) != num_items_total:
            logger.warning(f"Longitud final de embeddings ({len(all_embeddings)}) no coincide con datos de entrada ({num_items_total}). Esto puede indicar fallos parciales en lotes.")
            # Rellenar con None si es necesario para mantener la correspondencia de índices
            all_embeddings.extend([None] * (num_items_total - len(all_embeddings)))

        # Devolver la lista completa de embeddings (o Nones) para el lote original
        return all_embeddings


    def vectorize_images(
        self,
        images: List[Image.Image],
        batch_size: int,
        truncate_dim: Optional[int] = None,
    ) -> List[Optional[List[float]]]:
        """
        Vectoriza una lista de objetos PIL Image.

        Args:
            images: Lista de objetos PIL.Image.Image.
            batch_size: Número de imágenes a procesar en cada lote de inferencia.
            truncate_dim: Dimensión opcional a la que truncar los embeddings. Usa la dimensión completa si es None.

        Returns:
            Lista de embeddings (lista de floats). Devuelve None para imágenes que fallaron.
            La longitud de la lista devuelta coincide con la longitud de la lista 'images' de entrada.
        """
        if not self.is_ready:
            logger.error("Vectorizer no está listo. No se pueden vectorizar imágenes.")
            return [None] * len(images) if images else []
        if not images or not isinstance(images, list):
            logger.warning("No se proporcionó una lista válida de imágenes a vectorize_images. Devolviendo lista vacía.")
            return []
        if not all(isinstance(img, Image.Image) for img in images):
            logger.error("Lista de entrada para vectorize_images contiene objetos que no son PIL Image. Devolviendo lista de Nones.")
            return [None] * len(images)

        num_images = len(images)
        # Usar dimensión de config si truncate_dim no se proporciona explícitamente
        effective_truncate_dim = truncate_dim if truncate_dim is not None else VECTOR_DIMENSION
        logger.info(f"Iniciando vectorización de {num_images} imágenes (Tamaño Lote: {batch_size}, Dim Truncada: {effective_truncate_dim or 'Completa'})...")

        # Medición de tiempo
        start_time_cpu = time.time()
        start_event, end_event = None, None
        if self._resolved_device == 'cuda':
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

        # Llamar al método interno de procesamiento por lotes
        embeddings = self._process_batch(images, "image", batch_size, effective_truncate_dim)

        # Calcular tiempo transcurrido
        elapsed_time = 0
        if self._resolved_device == 'cuda' and start_event and end_event:
            end_event.record()
            torch.cuda.synchronize() # Esperar a que terminen las operaciones de GPU
            elapsed_time = start_event.elapsed_time(end_event) / 1000.0 # Tiempo en segundos
        else:
            end_time_cpu = time.time()
            elapsed_time = end_time_cpu - start_time_cpu

        # Resumen de resultados
        num_successful = sum(1 for emb in embeddings if emb is not None)
        num_failed = num_images - num_successful
        logger.info(f"Vectorización de {num_images} imágenes finalizada en {elapsed_time:.2f}s. Éxitos: {num_successful}, Fallos: {num_failed}")
        if num_failed > 0:
            logger.warning(f"{num_failed} imágenes fallaron en la vectorización (ver errores anteriores).")

        # Devolver la lista de embeddings (DENTRO de la función)
        return embeddings


    def vectorize_texts(
        self,
        texts: List[str],
        batch_size: int,
        truncate_dim: Optional[int] = None,
        is_query: bool = False, # Pasar indicador de query al procesador de lotes
    ) -> List[Optional[List[float]]]:
        """
        Vectoriza una lista de strings de texto.

        Args:
            texts: Lista de strings.
            batch_size: Número de textos a procesar en cada lote de inferencia.
            truncate_dim: Dimensión opcional a la que truncar los embeddings. Usa la dimensión completa si es None.
            is_query: Indicador de si estos textos son consultas de búsqueda.

        Returns:
            Lista de embeddings (lista de floats). Devuelve None para textos que fallaron.
            La longitud de la lista devuelta coincide con la longitud de la lista 'texts' de entrada.
        """
        if not self.is_ready:
            logger.error("Vectorizer no está listo. No se pueden vectorizar textos.")
            return [None] * len(texts) if texts else []
        if not texts or not isinstance(texts, list):
            logger.warning("No se proporcionó una lista válida de textos a vectorize_texts. Devolviendo lista vacía.")
            return []
        if not all(isinstance(t, str) for t in texts):
            logger.error("Lista de entrada para vectorize_texts contiene objetos que no son string. Devolviendo lista de Nones.")
            return [None] * len(texts)

        num_texts = len(texts)
        effective_truncate_dim = truncate_dim if truncate_dim is not None else VECTOR_DIMENSION
        logger.info(f"Iniciando vectorización de {num_texts} textos (Tamaño Lote: {batch_size}, Dim Truncada: {effective_truncate_dim or 'Completa'}, Es Query: {is_query})...")
        start_time = time.time()

        # Llamar al método interno de procesamiento por lotes
        embeddings = self._process_batch(texts, "text", batch_size, effective_truncate_dim, is_query=is_query)

        end_time = time.time()
        elapsed_time = end_time - start_time

        # Resumen de resultados
        num_successful = sum(1 for emb in embeddings if emb is not None)
        # Usar num_texts aquí, ya que es el total de ítems de entrada para esta función
        num_failed = num_texts - num_successful
        logger.info(f"Vectorización de {num_texts} textos finalizada en {elapsed_time:.2f}s. Éxitos: {num_successful}, Fallos: {num_failed}")
        if num_failed > 0:
            logger.warning(f"{num_failed} textos fallaron en la vectorización (ver errores anteriores).")

        # Devolver la lista de embeddings (DENTRO de la función)
        return embeddings