# core/vectorizer.py
import torch
from transformers import AutoModel, AutoProcessor, PreTrainedModel
from PIL import Image
from typing import List, Optional, Union, Any, Dict
import math
import logging
import time

from config import DEVICE, TRUST_REMOTE_CODE, VECTOR_DIMENSION
from app.exceptions import VectorizerError

logger = logging.getLogger(__name__)


class Vectorizer:
    """
    Maneja la carga del modelo de embedding y la vectorización de imágenes y texto.

    Gestiona la carga del modelo/procesador, la ubicación del dispositivo,
    el procesamiento por lotes, la normalización y el truncamiento opcional.

    Attributes:
        model_name: Nombre del modelo de Hugging Face.
        device: Dispositivo real utilizado para la inferencia ('cuda' o 'cpu').
        trust_remote_code: Booleano que indica si se confía en el código remoto del modelo.
        model: La instancia del modelo PreTrainedModel cargado.
        processor: La instancia del procesador cargado (puede ser None).
        is_ready: Booleano que indica si el vectorizador está listo para usarse.
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
            device: Dispositivo preferido para ejecutar la inferencia ('cuda' o 'cpu').
            trust_remote_code: Permite ejecutar código personalizado del repo del modelo.

        Raises:
            VectorizerError: Si el modelo o procesador no se cargan correctamente.
        """
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.trust_remote_code = trust_remote_code
        self.model: Optional[PreTrainedModel] = None
        self.processor: Optional[Any] = None
        self._is_loaded = False

        logger.info(
            f"Initializing Vectorizer with model: {self.model_name} on device: {self.device}"
        )
        self._load_model_and_processor()

    @property
    def is_ready(self) -> bool:
        """Verifica si el modelo (y el procesador, si aplica) están cargados."""
        return self._is_loaded and self.model is not None

    def _resolve_device(self, requested_device: str) -> str:
        """Determina el dispositivo real a usar, recurriendo a CPU si CUDA no está disponible."""
        resolved = "cpu"
        if requested_device.lower() == "cuda":
            if torch.cuda.is_available():
                logger.info("CUDA available. Using GPU.")
                resolved = "cuda"
            else:
                logger.warning("CUDA specified but not available. Falling back to CPU.")
        else:
            logger.info("Using CPU.")
        return resolved

    def _load_model_and_processor(self):
        """Carga el modelo y procesador de Hugging Face."""
        if self._is_loaded:
            logger.info("Model and processor already loaded.")
            return

        try:
            logger.info(f"Loading model '{self.model_name}'...")
            start_time = time.time()
            model = AutoModel.from_pretrained(
                self.model_name, trust_remote_code=self.trust_remote_code
            )
            self.model = model.to(self.device)
            self.model.eval()
            end_time = time.time()
            logger.info(
                f"Model loaded to {self.device} in {end_time - start_time:.2f}s."
            )

            logger.info(f"Loading processor for '{self.model_name}'...")
            start_time = time.time()
            try:
                self.processor = AutoProcessor.from_pretrained(
                    self.model_name, trust_remote_code=self.trust_remote_code
                )
            except ValueError as e:
                logger.warning(
                    f"Could not load AutoProcessor for '{self.model_name}': {e}. Assuming none required."
                )
                self.processor = None

            end_time = time.time()
            if self.processor:
                logger.info(f"Processor loaded in {end_time - start_time:.2f}s.")
            else:
                logger.info(
                    f"Processor loading step completed (none loaded/required) in {end_time - start_time:.2f}s."
                )

            self._is_loaded = True

        except OSError as e:
            msg = f"OSError loading model/processor '{self.model_name}': {e}. Correct name? Internet access?"
            logger.error(msg, exc_info=True)
            self._is_loaded = False
            raise VectorizerError(msg) from e
        except ImportError as e:
            msg = f"ImportError during loading: {e}. Missing dependencies? (e.g., 'pip install transformers[vision]')"
            logger.error(msg, exc_info=True)
            self._is_loaded = False
            raise VectorizerError(msg) from e
        except Exception as e:
            msg = (
                f"Unexpected error loading model or processor '{self.model_name}': {e}"
            )
            logger.error(msg, exc_info=True)
            self._is_loaded = False
            raise VectorizerError(msg) from e

    def _prepare_inputs(
        self, batch_data: Union[List[Image.Image], List[str]], batch_type: str
    ) -> Dict[str, torch.Tensor]:
        """
        Prepara los tensores de entrada para el modelo usando el procesador.

        Args:
            batch_data: Lista de imágenes PIL o strings de texto.
            batch_type: "image" o "text".

        Returns:
            Diccionario de tensores de entrada listos para el modelo.

        Raises:
            VectorizerError: Si el procesador es necesario pero no está disponible,
                             o si el tipo de lote es inválido.
            TypeError: Si los datos del lote no son del tipo esperado.
        """
        if batch_type == "image":
            if not self.processor:
                raise VectorizerError("Processor is None, cannot process images.")
            if not all(isinstance(item, Image.Image) for item in batch_data):
                raise TypeError("Image batch must contain only PIL Image objects.")
            inputs = self.processor(
                images=batch_data, return_tensors="pt", padding=True
            )

        elif batch_type == "text":
            if not all(isinstance(item, str) for item in batch_data):
                raise TypeError("Text batch must contain only strings.")
            if self.processor:
                inputs = self.processor(
                    text=batch_data, return_tensors="pt", padding=True, truncation=True
                )
            else:
                logger.warning(
                    "Processor is None for text vectorization. Model might handle raw text directly, or this might fail."
                )
                # This part is highly model-dependent if no processor exists
                # Assuming the model can take raw text list directly might be wrong
                # For simplicity, we'll raise if no processor for text for now.
                # If specific models handle raw text, adapt this logic.
                raise VectorizerError(
                    "Processor is None, cannot process text with standard pipeline."
                )
                # Example alternative (if model handles raw text):
                # return {"input_text": batch_data} # Adapt key based on model needs
        else:
            raise VectorizerError(f"Invalid batch type: {batch_type}")

        return inputs.to(self.device)

    def _run_inference(
        self, inputs: Dict[str, torch.Tensor], batch_type: str
    ) -> torch.Tensor:
        """
        Ejecuta la inferencia del modelo para obtener los embeddings.

        Args:
            inputs: Diccionario de tensores de entrada del preprocesamiento.
            batch_type: "image" o "text".

        Returns:
            Tensor de PyTorch con los embeddings crudos.

        Raises:
            VectorizerError: Si el modelo no está cargado o la inferencia falla.
        """
        if not self.model:
            raise VectorizerError("Model is not loaded, cannot run inference.")

        with torch.no_grad():
            if batch_type == "image":
                embeddings_tensor = self.model.get_image_features(**inputs)
            elif batch_type == "text":
                embeddings_tensor = self.model.get_text_features(**inputs)
                # Handle potential alternative for models without processor (if _prepare_inputs was adapted)
                # elif "input_text" in inputs:
                #     embeddings_tensor = self.model(inputs["input_text"]) # Adapt based on model API
                #     if not isinstance(embeddings_tensor, torch.Tensor):
                #         embeddings_tensor = torch.tensor(embeddings_tensor).to(self.device)
            else:
                # Should have been caught earlier, but defensive check
                raise VectorizerError(
                    f"Invalid batch type during inference: {batch_type}"
                )

        return embeddings_tensor

    def _postprocess_embeddings(
        self, embeddings_tensor: torch.Tensor, truncate_dim: Optional[int]
    ) -> List[List[float]]:
        """
        Normaliza, trunca (opcionalmente) y convierte los embeddings a listas de floats.

        Args:
            embeddings_tensor: Tensor de embeddings crudos del modelo.
            truncate_dim: Dimensión objetivo para truncar, o None para no truncar.

        Returns:
            Lista de embeddings procesados (listas de floats).
        """
        embeddings_tensor = torch.nn.functional.normalize(
            embeddings_tensor, p=2, dim=-1
        )

        was_truncated = False
        original_dim = embeddings_tensor.shape[-1]
        if truncate_dim is not None and original_dim > truncate_dim:
            embeddings_tensor = embeddings_tensor[:, :truncate_dim]
            was_truncated = True
            logger.debug(
                f"Truncated embeddings from {original_dim} to {truncate_dim} dimensions."
            )

        if was_truncated:
            embeddings_tensor = torch.nn.functional.normalize(
                embeddings_tensor, p=2, dim=-1
            )
            logger.debug("Re-normalized embeddings after truncation.")

        return embeddings_tensor.cpu().numpy().tolist()

    def _process_batch(
        self,
        batch_data: Union[List[Image.Image], List[str]],
        batch_type: str,
        batch_size: int,
        truncate_dim: Optional[int],
        is_query: bool = False,
    ) -> List[Optional[List[float]]]:
        """
        Método interno refactorizado para procesar un lote de imágenes o textos.

        Orquesta los pasos de preparación de entrada, inferencia y postprocesamiento
        para cada sub-lote.

        Args:
            batch_data: Lista de imágenes PIL o strings de texto.
            batch_type: "image" o "text".
            batch_size: Tamaño de los sub-lotes para la inferencia.
            truncate_dim: Dimensión objetivo para truncar, o None.
            is_query: Indicador para modelos que distinguen query/doc (actualmente no usado
                      en la lógica interna, pero mantenido por si acaso).

        Returns:
            Lista de embeddings procesados (lista de floats) o None para ítems fallidos.
            La longitud coincide con la longitud de batch_data.
        """
        if not self.is_ready:
            logger.error(
                "Vectorizer not ready (model not loaded). Cannot process batch."
            )
            return [None] * len(batch_data)

        all_embeddings: List[Optional[List[float]]] = []
        num_items_total = len(batch_data)
        num_sub_batches = math.ceil(num_items_total / batch_size)
        logger.debug(
            f"Processing {num_items_total} '{batch_type}' items in {num_sub_batches} sub-batches of size {batch_size}."
        )

        for i in range(num_sub_batches):
            sub_batch_start_time = time.time()
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, num_items_total)
            current_sub_batch = batch_data[start_idx:end_idx]
            sub_batch_len = len(current_sub_batch)

            if not current_sub_batch:
                logger.debug(f"Sub-batch {i+1}/{num_sub_batches} is empty, skipping.")
                continue

            sub_batch_results: List[Optional[List[float]]] = [None] * sub_batch_len

            try:
                inputs = self._prepare_inputs(current_sub_batch, batch_type)
                embeddings_tensor = self._run_inference(inputs, batch_type)
                valid_embeddings_list = self._postprocess_embeddings(
                    embeddings_tensor, truncate_dim
                )

                if len(valid_embeddings_list) == sub_batch_len:
                    sub_batch_results = valid_embeddings_list
                else:
                    logger.error(
                        f"Sub-batch {i+1}/{num_sub_batches}: Output size mismatch! Expected {sub_batch_len}, got {len(valid_embeddings_list)}. Filling with None."
                    )

            except (VectorizerError, TypeError, RuntimeError, ValueError) as e:
                logger.error(
                    f"Error vectorizing sub-batch {i+1}/{num_sub_batches} of type '{batch_type}': {e}",
                    exc_info=False,
                )  # Log less verbose traceback usually
            except Exception as e:
                logger.error(
                    f"Unexpected error vectorizing sub-batch {i+1}/{num_sub_batches} of type '{batch_type}': {e}",
                    exc_info=True,
                )

            all_embeddings.extend(sub_batch_results)
            sub_batch_end_time = time.time()
            logger.debug(
                f"Sub-batch {i+1}/{num_sub_batches} ({batch_type}) processed in {sub_batch_end_time - sub_batch_start_time:.3f}s."
            )

        if len(all_embeddings) != num_items_total:
            logger.warning(
                f"Final embeddings length ({len(all_embeddings)}) mismatch with input ({num_items_total}). Padding with None."
            )
            all_embeddings.extend([None] * (num_items_total - len(all_embeddings)))

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
            truncate_dim: Dimensión opcional a la que truncar los embeddings.
                          Usa config.VECTOR_DIMENSION si es None.

        Returns:
            Lista de embeddings (lista de floats). Devuelve None para imágenes que fallaron.
            La longitud de la lista devuelta coincide con la longitud de la lista 'images'.

        Raises:
            VectorizerError: Si el vectorizador no está listo.
            TypeError: Si la entrada 'images' no es una lista o contiene no-imágenes.
        """
        if not self.is_ready:
            raise VectorizerError("Vectorizer not ready. Cannot vectorize images.")
        if not images or not isinstance(images, list):
            logger.warning(
                "Invalid list of images provided to vectorize_images. Returning empty list."
            )
            return []
        if not all(isinstance(img, Image.Image) for img in images):
            raise TypeError(
                "Input list for vectorize_images contains non-PIL Image objects."
            )

        num_images = len(images)
        effective_truncate_dim = (
            truncate_dim if truncate_dim is not None else VECTOR_DIMENSION
        )
        logger.info(
            f"Starting vectorization of {num_images} images (Batch Size: {batch_size}, Truncate Dim: {effective_truncate_dim or 'Full'})..."
        )

        start_time_cpu = time.time()
        start_event, end_event = None, None
        if self.device == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

        embeddings = self._process_batch(
            images, "image", batch_size, effective_truncate_dim
        )

        elapsed_time = 0
        if self.device == "cuda" and start_event and end_event:
            end_event.record()
            torch.cuda.synchronize()
            elapsed_time = start_event.elapsed_time(end_event) / 1000.0
        else:
            end_time_cpu = time.time()
            elapsed_time = end_time_cpu - start_time_cpu

        num_successful = sum(1 for emb in embeddings if emb is not None)
        num_failed = num_images - num_successful
        logger.info(
            f"Image vectorization finished in {elapsed_time:.2f}s. Success: {num_successful}, Failed: {num_failed}"
        )
        if num_failed > 0:
            logger.warning(
                f"{num_failed} images failed vectorization (see previous errors)."
            )

        return embeddings

    def vectorize_texts(
        self,
        texts: List[str],
        batch_size: int,
        truncate_dim: Optional[int] = None,
        is_query: bool = False,
    ) -> List[Optional[List[float]]]:
        """
        Vectoriza una lista de strings de texto.

        Args:
            texts: Lista de strings.
            batch_size: Número de textos a procesar en cada lote de inferencia.
            truncate_dim: Dimensión opcional a la que truncar los embeddings.
                          Usa config.VECTOR_DIMENSION si es None.
            is_query: Indicador de si estos textos son consultas de búsqueda.

        Returns:
            Lista de embeddings (lista de floats). Devuelve None para textos que fallaron.
            La longitud de la lista devuelta coincide con la longitud de la lista 'texts'.

        Raises:
            VectorizerError: Si el vectorizador no está listo.
            TypeError: Si la entrada 'texts' no es una lista o contiene no-strings.
        """
        if not self.is_ready:
            raise VectorizerError("Vectorizer not ready. Cannot vectorize texts.")
        if not texts or not isinstance(texts, list):
            logger.warning(
                "Invalid list of texts provided to vectorize_texts. Returning empty list."
            )
            return []
        if not all(isinstance(t, str) for t in texts):
            raise TypeError(
                "Input list for vectorize_texts contains non-string objects."
            )

        num_texts = len(texts)
        effective_truncate_dim = (
            truncate_dim if truncate_dim is not None else VECTOR_DIMENSION
        )
        logger.info(
            f"Starting vectorization of {num_texts} texts (Batch Size: {batch_size}, Truncate Dim: {effective_truncate_dim or 'Full'}, Is Query: {is_query})..."
        )
        start_time = time.time()

        embeddings = self._process_batch(
            texts, "text", batch_size, effective_truncate_dim, is_query=is_query
        )

        end_time = time.time()
        elapsed_time = end_time - start_time

        num_successful = sum(1 for emb in embeddings if emb is not None)
        num_failed = num_texts - num_successful
        logger.info(
            f"Text vectorization finished in {elapsed_time:.2f}s. Success: {num_successful}, Failed: {num_failed}"
        )
        if num_failed > 0:
            logger.warning(
                f"{num_failed} texts failed vectorization (see previous errors)."
            )

        return embeddings
