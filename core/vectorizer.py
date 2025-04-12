import logging
import math
import time
from typing import Any, Dict, List, Optional, Union

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, PreTrainedModel

DEVICE = "cpu"
TRUST_REMOTE_CODE = True
VECTOR_DIMENSION = None

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
        native_dimension (property): La dimensión de embedding nativa del modelo.
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
        self._native_dimension: Optional[int] = None

        logger.info(
            f"Initializing Vectorizer with model: {self.model_name} on device: {self.device}"
        )
        self._load_model_and_processor()

    @property
    def is_ready(self) -> bool:
        """Verifica si el modelo (y el procesador, si aplica) están cargados."""
        return self._is_loaded and self.model is not None

    @property
    def native_dimension(self) -> Optional[int]:
        """Devuelve la dimensión nativa del embedding del modelo."""

        if self._native_dimension is None and self.is_ready:
            self._determine_native_dimension()
        return self._native_dimension

    def _resolve_device(self, requested_device: str) -> str:
        """Determina el dispositivo real a usar, recurriendo a CPU si CUDA no está disponible."""
        resolved = "cpu"
        if requested_device.lower() == "cuda":
            if torch.cuda.is_available():
                logger.info("CUDA available. Using GPU.")
                resolved = "cuda"
            else:
                logger.warning(
                    "CUDA specified but not available. Falling back to CPU."
                )
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
                logger.info(
                    f"Processor loaded in {end_time - start_time:.2f}s."
                )
            else:
                logger.info(
                    f"Processor loading step completed (none loaded/required) in {end_time - start_time:.2f}s."
                )

            self._is_loaded = True

            self._determine_native_dimension()

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
            msg = f"Unexpected error loading model or processor '{self.model_name}': {e}"
            logger.error(msg, exc_info=True)
            self._is_loaded = False
            raise VectorizerError(msg) from e

    def _determine_native_dimension(self):
        """Intenta determinar y almacenar la dimensión nativa del modelo."""
        if not self.is_ready or self._native_dimension is not None:
            return

        logger.debug(
            f"Attempting to determine native dimension for {self.model_name}..."
        )
        try:

            dummy_image = Image.new("RGB", (32, 32), color="black")

            inputs = self._prepare_inputs([dummy_image], "image")

            with torch.no_grad():

                if hasattr(self.model, "get_image_features"):
                    dummy_embedding_tensor = self.model.get_image_features(
                        **inputs
                    )
                elif hasattr(self.model, "encode_image"):
                    dummy_embedding_tensor = self.model.encode_image(
                        inputs["pixel_values"]
                    )
                elif hasattr(self.model, "forward"):
                    outputs = self.model(**inputs)

                    if hasattr(outputs, "pooler_output"):
                        dummy_embedding_tensor = outputs.pooler_output
                    elif hasattr(outputs, "last_hidden_state"):

                        dummy_embedding_tensor = outputs.last_hidden_state[
                            :, 0
                        ]
                    elif isinstance(outputs, torch.Tensor):
                        dummy_embedding_tensor = outputs
                    else:
                        raise VectorizerError(
                            "Could not determine embedding output from model forward pass."
                        )
                else:
                    raise VectorizerError(
                        "Model lacks known methods (get_image_features, encode_image, forward) to get embeddings."
                    )

            if dummy_embedding_tensor is not None and isinstance(
                dummy_embedding_tensor, torch.Tensor
            ):
                self._native_dimension = dummy_embedding_tensor.shape[-1]
                logger.info(
                    f"Determined native embedding dimension: {self._native_dimension}"
                )
            else:
                logger.warning(
                    f"Could not get a valid embedding tensor for dimension determination."
                )
                self._native_dimension = None

        except Exception as e:
            logger.warning(
                f"Could not automatically determine native dimension for model {self.model_name}: {e}. "
                f"Will rely solely on user-provided dimension or defaults.",
                exc_info=False,
            )
            self._native_dimension = None

    def _prepare_inputs(
        self, batch_data: Union[List[Image.Image], List[str]], batch_type: str
    ) -> Dict[str, torch.Tensor]:
        """
        Prepara los tensores de entrada para el modelo usando el procesador.
        (Sin cambios funcionales necesarios aquí para la gestión de dimensiones)
        """

        if batch_type == "image":
            if not self.processor:
                raise VectorizerError(
                    "Processor is None, cannot process images."
                )
            if not all(isinstance(item, Image.Image) for item in batch_data):
                raise TypeError(
                    "Image batch must contain only PIL Image objects."
                )

            inputs = self.processor(
                images=batch_data,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )

        elif batch_type == "text":
            if not all(isinstance(item, str) for item in batch_data):
                raise TypeError("Text batch must contain only strings.")
            if self.processor:
                inputs = self.processor(
                    text=batch_data,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
            else:

                raise VectorizerError(
                    "Processor is None, cannot process text with standard pipeline."
                )
        else:
            raise VectorizerError(f"Invalid batch type: {batch_type}")

        return inputs.to(self.device)

    def _run_inference(
        self, inputs: Dict[str, torch.Tensor], batch_type: str
    ) -> torch.Tensor:
        """
        Ejecuta la inferencia del modelo para obtener los embeddings.
        (Sin cambios funcionales necesarios aquí)
        """

        if not self.model:
            raise VectorizerError("Model is not loaded, cannot run inference.")

        with torch.no_grad():
            if batch_type == "image":
                if hasattr(self.model, "get_image_features"):
                    embeddings_tensor = self.model.get_image_features(**inputs)
                elif hasattr(self.model, "encode_image"):
                    embeddings_tensor = self.model.encode_image(
                        inputs["pixel_values"]
                    )
                else:
                    outputs = self.model(**inputs)
                    if hasattr(outputs, "pooler_output"):
                        embeddings_tensor = outputs.pooler_output
                    elif hasattr(outputs, "last_hidden_state"):
                        embeddings_tensor = outputs.last_hidden_state[:, 0]
                    elif isinstance(outputs, torch.Tensor):
                        embeddings_tensor = outputs
                    else:
                        raise VectorizerError(
                            "Could not determine embedding output from model."
                        )
            elif batch_type == "text":
                if hasattr(self.model, "get_text_features"):
                    embeddings_tensor = self.model.get_text_features(**inputs)
                elif hasattr(self.model, "encode_text"):
                    embeddings_tensor = self.model.encode_text(
                        inputs["input_ids"],
                        attention_mask=inputs.get("attention_mask"),
                    )
                else:
                    outputs = self.model(**inputs)
                    if hasattr(outputs, "pooler_output"):
                        embeddings_tensor = outputs.pooler_output
                    elif hasattr(outputs, "last_hidden_state"):
                        embeddings_tensor = outputs.last_hidden_state[:, 0]
                    elif isinstance(outputs, torch.Tensor):
                        embeddings_tensor = outputs
                    else:
                        raise VectorizerError(
                            "Could not determine embedding output from model."
                        )
            else:
                raise VectorizerError(
                    f"Invalid batch type during inference: {batch_type}"
                )

        if embeddings_tensor is None:
            raise VectorizerError("Inference returned None for embeddings.")

        return embeddings_tensor

    def _postprocess_embeddings(
        self, embeddings_tensor: torch.Tensor, truncate_dim: Optional[int]
    ) -> List[List[float]]:
        """
        Normaliza, trunca (opcionalmente) y convierte los embeddings a listas de floats.
        (Sin cambios funcionales necesarios aquí)
        """

        embeddings_tensor = torch.nn.functional.normalize(
            embeddings_tensor, p=2, dim=-1
        )

        was_truncated = False
        original_dim = embeddings_tensor.shape[-1]

        effective_truncate_dim = truncate_dim

        if (
            effective_truncate_dim is not None
            and original_dim > effective_truncate_dim
        ):
            if effective_truncate_dim <= 0:
                logger.warning(
                    f"Invalid truncate_dim requested ({effective_truncate_dim}). Using full dimension {original_dim}."
                )
            else:
                embeddings_tensor = embeddings_tensor[
                    :, :effective_truncate_dim
                ]
                was_truncated = True
                logger.debug(
                    f"Truncated embeddings from {original_dim} to {effective_truncate_dim} dimensions."
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
        (Sin cambios funcionales necesarios aquí)
        """

        if not self.is_ready:
            logger.error("Vectorizer not ready. Cannot process batch.")
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
                continue

            sub_batch_results: List[Optional[List[float]]] = [
                None
            ] * sub_batch_len
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
                    f"Error vectorizing sub-batch {i+1}/{num_sub_batches} ('{batch_type}'): {e}",
                    exc_info=False,
                )
            except Exception as e:
                logger.error(
                    f"Unexpected error vectorizing sub-batch {i+1}/{num_sub_batches} ('{batch_type}'): {e}",
                    exc_info=True,
                )

            all_embeddings.extend(sub_batch_results)
            sub_batch_end_time = time.time()
            logger.debug(
                f"Sub-batch {i+1}/{num_sub_batches} ({batch_type}) processed in {sub_batch_end_time - sub_batch_start_time:.3f}s."
            )

        return all_embeddings

    def vectorize_images(
        self,
        images: List[Image.Image],
        batch_size: int,
        truncate_dim: Optional[int] = None,
    ) -> List[Optional[List[float]]]:
        """
        Vectoriza una lista de objetos PIL Image.
        (Sin cambios funcionales necesarios aquí, solo pasa truncate_dim)
        """

        if not self.is_ready:
            raise VectorizerError("Vectorizer not ready.")
        if not images or not isinstance(images, list):
            return []
        if not all(isinstance(img, Image.Image) for img in images):
            raise TypeError("Input list contains non-PIL Image objects.")

        num_images = len(images)

        logger.info(
            f"Starting vectorization of {num_images} images (Batch Size: {batch_size}, Truncate Dim: {truncate_dim or 'Full'})..."
        )

        embeddings = self._process_batch(
            images, "image", batch_size, truncate_dim
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
        (Sin cambios funcionales necesarios aquí, solo pasa truncate_dim)
        """

        if not self.is_ready:
            raise VectorizerError("Vectorizer not ready.")
        if not texts or not isinstance(texts, list):
            return []
        if not all(isinstance(t, str) for t in texts):
            raise TypeError("Input list contains non-string objects.")

        num_texts = len(texts)

        logger.info(
            f"Starting vectorization of {num_texts} texts (Batch Size: {batch_size}, Truncate Dim: {truncate_dim or 'Full'}, Is Query: {is_query})..."
        )

        embeddings = self._process_batch(
            texts, "text", batch_size, truncate_dim, is_query=is_query
        )

        return embeddings
