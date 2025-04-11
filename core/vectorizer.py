import torch
from transformers import AutoModel, AutoProcessor, PreTrainedModel
from PIL import Image
from typing import List, Optional, Union, cast, Any
import math
import logging
import time

from config import DEVICE, TRUST_REMOTE_CODE

logger = logging.getLogger(__name__)

class Vectorizer:
    """
    Maneja la carga del modelo de embedding y la vectorización de imágenes y texto.
    """

    def __init__(
        self,
        model_name: str,
        device: str = DEVICE,
        trust_remote_code: bool = TRUST_REMOTE_CODE,
    ):
        self.model_name = model_name
        if device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA specified but not available. Falling back to CPU.")
            self.device = 'cpu'
        else:
            self.device = device
        self.trust_remote_code = trust_remote_code
        self.model: Optional[PreTrainedModel] = None
        self.processor: Optional[Any] = None # Usar Any para evitar error de importación

        logger.info(f"Initializing Vectorizer with model: {self.model_name} on device: {self.device}")
        self._load_model()

    def _load_model(self):
        if self.model is not None and self.processor is not None:
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
            logger.info(f"Model loaded successfully to {self.device} in {end_time - start_time:.2f}s.")

            logger.info(f"Loading processor for '{self.model_name}'...")
            start_time = time.time()
            self.processor = AutoProcessor.from_pretrained(
                self.model_name, trust_remote_code=self.trust_remote_code
            )
            end_time = time.time()
            logger.info(f"Processor loaded successfully in {end_time - start_time:.2f}s.")

        except OSError as e:
             logger.error(f"OSError loading model/processor '{self.model_name}': {e}.")
             self.model = None
             self.processor = None
             raise
        except ImportError as e:
             logger.error(f"ImportError during model loading: {e}. Missing dependencies?", exc_info=True)
             self.model = None
             self.processor = None
             raise
        except Exception as e:
            logger.error(f"Unexpected error loading model or processor '{self.model_name}': {e}", exc_info=True)
            self.model = None
            self.processor = None
            raise

    def _process_batch(
        self,
        batch_data: Union[List[Image.Image], List[str]],
        batch_type: str,
        batch_size: int,
        truncate_dim: Optional[int],
        is_query: bool = False # No usado actualmente en la lógica interna de CLIP
    ) -> List[Optional[List[float]]]:
        """Procesa datos (imágenes o texto) en lotes."""
        all_embeddings: List[Optional[List[float]]] = []
        num_items_in_batch_data = len(batch_data)
        num_sub_batches = math.ceil(num_items_in_batch_data / batch_size)

        if not self.model or not self.processor:
             logger.error("Model or processor not loaded. Cannot process batch.")
             return [None] * num_items_in_batch_data

        model = cast(PreTrainedModel, self.model)
        processor = self.processor
        if not processor:
            logger.error("Processor object is None in _process_batch.")
            return [None] * num_items_in_batch_data

        for i in range(num_sub_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, num_items_in_batch_data)
            current_sub_batch = batch_data[start_idx:end_idx]
            sub_batch_len = len(current_sub_batch)

            if not current_sub_batch:
                continue

            sub_batch_results: List[Optional[List[float]]] = [None] * sub_batch_len

            try:
                with torch.no_grad():
                    if batch_type == "image":
                        if not all(isinstance(item, Image.Image) for item in current_sub_batch):
                             raise TypeError("Input must be a list of PIL Images.")
                        inputs = processor(images=current_sub_batch, return_tensors="pt", padding=True).to(self.device)
                        # Asume modelo tipo CLIP. Para otros, la llamada puede variar.
                        embeddings_tensor = model.get_image_features(**inputs)
                    elif batch_type == "text":
                         if not all(isinstance(item, str) for item in current_sub_batch):
                             raise TypeError("Input must be a list of strings.")
                         inputs = processor(
                             text=current_sub_batch, return_tensors="pt", padding=True, truncation=True, max_length=77
                         ).to(self.device)
                         # Asume modelo tipo CLIP. Para otros (ej. Sentence Transformers), usar model.encode(...)
                         embeddings_tensor = model.get_text_features(**inputs)
                    else:
                        raise ValueError(f"Invalid batch_type: {batch_type}")

                    # 1. Normalizar embeddings originales
                    embeddings_tensor = torch.nn.functional.normalize(embeddings_tensor, p=2, dim=-1)

                    # 2. Truncar si se especifica
                    was_truncated = False
                    if truncate_dim and embeddings_tensor.shape[-1] > truncate_dim:
                        embeddings_tensor = embeddings_tensor[:, :truncate_dim]
                        logger.debug(f"Embeddings truncated to dimension {truncate_dim}")
                        was_truncated = True

                    # 3. Re-normalizar DESPUÉS del truncamiento (si se truncó)
                    #    Esto mantiene los vectores unitarios, importante para distancia coseno.
                    if was_truncated:
                         embeddings_tensor = torch.nn.functional.normalize(embeddings_tensor, p=2, dim=-1)
                         logger.debug("Re-normalized embeddings after truncation.")

                    valid_embeddings_list = embeddings_tensor.cpu().numpy().tolist()

                    if len(valid_embeddings_list) == sub_batch_len:
                        sub_batch_results = valid_embeddings_list
                    else:
                        logger.error(f"Sub-batch {i+1}/{num_sub_batches}: Output size mismatch.")

            except Exception as e:
                logger.error(f"Error vectorizing {batch_type} sub-batch {i+1}/{num_sub_batches}: {e}", exc_info=True)
                # sub_batch_results ya está lleno de None

            all_embeddings.extend(sub_batch_results)

        if len(all_embeddings) != num_items_in_batch_data:
             logger.warning(f"Final embeddings list length mismatch with input data length.")
             # Rellenar con None si faltan resultados por alguna razón
             all_embeddings.extend([None] * (num_items_in_batch_data - len(all_embeddings)))


        return all_embeddings

    def vectorize_images(
        self,
        images: List[Image.Image],
        batch_size: int,
        truncate_dim: Optional[int] = None,
    ) -> List[Optional[List[float]]]:
        """Vectoriza una lista de objetos PIL Image."""
        if not self.model or not self.processor:
            logger.error("Model/processor not loaded. Cannot vectorize images.")
            return [None] * len(images) if images else []
        if not images or not isinstance(images, list):
            logger.warning("No valid list of images provided. Returning empty list.")
            return []
        if not all(isinstance(img, Image.Image) for img in images):
             logger.error("Input list contains non-PIL Image objects.")
             # Devolver None para cada elemento para mantener la correspondencia de índices
             return [None] * len(images)


        num_images = len(images)
        logger.info(f"Starting image vectorization for {num_images} images with batch size {batch_size}...")

        start_time_event = None
        end_time_event = None
        start_time_cpu = None
        if self.device == 'cuda':
            start_time_event = torch.cuda.Event(enable_timing=True)
            end_time_event = torch.cuda.Event(enable_timing=True)
            start_time_event.record()
        else:
            start_time_cpu = time.time()

        embeddings = self._process_batch(images, "image", batch_size, truncate_dim)

        elapsed_time = 0
        if self.device == 'cuda' and start_time_event and end_time_event:
            end_time_event.record()
            torch.cuda.synchronize()
            elapsed_time = start_time_event.elapsed_time(end_time_event) / 1000.0
        elif start_time_cpu:
            end_time_cpu = time.time()
            elapsed_time = end_time_cpu - start_time_cpu

        num_successful = sum(1 for emb in embeddings if emb is not None)
        num_failed = num_images - num_successful
        logger.info(f"Finished vectorizing images in {elapsed_time:.2f}s. Successful: {num_successful}/{num_images}, Failed: {num_failed}")
        if num_failed > 0:
             logger.warning(f"{num_failed} images failed to vectorize.")

        return embeddings


    def vectorize_texts(
        self,
        texts: List[str],
        batch_size: int,
        truncate_dim: Optional[int] = None,
        is_query: bool = False, # Argumento mantenido por compatibilidad, no usado en CLIP interno
    ) -> List[Optional[List[float]]]:
        """Vectoriza una lista de strings de texto."""
        if not self.model or not self.processor:
            logger.error("Model/processor not loaded. Cannot vectorize texts.")
            return [None] * len(texts) if texts else []
        if not texts or not isinstance(texts, list):
            logger.warning("No valid list of texts provided. Returning empty list.")
            return []
        if not all(isinstance(t, str) for t in texts):
             logger.error("Input list contains non-string objects.")
             return [None] * len(texts)

        num_texts = len(texts)
        logger.info(f"Starting text vectorization for {num_texts} texts with batch size {batch_size}...")
        start_time = time.time()

        embeddings = self._process_batch(texts, "text", batch_size, truncate_dim, is_query=is_query)

        end_time = time.time()
        elapsed_time = end_time - start_time

        num_successful = sum(1 for emb in embeddings if emb is not None)
        num_failed = num_texts - num_successful
        logger.info(f"Finished vectorizing texts in {elapsed_time:.2f}s. Successful: {num_successful}/{num_texts}, Failed: {num_failed}")
        if num_failed > 0:
             logger.warning(f"{num_failed} texts failed to vectorize.")

        return embeddings
