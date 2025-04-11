import torch
from transformers import AutoModel, AutoProcessor
from PIL import Image
from typing import List, Optional, Union
import math
import logging
import time

# Import configuration constants
from config import DEVICE, TRUST_REMOTE_CODE

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Vectorizer:
    """
    Handles loading the embedding model and vectorizing images and text.
    Uses models via the transformers library.
    """

    def __init__(
        self,
        model_name: str,
        device: str = DEVICE,
        trust_remote_code: bool = TRUST_REMOTE_CODE,
    ):
        """
        Initializes the Vectorizer and loads the model and processor.
        """
        self.model_name = model_name
        self.device = device
        self.trust_remote_code = trust_remote_code
        self.model = None
        self.processor = None

        logging.info(f"Initializing Vectorizer with model: {self.model_name} on device: {self.device}")
        self._load_model()

    def _load_model(self):
        """Loads the model and processor from Hugging Face."""
        try:
            logging.info(f"Loading model '{self.model_name}'...")
            # Usar AutoModel y AutoProcessor es común para modelos CLIP
            self.model = AutoModel.from_pretrained(
                self.model_name, trust_remote_code=self.trust_remote_code
            ).to(self.device)

            logging.info(f"Loading processor for '{self.model_name}'...")
            self.processor = AutoProcessor.from_pretrained(
                self.model_name, trust_remote_code=self.trust_remote_code
            )

            self.model.eval() # Poner el modelo en modo evaluación
            logging.info("Model and processor loaded successfully.")

        # Capturar errores más específicos si es posible/necesario
        # except RepositoryNotFoundError:
        #     logging.error(f"Model '{self.model_name}' not found on Hugging Face Hub.")
        #     raise
        # except ConnectionError:
        #     logging.error(f"Connection error downloading '{self.model_name}'. Check internet.")
        #     raise
        except ImportError as e:
             logging.error(f"Import error during model loading. Missing dependencies? {e}", exc_info=True)
             raise
        except Exception as e: # Captura general final
            logging.error(f"Error loading model or processor '{self.model_name}': {e}", exc_info=True)
            logging.error("Check model name, internet access (first time), dependencies, and permissions.")
            raise # Re-lanzar para indicar fallo crítico

    def _process_batch(
        self,
        batch_data: Union[List[Image.Image], List[str]],
        batch_type: str,
        batch_size: int,
        truncate_dim: Optional[int],
        is_query: bool = False # Añadido por si el modelo diferencia query/document
    ) -> List[Optional[List[float]]]:
        """
        Helper function to process data in batches. Handles potential errors within a batch.
        Returns a list of embeddings or None for failed items, maintaining original batch order.
        """
        all_embeddings: List[Optional[List[float]]] = []
        num_items_in_batch_data = len(batch_data)
        num_batches = math.ceil(num_items_in_batch_data / batch_size)

        if not self.model or not self.processor:
             logging.error("Model or processor not loaded. Cannot process batch.")
             # Devolver Nones para toda la entrada original
             return [None] * num_items_in_batch_data

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, num_items_in_batch_data)
            batch = batch_data[start_idx:end_idx]
            batch_indices = list(range(start_idx, end_idx)) # Para logging de errores

            if not batch:
                continue

            # Inicializar resultados para este lote con None
            batch_results: List[Optional[List[float]]] = [None] * len(batch)

            try:
                with torch.no_grad(): # Desactivar gradientes para inferencia
                    if batch_type == "image":
                        inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
                        # La función puede variar: get_image_features, encode_image, etc.
                        embeddings_tensor = self.model.get_image_features(**inputs)
                    elif batch_type == "text":
                        # Algunos modelos usan prefijos (ej. Jina V2)
                        # texts_with_prefix = [f"query: {t}" if is_query else f"document: {t}" for t in batch] # Ejemplo
                        inputs = self.processor(
                            text=batch, # O texts_with_prefix si aplica
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                        ).to(self.device)
                        embeddings_tensor = self.model.get_text_features(**inputs)
                    else:
                        # Esto es un error de programación, debería lanzar excepción
                        raise ValueError(f"Invalid batch_type specified: {batch_type}")

                    # Normalizar embeddings (práctica común para similitud coseno)
                    embeddings_tensor = embeddings_tensor / embeddings_tensor.norm(p=2, dim=-1, keepdim=True)

                    # Truncamiento opcional
                    if truncate_dim and embeddings_tensor.shape[-1] > truncate_dim:
                        embeddings_tensor = embeddings_tensor[:, :truncate_dim]
                        # Considerar re-normalizar si es necesario después de truncar
                        # embeddings_tensor = embeddings_tensor / embeddings_tensor.norm(p=2, dim=-1, keepdim=True)

                    # Convertir a lista de listas (numpy primero)
                    valid_embeddings_list = embeddings_tensor.cpu().numpy().tolist()

                    # Verificar si la salida coincide con la entrada del lote
                    if len(valid_embeddings_list) == len(batch):
                        batch_results = valid_embeddings_list # Todos OK
                    else:
                        # Error inesperado: el modelo no devolvió un embedding por cada entrada
                        logging.error(f"Batch {i+1}: Output size ({len(valid_embeddings_list)}) mismatch with input size ({len(batch)}). Marking all as failed.")
                        # batch_results ya está lleno de None

            except Exception as e:
                # Error durante la vectorización del lote completo
                logging.error(f"Error vectorizing {batch_type} batch {i+1}/{num_batches}: {e}", exc_info=True)
                # Marcar todos los ítems de este lote como fallidos (ya son None)
                # Podrías intentar identificar ítems específicos si el error lo permite
                # for item_idx_in_batch, original_idx in enumerate(batch_indices):
                #    logging.error(f"  - Failed item at original index {original_idx}")
                pass # batch_results ya está como [None, None, ...]

            all_embeddings.extend(batch_results)

        # Verificación final de longitud
        if len(all_embeddings) != num_items_in_batch_data:
             logging.warning(f"Final embeddings list length ({len(all_embeddings)}) mismatch with input data length ({num_items_in_batch_data}).")

        return all_embeddings

    def vectorize_images(
        self,
        images: List[Image.Image],
        batch_size: int,
        truncate_dim: Optional[int] = None,
    ) -> List[Optional[List[float]]]:
        """
        Vectorizes a list of PIL Image objects, handling potential errors.

        Args:
            images: A list of PIL Image objects.
            batch_size: The number of images to process in each inference batch.
            truncate_dim: Optional dimension to truncate the embeddings to.

        Returns:
            A list where each element is either an embedding (list of floats)
            or None if vectorization failed for that specific image.
            The list maintains the order of the input images.
        """
        if not self.model or not self.processor:
            logging.error("Model/processor not loaded. Cannot vectorize images.")
            return [None] * len(images) if images else []
        if not images or not isinstance(images, list):
            logging.warning("No valid list of images provided to vectorize.")
            return []

        logging.info(f"Vectorizing {len(images)} images in batches of {batch_size}...")
        # Medición de tiempo (GPU o CPU)
        start_time_event = None
        end_time_event = None
        if self.device == 'cuda':
            start_time_event = torch.cuda.Event(enable_timing=True)
            end_time_event = torch.cuda.Event(enable_timing=True)
            start_time_event.record()
        else:
            start_time = time.time() # Usar time.time() para CPU

        embeddings = self._process_batch(images, "image", batch_size, truncate_dim)

        # Finalizar medición de tiempo
        if self.device == 'cuda' and start_time_event and end_time_event:
            end_time_event.record()
            torch.cuda.synchronize() # Esperar a que terminen las operaciones CUDA
            elapsed_time = start_time_event.elapsed_time(end_time_event) / 1000.0 # ms a s
        else:
            end_time = time.time()
            elapsed_time = end_time - start_time

        num_successful = sum(1 for emb in embeddings if emb is not None)
        logging.info(f"Finished vectorizing images in {elapsed_time:.2f}s. Got {num_successful}/{len(images)} successful embeddings.")
        return embeddings

    def vectorize_texts(
        self,
        texts: List[str],
        batch_size: int,
        truncate_dim: Optional[int] = None,
        is_query: bool = False, # Para modelos que distinguen
    ) -> List[Optional[List[float]]]:
        """
        Vectorizes a list of text strings, handling potential errors.

        Args:
            texts: A list of text strings.
            batch_size: The number of texts to process in each inference batch.
            truncate_dim: Optional dimension to truncate the embeddings to.
            is_query: Flag for models differentiating query/document encoding.

        Returns:
            A list where each element is either an embedding (list of floats)
            or None if vectorization failed for that specific text.
            The list maintains the order of the input texts.
        """
        if not self.model or not self.processor:
            logging.error("Model/processor not loaded. Cannot vectorize texts.")
            return [None] * len(texts) if texts else []
        if not texts or not isinstance(texts, list):
            logging.warning("No valid list of texts provided to vectorize.")
            return []

        logging.info(f"Vectorizing {len(texts)} texts in batches of {batch_size} (is_query={is_query})...")
        start_time = time.time() # CPU timing es suficiente para texto normalmente

        embeddings = self._process_batch(texts, "text", batch_size, truncate_dim, is_query=is_query)

        end_time = time.time()
        elapsed_time = end_time - start_time

        num_successful = sum(1 for emb in embeddings if emb is not None)
        logging.info(f"Finished vectorizing texts in {elapsed_time:.2f}s. Got {num_successful}/{len(texts)} successful embeddings.")
        return embeddings
