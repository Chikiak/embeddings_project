import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np

import config
from app.exceptions import (
    DatabaseError,
    ImageProcessingError,
    PipelineError,
    VectorizerError,
)
from app.models import SearchResultItem, SearchResults
from core.image_processor import load_image
from core.vectorizer import Vectorizer
from data_access.vector_db_interface import VectorDBInterface

logger = logging.getLogger(__name__)


def _get_db_collection_name(db: VectorDBInterface) -> str:
    return getattr(db, "collection_name", "N/A")


def search_by_text(
    query_text: str,
    vectorizer: Vectorizer,
    db: VectorDBInterface,
    n_results: int = config.DEFAULT_N_RESULTS,
    truncate_dim: Optional[int] = None,
) -> SearchResults:
    """
    Busca imágenes similares a una consulta de texto usando la BD y dimensión proporcionadas.
    """
    collection_name = _get_db_collection_name(db)
    effective_dim_str = str(truncate_dim) if truncate_dim else "Full"
    logger.info(
        f"--- Performing Text-to-Image Search on Collection '{collection_name}' ---"
    )
    logger.info(f"  Query: '{query_text}'")
    logger.info(f"  Using Vectorizer Model: {vectorizer.model_name}")
    logger.info(f"  Target Embedding Dimension: {effective_dim_str}")

    if not query_text or not isinstance(query_text, str):
        raise ValueError("Invalid query text.")
    if not db.is_initialized:
        raise PipelineError(
            f"Database (Collection: '{collection_name}') is not initialized."
        )
    if db.count() <= 0:
        logger.warning(
            f"Cannot search: Collection '{collection_name}' is empty."
        )
        return SearchResults(items=[])

    try:
        logger.info("Vectorizing query text...")

        query_embedding_list = vectorizer.vectorize_texts(
            [query_text],
            batch_size=1,
            truncate_dim=truncate_dim,
            is_query=True,
        )
        if not query_embedding_list or query_embedding_list[0] is None:
            raise PipelineError("Failed to vectorize the query text.")

        query_embedding = query_embedding_list[0]
        logger.info(
            f"Query text vectorized successfully (dim: {len(query_embedding)})."
        )

        logger.info(
            f"Querying collection '{collection_name}' for top {n_results} similar images..."
        )

        results: Optional[SearchResults] = db.query_similar(
            query_embedding=query_embedding, n_results=n_results
        )

        if results is None:
            raise PipelineError(
                f"Database query failed on collection '{collection_name}'."
            )
        elif results.is_empty:
            logger.info("Search successful, but no similar images found.")
        else:
            logger.info(f"Search successful. Found {results.count} results.")

        logger.info("--- Text-to-Image Search Finished ---")
        return results

    except (VectorizerError, DatabaseError) as e:
        logger.error(
            f"Error during text search pipeline on '{collection_name}': {e}",
            exc_info=True,
        )
        raise PipelineError(f"Text search failed: {e}") from e
    except Exception as e:
        logger.error(
            f"Unexpected error during text search pipeline on '{collection_name}': {e}",
            exc_info=True,
        )
        raise PipelineError(f"Unexpected text search failure: {e}") from e


def search_by_image(
    query_image_path: str,
    vectorizer: Vectorizer,
    db: VectorDBInterface,
    n_results: int = config.DEFAULT_N_RESULTS,
    truncate_dim: Optional[int] = None,
) -> SearchResults:
    """
    Busca imágenes similares a una imagen de consulta usando la BD y dimensión proporcionadas.
    """
    collection_name = _get_db_collection_name(db)
    effective_dim_str = str(truncate_dim) if truncate_dim else "Full"
    logger.info(
        f"--- Performing Image-to-Image Search on Collection '{collection_name}' ---"
    )
    logger.info(f"  Query Image: '{query_image_path}'")
    logger.info(f"  Using Vectorizer Model: {vectorizer.model_name}")
    logger.info(f"  Target Embedding Dimension: {effective_dim_str}")

    if not query_image_path or not isinstance(query_image_path, str):
        raise ValueError("Invalid query image path.")
    if not os.path.isfile(query_image_path):
        raise FileNotFoundError(
            f"Query image file not found: {query_image_path}"
        )
    if not db.is_initialized:
        raise PipelineError(
            f"Database (Collection: '{collection_name}') is not initialized."
        )
    if db.count() <= 0:
        logger.warning(
            f"Cannot search: Collection '{collection_name}' is empty."
        )
        return SearchResults(items=[])

    try:
        logger.info("Loading and vectorizing query image...")
        query_image = load_image(query_image_path)
        if not query_image:
            raise PipelineError(
                f"Failed to load query image: {query_image_path}."
            )

        query_embedding_list = vectorizer.vectorize_images(
            [query_image], batch_size=1, truncate_dim=truncate_dim
        )
        if not query_embedding_list or query_embedding_list[0] is None:
            raise PipelineError("Failed to vectorize the query image.")

        query_embedding = query_embedding_list[0]
        logger.info(
            f"Query image vectorized successfully (dim: {len(query_embedding)})."
        )

        effective_n_results = n_results + 1
        logger.info(
            f"Querying collection '{collection_name}' for top {effective_n_results} similar images..."
        )

        initial_results: Optional[SearchResults] = db.query_similar(
            query_embedding=query_embedding, n_results=effective_n_results
        )

        if initial_results is None:
            raise PipelineError(
                f"Database query failed on collection '{collection_name}'."
            )

        filtered_items: List[SearchResultItem] = []
        results_to_return: SearchResults

        if not initial_results.is_empty:
            logger.info(
                f"Search successful. Found {initial_results.count} potential results (before filtering)."
            )
            normalized_query_path = os.path.abspath(query_image_path)
            for item in initial_results.items:
                if item.id and isinstance(item.id, str):
                    try:

                        normalized_result_path = item.id
                        is_exact_match = (
                            normalized_result_path == normalized_query_path
                        ) and (
                            item.distance is not None and item.distance < 1e-6
                        )
                        if not is_exact_match:
                            filtered_items.append(item)
                        else:
                            logger.info(
                                f"  (Excluding query image '{item.id}' itself from results)"
                            )
                    except Exception as path_err:
                        logger.warning(
                            f"Could not compare path '{item.id}': {path_err}"
                        )
                        filtered_items.append(item)
                else:
                    filtered_items.append(item)

            final_items = filtered_items[:n_results]
            final_results = SearchResults(items=final_items)
            if final_results.is_empty:
                logger.info("No similar images found (after filtering).")
            else:
                logger.info(
                    f"Returning {final_results.count} results after filtering."
                )
            results_to_return = final_results
        else:
            logger.info(
                "Search successful, but no similar images found initially."
            )
            results_to_return = initial_results

    except (VectorizerError, DatabaseError, ImageProcessingError) as e:
        logger.error(
            f"Error during image search pipeline on '{collection_name}': {e}",
            exc_info=True,
        )
        raise PipelineError(f"Image search failed: {e}") from e
    except FileNotFoundError as e:
        raise e
    except Exception as e:
        logger.error(
            f"Unexpected error during image search pipeline on '{collection_name}': {e}",
            exc_info=True,
        )
        raise PipelineError(f"Unexpected image search failure: {e}") from e

    logger.info("--- Image-to-Image Search Finished ---")
    return results_to_return


def search_hybrid(
    query_text: str,
    query_image_path: str,
    vectorizer: Vectorizer,
    db: VectorDBInterface,
    n_results: int = config.DEFAULT_N_RESULTS,
    truncate_dim: Optional[int] = None,
    alpha: float = 0.5,
) -> SearchResults:
    """
    Busca imágenes combinando texto e imagen (interpolación) usando la BD y dimensión proporcionadas.
    """
    collection_name = _get_db_collection_name(db)
    effective_dim_str = str(truncate_dim) if truncate_dim else "Full"
    logger.info(
        f"--- Performing Hybrid Search (Interpolated, Alpha={alpha}) on Collection '{collection_name}' ---"
    )
    logger.info(f"  Text Query: '{query_text}'")
    logger.info(f"  Image Query Path: '{query_image_path}'")
    logger.info(f"  Using Vectorizer Model: {vectorizer.model_name}")
    logger.info(f"  Target Embedding Dimension: {effective_dim_str}")

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("Alpha must be between 0.0 and 1.0")
    if not query_text or not isinstance(query_text, str):
        raise ValueError("Invalid query text.")
    if not query_image_path or not isinstance(query_image_path, str):
        raise ValueError("Invalid query image path.")
    if not os.path.isfile(query_image_path):
        raise FileNotFoundError(
            f"Query image file not found: {query_image_path}"
        )
    if not db.is_initialized:
        raise PipelineError(
            f"Database (Collection: '{collection_name}') is not initialized."
        )
    if db.count() <= 0:
        logger.warning(
            f"Cannot search: Collection '{collection_name}' is empty."
        )
        return SearchResults(items=[])

    try:

        logger.debug("Vectorizing hybrid query text...")
        text_embedding_list = vectorizer.vectorize_texts(
            [query_text],
            batch_size=1,
            truncate_dim=truncate_dim,
            is_query=True,
        )
        if not text_embedding_list or text_embedding_list[0] is None:
            raise PipelineError("Failed to vectorize text.")
        text_embedding = np.array(text_embedding_list[0], dtype=np.float32)
        text_norm = np.linalg.norm(text_embedding)
        text_embedding /= text_norm if text_norm > 1e-6 else 1.0

        logger.debug("Loading and vectorizing hybrid query image...")
        query_image = load_image(query_image_path)
        if not query_image:
            raise PipelineError(
                f"Failed to load hybrid query image: {query_image_path}"
            )
        image_embedding_list = vectorizer.vectorize_images(
            [query_image], batch_size=1, truncate_dim=truncate_dim
        )
        if not image_embedding_list or image_embedding_list[0] is None:
            raise PipelineError("Failed to vectorize image.")
        image_embedding = np.array(image_embedding_list[0], dtype=np.float32)
        image_norm = np.linalg.norm(image_embedding)
        image_embedding /= image_norm if image_norm > 1e-6 else 1.0

        logger.info(f"Combining embeddings with alpha={alpha}")
        if text_embedding.shape != image_embedding.shape:
            raise PipelineError("Incompatible embedding dimensions.")
        hybrid_embedding_np = (alpha * text_embedding) + (
            (1.0 - alpha) * image_embedding
        )
        hybrid_norm = np.linalg.norm(hybrid_embedding_np)
        if hybrid_norm > 1e-6:
            hybrid_embedding_np /= hybrid_norm
        else:
            hybrid_embedding_np = text_embedding
        hybrid_embedding = hybrid_embedding_np.tolist()
        logger.info(
            f"Hybrid embedding generated (Dim: {len(hybrid_embedding)})."
        )

        logger.info(
            f"Querying collection '{collection_name}' with hybrid embedding (n_results={n_results})..."
        )
        results: Optional[SearchResults] = db.query_similar(
            query_embedding=hybrid_embedding, n_results=n_results
        )

        if results is None:
            raise PipelineError(
                f"Hybrid database query failed on collection '{collection_name}'."
            )
        elif results.is_empty:
            logger.info("Hybrid search successful, but no results found.")
        else:
            logger.info(
                f"Hybrid search successful. Found {results.count} results."
            )

        logger.info("--- Hybrid Search (Interpolated Embedding) Finished ---")
        return results

    except (
        VectorizerError,
        DatabaseError,
        ImageProcessingError,
        FileNotFoundError,
        ValueError,
    ) as e:
        logger.error(
            f"Error during hybrid search pipeline on '{collection_name}': {e}",
            exc_info=False,
        )
        raise PipelineError(f"Hybrid search failed: {e}") from e
    except Exception as e:
        logger.error(
            f"Unexpected error during hybrid pipeline on '{collection_name}': {e}",
            exc_info=True,
        )
        raise PipelineError(f"Unexpected hybrid search failure: {e}") from e


def search_hybrid_rrf(
    query_text: str,
    query_image_path: str,
    vectorizer: Vectorizer,
    db: VectorDBInterface,
    n_results: int = config.DEFAULT_N_RESULTS,
    truncate_dim: Optional[int] = None,
    k_rrf: int = 60,
) -> SearchResults:
    """
    Realiza búsqueda híbrida (RRF) usando la BD y dimensión proporcionadas.
    """
    collection_name = _get_db_collection_name(db)
    effective_dim_str = str(truncate_dim) if truncate_dim else "Full"
    logger.info(
        f"--- Performing Hybrid Search (RRF, k={k_rrf}) on Collection '{collection_name}' ---"
    )
    logger.info(f"  Text Query: '{query_text}'")
    logger.info(f"  Image Query Path: '{query_image_path}'")
    logger.info(f"  Using Vectorizer Model: {vectorizer.model_name}")
    logger.info(f"  Target Embedding Dimension: {effective_dim_str}")

    if not query_text or not isinstance(query_text, str):
        raise ValueError("Invalid query text.")
    if not query_image_path or not isinstance(query_image_path, str):
        raise ValueError("Invalid query image path.")
    if not os.path.isfile(query_image_path):
        raise FileNotFoundError(
            f"Query image file not found: {query_image_path}"
        )
    if not db.is_initialized:
        raise PipelineError(
            f"Database (Collection: '{collection_name}') is not initialized."
        )
    if db.count() <= 0:
        logger.warning(
            f"Cannot search: Collection '{collection_name}' is empty."
        )
        return SearchResults(items=[])

    try:

        fetch_n = max(n_results * 2, 20)
        logger.info(
            f"Performing independent text search (n_results={fetch_n}) on '{collection_name}'..."
        )
        text_results: SearchResults = search_by_text(
            query_text=query_text,
            vectorizer=vectorizer,
            db=db,
            n_results=fetch_n,
            truncate_dim=truncate_dim,
        )
        logger.info(f"Text search yielded {text_results.count} results.")

        logger.info(
            f"Performing independent image search (n_results={fetch_n}) on '{collection_name}'..."
        )
        image_results: SearchResults = search_by_image(
            query_image_path=query_image_path,
            vectorizer=vectorizer,
            db=db,
            n_results=fetch_n,
            truncate_dim=truncate_dim,
        )
        logger.info(f"Image search yielded {image_results.count} results.")

        logger.info(f"Fusing results using RRF (k={k_rrf})...")
        rrf_scores: Dict[str, float] = {}
        doc_metadata: Dict[str, Dict[str, Any]] = {}

        if not text_results.is_empty:
            for rank, item in enumerate(text_results.items):
                if item.id:
                    score = 1.0 / (k_rrf + rank + 1)
                    rrf_scores[item.id] = rrf_scores.get(item.id, 0.0) + score
                    if item.id not in doc_metadata:
                        doc_metadata[item.id] = item.metadata or {}

        if not image_results.is_empty:
            for rank, item in enumerate(image_results.items):
                if item.id:
                    score = 1.0 / (k_rrf + rank + 1)
                    rrf_scores[item.id] = rrf_scores.get(item.id, 0.0) + score
                    if item.id not in doc_metadata:
                        doc_metadata[item.id] = item.metadata or {}

        if not rrf_scores:
            logger.info(
                "No results found from either text or image search for RRF."
            )
            return SearchResults(items=[])

        sorted_ids = sorted(
            rrf_scores.keys(),
            key=lambda doc_id: rrf_scores[doc_id],
            reverse=True,
        )
        final_results_items: List[SearchResultItem] = []
        for rank, doc_id in enumerate(sorted_ids[:n_results]):

            final_meta = doc_metadata.get(doc_id, {})
            final_meta["fusion_method"] = "RRF"
            item = SearchResultItem(
                id=doc_id, distance=rrf_scores[doc_id], metadata=final_meta
            )
            final_results_items.append(item)

        logger.info(
            f"RRF fusion completed. Returning top {len(final_results_items)} results."
        )
        logger.info("--- Hybrid Search (RRF) Finished ---")
        return SearchResults(items=final_results_items)

    except (PipelineError, FileNotFoundError, ValueError) as e:
        logger.error(
            f"Error during RRF hybrid search pipeline on '{collection_name}': {e}",
            exc_info=False,
        )
        raise PipelineError(f"RRF hybrid search failed: {e}") from e
    except Exception as e:
        logger.error(
            f"Unexpected error during RRF hybrid pipeline on '{collection_name}': {e}",
            exc_info=True,
        )
        raise PipelineError(
            f"Unexpected RRF hybrid search failure: {e}"
        ) from e
