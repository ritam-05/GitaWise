"""Global singleton embedding model loader with CUDA support."""

from __future__ import annotations

import logging
import os
from typing import ClassVar

logger = logging.getLogger(__name__)
logger.info("[EMBEDDING_MODEL] Module starting...")

import torch
logger.info("[EMBEDDING_MODEL] Torch imported")

# DO NOT import SentenceTransformer here - defer until actually needed
logger.info("[EMBEDDING_MODEL] SentenceTransformer import deferred (lazy load)")


class EmbeddingModelSingleton:
    """Thread-safe singleton for SentenceTransformer embedding model.
    
    CUDA REQUIRED: This implementation forces GPU usage.
    Fails at startup if CUDA is not available.
    
    Loads ONCE at startup, reuses across all requests.
    """

    _instance: ClassVar[EmbeddingModelSingleton | None] = None
    _model: SentenceTransformer | None = None
    _device: str | None = None

    def __new__(cls) -> EmbeddingModelSingleton:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def initialize(cls, model_name: str = "BAAI/bge-large-en-v1.5") -> None:
        """Initialize the embedding model and detect CUDA.
        
        Call this ONCE during app startup, before any requests.
        """
        logger.info("[EMBEDDING] Initializing embedding model: %s", model_name)
        if cls._instance is not None and cls._model is not None:
            logger.info("[EMBEDDING] Model already initialized, skipping re-initialization.")
            return

        # Lazy import SentenceTransformer only when needed
        logger.info("[EMBEDDING] Importing SentenceTransformer (lazy)...")
        from sentence_transformers import SentenceTransformer
        logger.info("[EMBEDDING] ✓ SentenceTransformer imported")
        
        logger.info("[EMBEDDING] Checking CUDA availability...")
        force_cpu = os.getenv("FORCE_CPU", "").lower() in ("1", "true", "yes")
        cuda_available = torch.cuda.is_available()
        if not cuda_available and not force_cpu:
            logger.warning(
                "CUDA not available; falling back to CPU for embeddings (set FORCE_CPU=1 to force)."
            )

        if cuda_available and not force_cpu:
            device = "cuda"
            cls._device = device
            try:
                gpu_name = torch.cuda.get_device_name(0)
                logger.info("[EMBEDDING] Using GPU for embeddings: %s", gpu_name)
            except Exception:
                logger.info("[EMBEDDING] Using GPU for embeddings")
        else:
            device = "cpu"
            cls._device = device
            logger.info("[EMBEDDING] Using CPU for embeddings (development mode)")


        # Load model with device placement
        logger.info("[EMBEDDING] Loading embedding model: %s", model_name)
        try:
            cls._model = SentenceTransformer(
                model_name,
                device=device,
                trust_remote_code=True,
            )
            cls._model.eval()  # Set to eval mode
            logger.info("[EMBEDDING] ✓ Embedding model loaded successfully on %s", device)
        except Exception as e:
            logger.exception("Failed to load embedding model: %s", e)
            raise

    @classmethod
    def get_instance(cls) -> SentenceTransformer:
        """Get the singleton embedding model instance.
        
        Lazy-loads the model on first access if not already initialized.
        
        Returns:
            SentenceTransformer: The loaded model ready for inference.
        """
        if cls._model is None:
            logger.info("Embedding model not yet loaded, initializing on first request...")
            cls.initialize()
        return cls._model

    @classmethod
    def get_device(cls) -> str:
        """Get the device used for the embedding model.
        
        Returns:
            str: Always "cuda" (CUDA is required for production).
        """
        if cls._device is None:
            # Not initialized yet; assume CPU for safety
            cls._device = os.getenv("DEFAULT_DEVICE", "cpu")
        return cls._device

    @classmethod
    def encode(
        cls,
        texts: str | list[str],
        normalize_embeddings: bool = True,
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False,
        batch_size: int = 32,
    ) -> object:
        """Encode texts to embeddings using the singleton model.
        
        Args:
            texts: Single text or list of texts to encode.
            normalize_embeddings: Whether to normalize embeddings.
            convert_to_numpy: Whether to return numpy arrays.
            show_progress_bar: Whether to show progress bar (disabled by default for production).
            batch_size: Batch size for encoding (increase for faster throughput on GPU).
            
        Returns:
            Embeddings in the requested format.
        """
        model = cls.get_instance()
        with torch.no_grad():
            return model.encode(
                texts,
                normalize_embeddings=normalize_embeddings,
                convert_to_numpy=convert_to_numpy,
                show_progress_bar=show_progress_bar,
                batch_size=batch_size,
            )


# Public interface for import
get_embedding_model = EmbeddingModelSingleton.get_instance
initialize_embedding_model = EmbeddingModelSingleton.initialize
encode_embeddings = EmbeddingModelSingleton.encode

logger.info("[EMBEDDING_MODEL] ✓ Module fully initialized with public interface")
