"""Global singleton reranker model loader with CUDA and FP16 support."""

from __future__ import annotations

import logging
import os
from typing import ClassVar

logger = logging.getLogger(__name__)
logger.info("[RERANKER_MODEL] Module starting...")

import torch
logger.info("[RERANKER_MODEL] Torch imported")
# DO NOT import AutoModelForSequenceClassification, AutoTokenizer here - defer until actually needed
logger.info("[RERANKER_MODEL] Transformers import deferred (lazy load)")


class RerankerModelSingleton:
    """Thread-safe singleton for BGE reranker model.
    
    CUDA REQUIRED: This implementation forces GPU usage with FP16.
    Fails at startup if CUDA is not available.
    
    Loads ONCE at startup, reuses across all requests.
    """

    _instance: ClassVar[RerankerModelSingleton | None] = None
    _model: AutoModelForSequenceClassification | None = None
    _tokenizer: AutoTokenizer | None = None
    _device: str | None = None
    _use_fp16: bool = False

    def __new__(cls) -> RerankerModelSingleton:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def initialize(cls, model_name: str = "BAAI/bge-reranker-v2-m3") -> None:
        """Initialize the reranker model with CUDA and FP16 support.
        
        Call this ONCE during app startup, before any requests.
        """
        logger.info("[RERANKER] Initializing reranker model: %s", model_name)
        if cls._instance is not None and cls._model is not None:
            logger.info("[RERANKER] Model already initialized, skipping re-initialization.")
            return

        # Lazy import transformers only when needed
        logger.info("[RERANKER] Importing transformers (lazy)...")
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        logger.info("[RERANKER] ✓ Transformers imported")
        
        logger.info("[RERANKER] Checking CUDA availability...")
        force_cpu = os.getenv("FORCE_CPU", "").lower() in ("1", "true", "yes")
        cuda_available = torch.cuda.is_available()
        if not cuda_available and not force_cpu:
            logger.warning(
                "CUDA not available; falling back to CPU for reranker (set FORCE_CPU=1 to force)."
            )

        if cuda_available and not force_cpu:
            device = "cuda"
            use_fp16 = True
            cls._device = device
            cls._use_fp16 = use_fp16
            try:
                gpu_name = torch.cuda.get_device_name(0)
                logger.info("[RERANKER] Using GPU for reranker: %s (FP16 enabled)", gpu_name)
            except Exception:
                logger.info("[RERANKER] Using GPU for reranker (FP16 enabled)")
        else:
            device = "cpu"
            use_fp16 = False
            cls._device = device
            cls._use_fp16 = use_fp16
            logger.info("[RERANKER] Using CPU for reranker (development mode)")

        # Load tokenizer
        logger.info("[RERANKER] Loading reranker tokenizer: %s", model_name)
        try:
            cls._tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            logger.info("[RERANKER] ✓ Reranker tokenizer loaded")
        except Exception as e:
            logger.exception("[RERANKER] Failed to load reranker tokenizer: %s", e)
            raise

        # Load model
        logger.info("[RERANKER] Loading reranker model: %s", model_name)
        try:
            dtype = torch.float16 if use_fp16 else torch.float32
            cls._model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                torch_dtype=dtype,
                trust_remote_code=True,
            )
            if device == "cuda":
                cls._model.to(device)
            cls._model.eval()  # Set to eval mode
            logger.info("[RERANKER] ✓ Reranker model loaded successfully on %s (FP16: %s)", device, use_fp16)
        except Exception as e:
            logger.exception("Failed to load reranker model: %s", e)
            raise

    @classmethod
    def get_instance(cls) -> AutoModelForSequenceClassification:
        """Get the singleton reranker model instance.
        
        Lazy-loads the model on first access if not already initialized.
        
        Returns:
            AutoModelForSequenceClassification: The loaded model ready for inference.
        """
        if cls._model is None:
            logger.info("Reranker model not yet loaded, initializing on first request...")
            cls.initialize()
        return cls._model

    @classmethod
    def get_tokenizer(cls) -> AutoTokenizer:
        """Get the singleton reranker tokenizer instance.
        
        Lazy-loads the tokenizer on first access if not already initialized.
        
        Returns:
            AutoTokenizer: The loaded tokenizer.
        """
        if cls._tokenizer is None:
            logger.info("Reranker tokenizer not yet loaded, initializing on first request...")
            cls.initialize()
        return cls._tokenizer

    @classmethod
    def get_device(cls) -> str:
        """Get the device used for the reranker model.
        
        Returns:
            str: Always "cuda" (CUDA is required for production).
        """
        if cls._device is None:
            # Not initialized yet; assume CPU for safety
            cls._device = os.getenv("DEFAULT_DEVICE", "cpu")
        return cls._device

    @classmethod
    def is_fp16_enabled(cls) -> bool:
        """Check if FP16 precision is enabled.
        
        Returns:
            bool: True if FP16 is used, False otherwise.
        """
        return cls._use_fp16

    @classmethod
    def score_pairs(cls, pairs: list[list[str]], batch_size: int = 16) -> list[float]:
        """Score query-passage pairs using the reranker.
        
        Args:
            pairs: List of [query, passage] pairs.
            batch_size: Batch size for inference.
            
        Returns:
            List of scores (one per pair).
            
        Raises:
            RuntimeError: If model not initialized.
        """
        model = cls.get_instance()
        tokenizer = cls.get_tokenizer()
        device = cls.get_device()
        all_scores: list[float] = []

        for start in range(0, len(pairs), batch_size):
            batch_pairs = pairs[start : start + batch_size]
            inputs = tokenizer(
                batch_pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}

            with torch.no_grad():
                logits = model(**inputs, return_dict=True).logits.view(-1).float()

            # Sigmoid normalization
            scores = torch.sigmoid(logits).cpu().tolist()
            all_scores.extend(scores)

        return all_scores


# Public interface for import
get_reranker_model = RerankerModelSingleton.get_instance
get_reranker_tokenizer = RerankerModelSingleton.get_tokenizer
initialize_reranker_model = RerankerModelSingleton.initialize
score_reranker_pairs = RerankerModelSingleton.score_pairs

logger.info("[RERANKER_MODEL] ✓ Module fully initialized with public interface")
