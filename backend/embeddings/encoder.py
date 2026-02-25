from __future__ import annotations

import numpy as np

from .config import DEFAULT_EMBEDDING_CONFIG, EmbeddingConfig

try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except ImportError:
    _HAS_ST = False

_model = None


def _get_model(config: EmbeddingConfig = DEFAULT_EMBEDDING_CONFIG):
    global _model
    if not _HAS_ST:
        return None
    if _model is None:
        _model = SentenceTransformer(config.model_name)
    return _model


def encode_text(text: str, config: EmbeddingConfig = DEFAULT_EMBEDDING_CONFIG) -> np.ndarray:
    """Encode a single string into a 1-D embedding vector."""
    model = _get_model(config)
    if model is None:
        return np.zeros(384)
    return model.encode(text, show_progress_bar=False)


def encode_batch(texts: list[str], config: EmbeddingConfig = DEFAULT_EMBEDDING_CONFIG) -> np.ndarray:
    """Encode a list of strings into a 2-D array of shape (N, dim)."""
    model = _get_model(config)
    if model is None:
        return np.zeros((len(texts), 384))
    return model.encode(texts, show_progress_bar=True, batch_size=256)
