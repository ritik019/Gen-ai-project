from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str = "all-MiniLM-L6-v2"
    dimension: int = 384
    embeddings_path: Path = Path(__file__).resolve().parent.parent / "data" / "processed" / "embeddings.npy"


DEFAULT_EMBEDDING_CONFIG = EmbeddingConfig()
