"""
Offline script to precompute restaurant embeddings.

Usage:
    python -m backend.embeddings.precompute
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DEFAULT_EMBEDDING_CONFIG
from .encoder import encode_batch

_PROCESSED_CSV = DEFAULT_EMBEDDING_CONFIG.embeddings_path.parent / "restaurants.csv"


def _build_text(row: pd.Series) -> str:
    parts: list[str] = []
    if pd.notna(row.get("name")):
        parts.append(str(row["name"]))
    if pd.notna(row.get("cuisines")):
        parts.append(str(row["cuisines"]))
    if pd.notna(row.get("locality")):
        parts.append(str(row["locality"]))
    return " ".join(parts).strip().lower()


def run_precompute() -> None:
    df = pd.read_csv(_PROCESSED_CSV)
    texts = df.apply(_build_text, axis=1).tolist()

    print(f"Encoding {len(texts)} restaurants ...")
    embeddings = encode_batch(texts)

    out_path = DEFAULT_EMBEDDING_CONFIG.embeddings_path
    np.save(out_path, embeddings)
    print(f"Saved embeddings ({embeddings.shape}) to {out_path}")


if __name__ == "__main__":
    run_precompute()
