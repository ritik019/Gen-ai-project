from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
_PROCESSED_CSV = _PROCESSED_DIR / "restaurants.csv"
_EMBEDDINGS_NPY = _PROCESSED_DIR / "embeddings.npy"

_df: pd.DataFrame | None = None
_embeddings: np.ndarray | None = None


def _load() -> pd.DataFrame:
    df = pd.read_csv(_PROCESSED_CSV)

    # Pre-parse cuisines into lists and lowercase for matching
    df["cuisines_list"] = (
        df["cuisines"]
        .fillna("")
        .apply(lambda s: [c.strip().lower() for c in s.split(",") if c.strip()])
    )

    # Lowercase city and locality for case-insensitive lookup
    df["city_lower"] = df["city"].fillna("").str.lower()
    df["locality_lower"] = df["locality"].fillna("").str.lower()

    return df


def get_dataframe() -> pd.DataFrame:
    """Return the in-memory restaurant DataFrame, loading it on first call."""
    global _df
    if _df is None:
        _df = _load()
    return _df


def get_embeddings() -> np.ndarray | None:
    """Return precomputed restaurant embeddings, or None if file missing."""
    global _embeddings
    if _embeddings is None and _EMBEDDINGS_NPY.exists():
        _embeddings = np.load(_EMBEDDINGS_NPY)
    return _embeddings
