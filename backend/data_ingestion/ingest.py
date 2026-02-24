from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
from datasets import load_dataset

from .config import DEFAULT_INGESTION_CONFIG, IngestionConfig


CANONICAL_COLUMNS: List[str] = [
    "id",
    "name",
    "address",
    "city",
    "locality",
    "price_bucket",
    "avg_cost_for_two",
    "avg_rating",
    "cuisines",
]


def _map_price_to_bucket(cost_for_two: float | int | None) -> str | None:
    if cost_for_two is None:
        return None
    try:
        value = float(cost_for_two)
    except (TypeError, ValueError):
        return None

    if value <= 300:
        return "$"
    if value <= 700:
        return "$$"
    if value <= 1500:
        return "$$$"
    return "$$$$"


def _normalize_rating(rating: float | int | str | None) -> float | None:
    if rating is None:
        return None
    raw = str(rating).strip()
    # Handle "X/5" format (e.g. "4.1/5")
    if "/" in raw:
        raw = raw.split("/")[0].strip()
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None

    # Clamp to [0, 5]
    return max(0.0, min(5.0, value))


def run_ingestion(config: IngestionConfig = DEFAULT_INGESTION_CONFIG) -> Path:
    """
    Execute the Phase 1 ingestion pipeline.

    Steps:
    - Download dataset from Hugging Face.
    - Map raw fields into canonical Restaurant schema.
    - Persist cleaned data as CSV for downstream use.
    """

    # Ensure directories exist
    config.raw_data_dir.mkdir(parents=True, exist_ok=True)
    config.processed_data_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset from Hugging Face
    dataset = load_dataset(config.dataset_name, split="train")
    df = dataset.to_pandas()

    # Try to infer raw column names as they appear in the dataset.
    # These mappings are defensive so that minor column naming differences do not break ingestion.
    def _first_present(columns: List[str]) -> str | None:
        for col in columns:
            if col in df.columns:
                return col
        return None

    col_name = _first_present(["name", "restaurant_name", "res_name"])
    col_address = _first_present(["address", "full_address", "location"])
    col_city = _first_present(["city", "City", "listed_in(city)"])
    col_locality = _first_present(["locality", "location", "area", "subzone"])
    col_cost_for_two = _first_present(
        ["cost_for_two", "approx_cost_for_two", "approx_cost(for two people)"]
    )
    col_rating = _first_present(["rating", "rate", "aggregate_rating"])
    col_cuisines = _first_present(["cuisines", "cuisine"])

    # Build canonical DataFrame
    canonical = pd.DataFrame()
    canonical["id"] = df.index.astype(str)

    canonical["name"] = df[col_name] if col_name else ""
    canonical["address"] = df[col_address] if col_address else ""
    canonical["city"] = df[col_city] if col_city else ""
    canonical["locality"] = df[col_locality] if col_locality else ""

    if col_cost_for_two:
        canonical["avg_cost_for_two"] = pd.to_numeric(
            df[col_cost_for_two], errors="coerce"
        )
    else:
        canonical["avg_cost_for_two"] = pd.NA

    canonical["price_bucket"] = canonical["avg_cost_for_two"].apply(
        _map_price_to_bucket
    )

    if col_rating:
        canonical["avg_rating"] = df[col_rating].apply(_normalize_rating)
    else:
        canonical["avg_rating"] = pd.NA

    if col_cuisines:
        canonical["cuisines"] = df[col_cuisines].fillna("").astype(str)
    else:
        canonical["cuisines"] = ""

    # Ensure all expected columns exist and order them
    canonical = canonical[CANONICAL_COLUMNS]

    # Write processed CSV
    output_path = config.processed_path
    canonical.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    path = run_ingestion()
    print(f"Ingestion complete. Processed data saved to: {path}")
