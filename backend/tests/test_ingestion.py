from pathlib import Path

import pandas as pd

from backend.data_ingestion.config import IngestionConfig
from backend.data_ingestion.ingest import CANONICAL_COLUMNS, run_ingestion


def test_run_ingestion_creates_non_empty_processed_file(tmp_path: Path):
    """
    End-to-end test for Phase 1 ingestion.

    Uses a temporary output directory so we don't pollute real data directories.
    """
    cfg = IngestionConfig(
        raw_data_dir=tmp_path / "raw",
        processed_data_dir=tmp_path / "processed",
    )

    output_path = run_ingestion(config=cfg)

    assert output_path.is_file(), "Processed CSV should be created"

    df = pd.read_csv(output_path)
    assert not df.empty, "Processed dataset should not be empty"
    assert list(df.columns) == CANONICAL_COLUMNS

