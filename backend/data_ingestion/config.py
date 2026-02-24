from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class IngestionConfig:
    """
    Configuration for the Phase 1 data ingestion pipeline.
    """

    dataset_name: str = "ManikaSaini/zomato-restaurant-recommendation"
    raw_data_dir: Path = Path("backend/data/raw")
    processed_data_dir: Path = Path("backend/data/processed")
    processed_filename: str = "restaurants.csv"

    @property
    def processed_path(self) -> Path:
        return self.processed_data_dir / self.processed_filename


DEFAULT_INGESTION_CONFIG = IngestionConfig()

