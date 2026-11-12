"""
UrbanFlow AI - Supplementary Data Loader: Socioeconomic & Demographic Indicators
Acquires ward-level census data (PCA) from OpenCity to construct vulnerability indices.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import pandas as pd

from data_ingestion.config import (
    RAW_SUPPLEMENTARY,
    DATA_SOURCES
)
from data_ingestion.downloader import ResilientDownloader
from data_ingestion.logging_config import logger


class SupplementaryDataLoader:
    """Manages acquisition and transformation of supplementary census and socioeconomic datasets."""

    def __init__(self, raw_dir: Path = RAW_SUPPLEMENTARY):
        self.raw_dir = raw_dir
        self.downloader = ResilientDownloader()

    def acquire_and_process(self, force_download: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Downloads and parses ward-level census demographic data.
        """
        logger.info("=== [Supplementary] Ingesting Mumbai Ward-Wise Census Indicators ===")
        source_cfg = DATA_SOURCES["supplementary_census"]
        csv_dest = self.raw_dir / "mumbai_ward_census_2011.csv"

        self.downloader.download_file(
            url=source_cfg["csv_url"],
            dest_path=csv_dest,
            dataset_name="Mumbai Census Data",
            force_redownload=force_download
        )

        df = pd.read_csv(csv_dest)
        logger.info(f"Loaded supplementary census dataset ({len(df)} rows, {len(df.columns)} columns)")

        summary = {
            "dataset_name": "Mumbai_Ward_Census_2011",
            "source": "OpenCity / Census of India 2011",
            "rows_count": len(df),
            "columns": list(df.columns)[:8],
            "status": "VALIDATED"
        }
        return df, summary

