"""
UrbanFlow AI - GTFS (General Transit Feed Specification) Ingestion Utility
Parses feeds for transit agencies (e.g. BEST bus schedules, Mumbai Suburban rail, Metro).
"""

from pathlib import Path
from typing import Dict, Optional
import zipfile
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

from config.settings import settings
from utils.logger import logger
from utils.exceptions import DataIngestionError


class GTFSFeedLoader:
    """Loads and standardizes GTFS feed tables into pandas and GeoPandas structures."""

    REQUIRED_FILES = ["agency.txt", "routes.txt", "trips.txt", "stops.txt", "stop_times.txt"]

    def __init__(self, gtfs_path: Path):
        self.gtfs_path = gtfs_path
        self.tables: Dict[str, pd.DataFrame] = {}

    def load_feed(self) -> Dict[str, pd.DataFrame]:
        """Loads all standard CSV tables from a GTFS directory or zip archive."""
        if not self.gtfs_path.exists():
            raise DataIngestionError(f"GTFS source not found at: {self.gtfs_path}")

        logger.info(f"Loading GTFS feed from {self.gtfs_path.name}...")
        try:
            if self.gtfs_path.is_file() and self.gtfs_path.suffix == ".zip":
                with zipfile.ZipFile(self.gtfs_path, "r") as z:
                    for filename in z.namelist():
                        base = Path(filename).name
                        if base in self.REQUIRED_FILES:
                            with z.open(filename) as f:
                                self.tables[base.replace(".txt", "")] = pd.read_csv(f)
            elif self.gtfs_path.is_dir():
                for filename in self.REQUIRED_FILES:
                    file_path = self.gtfs_path / filename
                    if file_path.exists():
                        self.tables[filename.replace(".txt", "")] = pd.read_csv(file_path)

            logger.info(f"GTFS tables loaded: {list(self.tables.keys())}")
            return self.tables
        except Exception as e:
            raise DataIngestionError(f"Error parsing GTFS feed: {e}") from e

    def get_stops_geodataframe(self) -> gpd.GeoDataFrame:
        """Converts stops.txt to a GeoDataFrame in standard display CRS (EPSG:4326)."""
        if "stops" not in self.tables:
            self.load_feed()

        stops_df = self.tables.get("stops")
        if stops_df is None or stops_df.empty:
            raise DataIngestionError("No stops table found in loaded GTFS feed.")

        geometry = [Point(xy) for xy in zip(stops_df["stop_lon"], stops_df["stop_lat"])]
        return gpd.GeoDataFrame(stops_df, geometry=geometry, crs=settings.DISPLAY_CRS)

