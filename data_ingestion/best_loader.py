"""
UrbanFlow AI - Dataset A: BEST Bus Stops and Depots Loader
Automates downloading, parsing, validation, and standardization of Mumbai BEST bus infrastructure.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

from data_ingestion.config import (
    RAW_BEST,
    PROCESSED_TRANSPORT,
    DATA_SOURCES,
    DISPLAY_CRS
)
from data_ingestion.downloader import ResilientDownloader
from data_ingestion.source_discovery import DatasetSourceDiscovery
from data_ingestion.format_parser import FormatParser
from data_ingestion.validator import DatasetValidator
from data_ingestion.logging_config import logger


class BESTDataLoader:
    """Acquires, validates, and standardizes BEST bus stops and depots."""

    def __init__(self, raw_dir: Path = RAW_BEST, output_dir: Path = PROCESSED_TRANSPORT):
        self.raw_dir = raw_dir
        self.output_dir = output_dir
        self.downloader = ResilientDownloader()
        self.validator = DatasetValidator()

    def acquire_and_process(self, force_download: bool = False) -> Tuple[gpd.GeoDataFrame, Dict[str, Any]]:
        """
        Executes end-to-end pipeline for BEST dataset:
        1. Discover download URL from OpenCity CKAN API or fallback to direct URL
        2. Download raw KML to data/raw/best/
        3. Parse KML placemarks and attributes
        4. Validate coordinates, duplicates, and Mumbai study extent
        5. Save standardized GeoJSON to data/processed/transport/best_stops.geojson
        """
        logger.info("=== [Dataset A] Ingesting BEST Bus Stops and Depots ===")
        source_cfg = DATA_SOURCES["best_bus"]
        target_kml = self.raw_dir / "mumbai_best_bus_stops_and_depots.kml"

        # 1. Automated Discovery & Download
        download_url = source_cfg.get("direct_download_url")
        # Try dynamic discovery from OpenCity
        discovered = DatasetSourceDiscovery.discover_opencity_package("mumbai-best-bus-stops-and-depots-data")
        if discovered:
            for res in discovered.get("resources", []):
                if res.get("format", "").upper() == "KML" and res.get("url"):
                    download_url = res["url"]
                    break

        self.downloader.download_file(
            url=download_url,
            dest_path=target_kml,
            dataset_name="BEST Bus Stops",
            force_redownload=force_download
        )

        # 2. Parse KML
        raw_gdf = FormatParser.parse_kml(target_kml)

        # 3. Standardize Columns
        standardized = self._standardize_attributes(raw_gdf)

        # 4. Validate
        valid_gdf, outside_gdf, summary = self.validator.validate_spatial_dataset(
            gdf=standardized,
            dataset_name="BEST_Bus_Stops",
            id_col="stop_id",
            name_col="stop_name"
        )

        # 5. Save Processed GeoJSON
        self.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.output_dir / "best_stops.geojson"
        valid_gdf.to_file(out_path, driver="GeoJSON")
        logger.info(f"[Dataset A] Saved {len(valid_gdf)} validated BEST stops to {out_path.name}")

        return valid_gdf, summary

    def _standardize_attributes(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Standardizes attribute column names and extracts depot metadata."""
        df = gdf.copy()
        # Name column
        name_col = "Name" if "Name" in df.columns else df.columns[0]
        df["stop_name"] = df[name_col].astype(str)

        # Identify depots vs regular stops
        is_depot = df["stop_name"].str.contains(r"depot|bus station|chowki", case=False, na=False)
        df["is_depot"] = is_depot
        df["mode_id"] = "best_bus"
        df["agency_name"] = "Brihanmumbai Electric Supply and Transport (BEST)"

        # Generate clean synthetic/extracted stop_id
        df["stop_id"] = [f"BEST_{i+1:05d}" for i in range(len(df))]

        df["latitude"] = df.geometry.y
        df["longitude"] = df.geometry.x

        keep_cols = ["stop_id", "stop_name", "mode_id", "agency_name", "is_depot", "latitude", "longitude", "geometry"]
        return df[keep_cols]

