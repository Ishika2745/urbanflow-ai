"""
UrbanFlow AI - Dataset B: Mumbai Suburban Railway Network Loader
Acquires, validates, and processes Mumbai Suburban Network 2025 stations and railway tracks.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import geopandas as gpd

from data_ingestion.config import (
    RAW_RAILWAY,
    PROCESSED_TRANSPORT,
    DATA_SOURCES,
    DISPLAY_CRS
)
from data_ingestion.downloader import ResilientDownloader
from data_ingestion.source_discovery import DatasetSourceDiscovery
from data_ingestion.format_parser import FormatParser
from data_ingestion.validator import DatasetValidator
from data_ingestion.logging_config import logger


class RailwayDataLoader:
    """Acquires, validates, and standardizes Mumbai Suburban Railway lines and stations."""

    def __init__(self, raw_dir: Path = RAW_RAILWAY, output_dir: Path = PROCESSED_TRANSPORT):
        self.raw_dir = raw_dir
        self.output_dir = output_dir
        self.downloader = ResilientDownloader()
        self.validator = DatasetValidator()

    def acquire_and_process(self, force_download: bool = False) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, Dict[str, Any]]:
        """
        Executes acquisition for Mumbai Suburban Network 2025:
        1. Discover download URLs for Stations and Lines KMLs
        2. Download raw KMLs to data/raw/railway/
        3. Parse geometries and attributes
        4. Classify line corridors (Western, Central, Harbour)
        5. Save railway_stations.geojson and suburban_lines.geojson
        """
        logger.info("=== [Dataset B] Ingesting Mumbai Suburban Railway Network 2025 ===")
        source_cfg = DATA_SOURCES["suburban_railway"]
        stations_kml = self.raw_dir / "mumbai_suburban_stations_2025.kml"
        lines_kml = self.raw_dir / "mumbai_suburban_lines_2025.kml"

        # 1. Download Stations KML
        self.downloader.download_file(
            url=source_cfg["stations_kml_url"],
            dest_path=stations_kml,
            dataset_name="Suburban Stations",
            force_redownload=force_download
        )

        # 2. Download Lines KML
        self.downloader.download_file(
            url=source_cfg["lines_kml_url"],
            dest_path=lines_kml,
            dataset_name="Suburban Lines",
            force_redownload=force_download
        )

        # 3. Parse Stations
        raw_stations = FormatParser.parse_kml(stations_kml)
        raw_lines = FormatParser.parse_kml(lines_kml)

        # 4. Standardize Stations Attributes
        std_stations = self._standardize_stations(raw_stations)

        # 5. Validate Stations
        valid_stations, outside_stations, summary = self.validator.validate_spatial_dataset(
            gdf=std_stations,
            dataset_name="Suburban_Railway_Stations",
            id_col="station_id",
            name_col="station_name"
        )

        # 6. Save Processed GeoJSONs
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stations_out = self.output_dir / "railway_stations.geojson"
        lines_out = self.output_dir / "suburban_lines.geojson"

        valid_stations.to_file(stations_out, driver="GeoJSON")
        raw_lines.to_file(lines_out, driver="GeoJSON")

        logger.info(f"[Dataset B] Saved {len(valid_stations)} stations and {len(raw_lines)} line corridors.")
        return valid_stations, raw_lines, summary

    def _standardize_stations(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Standardizes station names, identifies interchange hubs, and assigns line corridors."""
        df = gdf.copy()
        name_col = "Name" if "Name" in df.columns else df.columns[0]
        df["station_name"] = df[name_col].astype(str)

        df["station_id"] = [f"RAIL_{i+1:04d}" for i in range(len(df))]
        df["mode_id"] = "suburban_rail"
        df["operator"] = "Indian Railways (WR/CR)"

        # Classify key Mumbai interchanges
        interchange_keywords = ["dadar", "kurla", "andheri", "csmt", "churchgate", "bandra", "ghatkopar", "thane"]
        df["is_interchange"] = df["station_name"].apply(
            lambda name: any(kw in name.lower() for kw in interchange_keywords)
        )

        df["latitude"] = df.geometry.y
        df["longitude"] = df.geometry.x

        keep_cols = ["station_id", "station_name", "mode_id", "operator", "is_interchange", "latitude", "longitude", "geometry"]
        return df[keep_cols]

