"""
UrbanFlow AI - Dataset C: Mumbai GTFS Data Loader (croyla/mumbai-gtfs)
Extracts, parses, validates, and standardizes scheduled transit feeds for BEST, TMT, and KDMT.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString

from data_ingestion.config import (
    RAW_GTFS,
    PROCESSED_TRANSPORT,
    DATA_SOURCES,
    DISPLAY_CRS
)
from data_ingestion.downloader import ResilientDownloader
from data_ingestion.file_extractor import ArchiveExtractor
from data_ingestion.validator import DatasetValidator
from data_ingestion.logging_config import logger


class GTFSDataLoader:
    """Acquires, extracts, validates, and processes scheduled GTFS transit data for Mumbai."""

    GTFS_FILES = [
        "agency.txt", "stops.txt", "routes.txt", "trips.txt",
        "stop_times.txt", "calendar.txt", "calendar_dates.txt",
        "shapes.txt", "transfers.txt", "feed_info.txt"
    ]

    def __init__(self, raw_dir: Path = RAW_GTFS, output_dir: Path = PROCESSED_TRANSPORT):
        self.raw_dir = raw_dir
        self.output_dir = output_dir
        self.downloader = ResilientDownloader()
        self.validator = DatasetValidator()
        self.tables: Dict[str, pd.DataFrame] = {}

    def acquire_and_process(self, force_download: bool = False) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, Dict[str, Any]]:
        """
        Full GTFS pipeline:
        1. Download master repository ZIP from GitHub
        2. Extract inner gtfs_compat.zip
        3. Parse all available text tables
        4. Validate integrity, service dates, and stop coordinates
        5. Build gtfs_stops.geojson and gtfs_routes.geojson
        """
        logger.info("=== [Dataset C] Ingesting Mumbai GTFS Bus Feeds (croyla/mumbai-gtfs) ===")
        repo_zip_path = self.raw_dir / "mumbai_gtfs_repo.zip"
        source_cfg = DATA_SOURCES["mumbai_gtfs"]

        # 1. Download GitHub repository ZIP
        self.downloader.download_file(
            url=source_cfg["master_zip_url"],
            dest_path=repo_zip_path,
            dataset_name="Mumbai GTFS Repo",
            force_redownload=force_download
        )

        # 2. Extract inner gtfs_compat.zip into raw/gtfs/extracted/
        extracted_dir = self.raw_dir / "extracted"
        extracted_dir.mkdir(parents=True, exist_ok=True)
        ArchiveExtractor.extract_nested_zip(repo_zip_path, "gtfs_compat.zip", extracted_dir)

        # 3. Parse all present GTFS tables
        for filename in self.GTFS_FILES:
            file_path = extracted_dir / filename
            table_key = filename.replace(".txt", "")
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path, low_memory=False)
                    self.tables[table_key] = df
                    logger.info(f"Loaded GTFS table '{table_key}' ({len(df)} rows)")
                except Exception as e:
                    logger.warning(f"Error reading {filename}: {e}")

        # 4. Validate Relational Integrity
        val_summary = self.validator.validate_gtfs_tables(self.tables)

        # 5. Build GTFS Stops GeoDataFrame
        stops_gdf = self._build_stops_geodataframe()

        # Validate stops spatially
        valid_stops_gdf, outside_stops, stop_summary = self.validator.validate_spatial_dataset(
            gdf=stops_gdf,
            dataset_name="GTFS_Bus_Stops",
            id_col="stop_id",
            name_col="stop_name"
        )

        # 6. Build GTFS Routes GeoDataFrame
        routes_gdf = self._build_routes_geodataframe()

        # 7. Save outputs
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stops_out = self.output_dir / "gtfs_stops.geojson"
        routes_out = self.output_dir / "gtfs_routes.geojson"

        valid_stops_gdf.to_file(stops_out, driver="GeoJSON")
        if not routes_gdf.empty:
            routes_gdf.to_file(routes_out, driver="GeoJSON")

        logger.info(f"[Dataset C] Saved {len(valid_stops_gdf)} GTFS stops and {len(routes_gdf)} route alignments.")
        return valid_stops_gdf, routes_gdf, val_summary

    def _build_stops_geodataframe(self) -> gpd.GeoDataFrame:
        """Converts stops.txt to a GeoDataFrame and standardizes operator attribution."""
        if "stops" not in self.tables:
            raise ValueError("No stops.txt found in extracted GTFS feed.")

        stops = self.tables["stops"].copy()
        # Drop coordinates with NaNs
        stops = stops.dropna(subset=["stop_lat", "stop_lon"])

        # Determine agency / operator details from routes/agency if available
        agency_map = {}
        if "agency" in self.tables:
            for _, r in self.tables["agency"].iterrows():
                agency_map[str(r.get("agency_id", ""))] = r.get("agency_name", "Public Transit")

        stops["mode_id"] = "bus_gtfs"
        geometry = [Point(xy) for xy in zip(stops["stop_lon"], stops["stop_lat"])]
        gdf = gpd.GeoDataFrame(stops, geometry=geometry, crs=DISPLAY_CRS)
        return gdf

    def _build_routes_geodataframe(self) -> gpd.GeoDataFrame:
        """Constructs synthetic route LineStrings from stop sequences if shapes.txt is missing."""
        if "routes" not in self.tables or "trips" not in self.tables or "stop_times" not in self.tables:
            return gpd.GeoDataFrame()

        routes = self.tables["routes"].copy()
        trips = self.tables["trips"]
        st = self.tables["stop_times"]
        stops = self.tables["stops"].set_index("stop_id")

        # Pick one representative trip per route
        rep_trips = trips.drop_duplicates(subset=["route_id"]).set_index("trip_id")

        route_features = []
        for trip_id, trip_row in rep_trips.iterrows():
            route_id = trip_row["route_id"]
            trip_stop_times = st[st["trip_id"] == trip_id].sort_values(by="stop_sequence")
            coords = []
            for _, row in trip_stop_times.iterrows():
                sid = row["stop_id"]
                if sid in stops.index:
                    stop_row = stops.loc[sid]
                    if isinstance(stop_row, pd.DataFrame):
                        stop_row = stop_row.iloc[0]
                    coords.append((float(stop_row["stop_lon"]), float(stop_row["stop_lat"])))

            if len(coords) >= 2:
                route_info = routes[routes["route_id"] == route_id].iloc[0] if not routes[routes["route_id"] == route_id].empty else {}
                route_features.append({
                    "route_id": route_id,
                    "route_short_name": route_info.get("route_short_name", str(route_id)),
                    "route_long_name": route_info.get("route_long_name", ""),
                    "agency_id": route_info.get("agency_id", "BEST"),
                    "geometry": LineString(coords)
                })

        return gpd.GeoDataFrame(route_features, crs=DISPLAY_CRS)

