"""
UrbanFlow AI - Dataset F: Greater Mumbai & BMC Ward Boundaries Loader
Acquires, standardizes, validates, and derives municipal administrative and outer study boundaries.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

from data_ingestion.config import (
    RAW_BOUNDARIES,
    PROCESSED_BOUNDARIES,
    DATA_SOURCES,
    DISPLAY_CRS
)
from data_ingestion.downloader import ResilientDownloader
from data_ingestion.format_parser import FormatParser
from data_ingestion.validator import DatasetValidator
from data_ingestion.logging_config import logger


class BoundaryDataLoader:
    """Acquires, validates, and builds official BMC Ward and Greater Mumbai perimeter boundaries."""

    def __init__(self, raw_dir: Path = RAW_BOUNDARIES, output_dir: Path = PROCESSED_BOUNDARIES):
        self.raw_dir = raw_dir
        self.output_dir = output_dir
        self.downloader = ResilientDownloader()
        self.validator = DatasetValidator()

    def acquire_and_process(self, force_download: bool = False) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, Dict[str, Any]]:
        """
        Full Boundary pipeline:
        1. Download Mumbai Wards KML from OpenCity
        2. Parse Ward Polygons
        3. Standardize Ward IDs (A to T) and names
        4. Validate geometries and compute unary union for Greater Mumbai perimeter
        5. Save ward_boundaries.geojson and greater_mumbai_boundary.geojson
        """
        logger.info("=== [Dataset F] Ingesting Greater Mumbai & BMC Ward Boundaries ===")
        source_cfg = DATA_SOURCES["mumbai_wards"]
        wards_kml_path = self.raw_dir / "mumbai_bmc_wards_map.kml"

        # 1. Download
        self.downloader.download_file(
            url=source_cfg["wards_kml_url"],
            dest_path=wards_kml_path,
            dataset_name="Mumbai Wards KML",
            force_redownload=force_download
        )

        # 2. Parse KML
        raw_wards_gdf = FormatParser.parse_kml(wards_kml_path)

        # 3. Standardize Attributes
        std_wards = self._standardize_wards(raw_wards_gdf)

        # 4. Validate
        valid_wards, outside_wards, summary = self.validator.validate_spatial_dataset(
            gdf=std_wards,
            dataset_name="BMC_Ward_Boundaries",
            id_col="ward_id",
            name_col="ward_name"
        )

        # 5. Derive Outer Greater Mumbai Boundary Polygon (Unary Union)
        study_boundary_poly = unary_union(valid_wards.geometry)
        boundary_gdf = gpd.GeoDataFrame(
            [{"name": "Greater Mumbai", "admin_level": "Municipal Corporation", "geometry": study_boundary_poly}],
            crs=DISPLAY_CRS
        )

        # 6. Save outputs
        self.output_dir.mkdir(parents=True, exist_ok=True)
        wards_out = self.output_dir / "ward_boundaries.geojson"
        boundary_out = self.output_dir / "greater_mumbai_boundary.geojson"

        valid_wards.to_file(wards_out, driver="GeoJSON")
        boundary_gdf.to_file(boundary_out, driver="GeoJSON")

        logger.info(f"[Dataset F] Saved {len(valid_wards)} BMC ward boundaries and integrated city perimeter.")
        return valid_wards, boundary_gdf, summary

    def _standardize_wards(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Standardizes ward identifiers, names, and centroids."""
        df = gdf.copy()
        name_col = "Name" if "Name" in df.columns else df.columns[0]
        df["raw_name"] = df[name_col].astype(str)

        # Extract or clean ward code (e.g. Ward A, Ward F/N, etc.)
        def extract_code(val: str, idx: int) -> str:
            val_clean = val.replace("Ward", "").replace("WARD", "").strip()
            if len(val_clean) <= 4 and val_clean:
                return val_clean.upper()
            return f"W_{idx+1:02d}"

        df["ward_id"] = [extract_code(row["raw_name"], i) for i, row in df.iterrows()]
        df["ward_name"] = df["raw_name"]

        # Ensure valid Polygon geometries
        df["geometry"] = df.geometry.buffer(0)

        # Centroid coordinates
        centroids = df.geometry.centroid
        df["centroid_lat"] = centroids.y
        df["centroid_lon"] = centroids.x

        keep_cols = ["ward_id", "ward_name", "centroid_lat", "centroid_lon", "geometry"]
        return df[keep_cols]

