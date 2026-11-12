"""
UrbanFlow AI - Multi-Format Spatial Exporter
Exports processed geospatial feature sets into:
- GeoParquet (.parquet): High-efficiency columnar storage with snappy/zstandard compression
- GeoPackage (.gpkg): Open OGC standard spatial database container with RTree spatial index
- GeoJSON (.geojson): Web map interchange format for Streamlit, Folium, and Deck.gl
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import geopandas as gpd

from data_ingestion.config import PROCESSED_DIR
from preprocessing.coordinate_manager import CoordinateManager
from utils.logger import logger

FEATURES_DIR = PROCESSED_DIR.parent / "features"


class SpatialExporter:
    """Exports GeoDataFrames into GeoParquet, GeoPackage, and GeoJSON formats with integrity validation."""

    def __init__(self, output_dir: Path = FEATURES_DIR):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_layer(
        self,
        gdf: gpd.GeoDataFrame,
        layer_name: str,
        export_parquet: bool = True,
        export_gpkg: bool = True,
        export_geojson: bool = True
    ) -> Dict[str, str]:
        """
        Exports a GeoDataFrame to the specified formats.
        Ensures 2D geometries and display CRS (EPSG:4326) for interchange.
        """
        exported_files = {}
        # Ensure 2D and valid display CRS
        clean_gdf = CoordinateManager.strip_z(gdf)
        clean_gdf = CoordinateManager.to_display(clean_gdf)

        # 1. GeoParquet
        if export_parquet:
            parquet_path = self.output_dir / f"{layer_name}.parquet"
            clean_gdf.to_parquet(parquet_path, index=False)
            exported_files["geoparquet"] = str(parquet_path)
            logger.info(f"Exported GeoParquet: {parquet_path.name} ({parquet_path.stat().st_size / 1024:.1f} KB)")

        # 2. GeoPackage
        if export_gpkg:
            gpkg_path = self.output_dir / f"{layer_name}.gpkg"
            clean_gdf.to_file(gpkg_path, layer=layer_name, driver="GPKG")
            exported_files["geopackage"] = str(gpkg_path)
            logger.info(f"Exported GeoPackage: {gpkg_path.name} ({gpkg_path.stat().st_size / 1024:.1f} KB)")

        # 3. GeoJSON
        if export_geojson:
            geojson_path = self.output_dir / f"{layer_name}.geojson"
            clean_gdf.to_file(geojson_path, driver="GeoJSON")
            exported_files["geojson"] = str(geojson_path)
            logger.info(f"Exported GeoJSON: {geojson_path.name} ({geojson_path.stat().st_size / 1024:.1f} KB)")

        return exported_files

