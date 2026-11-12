"""
UrbanFlow AI - Central Spatial Integration Processor
Integrates bus stops, suburban railway stations, GTFS nodes, ward boundaries, and population grids.
Enforces dual-CRS projection between EPSG:32643 (Metric) and EPSG:4326 (Display).
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

from data_ingestion.config import (
    PROCESSED_INTEGRATED,
    DISPLAY_CRS,
    METRIC_CRS
)
from data_ingestion.logging_config import logger


class SpatialIntegrator:
    """Combines individual acquired layers into cohesive, analytical networks."""

    def __init__(self, output_dir: Path = PROCESSED_INTEGRATED):
        self.output_dir = output_dir

    def integrate_all(
        self,
        best_stops_gdf: gpd.GeoDataFrame,
        railway_stations_gdf: gpd.GeoDataFrame,
        wards_gdf: gpd.GeoDataFrame,
        gtfs_stops_gdf: Optional[gpd.GeoDataFrame] = None,
        population_grid_gdf: Optional[gpd.GeoDataFrame] = None
    ) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, Dict[str, Any]]:
        """
        Integrates multimodal transport points, joins with ward polygons,
        and generates integrated_transport_network.geojson and ward_accessibility_profiles.geojson.
        """
        logger.info("=== Integrating Multimodal Transport & Ward Polygons ===")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Standardize transport layers
        best_clean = best_stops_gdf[["stop_id", "stop_name", "mode_id", "geometry"]].copy()
        best_clean["source"] = "OpenCity_BEST"
        best_clean["is_rail"] = False

        rail_clean = railway_stations_gdf[["station_id", "station_name", "mode_id", "is_interchange", "geometry"]].copy()
        rail_clean = rail_clean.rename(columns={"station_id": "stop_id", "station_name": "stop_name"})
        rail_clean["source"] = "OpenCity_SuburbanRail_2025"
        rail_clean["is_rail"] = True

        combined_records = [best_clean, rail_clean]

        if gtfs_stops_gdf is not None and not gtfs_stops_gdf.empty:
            gtfs_clean = gtfs_stops_gdf[["stop_id", "stop_name", "mode_id", "geometry"]].copy()
            gtfs_clean["source"] = "croyla_GTFS"
            gtfs_clean["is_rail"] = False
            combined_records.append(gtfs_clean)

        # Unified Transport GeoDataFrame
        unified_stops = pd.concat(combined_records, ignore_index=True)
        unified_stops_gdf = gpd.GeoDataFrame(unified_stops, crs=DISPLAY_CRS)

        # 2. Spatial Join stops to Wards in Metric CRS to ensure accurate intersection
        wards_metric = wards_gdf.to_crs(METRIC_CRS)
        stops_metric = unified_stops_gdf.to_crs(METRIC_CRS)

        joined_stops_metric = gpd.sjoin(stops_metric, wards_metric[["ward_id", "ward_name", "geometry"]], how="left", predicate="within")
        if "index_right" in joined_stops_metric.columns:
            joined_stops_metric = joined_stops_metric.drop(columns=["index_right"])
        joined_stops = joined_stops_metric.to_crs(DISPLAY_CRS)
        joined_stops["latitude"] = joined_stops.geometry.y
        joined_stops["longitude"] = joined_stops.geometry.x

        # 3. Aggregate stop counts per ward
        stop_counts = joined_stops.groupby("ward_id")["stop_id"].count().rename("total_transit_stops")
        rail_counts = joined_stops[joined_stops["is_rail"] == True].groupby("ward_id")["stop_id"].count().rename("rail_stations_count")

        integrated_wards = wards_gdf.merge(stop_counts, on="ward_id", how="left").fillna({"total_transit_stops": 0})
        integrated_wards = integrated_wards.merge(rail_counts, on="ward_id", how="left").fillna({"rail_stations_count": 0})

        # Calculate ward area in sq km via metric projection
        integrated_wards["area_sq_km"] = wards_metric.geometry.area / 1e6
        integrated_wards["transit_stop_density"] = integrated_wards["total_transit_stops"] / integrated_wards["area_sq_km"].clip(lower=0.1)

        # 4. Save integrated outputs
        unified_path = self.output_dir / "integrated_transport_network.geojson"
        wards_path = self.output_dir / "ward_accessibility_profiles.geojson"

        joined_stops.to_file(unified_path, driver="GeoJSON")
        integrated_wards.to_file(wards_path, driver="GeoJSON")

        summary = {
            "integrated_transport_stops": len(joined_stops),
            "rail_stations": len(rail_clean),
            "bus_stops": len(best_clean),
            "wards_profiled": len(integrated_wards),
            "output_network_geojson": str(unified_path.name),
            "output_wards_geojson": str(wards_path.name)
        }
        logger.info(f"Spatial Integration Complete: {len(joined_stops)} stops associated with {len(integrated_wards)} wards.")
        return joined_stops, integrated_wards, summary
