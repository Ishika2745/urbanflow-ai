"""
UrbanFlow AI - Configurable Spatial Grid Generator
Implements Part C:
- Generates uniform spatial tessellations clipped strictly to Greater Mumbai perimeter:
  * 500-metre micro-grid (mumbai_grid_500m)
  * 1-kilometre meso-grid (mumbai_grid_1km)
  * Ward-level administrative aggregation (24 BMC Wards)
- Executes conservative, non-double-counting areal weighted population interpolation
  from WorldPop high-resolution raster data.
- Enforces EPSG:32643 metric projection for planar operations and EPSG:4326 for interchange.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box, Point, Polygon, MultiPolygon
from shapely.validation import make_valid

from data_ingestion.config import (
    INTERIM_CLEANED,
    PROCESSED_BOUNDARIES,
    PROCESSED_POPULATION,
    METRIC_CRS,
    DISPLAY_CRS
)
from preprocessing.coordinate_manager import CoordinateManager
from utils.logger import logger


class SpatialGridGenerator:
    """Constructs configurable spatial analytical units and interpolates demographic mass without double-counting."""

    def __init__(self, interim_dir: Path = INTERIM_CLEANED):
        self.interim_dir = interim_dir

    def create_grid(
        self,
        resolution_meters: int,
        perimeter_gdf: gpd.GeoDataFrame,
        prefix: str = "GRID"
    ) -> gpd.GeoDataFrame:
        """
        Generates a regular metric tessellation clipped to the Greater Mumbai perimeter.
        Calculates exact metric area (sq km) in EPSG:32643 and exports in EPSG:4326.
        """
        logger.info(f"Generating {resolution_meters}m spatial grid across Greater Mumbai...")
        # 1. Project perimeter to metric CRS
        metric_perim = CoordinateManager.to_metric(CoordinateManager.strip_z(perimeter_gdf))
        boundary_geom = metric_perim.geometry.union_all() if hasattr(metric_perim.geometry, "union_all") else metric_perim.geometry.unary_union

        xmin, ymin, xmax, ymax = boundary_geom.bounds
        x_coords = np.arange(xmin, xmax, resolution_meters)
        y_coords = np.arange(ymin, ymax, resolution_meters)

        records = []
        cell_index = 0
        min_cell_area_m2 = max(500.0, (resolution_meters ** 2) * 0.01) # Keep cells with at least 1% land coverage

        for x in x_coords:
            for y in y_coords:
                cell_box = box(x, y, x + resolution_meters, y + resolution_meters)
                if boundary_geom.intersects(cell_box):
                    intersection = boundary_geom.intersection(cell_box)
                    if not intersection.is_empty and intersection.area >= min_cell_area_m2:
                        cell_index += 1
                        cell_id = f"{prefix}_{resolution_meters}M_{cell_index:04d}"
                        records.append({
                            "cell_id": cell_id,
                            "area_sq_km": round(intersection.area / 1e6, 4),
                            "geometry": intersection
                        })

        grid_metric = gpd.GeoDataFrame(records, crs=METRIC_CRS)
        grid_metric["geometry"] = grid_metric["geometry"].apply(make_valid)

        # Compute centroid coordinates accurately in metric CRS
        metric_centroids = grid_metric.geometry.centroid
        centroids_disp = gpd.GeoSeries(metric_centroids, crs=METRIC_CRS).to_crs(DISPLAY_CRS)
        
        # Transform to display CRS
        grid_disp = CoordinateManager.to_display(grid_metric)
        grid_disp["centroid_lat"] = centroids_disp.y
        grid_disp["centroid_lon"] = centroids_disp.x

        logger.info(f"Generated {len(grid_disp)} cells for {resolution_meters}m grid (Total area: {grid_disp['area_sq_km'].sum():.2f} sq km).")
        return grid_disp

    def allocate_population_areal(
        self,
        target_gdf: gpd.GeoDataFrame,
        id_col: str = "cell_id"
    ) -> gpd.GeoDataFrame:
        """
        Allocates WorldPop population to target spatial polygons using exact areal weighted interpolation.
        Mathematically guarantees that:
          Sum(allocated_pop) == Sum(source_pop) across the overlapping region.
        Strictly prevents double-counting raster population across cell boundaries.
        """
        pop_path = PROCESSED_POPULATION / "population_grid.geojson"
        if not pop_path.exists():
            logger.warning(f"Population raster grid not found at: {pop_path}; setting population to 0.")
            target_gdf["population"] = 0.0
            target_gdf["population_density_per_sq_km"] = 0.0
            return target_gdf

        logger.info(f"Performing conservative areal-weighted population interpolation for {len(target_gdf)} units...")
        pop_gdf = gpd.read_file(pop_path)
        pop_gdf = CoordinateManager.strip_z(pop_gdf)

        # Project both to metric CRS for exact planar area calculations
        target_m = CoordinateManager.to_metric(target_gdf[[id_col, "area_sq_km", "geometry"]].copy())
        target_m = target_m.rename(columns={id_col: "target_unit_id"})

        pop_m = CoordinateManager.to_metric(pop_gdf[["cell_id", "population", "geometry"]].copy())
        pop_m = pop_m.rename(columns={"cell_id": "source_pop_cell_id"})
        pop_m["source_cell_area"] = pop_m.geometry.area

        # Spatial overlay intersection
        overlay = gpd.overlay(
            target_m[["target_unit_id", "geometry"]],
            pop_m[["source_pop_cell_id", "population", "source_cell_area", "geometry"]],
            how="intersection"
        )

        if not overlay.empty:
            # Proportion of source raster cell falling inside target cell
            overlay["intersection_area"] = overlay.geometry.area
            overlay["area_ratio"] = np.clip(overlay["intersection_area"] / overlay["source_cell_area"], 0.0, 1.0)
            overlay["allocated_pop"] = overlay["population"] * overlay["area_ratio"]

            # Aggregate by target cell ID
            pop_per_target = overlay.groupby("target_unit_id")["allocated_pop"].sum().round(1)
        else:
            pop_per_target = pd.Series(dtype=float)

        # Join back
        res = target_gdf.copy()
        res["population"] = res[id_col].map(pop_per_target).fillna(0.0)
        res["population_density_per_sq_km"] = (res["population"] / np.maximum(res["area_sq_km"], 0.001)).round(2)

        total_alloc = res["population"].sum()
        logger.info(f"Allocated total population: {total_alloc:,.0f} across {len(res)} spatial units (Zero double-counting).")
        return res

    def assign_ward_membership(
        self,
        target_gdf: gpd.GeoDataFrame,
        wards_gdf: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """
        Assigns each grid cell to its enclosing administrative ward using centroid spatial containment.
        """
        logger.info("Assigning administrative ward membership to spatial grid cells...")
        target_disp = CoordinateManager.to_display(target_gdf)
        wards_disp = CoordinateManager.to_display(wards_gdf[["ward_id", "ward_name", "geometry"]])

        # Create centroid points using precomputed metric centroids
        centroids = gpd.GeoDataFrame(
            {"geometry": [Point(lon, lat) for lon, lat in zip(target_disp["centroid_lon"], target_disp["centroid_lat"])]},
            crs=DISPLAY_CRS,
            index=target_disp.index
        )

        joined = gpd.sjoin(centroids, wards_disp, how="left", predicate="within")
        joined = joined[~joined.index.duplicated(keep="first")]

        # Handle edge cells whose centroid might fall just outside a ward boundary
        if joined["ward_id"].isna().any():
            # Nearest ward match in metric CRS for unassigned cells
            wards_metric = CoordinateManager.to_metric(wards_disp)
            centroids_metric = CoordinateManager.to_metric(centroids)
            missing_idx = joined[joined["ward_id"].isna()].index
            for idx in missing_idx:
                c_geom = centroids_metric.loc[idx, "geometry"]
                nearest_ward = wards_metric.distance(c_geom).idxmin()
                joined.loc[idx, "ward_id"] = wards_disp.loc[nearest_ward, "ward_id"]
                joined.loc[idx, "ward_name"] = wards_disp.loc[nearest_ward, "ward_name"]

        res = target_gdf.copy()
        res["ward_id"] = joined["ward_id"].values
        res["ward_name"] = joined["ward_name"].values
        return res
