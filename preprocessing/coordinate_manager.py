"""
UrbanFlow AI - Coordinate Reference System (CRS) & Metric Spatial Engine
Enforces dual-CRS architecture:
- EPSG:32643 (UTM Zone 43N) for all metric distance calculations, buffer generation, and area measurements in Mumbai.
- EPSG:4326 (WGS84) for map display, GeoJSON interchange, and dashboard visualization.
Provides accelerated KDTree metric spatial indexing in projected space.
"""

from typing import Tuple, List, Optional, Union
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from shapely.geometry import Point, Polygon, MultiPolygon
from scipy.spatial import cKDTree

from config.settings import settings
from utils.logger import logger
from utils.exceptions import CRSMismatchError, SpatialDataError

METRIC_CRS: str = "EPSG:32643"
DISPLAY_CRS: str = "EPSG:4326"


class CoordinateManager:
    """Manages CRS reprojections, geometry dimensionality flattening, and metric spatial queries."""

    @staticmethod
    def to_metric(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Projects GeoDataFrame into metric coordinate system (EPSG:32643).
        Preserves original metadata and ensures compatible spatial operations.
        """
        if gdf.empty:
            return gdf.copy()
        res = gdf.copy()
        if res.crs is None:
            logger.warning("Input GeoDataFrame has no CRS set; defaulting to EPSG:4326 before projecting to metric.")
            res = res.set_crs(DISPLAY_CRS)
        if res.crs.to_string().upper() != METRIC_CRS.upper():
            res = res.to_crs(METRIC_CRS)
        return res

    @staticmethod
    def to_display(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Projects GeoDataFrame into display coordinate system (EPSG:4326).
        """
        if gdf.empty:
            return gdf.copy()
        res = gdf.copy()
        if res.crs is None:
            res = res.set_crs(DISPLAY_CRS)
            return res
        if res.crs.to_string().upper() != DISPLAY_CRS.upper():
            res = res.to_crs(DISPLAY_CRS)
        return res

    @staticmethod
    def strip_z(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Converts 3D geometries (Point Z, Polygon Z, MultiPolygon Z) to canonical 2D geometries.
        Eliminates GDAL/Fiona 3D warnings and guarantees clean GeoPackage/GeoParquet compatibility.
        """
        if gdf.empty or "geometry" not in gdf.columns:
            return gdf.copy()
        res = gdf.copy()
        res["geometry"] = res["geometry"].apply(lambda geom: shapely.force_2d(geom) if geom is not None else None)
        return res

    @staticmethod
    def build_metric_kdtree(points_gdf: gpd.GeoDataFrame) -> Tuple[cKDTree, np.ndarray]:
        """
        Constructs a fast scipy cKDTree in projected metric coordinates (EPSG:32643).
        Returns the tree and coordinates array [N, 2].
        """
        metric_gdf = CoordinateManager.to_metric(CoordinateManager.strip_z(points_gdf))
        coords = np.column_stack([metric_gdf.geometry.x.values, metric_gdf.geometry.y.values])
        tree = cKDTree(coords)
        return tree, coords

    @staticmethod
    def query_nearest_distances(
        source_points_gdf: gpd.GeoDataFrame,
        target_tree: cKDTree
    ) -> np.ndarray:
        """
        Computes distance (in meters) from each source point to its nearest target neighbor.
        """
        metric_source = CoordinateManager.to_metric(CoordinateManager.strip_z(source_points_gdf))
        source_coords = np.column_stack([metric_source.geometry.x.values, metric_source.geometry.y.values])
        distances, _ = target_tree.query(source_coords, k=1)
        return distances

    @staticmethod
    def query_radius_counts(
        source_points_gdf: gpd.GeoDataFrame,
        target_tree: cKDTree,
        radius_meters: float
    ) -> np.ndarray:
        """
        Computes count of target points within radius_meters for each source point.
        """
        metric_source = CoordinateManager.to_metric(CoordinateManager.strip_z(source_points_gdf))
        source_coords = np.column_stack([metric_source.geometry.x.values, metric_source.geometry.y.values])
        counts = target_tree.query_ball_point(source_coords, r=radius_meters, return_length=True)
        return np.array(counts, dtype=int)

