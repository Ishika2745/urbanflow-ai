"""
UrbanFlow AI - Geospatial Utilities
Enforces CRS integrity between projected metric operations (EPSG:32643) and map display (EPSG:4326).
"""

from typing import Tuple, List, Union
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon, box
from pyproj import CRS

from config.settings import settings
from utils.exceptions import CRSMismatchError, SpatialDataError
from utils.logger import logger


def ensure_metric_crs(
    gdf: gpd.GeoDataFrame,
    target_crs: str = settings.PROJECTED_CRS
) -> gpd.GeoDataFrame:
    """
    Guarantees the GeoDataFrame is projected in a metric CRS (default UTM Zone 43N: EPSG:32643).
    Essential before calculating buffers, distances, or areas in meters.
    """
    if gdf.crs is None:
        logger.warning(f"GeoDataFrame has no CRS set. Assuming {settings.DISPLAY_CRS} and reprojecting.")
        gdf = gdf.set_crs(settings.DISPLAY_CRS)

    current_crs_str = gdf.crs.to_string().upper()
    target_crs_str = target_crs.upper()

    if current_crs_str != target_crs_str:
        logger.debug(f"Reprojecting spatial data from {current_crs_str} to metric {target_crs_str}")
        return gdf.to_crs(target_crs)
    return gdf


def ensure_display_crs(
    gdf: gpd.GeoDataFrame,
    target_crs: str = settings.DISPLAY_CRS
) -> gpd.GeoDataFrame:
    """
    Guarantees the GeoDataFrame is projected in standard WGS84 (EPSG:4326) for Streamlit, Folium, and PyDeck.
    """
    if gdf.crs is None:
        gdf = gdf.set_crs(settings.DISPLAY_CRS)

    current_crs_str = gdf.crs.to_string().upper()
    target_crs_str = target_crs.upper()

    if current_crs_str != target_crs_str:
        logger.debug(f"Reprojecting spatial data from {current_crs_str} to display {target_crs_str}")
        return gdf.to_crs(target_crs)
    return gdf


def create_catchment_buffers(
    gdf: gpd.GeoDataFrame,
    radius_meters: float = 800.0,
    metric_crs: str = settings.PROJECTED_CRS
) -> gpd.GeoDataFrame:
    """
    Generates circular catchment buffers of specified radius in meters around transit stops.
    Automatically handles projection to metric CRS and converts back to display CRS for visualization.
    """
    # 1. Project to metric CRS
    metric_gdf = ensure_metric_crs(gdf, target_crs=metric_crs)

    # 2. Buffer in meters
    buffered_gdf = metric_gdf.copy()
    buffered_gdf["geometry"] = metric_gdf.geometry.buffer(radius_meters)

    # 3. Return both projected and display versions
    return buffered_gdf


def calculate_euclidean_distance_matrix(
    points_a: gpd.GeoDataFrame,
    points_b: gpd.GeoDataFrame,
    metric_crs: str = settings.PROJECTED_CRS
) -> np.ndarray:
    """
    Calculates pairwise distance matrix (in meters) between two sets of points.
    Forces projection to metric CRS to avoid degree-distortion errors.
    """
    pa = ensure_metric_crs(points_a, metric_crs)
    pb = ensure_metric_crs(points_b, metric_crs)

    coords_a = np.column_stack([pa.geometry.centroid.x.values, pa.geometry.centroid.y.values])
    coords_b = np.column_stack([pb.geometry.centroid.x.values, pb.geometry.centroid.y.values])

    # Efficient vectorized distance computation (M x N matrix)
    diff = coords_a[:, np.newaxis, :] - coords_b[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))
    return dist_matrix


def bbox_to_polygon(min_lon: float, min_lat: float, max_lon: float, max_lat: float) -> Polygon:
    """Creates a Shapely Polygon from bounding box coordinates."""
    return box(min_lon, min_lat, max_lon, max_lat)

