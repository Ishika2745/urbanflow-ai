"""
UrbanFlow AI - Spatial Data Cleaner
Cleans, snaps, and standardizes geospatial vector layers.
"""

from typing import Optional
import geopandas as gpd
from shapely.geometry import box
from config.settings import settings
from config.city_config import BoundingBox
from utils.logger import logger
from utils.geo_utils import ensure_display_crs


def clean_transit_stops(
    stops_gdf: gpd.GeoDataFrame,
    bbox: Optional[BoundingBox] = None,
    deduplicate_distance_meters: float = 15.0
) -> gpd.GeoDataFrame:
    """
    Filters invalid coordinates, clips stops to bounding box, and removes near-duplicate nodes.
    """
    gdf = ensure_display_crs(stops_gdf.copy())
    initial_count = len(gdf)

    # 1. Drop rows with null or invalid geometries
    gdf = gdf[gdf.geometry.is_valid & ~gdf.geometry.is_empty]

    # 2. Clip to bounding box if provided
    if bbox is not None:
        poly_bbox = box(bbox.min_lon, bbox.min_lat, bbox.max_lon, bbox.max_lat)
        gdf = gdf[gdf.geometry.within(poly_bbox)]

    # 3. Deduplicate based on stop_id or identical locations
    gdf = gdf.drop_duplicates(subset=["geometry"])

    logger.info(f"Cleaned transit stops: retained {len(gdf)} of {initial_count} records.")
    return gdf
