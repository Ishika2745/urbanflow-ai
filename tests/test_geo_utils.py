"""
Unit Tests for Geospatial Transformation Utilities
"""

import geopandas as gpd
from shapely.geometry import Point
from utils.geo_utils import (
    ensure_metric_crs,
    ensure_display_crs,
    calculate_euclidean_distance_matrix,
    create_catchment_buffers
)


def test_ensure_crs_transformations():
    """Verify reprojection between EPSG:4326 and EPSG:32643."""
    pt = Point(72.83, 18.93)
    gdf = gpd.GeoDataFrame([{"id": 1, "geometry": pt}], crs="EPSG:4326")

    # Reproject to metric
    gdf_metric = ensure_metric_crs(gdf, "EPSG:32643")
    assert gdf_metric.crs.to_string() == "EPSG:32643"

    # Reproject back to display
    gdf_disp = ensure_display_crs(gdf_metric, "EPSG:4326")
    assert gdf_disp.crs.to_string() == "EPSG:4326"


def test_distance_matrix_metric():
    """Verify distance matrix returns non-negative metric values in meters."""
    p1 = Point(72.82, 18.93)
    p2 = Point(72.84, 18.95)
    gdf_a = gpd.GeoDataFrame([{"geometry": p1}], crs="EPSG:4326")
    gdf_b = gpd.GeoDataFrame([{"geometry": p2}], crs="EPSG:4326")

    dist = calculate_euclidean_distance_matrix(gdf_a, gdf_b)
    assert dist.shape == (1, 1)
    assert dist[0, 0] > 1000.0 # Distance should be several kilometers

