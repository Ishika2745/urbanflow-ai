"""
UrbanFlow AI - Validation Utilities
Validates transit inputs, spatial coordinates, and tabular schema structures.
"""

from typing import List, Optional
import pandas as pd
import geopandas as gpd
from utils.exceptions import SpatialDataError


def validate_dataframe_columns(df: pd.DataFrame, required_columns: List[str], dataset_name: str = "Dataset") -> None:
    """Verifies that all required columns exist in the DataFrame."""
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"{dataset_name} is missing required columns: {missing}. Available: {list(df.columns)}")


def validate_coordinates(
    df: pd.DataFrame,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    min_lat: float = -90.0,
    max_lat: float = 90.0,
    min_lon: float = -180.0,
    max_lon: float = 180.0
) -> None:
    """Validates that latitude and longitude columns fall within valid geographical ranges."""
    validate_dataframe_columns(df, [lat_col, lon_col], "Coordinate DataFrame")

    invalid_lat = df[(df[lat_col] < min_lat) | (df[lat_col] > max_lat)]
    if not invalid_lat.empty:
        raise SpatialDataError(f"Found {len(invalid_lat)} rows with latitude outside [{min_lat}, {max_lat}]")

    invalid_lon = df[(df[lon_col] < min_lon) | (df[lon_col] > max_lon)]
    if not invalid_lon.empty:
        raise SpatialDataError(f"Found {len(invalid_lon)} rows with longitude outside [{min_lon}, {max_lon}]")


def validate_geodataframe_crs(gdf: gpd.GeoDataFrame) -> None:
    """Checks that GeoDataFrame has a valid CRS assigned."""
    if gdf.crs is None:
        raise SpatialDataError("GeoDataFrame does not have a defined Coordinate Reference System (CRS).")

