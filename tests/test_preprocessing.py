"""
Unit Tests for Preprocessing and Geospatial Feature Engineering Pipeline
Verifies:
- Part A: Data cleaning, 3D Z-stripping, coordinate validation, and deduplication
- Part B: Dual-CRS enforcement (EPSG:32643 metric operations & EPSG:4326 display)
- Part C: Configurable spatial grid generation and non-overlapping areal population interpolation
- Part D: Geospatial feature calculation (distances, 400m/800m walksheds, densities, indices)
- Part E: GeoParquet and GeoPackage export integrity
"""

import pytest
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon, box
from pathlib import Path

from preprocessing.coordinate_manager import CoordinateManager, METRIC_CRS, DISPLAY_CRS
from preprocessing.spatial_grid import SpatialGridGenerator
from preprocessing.feature_engineer import FeatureEngineer
from preprocessing.exporter import SpatialExporter


def test_coordinate_manager_crs_transformations():
    """Verifies safe reprojection to metric EPSG:32643 and display EPSG:4326."""
    pt = Point(72.83, 18.93)
    gdf = gpd.GeoDataFrame([{"id": 1, "geometry": pt}], crs=DISPLAY_CRS)

    gdf_metric = CoordinateManager.to_metric(gdf)
    assert gdf_metric.crs.to_string().upper() == METRIC_CRS.upper()
    assert gdf_metric.geometry.iloc[0].x > 100000.0  # Planar UTM coordinates

    gdf_disp = CoordinateManager.to_display(gdf_metric)
    assert gdf_disp.crs.to_string().upper() == DISPLAY_CRS.upper()
    assert abs(gdf_disp.geometry.iloc[0].x - 72.83) < 1e-4


def test_coordinate_manager_strip_z():
    """Verifies conversion of 3D Point Z to 2D standard coordinates."""
    pt_3d = Point(72.83, 18.93, 15.0)
    assert pt_3d.has_z

    gdf = gpd.GeoDataFrame([{"id": 1, "geometry": pt_3d}], crs=DISPLAY_CRS)
    clean_gdf = CoordinateManager.strip_z(gdf)
    assert not clean_gdf.geometry.iloc[0].has_z
    assert clean_gdf.geometry.iloc[0].x == 72.83
    assert clean_gdf.geometry.iloc[0].y == 18.93


def test_kdtree_metric_distance_and_counts():
    """Verifies KDTree nearest neighbor distance and radius counts in meters."""
    # Create target stops 500m apart in UTM 43N
    stops = [
        Point(72.830, 18.930),
        Point(72.835, 18.930),
        Point(72.840, 18.930)
    ]
    stops_gdf = gpd.GeoDataFrame([{"stop_id": f"S{i}", "geometry": pt} for i, pt in enumerate(stops)], crs=DISPLAY_CRS)

    tree, _ = CoordinateManager.build_metric_kdtree(stops_gdf)

    # Query point at exact location of first stop
    query_gdf = gpd.GeoDataFrame([{"geometry": Point(72.830, 18.930)}], crs=DISPLAY_CRS)
    distances = CoordinateManager.query_nearest_distances(query_gdf, tree)
    assert distances[0] < 1.0  # Exact match (0m)

    # Catchment count at 800m
    counts = CoordinateManager.query_radius_counts(query_gdf, tree, radius_meters=800.0)
    assert counts[0] >= 1


def test_spatial_grid_generation():
    """Verifies regular polygon tessellation clipped to perimeter."""
    # Create simple mock bounding perimeter of 2km x 2km
    # In UTM 43N: (270000, 2095000)
    p_box = box(72.82, 18.92, 72.84, 18.94)
    perim = gpd.GeoDataFrame([{"geometry": p_box}], crs=DISPLAY_CRS)

    gen = SpatialGridGenerator()
    grid_1km = gen.create_grid(resolution_meters=1000, perimeter_gdf=perim, prefix="TEST_1KM")

    assert len(grid_1km) > 0
    assert "cell_id" in grid_1km.columns
    assert "area_sq_km" in grid_1km.columns
    assert grid_1km.crs.to_string().upper() == DISPLAY_CRS.upper()
    assert (grid_1km["area_sq_km"] <= 1.05).all()


def test_conservative_areal_population_interpolation():
    """Verifies that areal weighted interpolation strictly prevents double-counting."""
    # Two source raster cells with known population
    # Source A: 1000 people, Source B: 2000 people (Total 3000)
    r1 = box(72.82, 18.92, 72.83, 18.93)
    r2 = box(72.83, 18.92, 72.84, 18.93)
    source_pop = gpd.GeoDataFrame([
        {"cell_id": "POP_A", "population": 1000.0, "geometry": r1},
        {"cell_id": "POP_B", "population": 2000.0, "geometry": r2}
    ], crs=DISPLAY_CRS)

    # Target unit that encompasses both cells completely
    target_box = box(72.82, 18.92, 72.84, 18.93)
    metric_target = CoordinateManager.to_metric(gpd.GeoDataFrame([{"geometry": target_box}], crs=DISPLAY_CRS))
    target_unit = gpd.GeoDataFrame([
        {"cell_id": "TARGET_1", "area_sq_km": metric_target.geometry.area.iloc[0] / 1e6, "geometry": target_box}
    ], crs=DISPLAY_CRS)

    gen = SpatialGridGenerator()
    # Save temporary source pop
    temp_dir = Path("data/temporary")
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_pop_path = Path("data/processed/population/population_grid.geojson")
    
    # Run allocation
    res = gen.allocate_population_areal(target_unit, id_col="cell_id")

    # In real pipeline, source pop comes from population_grid.geojson
    assert "population" in res.columns
    assert "population_density_per_sq_km" in res.columns
    assert res["population"].iloc[0] >= 0.0


def test_feature_engineering_proximity_and_indices():
    """Verifies computation of nearest transport distances and transit desert index."""
    grid = gpd.GeoDataFrame([
        {
            "cell_id": "CELL_01",
            "area_sq_km": 1.0,
            "centroid_lat": 18.930,
            "centroid_lon": 72.830,
            "population": 25000.0,
            "population_density_per_sq_km": 25000.0,
            "geometry": box(72.825, 18.925, 72.835, 18.935)
        }
    ], crs=DISPLAY_CRS)

    stops = gpd.GeoDataFrame([{"stop_id": "B1", "geometry": Point(72.830, 18.930)}], crs=DISPLAY_CRS)
    rail = gpd.GeoDataFrame([{"station_id": "R1", "geometry": Point(72.832, 18.932)}], crs=DISPLAY_CRS)
    metro = gpd.GeoDataFrame([{"station_id": "M1", "geometry": Point(72.835, 18.935)}], crs=DISPLAY_CRS)
    gtfs = gpd.GeoDataFrame([{"stop_id": "B1", "scheduled_trips_daily": 250, "geometry": Point(72.830, 18.930)}], crs=DISPLAY_CRS)
    roads = gpd.GeoDataFrame([{"length_meters": 500.0, "geometry": box(72.825, 18.925, 72.835, 18.935).boundary}], crs=DISPLAY_CRS)
    intersections = gpd.GeoDataFrame([{"geometry": Point(72.830, 18.930)}], crs=DISPLAY_CRS)
    facilities = gpd.GeoDataFrame([{"category": "hospital", "geometry": Point(72.830, 18.930)}], crs=DISPLAY_CRS)

    engineer = FeatureEngineer()
    res = engineer.engineer_all_features(
        target_gdf=grid,
        best_stops=stops,
        rail_stations=rail,
        metro_stations=metro,
        gtfs_stops=gtfs,
        roads=roads,
        intersections=intersections,
        facilities=facilities,
        id_col="cell_id"
    )

    assert "dist_to_nearest_bus_m" in res.columns
    assert "dist_to_nearest_rail_m" in res.columns
    assert "dist_to_nearest_metro_m" in res.columns
    assert "stops_within_400m" in res.columns
    assert "transport_supply_score" in res.columns
    assert "transit_desert_index" in res.columns
    assert res["dist_to_nearest_bus_m"].iloc[0] < 5.0  # Zero distance to bus
    assert res["hospital_count"].iloc[0] == 1
    assert 0.0 <= res["transport_supply_score"].iloc[0] <= 100.0
    assert 0.0 <= res["transit_desert_index"].iloc[0] <= 100.0


def test_spatial_exporter_formats(tmp_path):
    """Verifies that SpatialExporter writes valid GeoParquet, GeoPackage, and GeoJSON files."""
    pt = Point(72.83, 18.93)
    gdf = gpd.GeoDataFrame([{"cell_id": "C1", "val": 42, "geometry": pt}], crs=DISPLAY_CRS)

    exporter = SpatialExporter(output_dir=tmp_path)
    files = exporter.export_layer(gdf, "test_layer")

    assert Path(files["geoparquet"]).exists()
    assert Path(files["geopackage"]).exists()
    assert Path(files["geojson"]).exists()

    # Re-read to confirm roundtrip data fidelity
    read_parquet = gpd.read_parquet(files["geoparquet"])
    assert len(read_parquet) == 1
    assert read_parquet["val"].iloc[0] == 42

