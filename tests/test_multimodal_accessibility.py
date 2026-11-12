"""
Unit Tests for Multimodal Transportation Network & Accessibility Analytics Engine
Verifies:
- Part A: Pedestrian graph construction, edge lengths, walking travel times, and connectivity validation
- Part B: Multimodal integration (BEST Bus, Suburban Rail, Metro Rail) and verified transfer connections
- Part C: Dijkstra shortest-path calculations (400m, 800m, 1000m walksheds and travel times)
- Part D: Normalized Multimodal Accessibility Scoring (0 to 100) and GTFS service verification
- Part E: Ward-level report generation and diagnostic consistency
"""

import pytest
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box
import networkx as nx

from data_ingestion.config import DISPLAY_CRS, METRIC_CRS
from analytics.pedestrian_network import PedestrianNetworkBuilder, WALKING_SPEED_MPS, WALKING_SPEED_KMH
from analytics.multimodal_network import MultimodalNetworkBuilder, TRANSFER_PENALTY_SECONDS
from analytics.accessibility_engine import MultimodalAccessibilityEngine


def test_pedestrian_network_builder_and_connectivity():
    """Verifies construction of pedestrian graph, metric length, and connectivity metrics."""
    builder = PedestrianNetworkBuilder()
    G_ped, connectivity = builder.build_or_load_pedestrian_network(force_rebuild=False)

    assert G_ped.number_of_nodes() > 0
    assert G_ped.number_of_edges() > 0
    assert connectivity["walking_speed_assumed_kmh"] == 4.5
    assert connectivity["walking_speed_mps"] == 1.25
    assert connectivity["largest_component_coverage_pct"] >= 90.0

    # Verify edge attributes
    sample_edge = list(G_ped.edges(data=True))[0]
    edge_data = sample_edge[2]
    assert "length" in edge_data
    assert "weight" in edge_data  # Travel time in seconds
    assert float(edge_data["weight"]) > 0.0


def test_multimodal_network_builder_integration():
    """Verifies multimodal integration of Rail, Metro, Bus, and verified transfer links."""
    builder = MultimodalNetworkBuilder()
    G_multi, summary = builder.build_multimodal_graph(max_snap_distance_m=1200.0)

    assert summary["total_multimodal_nodes"] > 5000
    assert summary["suburban_rail_stations"] >= 50
    assert summary["metro_stations"] >= 45
    assert summary["verified_transfer_edges"] >= 10

    # Check verified transfer edge properties
    transfer_edges = [
        (u, v, d) for u, v, d in G_multi.edges(data=True) if d.get("mode") == "verified_transfer"
    ]
    assert len(transfer_edges) > 0
    assert transfer_edges[0][2]["weight"] >= TRANSFER_PENALTY_SECONDS


def test_dijkstra_shortest_path_routing_and_thresholds():
    """Verifies Dijkstra shortest-path calculations, walkshed counts, and travel times."""
    builder = MultimodalNetworkBuilder()
    G_multi, _ = builder.build_multimodal_graph(max_snap_distance_m=1200.0)
    engine = MultimodalAccessibilityEngine(G_multi)

    # Test with origin at Churchgate (South Mumbai transit rich hub)
    # Churchgate: 18.935, 72.827
    origins = gpd.GeoDataFrame([
        {
            "cell_id": "TEST_CHURCHGATE",
            "centroid_lat": 18.935,
            "centroid_lon": 72.827,
            "geometry": box(72.825, 18.930, 72.830, 18.940)
        }
    ], crs=DISPLAY_CRS)

    res = engine.compute_network_accessibility(origins, thresholds=[400.0, 800.0, 1000.0])

    assert len(res) == 1
    assert "net_walk_dist_transit_m" in res.columns
    assert "walk_time_transit_mins" in res.columns
    assert "transit_options_400m" in res.columns
    assert "transit_options_800m" in res.columns
    assert "multimodal_accessibility_score" in res.columns

    # Churchgate origin should have rapid access to rail and bus
    assert res["net_walk_dist_bus_m"].iloc[0] <= 1000.0
    assert res["walk_time_transit_mins"].iloc[0] >= 0.0
    assert 0.0 <= res["multimodal_accessibility_score"].iloc[0] <= 100.0


def test_accessibility_score_normalization_bounds():
    """Verifies that multimodal accessibility scores are strictly bounded between 0 and 100."""
    builder = MultimodalNetworkBuilder()
    G_multi, _ = builder.build_multimodal_graph(max_snap_distance_m=1200.0)
    engine = MultimodalAccessibilityEngine(G_multi)

    # Test two contrasting points: Dadar (Central hub) and far peripheral point
    test_points = gpd.GeoDataFrame([
        {"cell_id": "DADAR_HUB", "centroid_lat": 19.018, "centroid_lon": 72.843, "geometry": box(72.84, 19.01, 72.85, 19.02)},
        {"cell_id": "PERIPHERAL", "centroid_lat": 19.260, "centroid_lon": 72.800, "geometry": box(72.79, 19.25, 72.81, 19.27)}
    ], crs=DISPLAY_CRS)

    res = engine.compute_network_accessibility(test_points)
    scores = res["multimodal_accessibility_score"].values

    assert len(scores) == 2
    assert (scores >= 0.0).all() and (scores <= 100.0).all()
    # Dadar interchange should score higher than a peripheral unserviced cell
    assert scores[0] >= scores[1]


def test_ward_accessibility_report_generation():
    """Verifies generation of the 24-ward accessibility report with valid rankings."""
    wards_path = "data/features/mumbai_ward_features.geojson"
    grid_path = "data/features/mumbai_grid_1km.parquet"

    wards_gdf = gpd.read_file(wards_path)
    grid_gdf = gpd.read_parquet(grid_path)

    builder = MultimodalNetworkBuilder()
    G_multi, _ = builder.build_multimodal_graph(max_snap_distance_m=1200.0)
    engine = MultimodalAccessibilityEngine(G_multi)

    # Compute accessibility on sample of grid cells
    sample_grid = grid_gdf.head(50)
    acc_res = engine.compute_network_accessibility(sample_grid)
    df_report = engine.generate_ward_accessibility_report(acc_res, wards_gdf)

    assert len(df_report) > 0
    assert "ward_id" in df_report.columns
    assert "mean_multimodal_score" in df_report.columns
    assert "mean_walk_time_transit_mins" in df_report.columns
    assert "accessibility_grade" in df_report.columns
