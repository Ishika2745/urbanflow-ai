"""
Unit & Integration Tests for Transit Desert Detection and Spatial Hotspot Analysis
Validates:
- Part A: DTII calculation and constraint that low stops alone != transit desert
- Part B: K-Means, DBSCAN, Local Moran's I, Getis-Ord Gi*
- Part C: Multi-criteria priority ranking without double counting
- Part D: Intervention recommendations and nearest facility assignment
"""

import pytest
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon

from analytics.transit_desert_detector import TransitDesertDetector
from analytics.spatial_clustering import SpatialClusterAnalyzer


@pytest.fixture
def synthetic_spatial_grid():
    """Generates a synthetic 5x5 grid of spatial cells with varied mobility demand and transit access."""
    rows = []
    cell_id = 0
    # Coordinates in Mumbai UTM 43N (approx EPSG:32643)
    base_x, base_y = 280000.0, 2100000.0
    step = 1000.0

    for i in range(5):
        for j in range(5):
            x = base_x + i * step
            y = base_y + j * step
            poly = Polygon([
                (x, y), (x + step, y), (x + step, y + step), (x, y + step), (x, y)
            ])

            # Synthetic profiles:
            # i=0, j=0..2: Acute Transit Desert (High population, high demand, 0 stops, high walk distance)
            # i=4, j=4: Uninhabited / Nature buffer (0 population, 0 demand, 0 stops, high walk distance)
            # i=2, j=2: Well-serviced hub (High access, high stops, low walk distance)
            if i == 0 and j <= 2:
                pop = 45000.0
                pop_dense = 45000.0
                demand = 75.0
                access = 10.0
                supply = 15.0
                walk_dist = 1800.0
            elif i == 4 and j == 4:
                # Nature reserve / uninhabited: 0 stops, but 0 demand!
                pop = 0.0
                pop_dense = 0.0
                demand = 0.0
                access = 0.0
                supply = 0.0
                walk_dist = 2500.0
            elif i == 2 and j == 2:
                pop = 30000.0
                pop_dense = 30000.0
                demand = 60.0
                access = 85.0
                supply = 90.0
                walk_dist = 250.0
            else:
                pop = 15000.0 + (i + j) * 2000.0
                pop_dense = pop
                demand = 20.0 + (i * 8.0)
                access = 30.0 + (j * 8.0)
                supply = 40.0 + (j * 6.0)
                walk_dist = 600.0 + (i * 100.0)

            rows.append({
                "cell_id": f"cell_{cell_id}",
                "ward_id": f"W_{i}",
                "ward_name": f"Ward_{i}",
                "population": pop,
                "population_density_per_sq_km": pop_dense,
                "relative_demand_score": demand,
                "multimodal_accessibility_score": access,
                "transport_supply_score": supply,
                "net_walk_dist_transit_m": walk_dist,
                "geometry": poly
            })
            cell_id += 1

    return gpd.GeoDataFrame(rows, crs="EPSG:32643")


def test_dtii_computation_and_constraint(synthetic_spatial_grid):
    """
    CRITICAL CONSTRAINT: Do not classify a location as a transit desert
    solely because it has few nearby stops.
    """
    gdf = TransitDesertDetector.compute_dtii(synthetic_spatial_grid)

    assert "dtii" in gdf.columns
    assert "accessibility_deficit" in gdf.columns
    assert "supply_deficit" in gdf.columns

    # 1. Boundedness
    assert (gdf["dtii"] >= 0.0).all()
    assert (gdf["dtii"] <= 100.0).all()

    # 2. Check Acute Desert Cell (cell 0: high demand, low access)
    acute_cell = gdf[gdf["cell_id"] == "cell_0"].iloc[0]
    assert acute_cell["dtii"] > 40.0, "Acute desert must have a high DTII"

    # 3. Check Nature Reserve / Uninhabited Cell (cell 24: 0 demand, 0 stops)
    nature_cell = gdf[gdf["cell_id"] == "cell_24"].iloc[0]
    assert nature_cell["dtii"] == 0.0, "Unpopulated zero-demand area must NOT be a transit desert despite 0 stops"

    # 4. Check Well-serviced hub
    hub_cell = gdf[gdf["cell_id"] == "cell_12"].iloc[0]
    assert hub_cell["dtii"] < 15.0, "Well-serviced transit hub must have a low DTII"


def test_transit_desert_zone_identification(synthetic_spatial_grid):
    """Verifies identification of high-demand low-access, inadequate coverage, and critical zones."""
    gdf = TransitDesertDetector.identify_transit_desert_zones(synthetic_spatial_grid)

    assert "is_high_demand_low_access" in gdf.columns
    assert "is_inadequate_coverage" in gdf.columns
    assert "is_critical_inequality_zone" in gdf.columns
    assert "is_intervention_zone" in gdf.columns

    # Acute cells should be flagged as intervention zones
    assert gdf[gdf["cell_id"] == "cell_0"]["is_intervention_zone"].values[0] is True or gdf[gdf["cell_id"] == "cell_0"]["is_intervention_zone"].values[0] == 1

    # Nature reserve should NOT be flagged as high demand low access
    assert gdf[gdf["cell_id"] == "cell_24"]["is_high_demand_low_access"].values[0] is False or gdf[gdf["cell_id"] == "cell_24"]["is_high_demand_low_access"].values[0] == 0


def test_spatial_clustering_kmeans(synthetic_spatial_grid):
    """Verifies K-Means clustering assigns valid operational typologies."""
    gdf = TransitDesertDetector.compute_dtii(synthetic_spatial_grid)
    gdf = SpatialClusterAnalyzer.compute_kmeans_clustering(gdf, n_clusters=4)

    assert "kmeans_cluster" in gdf.columns
    assert "kmeans_label" in gdf.columns
    assert len(gdf["kmeans_cluster"].unique()) <= 4
    assert len(gdf["kmeans_label"].dropna()) == len(gdf)


def test_spatial_clustering_dbscan(synthetic_spatial_grid):
    """Verifies DBSCAN detects contiguous spatial clusters and handles noise."""
    gdf = TransitDesertDetector.compute_dtii(synthetic_spatial_grid)
    gdf = TransitDesertDetector.identify_transit_desert_zones(gdf)
    gdf = SpatialClusterAnalyzer.compute_dbscan_clustering(gdf, eps_meters=1500.0, min_samples=2)

    assert "dbscan_cluster" in gdf.columns
    assert "is_contiguous_cluster" in gdf.columns
    # Cells 0, 1, 2 are contiguous acute cells within 1000m of each other
    cluster_0 = gdf[gdf["cell_id"] == "cell_0"]["dbscan_cluster"].values[0]
    cluster_1 = gdf[gdf["cell_id"] == "cell_1"]["dbscan_cluster"].values[0]
    assert cluster_0 >= 0, "Contiguous acute cells should form a DBSCAN cluster"
    assert cluster_0 == cluster_1, "Adjacent acute cells should belong to the same spatial cluster"


def test_local_moran_and_getis_ord(synthetic_spatial_grid):
    """Verifies Local Moran's I (LISA) and Getis-Ord Gi* statistical computations."""
    gdf = TransitDesertDetector.compute_dtii(synthetic_spatial_grid)

    # Local Moran's I
    gdf = SpatialClusterAnalyzer.compute_local_moran(gdf, variable_col="dtii", k_neighbors=4)
    assert "lisa_quadrant" in gdf.columns
    assert "local_moran_i" in gdf.columns
    assert "lisa_p_value" in gdf.columns
    assert "is_lisa_hotspot" in gdf.columns

    # Getis-Ord Gi*
    gdf = SpatialClusterAnalyzer.compute_getis_ord_gi_star(gdf, variable_col="dtii", k_neighbors=4)
    assert "getis_ord_z" in gdf.columns
    assert "getis_ord_p" in gdf.columns
    assert "gi_category" in gdf.columns
    assert "is_gi_hotspot" in gdf.columns

    # Check that Z-scores are numerical
    assert not np.isnan(gdf["getis_ord_z"]).any()


def test_priority_ranking_and_recommendations(synthetic_spatial_grid):
    """Verifies multi-criteria priority ranking without double-counting."""
    gdf = TransitDesertDetector.compute_dtii(synthetic_spatial_grid)
    gdf = SpatialClusterAnalyzer.compute_local_moran(gdf, variable_col="dtii", k_neighbors=4)
    gdf = SpatialClusterAnalyzer.compute_getis_ord_gi_star(gdf, variable_col="dtii", k_neighbors=4)
    gdf = TransitDesertDetector.recommend_interventions(gdf)
    gdf = TransitDesertDetector.compute_priority_ranking(gdf)

    assert "priority_score" in gdf.columns
    assert "priority_rank" in gdf.columns
    assert "recommended_intervention" in gdf.columns

    # Priority rank 1 must have the maximum priority score
    rank_1 = gdf[gdf["priority_rank"] == 1].iloc[0]
    assert rank_1["priority_score"] == gdf["priority_score"].max()

    # Acute desert (high demand, low access) should outrank nature reserve (0 demand)
    acute_prio = gdf[gdf["cell_id"] == "cell_0"]["priority_score"].values[0]
    nature_prio = gdf[gdf["cell_id"] == "cell_24"]["priority_score"].values[0]
    assert acute_prio > nature_prio, "Acute desert must have significantly higher priority than uninhabited area"

