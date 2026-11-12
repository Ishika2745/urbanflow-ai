"""
Unit Tests for Mobility Demand Intelligence Module
Verifies:
- Part A: Baseline Relative Demand Proxy calculation, weight normalization, and [0, 100] bounds
- Part B: Spatially-Aware Regressor Benchmark (GroupKFold by Ward on Random Forest, GBR, XGBoost, Baseline)
- Part C: K-Means Neighbourhood Demand Segmentation into 4 monotonic tiers
- Part D: SHAP global feature attributions and local instance explanations
- Part E: Demand visualization artifact generation
"""

import pytest
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box

from models.mobility_demand import (
    DemandProxyCalculator,
    SpatiallyAwareModelBenchmark,
    DemandSegmentationEngine,
    DemandExplainer,
    DEFAULT_DEMAND_WEIGHTS
)
from models.demand_visualizer import DemandVisualizer


@pytest.fixture
def sample_spatial_df():
    """Generates synthetic grid data mimicking Greater Mumbai feature schema."""
    np.random.seed(42)
    n = 60
    wards = ["A", "B", "C", "D", "E", "F/N"]
    records = []
    for i in range(n):
        w_id = wards[i % len(wards)]
        pop_density = np.random.uniform(5000, 60000)
        comm_density = np.random.uniform(0, 15)
        records.append({
            "cell_id": f"TEST_CELL_{i:04d}",
            "ward_id": w_id,
            "ward_name": f"Ward {w_id}",
            "population": pop_density * 0.8,
            "population_density_per_sq_km": pop_density,
            "commercial_density_per_sq_km": comm_density,
            "school_density_per_sq_km": np.random.uniform(0, 10),
            "hospital_density_per_sq_km": np.random.uniform(0, 5),
            "market_density_per_sq_km": np.random.uniform(0, 8),
            "road_intersection_density": np.random.uniform(10, 80),
            "pedestrian_network_density_km_per_sq_km": np.random.uniform(2, 25),
            "multimodal_accessibility_score": np.random.uniform(10, 80),
            "commercial_count": int(comm_density * 2),
            "geometry": box(72.82 + (i * 0.001), 18.92 + (i * 0.001), 72.83 + (i * 0.001), 18.93 + (i * 0.001))
        })
    return gpd.GeoDataFrame(records, crs="EPSG:4326")


def test_demand_proxy_calculator_bounds(sample_spatial_df):
    """Verifies relative demand proxy generates scores bounded between 0 and 100."""
    calc = DemandProxyCalculator(DEFAULT_DEMAND_WEIGHTS)
    res = calc.compute_relative_demand_proxy(sample_spatial_df)

    assert "relative_demand_score" in res.columns
    scores = res["relative_demand_score"].values
    assert (scores >= 0.0).all() and (scores <= 100.0).all()
    # High population + commercial density should have higher score than low
    max_idx = res["population_density_per_sq_km"].idxmax()
    min_idx = res["population_density_per_sq_km"].idxmin()
    assert res.loc[max_idx, "relative_demand_score"] > res.loc[min_idx, "relative_demand_score"]


def test_spatially_aware_model_benchmark(sample_spatial_df):
    """Verifies GroupKFold cross-validation benchmark across all 4 regression models."""
    calc = DemandProxyCalculator(DEFAULT_DEMAND_WEIGHTS)
    df_with_proxy = calc.compute_relative_demand_proxy(sample_spatial_df)

    benchmark = SpatiallyAwareModelBenchmark(n_splits=3, random_state=42)
    df_bench = benchmark.benchmark_models(df_with_proxy, target_col="relative_demand_score", group_col="ward_id")

    assert len(df_bench) == 4
    model_names = set(df_bench["Model"])
    assert "Baseline (Mean Predictor)" in model_names
    assert "Random Forest Regressor" in model_names
    assert "XGBoost Regressor" in model_names
    assert "Gradient Boosting Regressor" in model_names

    # Non-baseline models should outperform baseline model
    best_r2 = df_bench.iloc[0]["R-Squared"]
    baseline_mae = df_bench.loc[df_bench["Model"] == "Baseline (Mean Predictor)", "MAE"].values[0]
    assert best_r2 > 0.65
    assert df_bench.iloc[0]["MAE"] < baseline_mae


def test_demand_segmentation_tiers(sample_spatial_df):
    """Verifies K-Means neighbourhood demand segmentation into 4 distinct tiers."""
    calc = DemandProxyCalculator(DEFAULT_DEMAND_WEIGHTS)
    df_with_proxy = calc.compute_relative_demand_proxy(sample_spatial_df)

    segmenter = DemandSegmentationEngine(n_clusters=4, random_state=42)
    res = segmenter.segment_neighbourhoods(df_with_proxy, demand_col="relative_demand_score")

    assert "demand_tier" in res.columns
    assert "demand_tier_rank" in res.columns
    tiers = set(res["demand_tier"])
    assert "Very High Relative Demand" in tiers
    assert "High Relative Demand" in tiers
    assert "Moderate Relative Demand" in tiers
    assert "Low Relative Demand" in tiers

    # Verify tiers are monotonically ordered by mean demand score
    means = res.groupby("demand_tier_rank")["relative_demand_score"].mean()
    assert means.loc[1] > means.loc[2] > means.loc[3] > means.loc[4]


def test_shap_explainability(sample_spatial_df):
    """Verifies TreeSHAP global and local attribution extraction."""
    calc = DemandProxyCalculator(DEFAULT_DEMAND_WEIGHTS)
    df_with_proxy = calc.compute_relative_demand_proxy(sample_spatial_df)

    benchmark = SpatiallyAwareModelBenchmark(n_splits=3, random_state=42)
    benchmark.benchmark_models(df_with_proxy, target_col="relative_demand_score", group_col="ward_id")

    best_model = benchmark.fitted_models["Random Forest Regressor"]
    explainer = DemandExplainer(best_model, benchmark.feature_cols)

    shap_global = explainer.explain_global(df_with_proxy)
    assert len(shap_global["ranking"]) > 0
    assert "mean_abs_shap" in shap_global["ranking"][0]
    assert shap_global["ranking"][0]["mean_abs_shap"] > 0.0

    # Local instance explanation
    local_shap = explainer.explain_local_instance(df_with_proxy.iloc[0])
    assert len(local_shap) == len(benchmark.feature_cols)


def test_demand_visualizer_artifacts(sample_spatial_df, tmp_path):
    """Verifies that DemandVisualizer exports valid HTML map and chart artifacts."""
    calc = DemandProxyCalculator(DEFAULT_DEMAND_WEIGHTS)
    df_with_proxy = calc.compute_relative_demand_proxy(sample_spatial_df)
    segmenter = DemandSegmentationEngine(n_clusters=4, random_state=42)
    df_segmented = segmenter.segment_neighbourhoods(df_with_proxy)

    vis = DemandVisualizer(output_dir=tmp_path)

    # 1. Heatmap
    vis.create_demand_folium_heatmap(df_segmented, "test_demand_heatmap.html")
    assert (tmp_path / "test_demand_heatmap.html").exists()

    # 2. Ward chart
    ward_agg = df_segmented.groupby("ward_id").agg({
        "ward_name": "first",
        "relative_demand_score": "mean"
    }).reset_index()
    vis.create_ward_demand_barchart(ward_agg, "test_ward_chart.html")
    assert (tmp_path / "test_ward_chart.html").exists()

    # 3. Cluster scatter
    vis.create_demand_clusters_scatter(df_segmented, "test_scatter.html")
    assert (tmp_path / "test_scatter.html").exists()

    # 4. Unmet demand matrix
    vis.create_unmet_demand_matrix(df_segmented, "test_unmet_matrix.html")
    assert (tmp_path / "test_unmet_matrix.html").exists()
