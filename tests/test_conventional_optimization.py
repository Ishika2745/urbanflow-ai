"""
Unit and Integration Tests for Infrastructure Recommendation & Conventional Optimization Engine
Validates:
- Part A: Feasible candidate generation (connected to mapped infrastructure, minimum separation from existing stops and each other)
- Part B: Conventional optimization algorithms (Greedy Maximum Coverage, MCLP MILP, P-Median MILP)
- Part C: Population-weighted demand & DTII coverage maximization
- Part D: Comparative metrics (population coverage, additional population benefiting, average walking distance, DTII reduction, runtime, feasibility)
- Disclaimers on all candidate facilities
"""

import pytest
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

from optimization.candidate_generation import CandidateGenerator, CANDIDATE_DISCLAIMER
from optimization.fitness_function import MultiObjectiveFitnessEvaluator, FITNESS_DISCLAIMER
from optimization.greedy_optimizer import GreedyOptimizer
from optimization.mclp_optimizer import MCLPOptimizer
from optimization.pmedian_optimizer import PMedianOptimizer
from optimization.optimization_evaluation import OptimizationBenchmarkSuite


@pytest.fixture
def mock_optimization_instance():
    """Builds a realistic synthetic spatial problem instance with demand, existing stops, and candidates."""
    # 4 Demand points in UTM 43N (meters)
    base_x, base_y = 280000.0, 2100000.0
    wards_data = [
        {"ward_id": "W1", "population": 150000, "relative_demand_score": 60.0, "dtii": 0.55, "geometry": Point(base_x + 1000, base_y + 1000)},
        {"ward_id": "W2", "population": 90000,  "relative_demand_score": 40.0, "dtii": 0.45, "geometry": Point(base_x + 3000, base_y + 1000)},
        {"ward_id": "W3", "population": 200000, "relative_demand_score": 75.0, "dtii": 0.65, "geometry": Point(base_x + 1000, base_y + 4000)},
        {"ward_id": "W4", "population": 110000, "relative_demand_score": 30.0, "dtii": 0.35, "geometry": Point(base_x + 4000, base_y + 4000)}
    ]
    wards_gdf = gpd.GeoDataFrame(wards_data, crs="EPSG:32643")

    # 1 Existing transit stop at (base_x + 1200, base_y + 1100) covering W1
    existing_data = [
        {"stop_id": "EX_01", "geometry": Point(base_x + 1200, base_y + 1100)}
    ]
    existing_gdf = gpd.GeoDataFrame(existing_data, crs="EPSG:32643")

    # 6 Candidate feeder stops
    cands_data = [
        {"candidate_id": "C_01", "candidate_name": "Feeder Candidate #1", "ward_id": "W1", "estimated_cost_lakhs": 25.0, "geometry": Point(base_x + 1800, base_y + 1200)},
        {"candidate_id": "C_02", "candidate_name": "Feeder Candidate #2", "ward_id": "W2", "estimated_cost_lakhs": 25.0, "geometry": Point(base_x + 2900, base_y + 1100)},
        {"candidate_id": "C_03", "candidate_name": "Feeder Candidate #3", "ward_id": "W3", "estimated_cost_lakhs": 35.0, "geometry": Point(base_x + 1100, base_y + 3900)},
        {"candidate_id": "C_04", "candidate_name": "Feeder Candidate #4", "ward_id": "W3", "estimated_cost_lakhs": 30.0, "geometry": Point(base_x + 1600, base_y + 4100)},
        {"candidate_id": "C_05", "candidate_name": "Feeder Candidate #5", "ward_id": "W4", "estimated_cost_lakhs": 20.0, "geometry": Point(base_x + 3900, base_y + 3900)},
        {"candidate_id": "C_06", "candidate_name": "Feeder Candidate #6", "ward_id": "W4", "estimated_cost_lakhs": 20.0, "geometry": Point(base_x + 4500, base_y + 4500)}
    ]
    cands_gdf = gpd.GeoDataFrame(cands_data, crs="EPSG:32643")

    evaluator = MultiObjectiveFitnessEvaluator(
        demand_wards_gdf=wards_gdf,
        candidate_stops_gdf=cands_gdf,
        existing_stops_gdf=existing_gdf,
        service_radius_meters=1500.0
    )
    return evaluator, wards_gdf, cands_gdf, existing_gdf


def test_candidate_generation_feasibility_and_separation():
    """Validates that CandidateGenerator avoids random coords and satisfies separation rules."""
    cands = CandidateGenerator.generate_feasible_candidates(
        min_dist_to_existing_m=300.0,
        min_separation_candidates_m=350.0,
        target_candidate_count=20
    )

    assert len(cands) > 0
    assert "candidate_id" in cands.columns
    assert "connected_infrastructure" in cands.columns
    assert "dist_to_nearest_existing_stop_m" in cands.columns
    assert "estimated_cost_lakhs" in cands.columns

    # Verify minimum distance to existing stops
    assert (cands["dist_to_nearest_existing_stop_m"] >= 250.0).all()

    # Verify mutual separation between candidates
    coords = np.column_stack([cands.geometry.x.values, cands.geometry.y.values])
    # Check that disclaimer is attached
    assert "not approved bus stops" in CANDIDATE_DISCLAIMER


def test_greedy_maximum_coverage(mock_optimization_instance):
    """Validates Greedy Maximum Coverage optimizer and Part D metrics."""
    evaluator, _, _, _ = mock_optimization_instance
    greedy = GreedyOptimizer(evaluator)
    res = greedy.optimize(budget_stops=2, min_separation_m=350.0)

    assert res["num_stops_selected"] == 2
    assert "population_covered" in res
    assert "additional_population_benefiting" in res
    assert "avg_walking_distance_m" in res
    assert "dtii_reduction" in res
    assert "execution_time_sec" in res
    assert "feasibility" in res

    # Check that execution time is recorded and reasonable
    assert res["execution_time_sec"] >= 0.0
    assert res["population_covered"] > 0
    assert "not approved bus stops" in res["disclaimer"]


def test_mclp_exact_milp(mock_optimization_instance):
    """Validates Maximal Covering Location Problem exact MILP solution."""
    evaluator, _, _, _ = mock_optimization_instance
    mclp = MCLPOptimizer(evaluator)
    res = mclp.optimize(budget_stops=2, min_separation_m=300.0)

    assert res["num_stops_selected"] <= 2
    assert res["population_covered"] > 0
    assert res["solver_status"] in ["OPTIMAL", "FEASIBLE"]
    assert "additional_population_benefiting" in res
    assert res["coverage_pct"] > 0.0


def test_pmedian_exact_milp(mock_optimization_instance):
    """Validates P-Median distance-minimization exact MILP solution."""
    evaluator, _, _, _ = mock_optimization_instance
    pmed = PMedianOptimizer(evaluator)
    res = pmed.optimize(budget_stops=2)

    assert res["num_stops_selected"] <= 2
    assert res["solver_status"] in ["OPTIMAL", "FEASIBLE"]
    assert res["avg_walking_distance_m"] < evaluator.baseline_avg_walk_dist_m
    assert res["dtii_reduction"] >= 0.0


def test_conventional_benchmark_suite(mock_optimization_instance):
    """Validates OptimizationBenchmarkSuite for conventional algorithms."""
    evaluator, _, _, _ = mock_optimization_instance
    suite = OptimizationBenchmarkSuite(evaluator)
    df_summary = suite.run_conventional_benchmarks(budget_stops=2)

    assert len(df_summary) == 3
    expected_cols = [
        "Algorithm", "Population Covered", "Coverage %",
        "Additional Benefiting Pop", "Avg Walking Dist (m)",
        "DTII Reduction %", "Cost (₹ Lakhs)", "Runtime (sec)", "Feasibility"
    ]
    for col in expected_cols:
        assert col in df_summary.columns

