"""
Unit Tests for Optimization Algorithms (GA, PSO, Greedy, MCLP, P-Median)
Tests Nature-Inspired Equity Optimization Engine (DN-DS-01)
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

from optimization.fitness_function import MultiObjectiveFitnessEvaluator
from optimization.greedy_optimizer import GreedyOptimizer
from optimization.mclp_optimizer import MCLPOptimizer
from optimization.pmedian_optimizer import PMedianOptimizer
from optimization.genetic_algorithm import GeneticAlgorithmOptimizer
from optimization.particle_swarm import ParticleSwarmOptimizer
from optimization.optimization_evaluation import OptimizationBenchmarkSuite


def create_mock_instance():
    """Builds a spatial instance with 4 wards and 8 candidates for optimizer verification."""
    # 4 demand wards
    wards_data = [
        {"ward_id": "W1", "population": 100000, "accessibility_score": 20.0, "relative_demand_score": 30.0, "dtii": 0.55, "geometry": Point(72.82, 18.92)},
        {"ward_id": "W2", "population": 80000,  "accessibility_score": 35.0, "relative_demand_score": 20.0, "dtii": 0.40, "geometry": Point(72.85, 18.94)},
        {"ward_id": "W3", "population": 120000, "accessibility_score": 15.0, "relative_demand_score": 45.0, "dtii": 0.65, "geometry": Point(72.88, 19.05)},
        {"ward_id": "W4", "population": 90000,  "accessibility_score": 25.0, "relative_demand_score": 25.0, "dtii": 0.48, "geometry": Point(72.90, 19.07)}
    ]
    wards_gdf = gpd.GeoDataFrame(wards_data, crs="EPSG:4326")

    # 8 candidate stops separated by ~1-2km
    cand_data = [
        {"candidate_id": f"C_{i}", "candidate_name": f"Stop #{i}", "estimated_cost_lakhs": 20.0, "geometry": Point(72.82 + i*0.012, 18.92 + i*0.015)}
        for i in range(8)
    ]
    cands_gdf = gpd.GeoDataFrame(cand_data, crs="EPSG:4326")

    evaluator = MultiObjectiveFitnessEvaluator(
        demand_wards_gdf=wards_gdf,
        candidate_stops_gdf=cands_gdf,
        service_radius_meters=3500.0,
        w_coverage=0.40,
        w_equity=0.30,
        w_dtii_reduction=0.30,
        min_separation_m=300.0
    )
    return evaluator


def test_greedy_optimizer():
    evaluator = create_mock_instance()
    greedy = GreedyOptimizer(evaluator)
    res = greedy.optimize(budget_stops=2)

    assert res["num_stops_selected"] == 2
    assert "fitness" in res
    assert res["coverage_pct"] > 0


def test_mclp_optimizer():
    evaluator = create_mock_instance()
    mclp = MCLPOptimizer(evaluator)
    res = mclp.optimize(budget_stops=2)

    assert res["num_stops_selected"] <= 2
    assert res["population_covered"] > 0
    assert "feasibility" in res


def test_pmedian_optimizer():
    evaluator = create_mock_instance()
    pmed = PMedianOptimizer(evaluator)
    res = pmed.optimize(budget_stops=2)

    assert res["num_stops_selected"] <= 2
    assert res["avg_walking_distance_m"] > 0


def test_genetic_algorithm_chromosome_uniqueness_and_separation():
    evaluator = create_mock_instance()
    ga = GeneticAlgorithmOptimizer(
        evaluator,
        population_size=15,
        generations=8,
        crossover_prob=0.8,
        mutation_prob=0.2,
        min_separation_m=300.0,
        random_seed=42
    )
    res = ga.optimize(budget_stops=3)

    assert res["num_stops_selected"] == 3
    assert len(set(res["selected_indices"])) == 3  # Unique stops guaranteed
    assert all(0 <= idx < evaluator.num_candidates for idx in res["selected_indices"])
    assert res["feasibility"]["is_feasible"] is True
    assert len(res["convergence_history"]) > 1


def test_discrete_pso_rank_key_decoding_and_separation():
    evaluator = create_mock_instance()
    pso = ParticleSwarmOptimizer(
        evaluator,
        num_particles=15,
        iterations=8,
        min_separation_m=300.0,
        random_seed=42
    )
    res = pso.optimize(budget_stops=3)

    assert res["num_stops_selected"] == 3
    assert len(set(res["selected_indices"])) == 3  # Unique stops guaranteed
    assert res["feasibility"]["is_feasible"] is True
    assert len(res["convergence_history"]) > 1


def test_equity_multi_objective_fitness_and_pareto():
    evaluator = create_mock_instance()
    
    # Evaluate 2 candidate indices
    fit, details = evaluator.evaluate([0, 3])
    assert fit >= 0.0
    assert "coverage_improvement_score" in details
    assert "equity_improvement_score" in details
    assert "dtii_reduction_score" in details

    # Pareto evaluation
    pareto_vec = evaluator.evaluate_pareto([0, 3])
    assert len(pareto_vec) == 3
    assert all(isinstance(v, float) for v in pareto_vec)

    # Weight sensitivity analysis
    sens = evaluator.evaluate_weights_sensitivity([0, 3])
    assert len(sens) == 4
    assert sens[0]["scenario"] == "Balanced (Default)"


def test_5way_benchmark_suite():
    evaluator = create_mock_instance()
    suite = OptimizationBenchmarkSuite(evaluator)
    df_summary, results = suite.run_all(budget_stops=2, seed=42, return_raw=True)

    assert len(df_summary) == 5
    assert "Genetic Algorithm (Equity-GA)" in df_summary["Algorithm"].values
    assert "Particle Swarm Optimization (Discrete PSO)" in df_summary["Algorithm"].values
    assert "Population Covered" in df_summary.columns
    assert "Runtime (sec)" in df_summary.columns


def test_multi_seed_stability_analysis():
    evaluator = create_mock_instance()
    suite = OptimizationBenchmarkSuite(evaluator)
    df_raw, df_stats = suite.run_stability_analysis(
        budget_stops=2,
        seeds=[42, 101, 777]
    )

    assert len(df_raw) == 6  # 2 algorithms * 3 seeds
    assert len(df_stats) == 2  # GA and PSO
    assert "Fitness (Mean ± Std)" in df_stats.columns
    assert "Stability (Fitness Coeff of Var)" in df_stats.columns
