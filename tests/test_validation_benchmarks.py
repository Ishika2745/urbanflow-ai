"""
UrbanFlow AI - Unit Tests for Analytical Validations & Baseline Comparisons
Tests PART D, PART E, and PART F requirements:
- DTII Monotonicity and Bounds [0, 1]
- Pedestrian network routing realism
- Non-overlapping population accounting
- Missing data explicit handling
- Candidate feasibility and separation constraints
- Scenario reproducibility across random seeds
- 7 Baseline comparisons (Straight-line vs Network, Fixed-buffer vs Network, Pop vs DTII, Greedy vs MCLP, Conventional vs GA/PSO, Existing vs Simulated)
- Centralized caching mechanisms
"""

import pytest
import numpy as np
import pandas as pd
import geopandas as gpd

from validation.analytical_validator import AnalyticalValidator, ValidationResult
from validation.baseline_comparisons import BaselineComparator, BaselineComparisonResult
from utils.caching import DatasetCache, ShortestPathCache, ModelArtifactCache


@pytest.fixture
def validator():
    return AnalyticalValidator()


@pytest.fixture
def comparator():
    return BaselineComparator()


# -----------------------------------------------------------------------------
# PART D: ANALYTICAL VALIDATION TESTS
# -----------------------------------------------------------------------------

def test_analytical_validation_dtii_monotonicity(validator):
    """Verifies that higher inequality drivers strictly increase DTII."""
    res = validator.validate_dtii_monotonicity()
    assert isinstance(res, ValidationResult)
    assert res.passed is True
    assert res.metric_value > 0.0


def test_analytical_validation_dtii_bounds(validator):
    """Verifies that DTII is strictly bounded within [0.0, 1.0]."""
    res = validator.validate_dtii_bounds()
    assert res.passed is True
    assert 0.0 <= res.metric_value <= 1.0


def test_analytical_validation_pedestrian_network_routing(validator):
    """Verifies pedestrian network routing realism."""
    res = validator.validate_pedestrian_network_routing()
    assert res.passed is True
    if res.metric_value is not None:
        assert res.metric_value >= 1.05


def test_analytical_validation_non_overlapping_pop(validator):
    """Verifies non-overlapping population accounting."""
    res = validator.validate_non_overlapping_population_accounting()
    assert res.passed is True


def test_analytical_validation_no_silent_zero(validator):
    """Verifies missing data triggers explicit reduced mode rather than silent zeros."""
    res = validator.validate_no_silent_zero_conversion()
    assert res.passed is True


def test_analytical_validation_candidate_feasibility(validator):
    """Verifies candidates satisfy road network connectivity and minimum clearance."""
    res = validator.validate_candidate_feasibility()
    assert res.passed is True
    assert res.metric_value >= 250.0


def test_analytical_validation_optimization_constraints(validator):
    """Verifies solver constraint satisfaction (budget K, uniqueness, separation)."""
    res = validator.validate_optimization_constraints()
    assert res.passed is True


def test_analytical_validation_scenario_reproducibility(validator):
    """Verifies scenario outputs are strictly reproducible across identical seeds."""
    res = validator.validate_scenario_reproducibility()
    assert res.passed is True


# -----------------------------------------------------------------------------
# PART F: BASELINE COMPARISON TESTS
# -----------------------------------------------------------------------------

def test_baseline_straight_line_vs_network_distance(comparator):
    """Comparison 1: Straight-line vs. walking-network distance."""
    res = comparator.compare_straight_line_vs_network_distance()
    assert isinstance(res, BaselineComparisonResult)
    assert res.comparison_id == 1
    assert res.proposed_metric > res.baseline_metric  # Network distance must be longer than straight-line


def test_baseline_fixed_buffer_vs_network_accessibility(comparator):
    """Comparison 2: Fixed-buffer vs. network accessibility."""
    res = comparator.compare_fixed_buffer_vs_network_accessibility()
    assert res.comparison_id == 2
    assert res.baseline_metric > res.proposed_metric  # Circular buffer overstates true reach


def test_baseline_population_only_vs_dtii(comparator):
    """Comparison 3: Population-only ranking vs. DTII ranking."""
    res = comparator.compare_population_only_vs_dtii_ranking()
    assert res.comparison_id == 3
    # Significant rank divergence between population alone and DTII
    assert res.proposed_metric > 3.0  # Mean rank displacement > 3


def test_baseline_greedy_vs_mclp(comparator):
    """Comparison 4: Greedy vs. MCLP Exact MILP."""
    res = comparator.compare_greedy_vs_mclp(budget_k=6)
    assert res.comparison_id == 4
    assert isinstance(res.baseline_metric, float)
    assert isinstance(res.proposed_metric, float)


def test_baseline_conventional_vs_ga(comparator):
    """Comparison 5: Conventional Optimization vs. Genetic Algorithm."""
    res = comparator.compare_conventional_vs_ga(budget_k=6)
    assert res.comparison_id == 5
    assert isinstance(res.baseline_metric, float)
    assert isinstance(res.proposed_metric, float)


def test_baseline_conventional_vs_pso(comparator):
    """Comparison 6: Conventional Optimization vs. Particle Swarm Optimization."""
    res = comparator.compare_conventional_vs_pso(budget_k=6)
    assert res.comparison_id == 6
    assert isinstance(res.baseline_metric, float)
    assert isinstance(res.proposed_metric, float)


def test_baseline_existing_vs_simulated(comparator):
    """Comparison 7: Existing Transport Baseline vs. Simulated Expansion."""
    res = comparator.compare_existing_vs_simulated_coverage(budget_k=5)
    assert res.comparison_id == 7
    assert res.proposed_metric < res.baseline_metric  # Simulated walk distance is lower (improved)


# -----------------------------------------------------------------------------
# PART B: CACHING TESTS
# -----------------------------------------------------------------------------

def test_caching_mechanisms():
    """Verifies in-memory dataset, shortest path, and model artifact caching."""
    # DatasetCache
    DatasetCache.clear()
    count = 0
    def loader():
        nonlocal count
        count += 1
        return {"data": [1, 2, 3]}

    d1 = DatasetCache.get_or_load("test_key", loader)
    d2 = DatasetCache.get_or_load("test_key", loader)
    assert d1 == d2
    assert count == 1  # Loader called only once

    # ShortestPathCache
    origins = np.array([[100.0, 100.0], [200.0, 200.0]])
    destinations = np.array([[150.0, 150.0], [250.0, 250.0]])
    mat1 = ShortestPathCache.get_distance_matrix("mat_key", origins, destinations)
    mat2 = ShortestPathCache.get_distance_matrix("mat_key", origins, destinations)
    assert np.array_equal(mat1, mat2)

    # ModelArtifactCache
    ModelArtifactCache.set_model("xgb_test", "mock_model_object")
    assert ModelArtifactCache.get_model("xgb_test") == "mock_model_object"
