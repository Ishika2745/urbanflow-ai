"""
UrbanFlow AI - Unit Tests for Explainability & Intelligent Decision-Support Module
Tests Part A, Part B, Part C, and Part D requirements:
- Neighbourhood explanation completeness & exact components
- Candidate selection explanation across solvers
- Algorithm comparison trade-off auditing
- Dynamic non-hardcoded natural language text generation
- Missing information identification
- Proper SHAP restriction to supervised ML
"""

import pytest
import pandas as pd
import geopandas as gpd
from explainability.decision_support import (
    TransitDecisionSupport,
    NeighbourhoodExplainer,
    OptimizationExplainer,
    AlgorithmComparisonExplainer,
    NeighbourhoodExplanation,
    OptimizationExplanation,
    AlgorithmTradeoffExplanation,
    get_decision_support_engine
)


@pytest.fixture
def dss_engine():
    """Initializes a fresh TransitDecisionSupport engine."""
    return TransitDecisionSupport()


def test_neighbourhood_explainer_completeness(dss_engine):
    """Verifies that all Part A required metrics are present and populated."""
    # Test on a priority ward: M/E
    exp = dss_engine.explain_neighbourhood("M/E")
    
    assert isinstance(exp, NeighbourhoodExplanation)
    assert exp.ward_id in ["M/E", "M/East", "ME"]
    assert exp.population > 0
    assert 0.0 <= exp.relative_demand <= 100.0
    assert len(exp.nearest_bus_stop) > 0
    assert exp.nearest_bus_distance_m >= 0.0
    assert len(exp.nearest_rail_station) > 0
    assert exp.nearest_rail_distance_m >= 0.0
    assert exp.walking_network_distance_m >= 0.0
    assert 0.0 <= exp.accessibility_score <= 100.0
    assert 0.0 <= exp.transport_supply_score <= 100.0
    assert 0.0 <= exp.dtii_score <= 1.0
    assert 1 <= exp.dtii_rank <= 24
    assert len(exp.hotspot_status) > 0
    
    # Check main inequality contributors
    contributors = exp.main_inequality_contributors
    assert "Demand (D)" in contributors
    assert "Accessibility Deficit (1-A)" in contributors
    assert "Supply Deficit (1-S)" in contributors
    assert "Vulnerability (V)" in contributors
    
    # Verify sum of contributions equals ~100%
    total_pct = sum(contributors.values())
    assert pytest.approx(total_pct, rel=0.05) == 100.0


def test_nearest_metro_handling_and_missing_info(dss_engine):
    """Verifies metro availability check and missing information logging."""
    # Test on ward M/E (Chembur/Govandi periphery where operational metro is > 5 km away)
    exp = dss_engine.explain_neighbourhood("M/E")
    
    # Should identify that metro is beyond walking catchment
    if exp.nearest_metro_distance_m and exp.nearest_metro_distance_m > 3000.0:
        assert exp.metro_available_within_catchment is False
        assert any("metro" in note.lower() for note in exp.missing_information_notes)


def test_optimization_explainer_fields_and_solvers(dss_engine):
    """Verifies Part B candidate selection explanation across multiple solvers."""
    # Test with Genetic Algorithm
    exp_ga = dss_engine.explain_candidate_selection("CAND_001", "Genetic Algorithm (Equity-GA)")
    assert isinstance(exp_ga, OptimizationExplanation)
    assert exp_ga.candidate_id in ["CAND_001", "CAND_0001"]
    assert "Genetic Algorithm" in exp_ga.selection_rationale or "Pareto" in exp_ga.selection_rationale
    assert len(exp_ga.benefiting_neighbourhoods) >= 1
    assert exp_ga.population_covered > 0
    assert exp_ga.distance_improvement_m > 0
    assert exp_ga.dtii_reduction_pts > 0
    assert exp_ga.equity_improvement_pct > 0
    assert exp_ga.algorithm_name == "Genetic Algorithm (Equity-GA)"

    # Test with MCLP
    exp_mclp = dss_engine.explain_candidate_selection("CAND_001", "MCLP (OR-Tools Exact MILP)")
    assert "Maximal Covering" in exp_mclp.selection_rationale or "MILP" in exp_mclp.selection_rationale

    # Test with P-Median
    exp_pmed = dss_engine.explain_candidate_selection("CAND_001", "P-Median (OR-Tools Exact MILP)")
    assert "P-Median" in exp_pmed.selection_rationale or "walking distance" in exp_pmed.selection_rationale

    # Test with Greedy
    exp_greedy = dss_engine.explain_candidate_selection("CAND_001", "Greedy Maximum Coverage")
    assert "greedily" in exp_greedy.selection_rationale.lower() or "marginal" in exp_greedy.selection_rationale.lower()


def test_algorithm_comparison_tradeoffs(dss_engine):
    """Verifies Part C trade-off explanation across Coverage, Equity, Runtime, Stability, and Feasibility."""
    tradeoff = dss_engine.explain_algorithm_tradeoffs()
    assert isinstance(tradeoff, AlgorithmTradeoffExplanation)
    
    # Must compare all primary algorithms
    algo_names = [a.lower() for a in tradeoff.algorithms_compared]
    assert any("mclp" in a for a in algo_names)
    assert any("p-median" in a for a in algo_names)
    assert any("genetic" in a for a in algo_names)
    assert any("pso" in a for a in algo_names)
    assert any("greedy" in a for a in algo_names)

    # Check that all 5 trade-off dimensions are covered
    for algo, dim_dict in tradeoff.tradeoffs.items():
        assert "Coverage" in dim_dict
        assert "Equity" in dim_dict
        assert "Runtime" in dim_dict
        assert "Stability" in dim_dict
        assert "Feasibility" in dim_dict

    assert len(tradeoff.why_configurations_differ) > 50
    assert len(tradeoff.planner_guidance) > 50


def test_dynamic_narrative_different_across_wards(dss_engine):
    """Verifies Part D dynamic natural-language generation: text must NOT be identical across wards."""
    exp_me = dss_engine.explain_neighbourhood("M/E")
    exp_a = dss_engine.explain_neighbourhood("A")  # Ward A is South Mumbai (Colaba/Fort), very low DTII
    
    assert exp_me.dynamic_explanation_text != exp_a.dynamic_explanation_text
    
    # Ward M/E must mention its high rank or priority
    assert "M/E" in exp_me.dynamic_explanation_text
    assert str(exp_me.population) in exp_me.dynamic_explanation_text or f"{exp_me.population:,}" in exp_me.dynamic_explanation_text
    
    # Ward A must have different numerical metrics and context
    assert "Ward A" in exp_a.dynamic_explanation_text or "A (A)" in exp_a.dynamic_explanation_text


def test_shap_restricted_to_supervised_models(dss_engine):
    """Verifies that SHAP is strictly reserved for supervised models and has explicit disclaimer."""
    shap_res = dss_engine.get_shap_demand_explanation()
    assert "feature_attributions" in shap_res
    assert len(shap_res["feature_attributions"]) > 0
    assert "Supervised" in shap_res["model_type"]
    
    # Check methodological notice
    disclaimer = shap_res["methodology_disclaimer"]
    assert "NOT applied as an artificial explanation for rule-based indices" in disclaimer


def test_executive_briefing_generation(dss_engine):
    """Verifies that complete executive memo generates properly in markdown."""
    briefing = dss_engine.generate_executive_briefing(ward_id="M/E", candidate_id="CAND_001")
    assert len(briefing) > 200
    assert "Municipal Decision Briefing" in briefing
    assert "Demographic Baseline" in briefing
    assert "Multimodal Transit Accessibility Audit" in briefing
    assert "Inequality Drivers" in briefing
    assert "Recommended Infrastructure Intervention" in briefing
    assert "Targeted Candidate Location Assessment" in briefing

