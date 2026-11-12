"""
Unit Tests for Before-and-After Impact Simulator (UrbanFlow AI)
Validates Dynamic Simulation, Immutability, Non-Overlapping Accounting, and Scenario Management.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon

from simulation.transit_simulator import TransitScenarioSimulator, SIMULATOR_DISCLAIMER
from simulation.scenario_manager import ScenarioManager
from simulation.impact_analyzer import SocioeconomicImpactAnalyzer


def create_mock_simulation_world():
    """Builds a miniature spatial world with 3 wards, 4 existing stops, and 3 candidates."""
    # 3 Wards
    w1_poly = Polygon([(72.80, 18.90), (72.85, 18.90), (72.85, 18.95), (72.80, 18.95)])
    w2_poly = Polygon([(72.85, 18.90), (72.90, 18.90), (72.90, 18.95), (72.85, 18.95)])
    w3_poly = Polygon([(72.80, 18.95), (72.85, 18.95), (72.85, 19.00), (72.80, 19.00)])

    wards_data = [
        {"ward_id": "W1", "ward_name": "Ward 1", "population": 100000, "accessibility_score": 25.0, "relative_demand_score": 30.0, "transport_supply_score": 20.0, "dtii": 0.50, "dist_to_nearest_transit_m": 750.0, "geometry": w1_poly},
        {"ward_id": "W2", "ward_name": "Ward 2", "population": 80000,  "accessibility_score": 40.0, "relative_demand_score": 20.0, "transport_supply_score": 35.0, "dtii": 0.35, "dist_to_nearest_transit_m": 400.0, "geometry": w2_poly},
        {"ward_id": "W3", "ward_name": "Ward 3", "population": 120000, "accessibility_score": 15.0, "relative_demand_score": 45.0, "transport_supply_score": 15.0, "dtii": 0.65, "dist_to_nearest_transit_m": 950.0, "geometry": w3_poly}
    ]
    wards_gdf = gpd.GeoDataFrame(wards_data, crs="EPSG:4326")

    # 4 Existing Stops
    stops_data = [
        {"stop_id": f"EX_{i}", "geometry": Point(72.82 + i*0.02, 18.92 + i*0.02)}
        for i in range(4)
    ]
    stops_gdf = gpd.GeoDataFrame(stops_data, crs="EPSG:4326")

    # 3 Proposed Candidates
    cand_data = [
        {"candidate_id": "C_01", "ward_id": "W1", "candidate_name": "Feeder 1", "estimated_cost_lakhs": 25.0, "geometry": Point(72.82, 18.92)},
        {"candidate_id": "C_02", "ward_id": "W3", "candidate_name": "Feeder 2", "estimated_cost_lakhs": 30.0, "geometry": Point(72.82, 18.97)},
        {"candidate_id": "C_03", "ward_id": "W3", "candidate_name": "Feeder 3", "estimated_cost_lakhs": 20.0, "geometry": Point(72.84, 18.98)}
    ]
    cands_gdf = gpd.GeoDataFrame(cand_data, crs="EPSG:4326")

    return wards_gdf, stops_gdf, cands_gdf


def test_dataset_immutability():
    """Ensures input datasets are not modified by the simulator."""
    wards, stops, cands = create_mock_simulation_world()
    orig_wards_cols = list(wards.columns)
    orig_stops_len = len(stops)

    simulator = TransitScenarioSimulator(wards, stops, cands)
    res = simulator.simulate_scenario(cands.iloc[:2], scenario_name="Test Immutability")

    # Check original wards unchanged
    assert list(wards.columns) == orig_wards_cols
    assert "simulated_accessibility" not in wards.columns
    assert len(stops) == orig_stops_len


def test_dynamic_recalculation_without_hardcoding():
    """Validates that accessibility, walk distance, and DTII are recalculated dynamically."""
    wards, stops, cands = create_mock_simulation_world()
    simulator = TransitScenarioSimulator(wards, stops, cands)

    res = simulator.simulate_scenario(cands, service_radius_meters=800.0)

    # 1. Walking distance must improve (decrease)
    walk = res["walking_distance"]
    assert walk["simulated_avg_walk_dist_m"] < walk["baseline_avg_walk_dist_m"]
    assert walk["walk_distance_saved_m"] > 0

    # 2. DTII must improve (decrease)
    dtii = res["dtii"]
    assert dtii["simulated_mean_dtii"] < dtii["baseline_mean_dtii"]
    assert dtii["dtii_reduction"] > 0

    # 3. Gini inequality must improve (decrease)
    eq = res["equity"]
    assert eq["simulated_gini"] <= eq["baseline_gini"]

    # 4. Beneficiary wards must have increased accessibility score
    sim_wards = res["simulated_wards_gdf"]
    assert (sim_wards["accessibility_delta"] >= 0).all()
    assert (sim_wards["accessibility_delta"] > 0).any()


def test_non_overlapping_population_accounting():
    """Verifies that newly covered and shortened-walk populations are strictly partitioned."""
    wards, stops, cands = create_mock_simulation_world()
    simulator = TransitScenarioSimulator(wards, stops, cands)

    res = simulator.simulate_scenario(cands, service_radius_meters=800.0)
    cov = res["coverage"]

    inc_pop = cov["incremental_population_newly_served"]
    short_pop = cov["commuters_shortened_walking_distance"]
    tot_benefiting = cov["total_population_benefiting"]

    # Total benefiting must equal sum of non-overlapping components
    assert tot_benefiting == inc_pop + short_pop
    assert tot_benefiting <= simulator.total_population


def test_subregion_filtering():
    """Verifies sub-region focus on a specific municipal ward."""
    wards, stops, cands = create_mock_simulation_world()
    simulator = TransitScenarioSimulator(wards, stops, cands)

    res = simulator.simulate_scenario(cands, ward_filter="W3")
    sub = res["subregion_summary"]

    assert sub is not None
    assert sub["ward_id"] == "W3"
    assert sub["new_stops_count"] >= 1
    assert sub["dtii_reduction"] > 0


def test_scenario_manager_crud_and_comparison():
    """Validates saving, comparing, deleting, and resetting simulation scenarios."""
    wards, stops, cands = create_mock_simulation_world()
    simulator = TransitScenarioSimulator(wards, stops, cands)
    manager = ScenarioManager()

    # Run and save scenario 1
    res1 = simulator.simulate_scenario(cands.iloc[:1], scenario_name="Scenario A: 1 Stop")
    id1 = manager.save_scenario(res1, algorithm_used="Greedy")

    # Run and save scenario 2
    res2 = simulator.simulate_scenario(cands, scenario_name="Scenario B: 3 Stops")
    id2 = manager.save_scenario(res2, algorithm_used="Equity-GA")

    # Check comparison dataframe
    df_comp = manager.get_comparison_dataframe()
    assert len(df_comp) == 2
    assert "Scenario Label" in df_comp.columns
    assert "Coverage %" in df_comp.columns

    # Delete scenario 1
    deleted = manager.delete_scenario(id1)
    assert deleted is True
    assert len(manager.get_all_scenarios()) == 1

    # Reset all
    manager.reset_all()
    assert len(manager.get_all_scenarios()) == 0


def test_socioeconomic_roi_calculation():
    """Validates calculation of commuter hours saved, economic ROI, and CO2 offset."""
    roi = SocioeconomicImpactAnalyzer.compute_policy_roi(
        population_gain=50000,
        gini_reduction=0.03,
        capital_cost_lakhs=75.0,
        benefiting_population=150000,
        avg_walk_dist_saved_m=35.0
    )

    assert roi["annual_hours_saved"] > 0
    assert roi["annual_economic_benefit_crores"] > 0
    assert roi["co2_metric_tons_avoided"] > 0
    assert roi["capital_cost_crores"] == 0.75
    assert roi["payback_period_years"] > 0

