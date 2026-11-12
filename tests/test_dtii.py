"""
Unit and Integration Tests for the Dynamic Transit Inequality Index (DTII) Engine
Validates:
1. Consistent normalization and component extraction.
2. Configurable weights and reduced 3-component fallback for missing vulnerability data.
3. Strict formula correctness and monotonicity:
       DTII = w1*D + w2*(1-A) + w3*(1-S) + w4*V
4. Component-wise contributions and dominant factor identification.
5. Sensitivity analysis across alternative policy profiles.
6. Dynamic explanation function using actual calculated values.
7. Presence of the official UrbanFlow AI proposal disclaimer.
"""

import pytest
import numpy as np
import pandas as pd

from analytics.dtii_engine import (
    DTIICalculator,
    DTIIWeights,
    DTII_DISCLAIMER
)


@pytest.fixture
def sample_mobility_df():
    """Generates synthetic mobility data for 4 distinct urban zones."""
    return pd.DataFrame([
        {
            "ward_id": "Zone_A",
            "ward_name": "Underserved Informal Settlement",
            "population": 250000,
            "relative_demand_score": 70.0,     # High Demand D=0.70
            "multimodal_accessibility_score": 15.0, # Low Access A=0.15 -> Deficit = 0.85
            "transport_supply_score": 20.0,    # Low Supply S=0.20 -> Deficit = 0.80
            "vulnerability_index": 0.45,       # High Vulnerability V=0.45
            "net_walk_dist_transit_m": 1600.0,
            "nearest_facility_name": "East Feeder Bus Stop",
            "recommended_intervention": "Rapid Transit Extension & Micro-Transit"
        },
        {
            "ward_id": "Zone_B",
            "ward_name": "Well-Serviced Central Business District",
            "population": 120000,
            "relative_demand_score": 80.0,     # High Demand D=0.80
            "multimodal_accessibility_score": 90.0, # High Access A=0.90 -> Deficit = 0.10
            "transport_supply_score": 85.0,    # High Supply S=0.85 -> Deficit = 0.15
            "vulnerability_index": 0.10,       # Low Vulnerability V=0.10
            "net_walk_dist_transit_m": 250.0,
            "nearest_facility_name": "Central Metro Station",
            "recommended_intervention": "Maintain Intermodal Hub"
        },
        {
            "ward_id": "Zone_C",
            "ward_name": "Suburban Commuter Belt",
            "population": 80000,
            "relative_demand_score": 30.0,     # Low Demand D=0.30
            "multimodal_accessibility_score": 40.0, # Moderate Access A=0.40 -> Deficit = 0.60
            "transport_supply_score": 50.0,    # Moderate Supply S=0.50 -> Deficit = 0.50
            "vulnerability_index": 0.20,       # Low Vulnerability V=0.20
            "net_walk_dist_transit_m": 850.0,
            "nearest_facility_name": "Suburban Railway Halt",
            "recommended_intervention": "Feeder Bus Optimization"
        },
        {
            "ward_id": "Zone_D",
            "ward_name": "Unpopulated Wetland / Coastal Buffer",
            "population": 0,
            "relative_demand_score": 0.0,      # Zero Demand D=0.00
            "multimodal_accessibility_score": 5.0,  # Zero Access A=0.05 -> Deficit = 0.95
            "transport_supply_score": 5.0,     # Zero Supply S=0.05 -> Deficit = 0.95
            "vulnerability_index": 0.00,       # Zero Vulnerability V=0.00
            "net_walk_dist_transit_m": 2500.0,
            "nearest_facility_name": "Remote Highway Depot",
            "recommended_intervention": "Active Mobility Trail"
        }
    ])


def test_weights_validation_and_reduced_mode():
    """Validates normalization of custom weights and reduced 3-component mode."""
    # 1. Custom non-normalized weights
    w = DTIIWeights(w_demand=3.0, w_access=3.0, w_supply=2.0, w_vulnerability=2.0)
    assert abs(w.w_demand + w.w_access + w.w_supply + w.w_vulnerability - 1.0) < 1e-5
    assert w.w_demand == 0.30

    # 2. Reduced 3-component transformation
    w_red = w.to_reduced_3_component()
    assert w_red.w_vulnerability == 0.0
    assert abs(w_red.w_demand + w_red.w_access + w_red.w_supply - 1.0) < 1e-5
    assert abs(w_red.w_demand - 0.375) < 1e-4  # 0.30 / 0.80 = 0.375


def test_dtii_formula_and_monotonicity(sample_mobility_df):
    """
    Verifies the mathematical correctness of:
        DTII = w1*D + w2*(1-A) + w3*(1-S) + w4*V
    and monotonicity: worsening conditions must strictly increase DTII.
    """
    calc = DTIICalculator(weights=DTIIWeights(0.30, 0.30, 0.20, 0.20))
    res = calc.calculate(sample_mobility_df)

    # 1. Bounded strictly in [0.0, 1.0]
    assert (res["dtii"] >= 0.0).all()
    assert (res["dtii"] <= 1.0).all()

    # 2. Manual verification for Zone A:
    # D = 0.70, 1-A = 0.85, 1-S = 0.80, V = 0.45
    # Expected = 0.30*0.70 + 0.30*0.85 + 0.20*0.80 + 0.20*0.45
    #          = 0.210 + 0.255 + 0.160 + 0.090 = 0.715
    row_a = res[res["ward_id"] == "Zone_A"].iloc[0]
    assert abs(row_a["dtii"] - 0.715) < 0.01

    # 3. Zone A (underserved informal settlement) must have substantially higher DTII than Zone B (CBD)
    row_b = res[res["ward_id"] == "Zone_B"].iloc[0]
    assert row_a["dtii"] > row_b["dtii"]
    assert row_a["dtii_category"] == "Critical Inequality"
    assert row_b["dtii_category"] == "Low Inequality" or row_b["dtii_category"] == "Moderate Inequality"


def test_missing_vulnerability_reduced_mode(sample_mobility_df):
    """Verifies that missing vulnerability data triggers reduced 3-component mode smoothly."""
    df_no_vuln = sample_mobility_df.drop(columns=["vulnerability_index"])
    calc = DTIICalculator()
    res = calc.calculate(df_no_vuln)

    assert "dtii" in res.columns
    assert res["is_reduced_dtii"].all()
    assert (res["contrib_vuln"] == 0.0).all()
    assert (res["dtii"] >= 0.0).all() and (res["dtii"] <= 1.0).all()


def test_component_contributions_exact_sum(sample_mobility_df):
    """Verifies that component contributions sum exactly to total DTII and 100%."""
    calc = DTIICalculator()
    res = calc.calculate(sample_mobility_df)

    for _, row in res.iterrows():
        # Absolute contributions sum to DTII
        c_sum = row["contrib_demand"] + row["contrib_access"] + row["contrib_supply"] + row["contrib_vuln"]
        assert abs(c_sum - row["dtii"]) < 1e-3

        # Percentage contributions sum to 100%
        if row["dtii"] > 0.01:
            pct_sum = row["pct_contrib_demand"] + row["pct_contrib_access"] + row["pct_contrib_supply"] + row["pct_contrib_vuln"]
            assert abs(pct_sum - 100.0) < 0.2

    # Check dominant factor assigned
    assert "dominant_inequality_factor" in res.columns
    assert not res["dominant_inequality_factor"].isna().any()


def test_sensitivity_analysis(sample_mobility_df):
    """Verifies sensitivity analysis across alternative policy profiles."""
    calc = DTIICalculator()
    sens = calc.run_sensitivity_analysis(sample_mobility_df)

    assert "summary_table" in sens
    assert "rank_matrix" in sens
    summary = sens["summary_table"]

    # Must evaluate all 6 pre-defined profiles
    assert len(summary) == 6
    assert "spearman_rho" in summary.columns
    assert "mean_rank_displacement" in summary.columns

    # Balanced baseline self-correlation must be 1.0
    base_row = summary[summary["profile_key"] == "balanced_baseline"].iloc[0]
    assert base_row["spearman_rho"] == 1.0000
    assert base_row["mean_rank_displacement"] == 0.0


def test_dynamic_explanation_function(sample_mobility_df):
    """Verifies that the explanation function uses actual numerical values and provides disclaimer."""
    calc = DTIICalculator()
    res = calc.calculate(sample_mobility_df)

    explanation = DTIICalculator.explain_area_inequality(res, area_id="Zone_A")

    assert explanation["area_name"] == "Underserved Informal Settlement"
    assert explanation["ward_id"] == "Zone_A"
    assert explanation["dtii"] > 0.60
    assert "dominant_factor" in explanation
    assert "components" in explanation

    narrative = explanation["narrative_explanation"]
    # Must contain actual numerical values
    assert "250,000" in narrative
    assert "1600" in narrative or "1,600" in narrative
    assert "East Feeder Bus Stop" in narrative
    # Must contain official disclaimer
    assert "not an officially gazetted government standard" in narrative
