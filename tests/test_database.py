"""
UrbanFlow AI - Unit Tests for Relational Database Layer (PART A)
Tests:
- Database initialization and schema creation
- Table existence across all 15 required entities
- TransitRepository CRUD operations
- Relational integrity and modularity for PostGIS migration
"""

import sqlite3
import pytest
import pandas as pd
from database.connection import db_manager
from database.schema import init_database
from database.repository import TransitRepository
from database.populate_db import populate_all_tables


@pytest.fixture(scope="module", autouse=True)
def setup_db():
    """Initializes and populates database for testing."""
    populate_all_tables()


def test_database_tables_exist():
    """Verifies that all 15 required relational tables exist in urbanflow.db."""
    expected_tables = {
        "dataset_metadata",
        "wards",
        "transport_stops",
        "railway_stations",
        "population_grids",
        "spatial_features",
        "accessibility_scores",
        "demand_scores",
        "dtii_results",
        "hotspot_classifications",
        "candidate_infrastructure",
        "optimization_results",
        "metaheuristic_configurations",
        "simulation_scenarios",
        "evaluation_metrics"
    }

    with db_manager.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = {row[0] for row in cursor.fetchall()}

    missing = expected_tables - tables
    assert not missing, f"Missing required database tables: {missing}"


def test_repository_get_wards():
    """Verifies retrieving all 24 administrative municipal wards."""
    df_wards = TransitRepository.get_all_wards()
    assert isinstance(df_wards, pd.DataFrame)
    assert len(df_wards) == 24
    assert "ward_id" in df_wards.columns
    assert "ward_name" in df_wards.columns
    assert "population" in df_wards.columns


def test_repository_get_stops_and_rail():
    """Verifies retrieval of bus stops and railway stations."""
    df_stops = TransitRepository.get_all_stops()
    assert len(df_stops) > 5000
    assert "latitude" in df_stops.columns

    df_rail = TransitRepository.get_railway_stations()
    assert len(df_rail) > 50
    assert "station_name" in df_rail.columns


def test_repository_get_dtii_and_hotspots():
    """Verifies retrieval of DTII results and spatial hotspot records."""
    df_dtii = TransitRepository.get_dtii_results()
    assert len(df_dtii) == 24
    assert "dtii" in df_dtii.columns
    assert "dtii_rank" in df_dtii.columns

    df_hot = TransitRepository.get_hotspot_classifications()
    assert len(df_hot) == 24
    assert "lisa_quadrant" in df_hot.columns
    assert "gi_category" in df_hot.columns


def test_repository_save_and_retrieve_optimization_run():
    """Verifies saving and querying optimization benchmark run records."""
    test_run = {
        "run_id": "TEST_RUN_999",
        "algorithm_name": "Test Optimizer",
        "budget_stops": 5,
        "population_covered": 12000000,
        "coverage_percentage": 98.5,
        "avg_walking_dist_m": 210.0,
        "dtii_reduction_pct": 5.2,
        "gini_improvement_pct": 8.1,
        "cost_lakhs": 75.0,
        "runtime_sec": 0.12,
        "fitness_score": 0.885,
        "selected_stop_ids": "CAND_001,CAND_002"
    }
    TransitRepository.save_optimization_run(test_run)
    
    df_runs = TransitRepository.get_optimization_runs()
    matches = df_runs[df_runs["run_id"] == "TEST_RUN_999"]
    assert len(matches) == 1
    assert matches.iloc[0]["algorithm_name"] == "Test Optimizer"


def test_repository_evaluation_metrics():
    """Verifies recording and retrieval of verified validation metrics."""
    TransitRepository.record_evaluation_metric(
        module_name="Test_Validation",
        metric_name="Unit_Test_Metric",
        metric_value=0.999,
        validation_strategy="Automated PyTest",
        notes="Testing metric persistence"
    )

    df_metrics = TransitRepository.get_evaluation_metrics()
    assert len(df_metrics) >= 1
    test_rows = df_metrics[df_metrics["metric_name"] == "Unit_Test_Metric"]
    assert len(test_rows) >= 1
    assert pytest.approx(test_rows.iloc[0]["metric_value"], 1e-3) == 0.999

