"""
UrbanFlow AI - Data Access Repository Layer
Clean querying abstraction over wards, transit networks, spatial features, DTII, optimization runs, and simulation scenarios.
Designed for SQLite with transparent mapping to PostgreSQL/PostGIS.
"""

from typing import List, Dict, Any, Optional
import pandas as pd
from database.connection import db_manager
from utils.logger import logger


class TransitRepository:
    """Repository handling all UrbanFlow AI database operations."""

    @staticmethod
    def get_all_wards() -> pd.DataFrame:
        """Retrieves all city wards as a DataFrame."""
        with db_manager.get_connection() as conn:
            return pd.read_sql_query("SELECT * FROM wards ORDER BY ward_id ASC", conn)

    @staticmethod
    def get_all_stops(mode_id: Optional[str] = None) -> pd.DataFrame:
        """Retrieves transit stops, optionally filtered by mode."""
        query = "SELECT * FROM transport_stops"
        params = []
        if mode_id:
            query += " WHERE mode_id = ?"
            params.append(mode_id)
        query += " ORDER BY stop_name ASC"

        with db_manager.get_connection() as conn:
            return pd.read_sql_query(query, conn, params=params)

    @staticmethod
    def get_railway_stations() -> pd.DataFrame:
        """Retrieves suburban railway and metro stations."""
        with db_manager.get_connection() as conn:
            return pd.read_sql_query("SELECT * FROM railway_stations ORDER BY station_name ASC", conn)

    @staticmethod
    def get_dtii_results() -> pd.DataFrame:
        """Retrieves DTII inequality results."""
        with db_manager.get_connection() as conn:
            return pd.read_sql_query("SELECT * FROM dtii_results ORDER BY dtii_rank ASC", conn)

    @staticmethod
    def get_hotspot_classifications() -> pd.DataFrame:
        """Retrieves spatial hotspot and LISA results."""
        with db_manager.get_connection() as conn:
            return pd.read_sql_query("SELECT * FROM hotspot_classifications ORDER BY ward_id ASC", conn)

    @staticmethod
    def get_candidate_infrastructure() -> pd.DataFrame:
        """Retrieves feasible candidate stop locations."""
        with db_manager.get_connection() as conn:
            return pd.read_sql_query("SELECT * FROM candidate_infrastructure ORDER BY candidate_id ASC", conn)

    @staticmethod
    def save_optimization_run(run_data: Dict[str, Any]) -> None:
        """Persists an optimization run's results."""
        sql = """
        INSERT OR REPLACE INTO optimization_results (
            run_id, algorithm_name, budget_stops, population_covered,
            coverage_percentage, avg_walking_dist_m, dtii_reduction_pct,
            gini_improvement_pct, cost_lakhs, runtime_sec, fitness_score, selected_stop_ids
        ) VALUES (
            :run_id, :algorithm_name, :budget_stops, :population_covered,
            :coverage_percentage, :avg_walking_dist_m, :dtii_reduction_pct,
            :gini_improvement_pct, :cost_lakhs, :runtime_sec, :fitness_score, :selected_stop_ids
        )
        """
        with db_manager.get_connection() as conn:
            conn.execute(sql, run_data)
        logger.info(f"Persisted optimization result {run_data.get('run_id')} ({run_data.get('algorithm_name')})")

    @staticmethod
    def get_optimization_runs() -> pd.DataFrame:
        """Retrieves history of all benchmarked optimization runs."""
        with db_manager.get_connection() as conn:
            return pd.read_sql_query("SELECT * FROM optimization_results ORDER BY created_at DESC", conn)

    @staticmethod
    def save_simulation_scenario(scenario_data: Dict[str, Any]) -> None:
        """Persists a simulated policy scenario."""
        sql = """
        INSERT OR REPLACE INTO simulation_scenarios (
            scenario_id, scenario_name, algorithm, num_facilities_k, walking_radius_d0,
            subregion_ward, incremental_population_newly_served, commuters_shortened_walking_distance,
            walk_dist_saved_m, dtii_reduction_pct, gini_improvement_pct, annual_commute_hours_saved,
            economic_productivity_unlocked_cr, co2_tons_avoided_yr
        ) VALUES (
            :scenario_id, :scenario_name, :algorithm, :num_facilities_k, :walking_radius_d0,
            :subregion_ward, :incremental_population_newly_served, :commuters_shortened_walking_distance,
            :walk_dist_saved_m, :dtii_reduction_pct, :gini_improvement_pct, :annual_commute_hours_saved,
            :economic_productivity_unlocked_cr, :co2_tons_avoided_yr
        )
        """
        with db_manager.get_connection() as conn:
            conn.execute(sql, scenario_data)
        logger.info(f"Persisted simulation scenario {scenario_data.get('scenario_id')}")

    @staticmethod
    def get_simulation_scenarios() -> pd.DataFrame:
        """Retrieves all registered simulation scenarios."""
        with db_manager.get_connection() as conn:
            return pd.read_sql_query("SELECT * FROM simulation_scenarios ORDER BY created_at DESC", conn)

    @staticmethod
    def record_evaluation_metric(
        module_name: str,
        metric_name: str,
        metric_value: float,
        validation_strategy: str,
        notes: str = ""
    ) -> None:
        """Records a verified statistical/ML evaluation metric."""
        sql = """
        INSERT INTO evaluation_metrics (module_name, metric_name, metric_value, validation_strategy, notes)
        VALUES (?, ?, ?, ?, ?)
        """
        with db_manager.get_connection() as conn:
            conn.execute(sql, (module_name, metric_name, metric_value, validation_strategy, notes))

    @staticmethod
    def get_evaluation_metrics() -> pd.DataFrame:
        """Retrieves all stored model and statistical validation metrics."""
        with db_manager.get_connection() as conn:
            return pd.read_sql_query("SELECT * FROM evaluation_metrics ORDER BY metric_id ASC", conn)
