"""
UrbanFlow AI - Modular Relational Schema Definitions
Engineered for SQLite (Initial) with clear abstraction for PostgreSQL / PostGIS migration.
Stores metadata, transit nodes, spatial grids, features, accessibility, demand, DTII,
hotspots, candidates, optimization runs, metaheuristic parameters, simulation scenarios, and evaluation metrics.
"""

from database.connection import db_manager
from utils.logger import logger


CREATE_TABLES_SQL = """
-- 1. Dataset Metadata & Provenance
CREATE TABLE IF NOT EXISTS dataset_metadata (
    dataset_key TEXT PRIMARY KEY,
    dataset_name TEXT NOT NULL,
    category TEXT NOT NULL,
    source_organization TEXT NOT NULL,
    source_url TEXT,
    reference_year TEXT,
    record_count INTEGER NOT NULL,
    crs TEXT NOT NULL,
    license TEXT,
    validation_status TEXT DEFAULT 'VALIDATED',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Administrative Wards / Zones
CREATE TABLE IF NOT EXISTS wards (
    ward_id TEXT PRIMARY KEY,
    ward_name TEXT NOT NULL,
    population INTEGER NOT NULL,
    area_sq_km REAL NOT NULL,
    centroid_lat REAL NOT NULL,
    centroid_lon REAL NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 3. Multimodal Transit Stops (Bus)
CREATE TABLE IF NOT EXISTS transport_stops (
    stop_id TEXT PRIMARY KEY,
    stop_name TEXT NOT NULL,
    mode_id TEXT NOT NULL,
    agency_name TEXT,
    ward_id TEXT,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    is_depot BOOLEAN DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(ward_id) REFERENCES wards(ward_id)
);

-- 4. Railway Stations (Suburban Rail & Metro)
CREATE TABLE IF NOT EXISTS railway_stations (
    station_id TEXT PRIMARY KEY,
    station_name TEXT NOT NULL,
    mode_id TEXT NOT NULL,
    operator TEXT,
    line TEXT,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    is_interchange BOOLEAN DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 5. Population Grids (1km / 500m Cells)
CREATE TABLE IF NOT EXISTS population_grids (
    cell_id TEXT PRIMARY KEY,
    ward_id TEXT,
    pop_allocated REAL NOT NULL,
    area_sq_km REAL NOT NULL,
    centroid_lat REAL NOT NULL,
    centroid_lon REAL NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(ward_id) REFERENCES wards(ward_id)
);

-- 6. Spatial Urban Features
CREATE TABLE IF NOT EXISTS spatial_features (
    ward_id TEXT PRIMARY KEY,
    bus_stops_count INTEGER DEFAULT 0,
    rail_stations_count INTEGER DEFAULT 0,
    metro_stations_count INTEGER DEFAULT 0,
    road_length_km REAL DEFAULT 0.0,
    intersection_density REAL DEFAULT 0.0,
    hospital_count INTEGER DEFAULT 0,
    school_count INTEGER DEFAULT 0,
    commercial_count INTEGER DEFAULT 0,
    facility_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(ward_id) REFERENCES wards(ward_id)
);

-- 7. Accessibility Scores
CREATE TABLE IF NOT EXISTS accessibility_scores (
    ward_id TEXT PRIMARY KEY,
    multimodal_accessibility_score REAL NOT NULL,
    dist_to_nearest_bus_m REAL NOT NULL,
    dist_to_nearest_rail_m REAL NOT NULL,
    dist_to_nearest_metro_m REAL,
    net_walk_dist_transit_m REAL NOT NULL,
    accessibility_grade TEXT,
    pop_within_400m_pct REAL,
    pop_within_800m_pct REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(ward_id) REFERENCES wards(ward_id)
);

-- 8. Relative Demand Scores
CREATE TABLE IF NOT EXISTS demand_scores (
    ward_id TEXT PRIMARY KEY,
    relative_demand_score REAL NOT NULL,
    mean_demand_score REAL NOT NULL,
    demand_tier TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(ward_id) REFERENCES wards(ward_id)
);

-- 9. Dynamic Transit Inequality Index (DTII) Results
CREATE TABLE IF NOT EXISTS dtii_results (
    ward_id TEXT PRIMARY KEY,
    dtii REAL NOT NULL,
    dtii_score REAL NOT NULL,
    dtii_rank INTEGER NOT NULL,
    dtii_category TEXT NOT NULL,
    norm_demand REAL NOT NULL,
    accessibility_deficit REAL NOT NULL,
    supply_deficit REAL NOT NULL,
    norm_vulnerability REAL,
    dominant_factor TEXT NOT NULL,
    recommended_intervention TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(ward_id) REFERENCES wards(ward_id)
);

-- 10. Spatial Hotspot & Autocorrelation Classifications
CREATE TABLE IF NOT EXISTS hotspot_classifications (
    ward_id TEXT PRIMARY KEY,
    lisa_quadrant TEXT NOT NULL,
    local_moran_i REAL NOT NULL,
    lisa_p_value REAL NOT NULL,
    getis_ord_z REAL NOT NULL,
    getis_ord_p REAL NOT NULL,
    gi_category TEXT NOT NULL,
    is_lisa_hotspot BOOLEAN DEFAULT 0,
    is_gi_hotspot BOOLEAN DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(ward_id) REFERENCES wards(ward_id)
);

-- 11. Candidate Infrastructure Feasibility Locations
CREATE TABLE IF NOT EXISTS candidate_infrastructure (
    candidate_id TEXT PRIMARY KEY,
    candidate_name TEXT NOT NULL,
    ward_id TEXT NOT NULL,
    connected_infrastructure TEXT NOT NULL,
    dtii_at_location REAL NOT NULL,
    relative_demand_at_location REAL NOT NULL,
    dist_to_nearest_existing_stop_m REAL NOT NULL,
    estimated_cost_lakhs REAL NOT NULL,
    is_feasibility_verified BOOLEAN DEFAULT 1,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(ward_id) REFERENCES wards(ward_id)
);

-- 12. Optimization Results & Multi-Solver Benchmarks
CREATE TABLE IF NOT EXISTS optimization_results (
    run_id TEXT PRIMARY KEY,
    algorithm_name TEXT NOT NULL,
    budget_stops INTEGER NOT NULL,
    population_covered INTEGER NOT NULL,
    coverage_percentage REAL NOT NULL,
    avg_walking_dist_m REAL NOT NULL,
    dtii_reduction_pct REAL NOT NULL,
    gini_improvement_pct REAL NOT NULL,
    cost_lakhs REAL NOT NULL,
    runtime_sec REAL NOT NULL,
    fitness_score REAL NOT NULL,
    selected_stop_ids TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 13. Metaheuristic (GA & PSO) Parameter Configurations
CREATE TABLE IF NOT EXISTS metaheuristic_configurations (
    config_id TEXT PRIMARY KEY,
    algorithm TEXT NOT NULL,
    population_size INTEGER,
    max_generations INTEGER,
    crossover_prob REAL,
    mutation_prob REAL,
    swarm_size INTEGER,
    iterations INTEGER,
    c1 REAL,
    c2 REAL,
    w_coverage REAL NOT NULL,
    w_equity REAL NOT NULL,
    w_dtii REAL NOT NULL,
    min_separation_m REAL NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 14. Policy Simulation Scenarios
CREATE TABLE IF NOT EXISTS simulation_scenarios (
    scenario_id TEXT PRIMARY KEY,
    scenario_name TEXT NOT NULL,
    algorithm TEXT NOT NULL,
    num_facilities_k INTEGER NOT NULL,
    walking_radius_d0 REAL NOT NULL,
    subregion_ward TEXT,
    incremental_population_newly_served INTEGER NOT NULL,
    commuters_shortened_walking_distance INTEGER NOT NULL,
    walk_dist_saved_m REAL NOT NULL,
    dtii_reduction_pct REAL NOT NULL,
    gini_improvement_pct REAL NOT NULL,
    annual_commute_hours_saved REAL NOT NULL,
    economic_productivity_unlocked_cr REAL NOT NULL,
    co2_tons_avoided_yr REAL NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 15. Statistical & Machine Learning Evaluation Metrics
CREATE TABLE IF NOT EXISTS evaluation_metrics (
    metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
    module_name TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value REAL NOT NULL,
    validation_strategy TEXT NOT NULL,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


def init_database() -> None:
    """Initializes all 15 relational tables and indices in SQLite."""
    with db_manager.get_connection() as conn:
        logger.info("Initializing complete UrbanFlow AI relational schema...")
        conn.executescript(CREATE_TABLES_SQL)
        logger.info("Complete relational schema initialized successfully.")
