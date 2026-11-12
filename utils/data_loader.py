"""
UrbanFlow AI - Central Data & Pipeline Cache Loader
Initializes datasets, generates synthetic data on first run, and caches objects in memory.
"""

from typing import Tuple
import pandas as pd
import geopandas as gpd
from pathlib import Path

from config.settings import settings
from config.mumbai_config import get_mumbai_config
from database.schema import init_database
from ingestion.synthetic_generator import save_synthetic_data_to_disk
from preprocessing.feature_engineering import FeatureEngineeringPipeline
from preprocessing.demand_aggregator import DemandAggregator
from analytics.spatial_clustering import SpatialClusterAnalyzer
from utils.logger import logger


def load_or_initialize_project_data() -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, pd.DataFrame]:
    """
    Checks if processed data exists on disk. If not, generates high-fidelity Mumbai datasets,
    computes spatial features, and seeds the database.
    """
    init_database()

    wards_file = settings.RAW_DATA_DIR / "mumbai_wards.geojson"
    stops_file = settings.RAW_DATA_DIR / "mumbai_transit_stops.geojson"
    candidates_file = settings.RAW_DATA_DIR / "mumbai_candidates.geojson"
    features_file = settings.FEATURES_DATA_DIR / "mumbai_ward_features.geojson"

    if not (wards_file.exists() and stops_file.exists() and candidates_file.exists()):
        logger.info("Initializing raw Mumbai spatial data from scratch...")
        wards_gdf, stops_gdf, candidates_gdf = save_synthetic_data_to_disk()
    else:
        wards_gdf = gpd.read_file(wards_file)
        stops_gdf = gpd.read_file(stops_file)
        candidates_gdf = gpd.read_file(candidates_file)

    real_features_file = settings.FEATURES_DATA_DIR / "mumbai_ward_features.geojson"
    real_stops_file = settings.FEATURES_DATA_DIR / "cleaned_best_stops.geojson"

    # Prefer real-world engineered layers if available
    if real_features_file.exists():
        logger.info("Loading real-world engineered ward features from features/...")
        wards_featured = gpd.read_file(real_features_file)
        if "is_transit_desert" not in wards_featured.columns:
            wards_featured = SpatialClusterAnalyzer.detect_transit_deserts(wards_featured)
    elif not features_file.exists():
        logger.info("Generating ward transit features and spatial clusters...")
        wards_featured = FeatureEngineeringPipeline.build_ward_features(wards_gdf, stops_gdf)
        wards_featured = SpatialClusterAnalyzer.detect_transit_deserts(wards_featured)
        wards_featured.to_file(features_file, driver="GeoJSON")
    else:
        wards_featured = gpd.read_file(features_file)

    if real_stops_file.exists():
        stops_gdf = gpd.read_file(real_stops_file)

    # OD Flows
    flows_file = settings.PROCESSED_DATA_DIR / "mumbai_od_demand_flows.parquet"
    if not flows_file.exists():
        df_flows = DemandAggregator.generate_od_demand_matrix(wards_featured)
        df_flows.to_parquet(flows_file)
    else:
        df_flows = pd.read_parquet(flows_file)

    return wards_featured, stops_gdf, candidates_gdf, df_flows

