"""
UrbanFlow AI - Feature Engineering Pipeline
Constructs spatial lag, transit deficit indices, demographic vulnerability scores, and stop density.
"""

from typing import Tuple
import numpy as np
import pandas as pd
import geopandas as gpd

from config.settings import settings
from utils.geo_utils import ensure_metric_crs, ensure_display_crs
from utils.logger import logger


class FeatureEngineeringPipeline:
    """Builds spatial and socio-economic feature sets for demand modeling and transit desert detection."""

    @staticmethod
    def build_ward_features(
        wards_gdf: gpd.GeoDataFrame,
        stops_gdf: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """
        Extracts ward-level multimodal transit features:
        - Total stop count by mode (rail vs bus)
        - Stop density per km²
        - Stops per 10,000 residents
        - Transit Deficit Score (Demand / Supply discrepancy)
        """
        logger.info("Computing ward spatial transit accessibility features...")
        wards = ensure_metric_crs(wards_gdf.copy())
        stops = ensure_metric_crs(stops_gdf.copy())

        # Spatial Join stops to wards
        joined = gpd.sjoin(stops, wards, how="inner", predicate="within")

        # Counts by ward
        total_stops_per_ward = joined.groupby("ward_id")["stop_id"].count().rename("total_stops")
        rail_stops_per_ward = joined[joined["mode_id"].str.contains("suburban|metro")].groupby("ward_id")["stop_id"].count().rename("rail_metro_stops")
        bus_stops_per_ward = joined[joined["mode_id"] == "best_bus"].groupby("ward_id")["stop_id"].count().rename("bus_stops")

        # Merge back to wards
        wards = wards.merge(total_stops_per_ward, on="ward_id", how="left").fillna({"total_stops": 0})
        wards = wards.merge(rail_stops_per_ward, on="ward_id", how="left").fillna({"rail_metro_stops": 0})
        wards = wards.merge(bus_stops_per_ward, on="ward_id", how="left").fillna({"bus_stops": 0})

        # Derived metrics
        wards["stop_density_per_sq_km"] = wards["total_stops"] / wards["area_sq_km"]
        wards["stops_per_10k_population"] = (wards["total_stops"] / wards["population"]) * 10000.0

        # Baseline Accessibility Score (0 to 100)
        # Scaled by rail presence (heavily weighted in Mumbai) + bus frequency + per-capita stops
        raw_access = (
            (wards["rail_metro_stops"] * 12.0) +
            (wards["bus_stops"] * 1.5) +
            (wards["stops_per_10k_population"] * 8.0)
        )
        wards["accessibility_score"] = np.clip(raw_access, 5.0, 100.0)

        # Transit Desert Index (High vulnerability + High population + Low accessibility)
        # Higher index = worse transit poverty
        normalized_pop_density = wards["population"] / wards["area_sq_km"]
        normalized_pop_density = normalized_pop_density / normalized_pop_density.max()
        
        normalized_access = 1.0 - (wards["accessibility_score"] / 100.0)
        
        wards["transit_desert_index"] = (
            0.40 * normalized_access +
            0.35 * wards["vulnerability_index"] +
            0.25 * normalized_pop_density
        )

        wards_display = ensure_display_crs(wards)
        
        # Save features to disk
        feature_path = settings.FEATURES_DATA_DIR / "mumbai_ward_features.geojson"
        wards_display.to_file(feature_path, driver="GeoJSON")
        logger.info(f"Saved engineered spatial features to {feature_path.name}")

        return wards_display

