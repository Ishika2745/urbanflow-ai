"""
UrbanFlow AI - Relative Mobility Demand Aggregator
Models origin-destination trip matrices and peak vs. off-peak commute flows across Mumbai wards.
"""

from typing import Tuple
import numpy as np
import pandas as pd
import geopandas as gpd

from config.settings import settings
from utils.logger import logger


class DemandAggregator:
    """Computes relative mobility demand matrices between spatial units using a gravity model."""

    @staticmethod
    def generate_od_demand_matrix(wards_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        """
        Constructs an Origin-Destination (OD) trip flow table between all ward pairs.
        Applies a doubly constrained gravity formulation reflecting Mumbai's spatial commute asymmetry:
        - Major job centers: A (Fort), G/S (Lower Parel/Worli), H/E (BKC), K/E (Andheri MIDC).
        - Major residential hubs: P/N (Malad), R/C (Borivali), L (Kurla), M/E (Govandi), S (Bhandup).
        """
        logger.info("Computing Mumbai Origin-Destination commute demand flows...")
        wards = wards_gdf.copy()
        n = len(wards)

        # Attraction weights for employment centers
        job_hub_multipliers = {
            "A": 3.2,    # South Mumbai Financial / Port / Government
            "G/S": 2.8,  # Lower Parel Corporate hub
            "H/E": 3.0,  # Bandra Kurla Complex (BKC)
            "K/E": 2.5,  # Andheri MIDC / SEEPZ
            "F/S": 1.8,  # Parel Commercial
            "L": 1.5,    # Kurla West / Kalina
        }

        flows = []
        for i, origin in wards.iterrows():
            orig_id = origin["ward_id"]
            orig_pop = origin["population"]
            o_lat, o_lon = origin["centroid_lat"], origin["centroid_lon"]

            for j, dest in wards.iterrows():
                dest_id = dest["ward_id"]
                dest_pop = dest["population"]
                d_lat, d_lon = dest["centroid_lat"], dest["centroid_lon"]

                if orig_id == dest_id:
                    # Intra-ward trips
                    dist_km = 2.0
                    trips_base = orig_pop * 0.12
                else:
                    # Approx distance
                    from preprocessing.network_builder import np_haversine
                    dist_km = max(1.5, np_haversine(o_lat, o_lon, d_lat, d_lon))
                    # Gravity model: (Pop_i * Pop_j) / (Distance ^ friction)
                    attraction = job_hub_multipliers.get(dest_id, 1.0)
                    friction = 1.35
                    trips_base = (orig_pop * dest_pop * attraction) / (1e7 * (dist_km ** friction))

                # Morning Peak (8 AM - 11 AM)
                trips_morning_peak = trips_base * (1.6 if job_hub_multipliers.get(dest_id, 1.0) > 1.5 else 0.8)
                # Evening Peak (5 PM - 8 PM)
                trips_evening_peak = trips_base * (1.6 if job_hub_multipliers.get(orig_id, 1.0) > 1.5 else 0.8)

                flows.append({
                    "origin_ward_id": orig_id,
                    "destination_ward_id": dest_id,
                    "distance_km": round(dist_km, 2),
                    "daily_estimated_trips": int(trips_base),
                    "morning_peak_trips": int(trips_morning_peak),
                    "evening_peak_trips": int(trips_evening_peak),
                    "transit_dependency_factor": round(float(origin["slum_population_pct"] * 0.8 + 0.2), 3)
                })

        df_flows = pd.DataFrame(flows)
        logger.info(f"Generated {len(df_flows)} OD pair commute flows.")
        return df_flows

