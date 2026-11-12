"""
UrbanFlow AI - Multimodal Accessibility Analytics
Implements Two-Step Floating Catchment Area (2SFCA) and Cumulative Opportunities Access.
"""

from typing import Tuple
import numpy as np
import pandas as pd
import geopandas as gpd

from config.settings import settings
from utils.geo_utils import ensure_metric_crs, calculate_euclidean_distance_matrix
from utils.logger import logger


class AccessibilityCalculator:
    """Calculates spatial accessibility metrics between population centers and transit services."""

    @staticmethod
    def calculate_2sfca(
        demand_points: gpd.GeoDataFrame,
        transit_stops: gpd.GeoDataFrame,
        threshold_distance_meters: float = 1000.0
    ) -> gpd.GeoDataFrame:
        """
        Two-Step Floating Catchment Area (2SFCA) algorithm:
        Step 1: Compute provider-to-population ratio (R_j) for each transit stop j within threshold d_0.
        Step 2: Sum R_j for each demand ward i within threshold d_0 around i.
        """
        dem_proj = ensure_metric_crs(demand_points.copy())
        stops_proj = ensure_metric_crs(transit_stops.copy())

        # Pairwise distance matrix (stops x demand points)
        dist_matrix = calculate_euclidean_distance_matrix(stops_proj, dem_proj)

        # Step 1: Stop Capacity / Demand in catchment
        # Stop capacity approx from daily ridership or vehicle capacity
        capacities = stops_proj.get("daily_ridership", pd.Series(5000, index=stops_proj.index)).values
        demand_pops = dem_proj["population"].values

        in_catchment_1 = dist_matrix <= threshold_distance_meters

        # Total population served by each stop
        served_pop_per_stop = np.sum(in_catchment_1 * demand_pops[np.newaxis, :], axis=1)
        served_pop_per_stop = np.maximum(served_pop_per_stop, 1.0) # Avoid div by zero

        r_j = capacities / served_pop_per_stop

        # Step 2: Sum R_j for each demand ward
        in_catchment_2 = in_catchment_1.T # shape: (demand_points x stops)
        access_scores = np.sum(in_catchment_2 * r_j[np.newaxis, :], axis=1)

        # Normalize score to 0 - 100 range
        max_score = np.max(access_scores) if np.max(access_scores) > 0 else 1.0
        normalized_scores = (access_scores / max_score) * 100.0

        dem_proj["access_2sfca"] = np.round(normalized_scores, 2)
        return dem_proj

    @staticmethod
    def calculate_population_coverage(
        wards_gdf: gpd.GeoDataFrame,
        stops_gdf: gpd.GeoDataFrame,
        service_radius_meters: float = 800.0
    ) -> Tuple[int, float]:
        """
        Calculates total population residing within the service radius buffer of any transit stop.
        Returns (covered_population, coverage_percentage).
        """
        wards = ensure_metric_crs(wards_gdf.copy())
        stops = ensure_metric_crs(stops_gdf.copy())

        # Buffer stops
        stop_buffers = stops.geometry.buffer(service_radius_meters).union_all()

        total_pop = wards["population"].sum()
        covered_pop = 0

        for _, ward in wards.iterrows():
            ward_geom = ward.geometry
            intersection = ward_geom.intersection(stop_buffers)
            if not intersection.is_empty:
                area_ratio = intersection.area / max(1e-6, ward_geom.area)
                covered_pop += int(ward["population"] * min(1.0, area_ratio))

        pct = (covered_pop / max(1.0, total_pop)) * 100.0
        return covered_pop, round(pct, 2)

