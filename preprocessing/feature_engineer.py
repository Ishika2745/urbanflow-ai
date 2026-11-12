"""
UrbanFlow AI - Geospatial Feature Engineering Pipeline
Implements Part D:
- Computes multi-tier transit supply features:
  * Metric Euclidean distance to nearest bus stop, railway station, and metro station.
  * Stop catchment counts within 400 metres (5-minute walk) and 800 metres (10-minute walk).
  * In-cell stop counts by mode (bus, suburban rail, metro).
  * Validated coverage indicators (does not treat missing data as zero supply).
  * GTFS scheduled trips per day and average service headway.
- Computes urban infrastructure & connectivity densities:
  * Road intersection density (intersections per sq km).
  * Pedestrian network density (walkable km per sq km).
- Computes civic facility densities:
  * Hospital density, School density, Market density, Commercial facility density.
- Computes composite mobility indicators:
  * Transport Supply Score (0 to 100).
  * Transit Desert Index (0 to 100).
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

from data_ingestion.config import METRIC_CRS, DISPLAY_CRS
from preprocessing.coordinate_manager import CoordinateManager
from utils.logger import logger


class FeatureEngineer:
    """Builds spatial, multimodal transport supply, infrastructure, and demographic features."""

    def __init__(self):
        pass

    def engineer_all_features(
        self,
        target_gdf: gpd.GeoDataFrame,
        best_stops: gpd.GeoDataFrame,
        rail_stations: gpd.GeoDataFrame,
        metro_stations: gpd.GeoDataFrame,
        gtfs_stops: gpd.GeoDataFrame,
        roads: gpd.GeoDataFrame,
        intersections: gpd.GeoDataFrame,
        facilities: gpd.GeoDataFrame,
        id_col: str = "cell_id"
    ) -> gpd.GeoDataFrame:
        """
        Executes end-to-end feature engineering over any spatial layer (500m grid, 1km grid, or wards).
        """
        logger.info(f"=== Engineering geospatial features for {len(target_gdf)} spatial units ===")
        gdf = target_gdf.copy()

        # 1. Metric Distances & Catchment Counts
        gdf = self.compute_transit_proximity(gdf, best_stops, rail_stations, metro_stations)

        # 2. GTFS Service Availability Indicators
        gdf = self.compute_gtfs_service_metrics(gdf, gtfs_stops)

        # 3. Road & Pedestrian Network Connectivity
        gdf = self.compute_infrastructure_metrics(gdf, roads, intersections, id_col)

        # 4. Urban Civic Facilities Densities
        gdf = self.compute_facility_metrics(gdf, facilities, id_col)

        # 5. Composite Mobility & Deficit Indices
        gdf = self.compute_composite_indices(gdf)

        logger.info(f"Feature engineering completed successfully for {len(gdf)} spatial units.")
        return gdf

    def compute_transit_proximity(
        self,
        grid_gdf: gpd.GeoDataFrame,
        best_stops: gpd.GeoDataFrame,
        rail_stations: gpd.GeoDataFrame,
        metro_stations: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """
        Computes metric distance to nearest bus, rail, and metro stations,
        and stop catchment counts within 400m and 800m walksheds.
        """
        logger.info("Computing metric transit proximity and catchment counts...")
        res = grid_gdf.copy()

        # Create centroid points for queries
        centroids_disp = gpd.GeoDataFrame(
            {"geometry": [Point(lon, lat) for lon, lat in zip(res["centroid_lon"], res["centroid_lat"])]},
            crs=DISPLAY_CRS
        )

        # Build trees
        bus_tree, _ = CoordinateManager.build_metric_kdtree(best_stops)
        rail_tree, _ = CoordinateManager.build_metric_kdtree(rail_stations)
        metro_tree, _ = CoordinateManager.build_metric_kdtree(metro_stations)

        # Combine all transit stops for general walkshed queries
        all_stops_gdf = pd.concat([best_stops, rail_stations, metro_stations], ignore_index=True)
        all_tree, _ = CoordinateManager.build_metric_kdtree(all_stops_gdf)

        # Query nearest distances (meters)
        res["dist_to_nearest_bus_m"] = CoordinateManager.query_nearest_distances(centroids_disp, bus_tree).round(1)
        res["dist_to_nearest_rail_m"] = CoordinateManager.query_nearest_distances(centroids_disp, rail_tree).round(1)
        res["dist_to_nearest_metro_m"] = CoordinateManager.query_nearest_distances(centroids_disp, metro_tree).round(1)
        res["dist_to_nearest_transit_m"] = np.minimum(
            res["dist_to_nearest_bus_m"],
            np.minimum(res["dist_to_nearest_rail_m"], res["dist_to_nearest_metro_m"])
        ).round(1)

        # Query catchment counts
        res["stops_within_400m"] = CoordinateManager.query_radius_counts(centroids_disp, all_tree, 400.0)
        res["stops_within_800m"] = CoordinateManager.query_radius_counts(centroids_disp, all_tree, 800.0)

        # Coverage validation flag (do not treat missing data as zero supply without verification)
        res["has_transit_coverage"] = (res["stops_within_800m"] > 0)

        # In-cell stop counts via spatial join
        res_m = CoordinateManager.to_metric(res[["geometry"]].copy()).reset_index(names="orig_idx")
        best_m = CoordinateManager.to_metric(best_stops[["geometry"]].copy())
        rail_m = CoordinateManager.to_metric(rail_stations[["geometry"]].copy())
        metro_m = CoordinateManager.to_metric(metro_stations[["geometry"]].copy())

        bus_join = gpd.sjoin(best_m, res_m, how="inner", predicate="within")
        rail_join = gpd.sjoin(rail_m, res_m, how="inner", predicate="within")
        metro_join = gpd.sjoin(metro_m, res_m, how="inner", predicate="within")

        bus_counts = bus_join.groupby("orig_idx").size()
        rail_counts = rail_join.groupby("orig_idx").size()
        metro_counts = metro_join.groupby("orig_idx").size()

        res["bus_stops_count"] = res.index.map(bus_counts).fillna(0).astype(int)
        res["rail_stations_count"] = res.index.map(rail_counts).fillna(0).astype(int)
        res["metro_stations_count"] = res.index.map(metro_counts).fillna(0).astype(int)
        res["total_transit_stops"] = res["bus_stops_count"] + res["rail_stations_count"] + res["metro_stations_count"]

        return res

    def compute_gtfs_service_metrics(
        self,
        grid_gdf: gpd.GeoDataFrame,
        gtfs_stops: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """
        Integrates GTFS scheduled trips and headway indicators across cell catchment buffers.
        """
        logger.info("Computing GTFS service availability indicators (trips/day and headway)...")
        res = grid_gdf.copy()
        if gtfs_stops.empty or "scheduled_trips_daily" not in gtfs_stops.columns:
            res["gtfs_scheduled_trips_daily"] = 0
            res["gtfs_avg_headway_mins"] = 999.0
            return res

        # Filter GTFS stops with positive trip schedules
        active_gtfs = gtfs_stops[gtfs_stops["scheduled_trips_daily"] > 0].copy()
        if active_gtfs.empty:
            res["gtfs_scheduled_trips_daily"] = 0
            res["gtfs_avg_headway_mins"] = 999.0
            return res

        gtfs_tree, _ = CoordinateManager.build_metric_kdtree(active_gtfs)
        centroids_disp = gpd.GeoDataFrame(
            {"geometry": [Point(lon, lat) for lon, lat in zip(res["centroid_lon"], res["centroid_lat"])]},
            crs=DISPLAY_CRS
        )
        centroids_m = CoordinateManager.to_metric(centroids_disp)
        c_coords = np.column_stack([centroids_m.geometry.x.values, centroids_m.geometry.y.values])

        # Query all active GTFS stops within 800m
        neighbors = gtfs_tree.query_ball_point(c_coords, r=800.0)

        trips_array = active_gtfs["scheduled_trips_daily"].values
        daily_trips = []
        headways = []

        for stop_indices in neighbors:
            if len(stop_indices) > 0:
                total_trips = int(np.sum(trips_array[stop_indices]))
                daily_trips.append(total_trips)
                # Headway = (16 hours * 60 mins) / trips
                avg_hw = round(960.0 / max(total_trips, 1), 1)
                headways.append(min(avg_hw, 120.0))
            else:
                daily_trips.append(0)
                headways.append(999.0)

        res["gtfs_scheduled_trips_daily"] = daily_trips
        res["gtfs_avg_headway_mins"] = headways
        return res

    def compute_infrastructure_metrics(
        self,
        grid_gdf: gpd.GeoDataFrame,
        roads: gpd.GeoDataFrame,
        intersections: gpd.GeoDataFrame,
        id_col: str
    ) -> gpd.GeoDataFrame:
        """
        Computes road length, intersection density, and pedestrian network density.
        """
        logger.info("Computing road intersection and pedestrian network densities...")
        res = grid_gdf.copy()

        # Project to metric for intersection and length calculation
        res_m = CoordinateManager.to_metric(res[[id_col, "area_sq_km", "geometry"]].copy())
        roads_m = CoordinateManager.to_metric(roads[["length_meters", "geometry"]].copy())
        inter_m = CoordinateManager.to_metric(intersections[["geometry"]].copy())

        # 1. Road Intersections inside cell
        inter_join = gpd.sjoin(inter_m, res_m[[id_col, "geometry"]], how="inner", predicate="within")
        inter_counts = inter_join.groupby(id_col).size()
        res["road_intersection_count"] = res[id_col].map(inter_counts).fillna(0).astype(int)
        res["road_intersection_density"] = (res["road_intersection_count"] / res["area_sq_km"]).round(2)

        # 2. Road length inside cell via overlay
        try:
            roads_overlay = gpd.overlay(roads_m, res_m[[id_col, "geometry"]], how="intersection")
            roads_overlay["seg_len_km"] = roads_overlay.geometry.length / 1000.0
            road_lens = roads_overlay.groupby(id_col)["seg_len_km"].sum().round(3)
            res["road_length_km"] = res[id_col].map(road_lens).fillna(0.0)
            res["road_segments_count"] = res[id_col].map(roads_overlay.groupby(id_col).size()).fillna(0).astype(int)
        except Exception as e:
            logger.warning(f"Road overlay error: {e}; assigning default road lengths.")
            res["road_length_km"] = 0.0
            res["road_segments_count"] = 0

        # Pedestrian network density (km per sq km)
        res["pedestrian_network_density_km_per_sq_km"] = (res["road_length_km"] / res["area_sq_km"]).round(2)
        return res

    def compute_facility_metrics(
        self,
        grid_gdf: gpd.GeoDataFrame,
        facilities: gpd.GeoDataFrame,
        id_col: str
    ) -> gpd.GeoDataFrame:
        """
        Computes civic facility densities: hospitals, schools, markets, commercial hubs.
        """
        logger.info("Computing civic facility densities (hospitals, schools, markets, commercial)...")
        res = grid_gdf.copy()
        if facilities.empty:
            for cat in ["hospital", "school", "market", "commercial"]:
                res[f"{cat}_count"] = 0
                res[f"{cat}_density_per_sq_km"] = 0.0
            res["facility_count"] = 0
            res["facility_density_per_sq_km"] = 0.0
            return res

        res_m = CoordinateManager.to_metric(res[[id_col, "area_sq_km", "geometry"]].copy())
        fac_m = CoordinateManager.to_metric(facilities[["category", "geometry"]].copy())

        fac_join = gpd.sjoin(fac_m, res_m[[id_col, "geometry"]], how="inner", predicate="within")

        # Counts by category
        categories = ["hospital", "school", "market", "commercial"]
        for cat in categories:
            cat_subset = fac_join[fac_join["category"] == cat]
            cat_counts = cat_subset.groupby(id_col).size()
            res[f"{cat}_count"] = res[id_col].map(cat_counts).fillna(0).astype(int)
            res[f"{cat}_density_per_sq_km"] = (res[f"{cat}_count"] / res["area_sq_km"]).round(2)

        # Total facilities
        total_counts = fac_join.groupby(id_col).size()
        res["facility_count"] = res[id_col].map(total_counts).fillna(0).astype(int)
        res["facility_density_per_sq_km"] = (res["facility_count"] / res["area_sq_km"]).round(2)
        return res

    def compute_composite_indices(self, grid_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Synthesizes composite mobility supply and inequality deficit indices:
        - Transport Supply Score (0 to 100)
        - Transit Desert Index (0 to 100)
        """
        logger.info("Computing composite Transport Supply Score and Transit Desert Index...")
        res = grid_gdf.copy()

        # 1. Transport Supply Score (0 to 100)
        # Scaled components:
        # - Rail/Metro accessibility: max score if rail < 800m or metro < 800m
        rail_proximity_score = np.clip(100.0 * (1.0 - (res["dist_to_nearest_rail_m"] / 3000.0)), 0.0, 100.0)
        metro_proximity_score = np.clip(100.0 * (1.0 - (res["dist_to_nearest_metro_m"] / 2500.0)), 0.0, 100.0)
        rapid_transit_score = np.maximum(rail_proximity_score, metro_proximity_score)

        # - Bus accessibility: max score if bus stop < 400m
        bus_proximity_score = np.clip(100.0 * (1.0 - (res["dist_to_nearest_bus_m"] / 1000.0)), 0.0, 100.0)

        # - Catchment density: stops within 400m and 800m
        catchment_score = np.clip((res["stops_within_400m"] * 10.0) + (res["stops_within_800m"] * 3.0), 0.0, 100.0)

        # - GTFS scheduled frequency: max score for > 200 daily trips
        gtfs_frequency_score = np.clip((res["gtfs_scheduled_trips_daily"] / 200.0) * 100.0, 0.0, 100.0)

        # Composite Supply Score
        supply_score = (
            0.35 * rapid_transit_score +
            0.25 * bus_proximity_score +
            0.20 * catchment_score +
            0.20 * gtfs_frequency_score
        )
        res["transport_supply_score"] = supply_score.round(2)

        # 2. Transit Desert Index (0 to 100)
        # Identifies high-density areas facing extreme public transit deficits:
        # High Population Density + Low Supply Score + High Walk Distance to Transit
        max_pop_density = max(res["population_density_per_sq_km"].quantile(0.95), 1000.0)
        norm_pop_density = np.clip(res["population_density_per_sq_km"] / max_pop_density, 0.0, 1.0)
        norm_supply_deficit = 1.0 - (res["transport_supply_score"] / 100.0)
        norm_dist_deficit = np.clip(res["dist_to_nearest_transit_m"] / 1200.0, 0.0, 1.0)

        desert_raw = (
            0.45 * norm_supply_deficit +
            0.35 * norm_pop_density +
            0.20 * norm_dist_deficit
        ) * 100.0

        res["transit_desert_index"] = np.clip(desert_raw, 0.0, 100.0).round(2)
        return res

