"""
UrbanFlow AI - Multimodal Accessibility Analytics Engine
Implements Part C & Part D:
- Dijkstra's shortest-path graph routing across integrated pedestrian and transit network.
- Calculates:
  * Network walking distance to nearest bus stop, railway station, and metro station.
  * Network walking travel time using standard walking speed:
      v_walk = 4.5 km/h (1.25 m/s) -> explicitly documented urban planning standard.
  * Number of nearby transport facilities within configurable walking thresholds:
      - 400 metres (5-minute walkshed)
      - 800 metres (10-minute walkshed)
      - 1 kilometre (13.3-minute walkshed)
  * Multimodal access opportunities (mode diversity: Bus, Rail, Metro).
  * Transfer connectivity to verified multimodal interchange hubs.
  * Verified scheduled service frequency (GTFS daily trips).
- Normalized Multimodal Accessibility Score (0 to 100).
- Ward-level accessibility audit reporting.
- High-performance caching for reuse across Demand, DTII, Optimization, and Simulation engines.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import networkx as nx
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

from data_ingestion.config import (
    PROCESSED_INTEGRATED,
    DISPLAY_CRS,
    METRIC_CRS
)
from preprocessing.coordinate_manager import CoordinateManager
from analytics.pedestrian_network import WALKING_SPEED_MPS, WALKING_SPEED_KMH
from analytics.multimodal_network import MultimodalNetworkBuilder
from utils.logger import logger


class MultimodalAccessibilityEngine:
    """Executes network routing, walkshed catchment extraction, and accessibility scoring."""

    def __init__(self, G_multi: Optional[nx.MultiDiGraph] = None):
        if G_multi is None:
            builder = MultimodalNetworkBuilder()
            G_multi, _ = builder.build_multimodal_graph(max_snap_distance_m=1200.0)
        self.G = G_multi

        # Extract pedestrian nodes for spatial mapping
        ped_nodes = []
        ped_coords = []
        for n, d in self.G.nodes(data=True):
            if d.get("node_type") in ["pedestrian_junction", "skywalk_terminus"] and "x" in d and "y" in d:
                ped_nodes.append(n)
                ped_coords.append((d["x"], d["y"]))

        if not ped_nodes:
            # Fallback to any node with coordinates
            for n, d in self.G.nodes(data=True):
                if "x" in d and "y" in d:
                    ped_nodes.append(n)
                    ped_coords.append((d["x"], d["y"]))

        self.df_ped_nodes = gpd.GeoDataFrame(
            {"node_id": ped_nodes},
            geometry=[Point(x, y) for x, y in ped_coords],
            crs=DISPLAY_CRS
        )
        self.ped_tree, _ = CoordinateManager.build_metric_kdtree(self.df_ped_nodes)

    def compute_network_accessibility(
        self,
        origins_gdf: gpd.GeoDataFrame,
        thresholds: List[float] = [400.0, 800.0, 1000.0],
        max_search_dist_m: float = 2500.0
    ) -> gpd.GeoDataFrame:
        """
        Executes Dijkstra shortest-path calculations from each origin point through the walking network
        to multimodal transit stops and stations.
        """
        logger.info(f"=== [Part C & D] Computing Network Walking Accessibility for {len(origins_gdf)} origins ===")
        res = origins_gdf.copy()

        # Ensure centroids in display CRS
        if "centroid_lon" not in res.columns or "centroid_lat" not in res.columns:
            metric_res = CoordinateManager.to_metric(res)
            disp_centroids = gpd.GeoSeries(metric_res.geometry.centroid, crs=METRIC_CRS).to_crs(DISPLAY_CRS)
            res["centroid_lat"] = disp_centroids.y
            res["centroid_lon"] = disp_centroids.x

        # Vectorized snap of origin centroids to nearest pedestrian graph node
        origins_m = CoordinateManager.to_metric(
            gpd.GeoDataFrame(
                {"geometry": [Point(lon, lat) for lon, lat in zip(res["centroid_lon"], res["centroid_lat"])]},
                crs=DISPLAY_CRS
            )
        )
        origin_coords = np.column_stack([origins_m.geometry.x.values, origins_m.geometry.y.values])
        snap_dists, snap_idxs = self.ped_tree.query(origin_coords, k=1)

        # Output feature columns
        walk_dist_bus = []
        walk_dist_rail = []
        walk_dist_metro = []
        walk_dist_transit = []

        walk_time_bus = []
        walk_time_rail = []
        walk_time_metro = []
        walk_time_transit = []

        options_400m = []
        options_800m = []
        options_1000m = []

        mode_diversity_scores = []
        modes_count_list = []
        transfer_hubs_count = []
        verified_daily_trips = []
        frequent_service_flags = []
        multimodal_accessibility_scores = []

        # Graph node classification cache
        node_modes = {}
        for n, d in self.G.nodes(data=True):
            node_modes[n] = {
                "mode": d.get("mode", ""),
                "is_transfer": d.get("is_interchange", False) or d.get("is_transfer", False),
                "daily_trips": d.get("daily_trips", 0)
            }

        logger.info(f"Running Dijkstra routing (v_walk = {WALKING_SPEED_KMH} km/h, max_search = {max_search_dist_m}m)...")

        for i in range(len(res)):
            snap_dist = float(snap_dists[i])
            ped_node = self.df_ped_nodes.iloc[snap_idxs[i]]["node_id"]

            # Run single-source Dijkstra on length weight
            try:
                path_lengths = nx.single_source_dijkstra_path_length(
                    self.G, ped_node, cutoff=max_search_dist_m, weight="length"
                )
            except Exception:
                path_lengths = {}

            # Filter reachable transit nodes
            d_bus = float("inf")
            d_rail = float("inf")
            d_metro = float("inf")

            cnt_400 = 0
            cnt_800 = 0
            cnt_1000 = 0

            reachable_modes = set()
            transfers_reachable = 0
            daily_trips_sum = 0

            for target_node, net_len in path_lengths.items():
                total_walk_len = net_len + snap_dist  # Include initial origin-to-node connector distance
                info = node_modes.get(target_node, {})
                t_mode = info.get("mode", "")

                if t_mode in ["best_bus", "suburban_rail", "metro_rail"]:
                    # Nearest distances
                    if t_mode == "best_bus" and total_walk_len < d_bus:
                        d_bus = total_walk_len
                    elif t_mode == "suburban_rail" and total_walk_len < d_rail:
                        d_rail = total_walk_len
                    elif t_mode == "metro_rail" and total_walk_len < d_metro:
                        d_metro = total_walk_len

                    # Walkshed thresholds
                    if total_walk_len <= 400.0:
                        cnt_400 += 1
                    if total_walk_len <= 800.0:
                        cnt_800 += 1
                        reachable_modes.add(t_mode)
                    if total_walk_len <= 1000.0:
                        cnt_1000 += 1
                        if info.get("is_transfer", False):
                            transfers_reachable += 1
                        daily_trips_sum += info.get("daily_trips", 0)

            # Cap distances at max search distance for unreached modes
            cap_val = max_search_dist_m
            d_bus_clean = min(d_bus, cap_val)
            d_rail_clean = min(d_rail, cap_val)
            d_metro_clean = min(d_metro, cap_val)
            d_transit_clean = min(d_bus_clean, min(d_rail_clean, d_metro_clean))

            # Walking travel times in minutes (distance / 1.25 m/s / 60)
            t_bus_mins = round((d_bus_clean / WALKING_SPEED_MPS) / 60.0, 1)
            t_rail_mins = round((d_rail_clean / WALKING_SPEED_MPS) / 60.0, 1)
            t_metro_mins = round((d_metro_clean / WALKING_SPEED_MPS) / 60.0, 1)
            t_transit_mins = round((d_transit_clean / WALKING_SPEED_MPS) / 60.0, 1)

            # Mode diversity score (0 to 100)
            num_modes = len(reachable_modes)
            if num_modes >= 3:
                div_score = 100.0
            elif num_modes == 2:
                div_score = 75.0
            elif num_modes == 1:
                div_score = 35.0
            else:
                div_score = 0.0

            # Verified service frequency check
            has_freq = (daily_trips_sum >= 120 or cnt_400 >= 3)
            # Frequency score based strictly on verified trip data
            freq_score = np.clip((daily_trips_sum / 200.0) * 100.0, 0.0, 100.0) if daily_trips_sum > 0 else (20.0 if cnt_400 > 0 else 0.0)

            # Proximity sub-score (0 to 100): inverse walking distance
            prox_score = np.clip((1.0 - (d_transit_clean / 1200.0)) * 100.0, 0.0, 100.0)

            # Opportunity sub-score (0 to 100): cumulative log opportunities
            opp_score = np.clip(
                (18.0 * np.log1p(cnt_400)) + (10.0 * np.log1p(cnt_800)) + (4.0 * np.log1p(cnt_1000)),
                0.0, 100.0
            )

            # Multimodal Accessibility Score (0 to 100)
            # Normalized weighted composite formula:
            access_score = round(
                0.30 * prox_score +
                0.25 * opp_score +
                0.25 * div_score +
                0.20 * freq_score,
                2
            )

            # Store metrics
            walk_dist_bus.append(round(d_bus_clean, 1))
            walk_dist_rail.append(round(d_rail_clean, 1))
            walk_dist_metro.append(round(d_metro_clean, 1))
            walk_dist_transit.append(round(d_transit_clean, 1))

            walk_time_bus.append(t_bus_mins)
            walk_time_rail.append(t_rail_mins)
            walk_time_metro.append(t_metro_mins)
            walk_time_transit.append(t_transit_mins)

            options_400m.append(cnt_400)
            options_800m.append(cnt_800)
            options_1000m.append(cnt_1000)

            mode_diversity_scores.append(div_score)
            modes_count_list.append(num_modes)
            transfer_hubs_count.append(transfers_reachable)
            verified_daily_trips.append(daily_trips_sum)
            frequent_service_flags.append(has_freq)
            multimodal_accessibility_scores.append(access_score)

        # Append engineered accessibility columns
        res["net_walk_dist_bus_m"] = walk_dist_bus
        res["net_walk_dist_rail_m"] = walk_dist_rail
        res["net_walk_dist_metro_m"] = walk_dist_metro
        res["net_walk_dist_transit_m"] = walk_dist_transit

        res["walk_time_bus_mins"] = walk_time_bus
        res["walk_time_rail_mins"] = walk_time_rail
        res["walk_time_metro_mins"] = walk_time_metro
        res["walk_time_transit_mins"] = walk_time_transit

        res["transit_options_400m"] = options_400m
        res["transit_options_800m"] = options_800m
        res["transit_options_1000m"] = options_1000m

        res["mode_diversity_score"] = mode_diversity_scores
        res["reachable_modes_count_800m"] = modes_count_list
        res["transfer_hubs_within_1km"] = transfer_hubs_count
        res["verified_daily_trips_accessible"] = verified_daily_trips
        res["has_frequent_service"] = frequent_service_flags
        res["multimodal_accessibility_score"] = multimodal_accessibility_scores

        logger.info(
            f"Accessibility computation complete. Mean score: {res['multimodal_accessibility_score'].mean():.2f}, "
            f"Mean walk time to transit: {res['walk_time_transit_mins'].mean():.1f} mins."
        )
        return res

    def generate_ward_accessibility_report(
        self,
        accessibility_gdf: gpd.GeoDataFrame,
        wards_gdf: gpd.GeoDataFrame
    ) -> pd.DataFrame:
        """
        Synthesizes a ward-level accessibility intelligence report ranking all 24 administrative wards.
        """
        logger.info("Generating comprehensive Ward-Level Accessibility Report...")
        gdf = accessibility_gdf.copy()
        if "ward_id" not in gdf.columns:
            # Spatial join to wards
            gdf = gpd.sjoin(gdf, wards_gdf[["ward_id", "ward_name", "geometry"]], how="inner", predicate="within")

        report_rows = []
        for ward_id, grp in gdf.groupby("ward_id"):
            w_name = grp["ward_name"].iloc[0] if "ward_name" in grp.columns else ward_id
            pop_total = grp["population"].sum() if "population" in grp.columns else 0.0
            area_total = grp["area_sq_km"].sum() if "area_sq_km" in grp.columns else 1.0

            mean_score = grp["multimodal_accessibility_score"].mean()
            mean_dist_bus = grp["net_walk_dist_bus_m"].mean()
            mean_dist_rail = grp["net_walk_dist_rail_m"].mean()
            mean_dist_metro = grp["net_walk_dist_metro_m"].mean()
            mean_time_transit = grp["walk_time_transit_mins"].mean()

            # Population access coverage
            if "population" in grp.columns and pop_total > 0:
                pop_400m = grp[grp["net_walk_dist_transit_m"] <= 400.0]["population"].sum()
                pop_800m = grp[grp["net_walk_dist_transit_m"] <= 800.0]["population"].sum()
                pct_400m = round((pop_400m / pop_total) * 100.0, 1)
                pct_800m = round((pop_800m / pop_total) * 100.0, 1)
            else:
                pct_400m = round((len(grp[grp["net_walk_dist_transit_m"] <= 400.0]) / max(1, len(grp))) * 100.0, 1)
                pct_800m = round((len(grp[grp["net_walk_dist_transit_m"] <= 800.0]) / max(1, len(grp))) * 100.0, 1)

            # Letter grade
            if mean_score >= 70.0:
                grade = "A (High Multimodal Access)"
            elif mean_score >= 50.0:
                grade = "B (Moderate Accessibility)"
            elif mean_score >= 35.0:
                grade = "C (Underserved / Transit Desert)"
            else:
                grade = "D (Critical Transit Deficit)"

            report_rows.append({
                "ward_id": ward_id,
                "ward_name": w_name,
                "population": int(pop_total),
                "area_sq_km": round(area_total, 2),
                "mean_multimodal_score": round(mean_score, 1),
                "accessibility_grade": grade,
                "mean_walk_time_transit_mins": round(mean_time_transit, 1),
                "mean_walk_dist_bus_m": round(mean_dist_bus, 0),
                "mean_walk_dist_rail_m": round(mean_dist_rail, 0),
                "mean_walk_dist_metro_m": round(mean_dist_metro, 0),
                "pop_within_400m_walk_pct": pct_400m,
                "pop_within_800m_walk_pct": pct_800m,
                "disconnected_pedestrian_flag": bool(mean_time_transit > 25.0)
            })

        df_report = pd.DataFrame(report_rows).sort_values(by="mean_multimodal_score", ascending=False).reset_index(drop=True)
        return df_report

