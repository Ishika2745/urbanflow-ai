"""
UrbanFlow AI - Multimodal Transit Network Builder
Constructs unified NetworkX graphs across rail corridors, bus lines, and pedestrian transfer links.
"""

from typing import Dict, List, Tuple
import networkx as nx
import geopandas as gpd
from shapely.geometry import Point

from config.settings import settings
from config.city_config import CityConfig
from utils.geo_utils import ensure_metric_crs, calculate_euclidean_distance_matrix
from utils.logger import logger


class MultimodalNetworkBuilder:
    """Combines stops, line sequences, and transfer edges into a routable graph."""

    def __init__(self, city_config: CityConfig):
        self.config = city_config
        self.graph = nx.DiGraph()

    def add_transit_nodes(self, stops_gdf: gpd.GeoDataFrame) -> None:
        """Adds stops as graph nodes with spatial coordinates and attributes."""
        for _, row in stops_gdf.iterrows():
            self.graph.add_node(
                row["stop_id"],
                stop_name=row["stop_name"],
                mode_id=row["mode_id"],
                lat=row["latitude"],
                lon=row["longitude"],
                is_interchange=row.get("is_interchange", False),
                daily_ridership=row.get("daily_ridership", 0)
            )
        logger.info(f"Added {len(stops_gdf)} transit nodes to network graph.")

    def add_corridor_edges(self, stop_sequences: List[List[str]], mode_id: str) -> None:
        """
        Adds directed edges representing transit routes along stop sequences.
        Weight represents in-vehicle travel time in minutes.
        """
        mode_cfg = self.config.transit_modes.get(mode_id)
        speed_kmh = mode_cfg.average_speed_kmh if mode_cfg else 25.0

        edge_count = 0
        for seq in stop_sequences:
            for i in range(len(seq) - 1):
                u, v = seq[i], seq[i + 1]
                if self.graph.has_node(u) and self.graph.has_node(v):
                    # Approximate distance between consecutive stops
                    p1 = (self.graph.nodes[u]["lat"], self.graph.nodes[u]["lon"])
                    p2 = (self.graph.nodes[v]["lat"], self.graph.nodes[v]["lon"])
                    dist_km = np_haversine(p1[0], p1[1], p2[0], p2[1])
                    travel_time_mins = (dist_km / speed_kmh) * 60.0

                    self.graph.add_edge(
                        u, v,
                        mode=mode_id,
                        distance_km=dist_km,
                        weight=travel_time_mins
                    )
                    edge_count += 1
        logger.info(f"Added {edge_count} corridor edges for mode '{mode_id}'.")

    def add_transfer_edges(self, max_walk_distance_m: float = 300.0) -> None:
        """Adds walking interchange edges between stations within walking proximity."""
        stops = [
            (n, data["lat"], data["lon"], data["mode_id"])
            for n, data in self.graph.nodes(data=True)
        ]
        transfer_count = 0
        walk_speed_kmh = self.config.standard_walking_speed_kmh

        for i in range(len(stops)):
            n1, lat1, lon1, m1 = stops[i]
            for j in range(i + 1, len(stops)):
                n2, lat2, lon2, m2 = stops[j]
                if m1 == m2:
                    continue  # Only transfer between different modes/routes
                dist_km = np_haversine(lat1, lon1, lat2, lon2)
                dist_m = dist_km * 1000.0
                if dist_m <= max_walk_distance_m:
                    # Walking transfer time + 3 min transfer penalty
                    walk_time_mins = (dist_km / walk_speed_kmh) * 60.0 + 3.0
                    self.graph.add_edge(n1, n2, mode="walk_transfer", distance_km=dist_km, weight=walk_time_mins)
                    self.graph.add_edge(n2, n1, mode="walk_transfer", distance_km=dist_km, weight=walk_time_mins)
                    transfer_count += 2
        logger.info(f"Added {transfer_count} multimodal transfer edges.")


def np_haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Computes great-circle distance in kilometers."""
    import numpy as np
    r = 6371.0  # Earth radius in km
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return float(2 * r * np.arcsin(np.sqrt(a)))

