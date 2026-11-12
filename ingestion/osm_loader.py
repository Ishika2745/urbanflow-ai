"""
UrbanFlow AI - OpenStreetMap (OSM) Ingestion Utility
Queries OSM via OSMnx for Mumbai road networks, walking graphs, and railway transit lines with local caching.
"""

from pathlib import Path
from typing import Optional
import osmnx as ox
import networkx as nx
import geopandas as gpd

from config.settings import settings
from config.city_config import CityConfig
from utils.logger import logger
from utils.exceptions import DataIngestionError


class OSMNetworkLoader:
    """Extracts and caches multi-modal street and rail graphs from OpenStreetMap."""

    def __init__(self, city_config: CityConfig, cache_dir: Path = settings.RAW_DATA_DIR):
        self.config = city_config
        self.cache_dir = cache_dir

    def get_walking_graph(self, reload_cache: bool = False) -> nx.MultiDiGraph:
        """
        Retrieves pedestrian walkable street graph for pedestrian catchment & accessibility analysis.
        Caches graph to GraphML format.
        """
        cache_file = self.cache_dir / f"{self.config.city_name.lower().replace(' ', '_')}_walk_network.graphml"

        if cache_file.exists() and not reload_cache:
            logger.info(f"Loading cached walking graph from {cache_file.name}")
            try:
                return ox.load_graphml(cache_file)
            except Exception as e:
                logger.warning(f"Failed to load cached graph: {e}. Will attempt fresh generation.")

        bbox = self.config.bbox
        logger.info(f"Querying OSM walking network for {self.config.city_name} bbox: {bbox.bounds}")
        try:
            # Query OSM within bounding box
            G = ox.graph_from_bbox(
                bbox=bbox.bounds,
                network_type="walk",
                simplify=True
            )
            # Save to cache
            ox.save_graphml(G, cache_file)
            return G
        except Exception as e:
            logger.error(f"Failed to download OSM network: {e}")
            raise DataIngestionError(f"OSM download failed: {e}") from e

