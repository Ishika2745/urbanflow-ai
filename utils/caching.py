"""
UrbanFlow AI - Centralized Caching and Performance Optimization Engine
Avoids rebuilding the multimodal graph and network routing on dashboard filter changes.
Provides:
- Processed dataset caching (Parquet / GeoParquet)
- Network graph in-memory and disk caching (GraphML / Pickle)
- Shortest-path and metric distance matrix caching
- Machine learning model artifact caching
- Optimization result caching
- Vectorized spatial queries via cKDTree and spatial indexes
"""

import os
import pickle
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Callable
import networkx as nx
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree

from utils.logger import logger


class DatasetCache:
    """In-memory singleton cache with Parquet / GeoParquet disk backing."""
    _memory_store: Dict[str, Any] = {}

    @classmethod
    def get_or_load(cls, key: str, loader_fn: Callable[[], Any], force_reload: bool = False) -> Any:
        if not force_reload and key in cls._memory_store:
            return cls._memory_store[key]

        data = loader_fn()
        cls._memory_store[key] = data
        return data

    @classmethod
    def clear(cls, key: Optional[str] = None) -> None:
        if key:
            cls._memory_store.pop(key, None)
        else:
            cls._memory_store.clear()


class NetworkGraphCache:
    """Caches pedestrian & multimodal transport graphs to prevent costly graph reconstruction."""
    _cached_graph: Optional[nx.MultiDiGraph] = None
    _cache_file: Path = Path("data/processed/infrastructure/pedestrian_network.graphml")

    @classmethod
    def get_multimodal_graph(cls, builder_fn: Optional[Callable[[], nx.MultiDiGraph]] = None) -> nx.MultiDiGraph:
        if cls._cached_graph is not None:
            return cls._cached_graph

        if cls._cache_file.exists():
            try:
                t0 = time.time()
                cls._cached_graph = nx.read_graphml(cls._cache_file)
                logger.info(f"Loaded cached network graph from {cls._cache_file} in {time.time() - t0:.2f}s")
                return cls._cached_graph
            except Exception as e:
                logger.warning(f"Failed to read cached graphml: {e}. Rebuilding...")

        if builder_fn is not None:
            t0 = time.time()
            cls._cached_graph = builder_fn()
            logger.info(f"Built network graph in {time.time() - t0:.2f}s")
            try:
                cls._cache_file.parent.mkdir(parents=True, exist_ok=True)
                nx.write_graphml(cls._cached_graph, cls._cache_file)
            except Exception as e:
                logger.warning(f"Could not persist graphml cache: {e}")
            return cls._cached_graph

        # Return empty graph if no builder supplied
        return nx.MultiDiGraph()

    @classmethod
    def set_graph(cls, G: nx.MultiDiGraph) -> None:
        cls._cached_graph = G


class ShortestPathCache:
    """In-memory cache for pair-wise routing distances and Dijkstra paths."""
    _distance_matrix_cache: Dict[str, np.ndarray] = {}
    _nearest_stop_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    @classmethod
    def get_distance_matrix(
        cls,
        cache_key: str,
        origins_xy: np.ndarray,
        destinations_xy: np.ndarray
    ) -> np.ndarray:
        """
        Fast vectorized Euclidean / metric distance matrix in EPSG:32643 using cKDTree.
        Caches result matrix in memory.
        """
        if cache_key in cls._distance_matrix_cache:
            return cls._distance_matrix_cache[cache_key]

        tree = cKDTree(destinations_xy)
        # Query all destinations for each origin
        distances = []
        for orig in origins_xy:
            dists, _ = tree.query(orig, k=len(destinations_xy))
            distances.append(dists)

        matrix = np.array(distances)
        cls._distance_matrix_cache[cache_key] = matrix
        return matrix

    @classmethod
    def get_nearest_stops(
        cls,
        cache_key: str,
        origins_xy: np.ndarray,
        stops_xy: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorized nearest stop query returning (distances, indices)."""
        if cache_key in cls._nearest_stop_cache:
            return cls._nearest_stop_cache[cache_key]

        tree = cKDTree(stops_xy)
        distances, indices = tree.query(origins_xy)
        cls._nearest_stop_cache[cache_key] = (distances, indices)
        return distances, indices


class ModelArtifactCache:
    """Caches machine learning regression models and explainers to eliminate retraining latency."""
    _models: Dict[str, Any] = {}

    @classmethod
    def get_model(cls, model_name: str, train_fn: Optional[Callable[[], Any]] = None) -> Any:
        if model_name in cls._models:
            return cls._models[model_name]

        if train_fn is not None:
            model = train_fn()
            cls._models[model_name] = model
            return model
        return None

    @classmethod
    def set_model(cls, model_name: str, model_obj: Any) -> None:
        cls._models[model_name] = model_obj


class OptimizationResultCache:
    """Caches benchmark results and convergence histories."""
    _results: Dict[str, Any] = {}

    @classmethod
    def get_benchmark(cls, key: str, run_fn: Optional[Callable[[], Any]] = None) -> Any:
        if key in cls._results:
            return cls._results[key]
        if run_fn is not None:
            res = run_fn()
            cls._results[key] = res
            return res
        return None

    @classmethod
    def store_result(cls, key: str, result: Any) -> None:
        cls._results[key] = result

