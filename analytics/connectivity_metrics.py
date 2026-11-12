"""
UrbanFlow AI - Network Connectivity & Interchange Centrality
Computes graph centrality, hub vulnerabilities, and bottleneck metrics across transit lines.
"""

from typing import Dict, Any
import networkx as nx
import pandas as pd
from utils.logger import logger


class NetworkConnectivityAnalyzer:
    """Calculates network centrality metrics for interchange stops."""

    @staticmethod
    def calculate_centralities(graph: nx.DiGraph) -> pd.DataFrame:
        """
        Computes Betweenness Centrality, Closeness Centrality, Degree, and PageRank.
        Identifies critical multi-modal interchanges (e.g., Dadar, Kurla, Andheri).
        """
        if graph.number_of_nodes() == 0:
            return pd.DataFrame()

        logger.info(f"Computing centrality metrics on graph ({graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges)...")

        # Convert to undirected copy for bidirectional connectivity analysis
        undirected = graph.to_undirected()

        degree = dict(undirected.degree())
        betweenness = nx.betweenness_centrality(undirected, weight="weight", normalized=True)
        closeness = nx.closeness_centrality(undirected, distance="weight")
        pagerank = nx.pagerank(graph, weight="weight")

        records = []
        for node in graph.nodes():
            node_data = graph.nodes[node]
            records.append({
                "stop_id": node,
                "stop_name": node_data.get("stop_name", node),
                "mode_id": node_data.get("mode_id", "transit"),
                "is_interchange": node_data.get("is_interchange", False),
                "degree_centrality": degree.get(node, 0),
                "betweenness_centrality": round(betweenness.get(node, 0.0), 5),
                "closeness_centrality": round(closeness.get(node, 0.0), 5),
                "pagerank": round(pagerank.get(node, 0.0), 5)
            })

        df = pd.DataFrame(records).sort_values(by="betweenness_centrality", ascending=False).reset_index(drop=True)
        return df

