"""
UrbanFlow AI - Spatial Clustering & Hotspot Analytics Engine
Implements Part B Spatial Analysis:
1. K-Means clustering for multivariate transit typologies.
2. DBSCAN for spatial cluster detection of contiguous transit deserts.
3. Local Moran's I (LISA) for spatial autocorrelation and outlier detection.
4. Getis-Ord Gi* for statistically significant hotspot and coldspot identification.

Grounded in PySAL (libpysal, esda) and scikit-learn with valid spatial weights.
"""

from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from libpysal.weights import DistanceBand, Queen, KNN, util
from esda.moran import Moran, Moran_Local
from esda.getisord import G_Local

from utils.logger import logger
from utils.geo_utils import ensure_metric_crs, ensure_display_crs


class SpatialClusterAnalyzer:
    """Detects spatial autocorrelation, spatial clusters, and statistically significant transit hotspots."""

    @staticmethod
    def compute_kmeans_clustering(
        gdf: gpd.GeoDataFrame,
        feature_cols: Optional[List[str]] = None,
        n_clusters: int = 4,
        random_state: int = 42
    ) -> gpd.GeoDataFrame:
        """
        K-Means clustering in normalized multivariate feature space:
        Groups urban areas into operational mobility typologies:
        - Acute Transit Desert (High Demand, Low Accessibility/Supply)
        - Developing Transit Corridor (Moderate Demand, Moderate Supply)
        - Well-Serviced Multimodal Hub (High Accessibility, High Supply)
        - Low-Density / Peripheral Buffer (Low Demand, Low Supply)
        """
        res = gdf.copy()
        if feature_cols is None:
            # Default to key mobility criteria
            candidates = [
                "dtii", "transit_desert_index", "relative_demand_score",
                "multimodal_accessibility_score", "transport_supply_score",
                "population_density_per_sq_km", "net_walk_dist_transit_m"
            ]
            feature_cols = [c for c in candidates if c in res.columns]

        if not feature_cols:
            logger.warning("No valid feature columns found for K-Means clustering.")
            res["kmeans_cluster"] = 0
            res["kmeans_label"] = "Cluster 0"
            return res

        logger.info(f"Computing K-Means clustering ({n_clusters} clusters) on features: {feature_cols}...")
        X = res[feature_cols].fillna(0.0).values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        labels = km.fit_predict(X_scaled)
        res["kmeans_cluster"] = labels

        # Identify typologies by profiling cluster centroids
        centroids = pd.DataFrame(scaler.inverse_transform(km.cluster_centers_), columns=feature_cols)
        
        # Profile clusters based on DTII / demand vs accessibility
        dtii_col = "dtii" if "dtii" in centroids.columns else ("transit_desert_index" if "transit_desert_index" in centroids.columns else None)
        acc_col = "multimodal_accessibility_score" if "multimodal_accessibility_score" in centroids.columns else None
        dem_col = "relative_demand_score" if "relative_demand_score" in centroids.columns else None

        cluster_names = {}
        for c_idx in range(n_clusters):
            c_row = centroids.iloc[c_idx]
            c_dtii = c_row.get(dtii_col, 50.0) if dtii_col else 50.0
            c_acc = c_row.get(acc_col, 50.0) if acc_col else 50.0
            c_dem = c_row.get(dem_col, 50.0) if dem_col else 50.0

            if c_dtii >= 50.0 or (c_dem > 25.0 and c_acc < 25.0):
                cluster_names[c_idx] = "Acute Transit Desert"
            elif c_acc >= 40.0:
                cluster_names[c_idx] = "Well-Serviced Multimodal Hub"
            elif c_dem >= 20.0 and c_acc >= 20.0:
                cluster_names[c_idx] = "Developing Transit Corridor"
            else:
                cluster_names[c_idx] = "Low-Density / Buffer Zone"

        res["kmeans_label"] = res["kmeans_cluster"].map(cluster_names)
        return res

    @staticmethod
    def compute_dbscan_clustering(
        gdf: gpd.GeoDataFrame,
        eps_meters: float = 1200.0,
        min_samples: int = 3,
        desert_filter_col: Optional[str] = "is_transit_desert"
    ) -> gpd.GeoDataFrame:
        """
        DBSCAN for spatial cluster detection:
        Identifies spatially contiguous transit desert clusters using projected coordinates (EPSG:32643).
        Noise points (cluster = -1) represent isolated pockets, while labeled clusters (>= 0)
        represent spatially contiguous regional transit deficit belts.
        """
        res = ensure_metric_crs(gdf.copy())
        coords = np.column_stack([res.geometry.centroid.x.values, res.geometry.centroid.y.values])

        # If a filter column is provided, we can run DBSCAN primarily on desert cells
        if desert_filter_col and desert_filter_col in res.columns:
            mask = res[desert_filter_col].fillna(False).astype(bool).values
            logger.info(f"Running DBSCAN on {int(mask.sum())} identified transit desert cells (eps={eps_meters}m, min_samples={min_samples})...")
            
            cluster_labels = np.full(len(res), -1, dtype=int)
            if mask.sum() >= min_samples:
                db = DBSCAN(eps=eps_meters, min_samples=min_samples, metric="euclidean")
                sub_labels = db.fit_predict(coords[mask])
                cluster_labels[mask] = sub_labels
        else:
            logger.info(f"Running spatial DBSCAN on all {len(res)} locations (eps={eps_meters}m, min_samples={min_samples})...")
            db = DBSCAN(eps=eps_meters, min_samples=min_samples, metric="euclidean")
            cluster_labels = db.fit_predict(coords)

        res["dbscan_cluster"] = cluster_labels
        res["is_contiguous_cluster"] = res["dbscan_cluster"] >= 0

        num_clusters = len(set(cluster_labels[cluster_labels >= 0]))
        num_clustered_pts = int((cluster_labels >= 0).sum())
        logger.info(f"DBSCAN detected {num_clusters} contiguous spatial clusters containing {num_clustered_pts} locations.")
        return res

    @staticmethod
    def compute_local_moran(
        gdf: gpd.GeoDataFrame,
        variable_col: str = "dtii",
        k_neighbors: int = 6,
        significance_level: float = 0.05
    ) -> gpd.GeoDataFrame:
        """
        Local Indicators of Spatial Association (LISA) Local Moran's I.
        Classifies locations into:
        - 'High-High' (Transit Desert Hotspot: high inequality surrounded by high inequality)
        - 'Low-Low' (Transit Oasis: well-served zone surrounded by well-served zones)
        - 'Low-High' (Spatial Outlier: underserved zone surrounded by transit-rich areas)
        - 'High-Low' (Spatial Outlier: transit island surrounded by underserved areas)
        - 'Not Significant' (p > significance_level)
        """
        res = ensure_metric_crs(gdf.copy())
        if variable_col not in res.columns:
            fallback = "transit_desert_index" if "transit_desert_index" in res.columns else res.columns[0]
            logger.warning(f"Column '{variable_col}' not found. Falling back to '{fallback}'.")
            variable_col = fallback

        y = res[variable_col].fillna(0.0).values
        n_samples = len(res)
        k = min(k_neighbors, max(1, n_samples - 1))

        logger.info(f"Computing Local Moran's I on '{variable_col}' (N={n_samples}, k={k})...")
        w = KNN.from_dataframe(res, k=k)
        w.transform = "R"  # Row-standardized spatial weights

        # Global Moran's I for overall spatial autocorrelation
        try:
            moran_global = Moran(y, w)
            logger.info(f"Global Moran's I: {moran_global.I:.3f} (p-value: {moran_global.p_sim:.4f})")
            res.attrs["global_moran_i"] = float(moran_global.I)
            res.attrs["global_moran_p"] = float(moran_global.p_sim)
        except Exception as e:
            logger.warning(f"Global Moran calculation note: {e}")

        # Local Moran's I
        lm = Moran_Local(y, w, transformation="r", permutations=999, seed=42, alternative="two-sided")

        quad_names = {
            1: "High-High (Transit Desert Hotspot)",
            2: "Low-High (Spatial Outlier)",
            3: "Low-Low (Transit Oasis)",
            4: "High-Low (Spatial Outlier)"
        }

        labels = []
        for q, p in zip(lm.q, lm.p_sim):
            if p <= significance_level:
                labels.append(quad_names.get(q, "Not Significant"))
            else:
                labels.append("Not Significant")

        res["lisa_quadrant"] = labels
        res["local_moran_i"] = np.round(lm.Is, 4)
        res["lisa_p_value"] = np.round(lm.p_sim, 4)
        res["lisa_z_score"] = np.round(lm.z_sim, 3)
        res["is_lisa_hotspot"] = res["lisa_quadrant"] == "High-High (Transit Desert Hotspot)"
        res["is_transit_desert"] = res["is_lisa_hotspot"]

        return res

    @staticmethod
    def compute_getis_ord_gi_star(
        gdf: gpd.GeoDataFrame,
        variable_col: str = "dtii",
        k_neighbors: int = 6,
        significance_level: float = 0.05
    ) -> gpd.GeoDataFrame:
        """
        Getis-Ord Gi* statistic:
        Identifies statistically significant spatial hotspots (clusters of high values)
        and coldspots (clusters of low values).
        Uses spatial weights with self-weight on the diagonal (star=True) and row-standardization.
        Reports z-scores Z(Gi*) and pseudo p-values.
        """
        res = ensure_metric_crs(gdf.copy())
        if variable_col not in res.columns:
            fallback = "transit_desert_index" if "transit_desert_index" in res.columns else res.columns[0]
            logger.warning(f"Column '{variable_col}' not found. Falling back to '{fallback}'.")
            variable_col = fallback

        y = res[variable_col].fillna(0.0).values
        n_samples = len(res)
        k = min(k_neighbors, max(1, n_samples - 1))

        logger.info(f"Computing Getis-Ord Gi* on '{variable_col}' (N={n_samples}, k={k})...")
        
        # Spatial weights with self included
        w = KNN.from_dataframe(res, k=k)
        w_star = util.fill_diagonal(w, 1.0)

        # G_Local with star=True
        g_local = G_Local(y, w_star, transform="R", star=True, permutations=999, seed=42, alternative="two-sided")

        z_scores = g_local.Zs
        p_vals = g_local.p_sim

        # Categorize confidence levels
        gi_categories = []
        for z, p in zip(z_scores, p_vals):
            if p <= 0.01 and z > 2.58:
                gi_categories.append("Hotspot - 99% Confidence")
            elif p <= 0.05 and z > 1.96:
                gi_categories.append("Hotspot - 95% Confidence")
            elif p <= 0.10 and z > 1.65:
                gi_categories.append("Hotspot - 90% Confidence")
            elif p <= 0.01 and z < -2.58:
                gi_categories.append("Coldspot - 99% Confidence")
            elif p <= 0.05 and z < -1.96:
                gi_categories.append("Coldspot - 95% Confidence")
            elif p <= 0.10 and z < -1.65:
                gi_categories.append("Coldspot - 90% Confidence")
            else:
                gi_categories.append("Not Significant")

        res["getis_ord_z"] = np.round(z_scores, 3)
        res["getis_ord_p"] = np.round(p_vals, 4)
        res["gi_category"] = gi_categories
        res["is_gi_hotspot"] = res["gi_category"].str.startswith("Hotspot")

        hotspot_cnt = int(res["is_gi_hotspot"].sum())
        logger.info(f"Getis-Ord Gi* identified {hotspot_cnt} statistically significant hotspots (z > 1.65, p <= 0.10).")
        return res

    @staticmethod
    def detect_transit_deserts(
        wards_gdf: gpd.GeoDataFrame,
        variable_col: str = "transit_desert_index",
        significance_level: float = 0.05
    ) -> gpd.GeoDataFrame:
        """
        Backward-compatible wrapper for Local Moran's I transit desert detection.
        """
        return SpatialClusterAnalyzer.compute_local_moran(
            gdf=wards_gdf,
            variable_col=variable_col,
            k_neighbors=4,
            significance_level=significance_level
        )
