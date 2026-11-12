"""
UrbanFlow AI - Mobility Demand Intelligence Engine
Implements Parts A, B, C, and D:
- Part A: Baseline Relative Demand Proxy (Multi-Criteria Urban Mobility Formulation)
  * Transparent, documented, normalized weights (Population, Commercial, Education, Market, Healthcare, Road Connectivity).
- Part B: Spatially-Aware Machine Learning Benchmark
  * GroupKFold cross-validation by administrative ward (eliminating location leakage).
  * Evaluates Baseline Dummy, Random Forest, XGBoost, and Gradient Boosting regressors on MAE, RMSE, R².
  * Strict data integrity: labels results as 'Estimated Relative Demand' (not actual passenger tap-in ridership).
- Part C: Unsupervised Neighbourhood Demand Segmentation
  * K-Means clustering classifying areas into 4 tiers:
    Very High, High, Moderate, and Low Relative Demand.
- Part D: SHAP Feature Explainability
  * TreeSHAP feature attributions and local contribution breakdowns for any ward or grid cell.
"""

from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import shap

from utils.logger import logger

# Documented Multi-Criteria Demand Weights (Sum = 1.00)
# Grounded in urban planning travel generation/attraction principles
DEFAULT_DEMAND_WEIGHTS: Dict[str, float] = {
    "population_density": 0.35,      # Residential trip generation & commuter baseline
    "commercial_density": 0.20,      # Employment centers, corporate offices & morning peak attractors
    "road_connectivity": 0.15,       # Intersection & street network capacity to support movement
    "school_density": 0.12,          # Educational commuting (schools, colleges, universities)
    "market_density": 0.10,          # Daily retail & informal commercial trips
    "hospital_density": 0.08         # Healthcare & essential civic service access
}


class DemandProxyCalculator:
    """Calculates reproducible, multi-criteria estimated relative mobility demand proxies."""

    def __init__(self, weights: Dict[str, float] = DEFAULT_DEMAND_WEIGHTS):
        # Validate weights sum to 1.0
        total_w = sum(weights.values())
        if abs(total_w - 1.0) > 1e-4:
            logger.warning(f"Demand weights sum to {total_w:.4f}; normalizing to 1.00.")
            self.weights = {k: v / total_w for k, v in weights.items()}
        else:
            self.weights = weights

    def compute_relative_demand_proxy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes normalized Relative Demand Score (0 to 100) using multi-criteria weighted summation.
        Applies 99th percentile clipping to prevent outlier distortion.
        """
        logger.info("Computing multi-criteria baseline relative mobility demand proxy...")
        res = df.copy()

        # Map feature columns
        feature_map = {
            "population_density": "population_density_per_sq_km",
            "commercial_density": "commercial_density_per_sq_km",
            "road_connectivity": "road_intersection_density",
            "school_density": "school_density_per_sq_km",
            "market_density": "market_density_per_sq_km",
            "hospital_density": "hospital_density_per_sq_km"
        }

        normalized_scores = {}
        for factor, col in feature_map.items():
            if col in res.columns:
                raw_vals = res[col].fillna(0.0).values
                # 99th percentile clipping for robustness
                cap_val = np.percentile(raw_vals, 99) if len(raw_vals) > 0 else 1.0
                clipped_vals = np.clip(raw_vals, 0.0, max(cap_val, 1e-4))
                min_v, max_v = np.min(clipped_vals), np.max(clipped_vals)
                if max_v > min_v:
                    norm_v = (clipped_vals - min_v) / (max_v - min_v)
                else:
                    norm_v = np.zeros_like(clipped_vals)
                normalized_scores[factor] = norm_v
            else:
                logger.warning(f"Feature column '{col}' not found; assuming zero.")
                normalized_scores[factor] = np.zeros(len(res))

        # Weighted combination
        composite_demand = np.zeros(len(res))
        for factor, weight in self.weights.items():
            composite_demand += weight * normalized_scores[factor]

        # Scale to 0 - 100 range
        res["relative_demand_score"] = np.round(composite_demand * 100.0, 2)
        logger.info(
            f"Relative demand proxy computed. Mean: {res['relative_demand_score'].mean():.2f}, "
            f"Median: {res['relative_demand_score'].median():.2f}, Max: {res['relative_demand_score'].max():.2f}."
        )
        return res


class SpatiallyAwareModelBenchmark:
    """Trains and compares regression models using GroupKFold cross-validation by administrative ward."""

    def __init__(self, n_splits: int = 5, random_state: int = 42):
        self.n_splits = n_splits
        self.random_state = random_state
        self.models = {
            "Baseline (Mean Predictor)": DummyRegressor(strategy="mean"),
            "Random Forest Regressor": RandomForestRegressor(n_estimators=100, max_depth=6, random_state=random_state),
            "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.08, random_state=random_state),
            "XGBoost Regressor": xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.08, random_state=random_state)
        }
        self.feature_cols = [
            "population_density_per_sq_km",
            "commercial_density_per_sq_km",
            "school_density_per_sq_km",
            "hospital_density_per_sq_km",
            "market_density_per_sq_km",
            "road_intersection_density",
            "pedestrian_network_density_km_per_sq_km"
        ]
        self.fitted_models: Dict[str, Any] = {}
        self.evaluation_results: Dict[str, Dict[str, float]] = {}

    def benchmark_models(
        self,
        df: pd.DataFrame,
        target_col: str = "relative_demand_score",
        group_col: str = "ward_id"
    ) -> pd.DataFrame:
        """
        Executes spatially-aware GroupKFold cross-validation to prevent geographic data leakage.
        Evaluates MAE, RMSE, and R-squared across all models.
        """
        logger.info(f"=== [Part B] Benchmarking Regressors with {self.n_splits}-Fold Spatial Cross-Validation ===")
        # Filter available feature columns
        active_features = [c for c in self.feature_cols if c in df.columns]
        X = df[active_features].fillna(0.0)
        y = df[target_col].values
        groups = df[group_col].values if group_col in df.columns else np.random.randint(0, self.n_splits, size=len(df))

        gkf = GroupKFold(n_splits=self.n_splits)
        benchmark_records = []

        for name, model in self.models.items():
            maes, rmses, r2s = [], [], []

            for train_idx, test_idx in gkf.split(X, y, groups=groups):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # Train
                model.fit(X_train, y_train)
                preds = model.predict(X_test)

                maes.append(mean_absolute_error(y_test, preds))
                rmses.append(np.sqrt(mean_squared_error(y_test, preds)))
                r2s.append(r2_score(y_test, preds))

            mean_mae = float(np.mean(maes))
            mean_rmse = float(np.mean(rmses))
            mean_r2 = float(np.mean(r2s))

            self.evaluation_results[name] = {
                "MAE": round(mean_mae, 2),
                "RMSE": round(mean_rmse, 2),
                "R2": round(mean_r2, 3)
            }
            benchmark_records.append({
                "Model": name,
                "MAE": round(mean_mae, 2),
                "RMSE": round(mean_rmse, 2),
                "R-Squared": round(mean_r2, 3),
                "Spatial CV Strategy": f"GroupKFold ({self.n_splits} Ward Splits)"
            })
            logger.info(f"Model [{name}] -> MAE: {mean_mae:.2f}, RMSE: {mean_rmse:.2f}, R²: {mean_r2:.3f}")

        # Retrain full models on complete dataset for inference and SHAP
        for name, model in self.models.items():
            model.fit(X, y)
            self.fitted_models[name] = model

        df_bench = pd.DataFrame(benchmark_records).sort_values(by="R-Squared", ascending=False).reset_index(drop=True)
        return df_bench


class DemandSegmentationEngine:
    """Segments neighbourhoods into 4 intuitive mobility demand tiers using K-Means."""

    def __init__(self, n_clusters: int = 4, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.scaler = StandardScaler()

    def segment_neighbourhoods(
        self,
        df: pd.DataFrame,
        demand_col: str = "relative_demand_score"
    ) -> pd.DataFrame:
        """
        Applies K-Means clustering to classify spatial units into:
        - Tier 1: Very High Relative Demand
        - Tier 2: High Relative Demand
        - Tier 3: Moderate Relative Demand
        - Tier 4: Low Relative Demand
        """
        logger.info("=== [Part C] Segmenting Neighbourhoods into Relative Demand Tiers ===")
        res = df.copy()

        # Multi-attribute segmentation vector
        cluster_cols = [
            demand_col,
            "population_density_per_sq_km",
            "commercial_density_per_sq_km",
            "road_intersection_density"
        ]
        available_cols = [c for c in cluster_cols if c in res.columns]
        X = res[available_cols].fillna(0.0)
        X_scaled = self.scaler.fit_transform(X)

        raw_clusters = self.kmeans.fit_predict(X_scaled)
        res["cluster_id"] = raw_clusters

        # Order clusters by mean relative demand score so tiers are strictly sorted
        cluster_means = res.groupby("cluster_id")[demand_col].mean().sort_values(ascending=False)
        tier_names = [
            "Very High Relative Demand",
            "High Relative Demand",
            "Moderate Relative Demand",
            "Low Relative Demand"
        ]
        cluster_tier_map = {cid: tier_names[i] for i, cid in enumerate(cluster_means.index)}
        cluster_rank_map = {cid: i + 1 for i, cid in enumerate(cluster_means.index)}

        res["demand_tier"] = res["cluster_id"].map(cluster_tier_map)
        res["demand_tier_rank"] = res["cluster_id"].map(cluster_rank_map)

        tier_counts = res["demand_tier"].value_counts().to_dict()
        logger.info(f"Demand segmentation complete: {tier_counts}")
        return res


class DemandExplainer:
    """Calculates SHAP feature importances and local factor attributions for demand scores."""

    def __init__(self, model, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        try:
            self.explainer = shap.TreeExplainer(self.model)
        except Exception:
            self.explainer = shap.Explainer(self.model)

    def explain_global(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculates global mean absolute SHAP feature importance."""
        logger.info("Computing global SHAP feature attributions...")
        X = df[self.feature_names].fillna(0.0)
        shap_vals = self.explainer.shap_values(X)

        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]

        mean_abs_shap = np.abs(shap_vals).mean(axis=0)
        total_shap = sum(mean_abs_shap) if sum(mean_abs_shap) > 0 else 1.0

        ranking = []
        for feat, val in sorted(zip(self.feature_names, mean_abs_shap), key=lambda x: x[1], reverse=True):
            pct = round((val / total_shap) * 100.0, 1)
            ranking.append({
                "feature": feat,
                "mean_abs_shap": round(float(val), 4),
                "relative_contribution_pct": pct
            })

        logger.info(f"Top SHAP feature drivers: {[r['feature'] for r in ranking[:3]]}")
        return {
            "ranking": ranking,
            "feature_names": self.feature_names,
            "shap_values": shap_vals
        }

    def explain_local_instance(self, row: pd.Series) -> Dict[str, float]:
        """Explains feature contributions for an individual ward or grid cell."""
        X_inst = pd.DataFrame([row[self.feature_names].fillna(0.0)])
        shap_inst = self.explainer.shap_values(X_inst)
        if isinstance(shap_inst, list):
            shap_inst = shap_inst[0]
        contributions = {}
        for feat, val in zip(self.feature_names, shap_inst[0]):
            contributions[feat] = round(float(val), 3)
        return contributions

