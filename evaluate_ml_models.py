"""
UrbanFlow AI - Machine Learning & Spatial Data Science Model Evaluation CLI
Calculates and prints core data science models, cross-validation metrics, and spatial statistics directly in the terminal (zero UI impact).

Usage:
    python evaluate_ml_models.py
"""

import sys
import os
import time
from pathlib import Path
import warnings

# Suppress minor library warnings for clean terminal display
warnings.filterwarnings("ignore")

# Ensure project root is in sys.path regardless of execution directory
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
os.chdir(BASE_DIR)

# Auto-detect project virtualenv: if system python was used and lacks libraries, re-exec with .venv python
venv_python = BASE_DIR / ".venv" / "Scripts" / "python.exe"
if venv_python.exists() and sys.executable.lower() != str(venv_python).lower():
    try:
        import xgboost
    except ImportError:
        import subprocess
        res = subprocess.run([str(venv_python), str(__file__)] + sys.argv[1:])
        sys.exit(res.returncode)

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, r2_score, mean_absolute_error, mean_squared_error

from models.mobility_demand import (
    DemandProxyCalculator,
    SpatiallyAwareModelBenchmark,
    DemandSegmentationEngine,
    DemandExplainer,
    DEFAULT_DEMAND_WEIGHTS
)
from analytics.dtii_engine import DTIICalculator, DTIIWeights


def print_header(title: str):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_section(title: str):
    print(f"\n--- {title} ---")


def main():
    start_total = time.time()
    print_header("URBANFLOW AI - CORE DATA SCIENCE & ML MODEL EVALUATION SUITE")
    print(f"Project Directory  : {BASE_DIR}")
    print("Geographical Scope : Greater Mumbai (MCGM 24 Administrative Wards)")
    print("Target Evaluation  : Spatially-Aware GroupKFold (Zero Geographic Leakage)")
    print("Evaluation Mode    : Terminal Output Only (Not included in UI dashboard)")

    # 1. Load Data
    grid_path = BASE_DIR / "data" / "features" / "mumbai_accessibility_grid.parquet"
    if not grid_path.exists():
        grid_path = BASE_DIR / "data" / "features" / "mumbai_grid_1km.parquet"
    
    if not grid_path.exists():
        print(f"\nError: Dataset {grid_path} not found.")
        sys.exit(1)

    print(f"\n[1/5] Loading spatial dataset from {grid_path}...")
    grid_gdf = gpd.read_parquet(grid_path)
    print(f"      Loaded {len(grid_gdf)} spatial units across {grid_gdf['ward_id'].nunique()} municipal wards.")

    # 2. Part A: Multi-Criteria Demand Proxy
    print_section("STAGE 1: Baseline Multi-Criteria Relative Demand Formulation")
    calc = DemandProxyCalculator(DEFAULT_DEMAND_WEIGHTS)
    grid_gdf = calc.compute_relative_demand_proxy(grid_gdf)
    dem_mean = grid_gdf["relative_demand_score"].mean()
    dem_std = grid_gdf["relative_demand_score"].std()
    dem_max = grid_gdf["relative_demand_score"].max()
    print(f"      Demand Score Distribution: Mean = {dem_mean:.2f}, Std = {dem_std:.2f}, Max = {dem_max:.2f} (0-100 scale)")

    # 3. Part B: Supervised Spatially-Aware Regressors Benchmark
    print_section("STAGE 2: Supervised Regressor Benchmark (5-Fold Spatial GroupKFold by Ward)")
    print("      Evaluating models with zero location leakage across geographic boundaries...")
    
    benchmark = SpatiallyAwareModelBenchmark(n_splits=5, random_state=42)
    df_bench = benchmark.benchmark_models(
        df=grid_gdf,
        target_col="relative_demand_score",
        group_col="ward_id"
    )

    print("\n" + "-" * 78)
    print(f"{'Model Name':<30} | {'MAE':<8} | {'RMSE':<8} | {'R-Squared':<10} | {'Validation Strategy'}")
    print("-" * 78)
    for _, row in df_bench.iterrows():
        m_name = row['Model']
        mae = f"{row['MAE']:.3f}"
        rmse = f"{row['RMSE']:.3f}"
        r2 = f"{row['R-Squared']:.3f}"
        strat = row.get('Spatial CV Strategy', 'GroupKFold (5 Wards)')
        print(f"{m_name:<30} | {mae:<8} | {rmse:<8} | {r2:<10} | {strat}")
    print("-" * 78)
    best_m = df_bench.iloc[0]['Model']
    best_r2 = df_bench.iloc[0]['R-Squared']
    print(f"      Best Performing Model: {best_m} (R² = {best_r2:.3f})")

    # 4. Part C: Unsupervised K-Means Demand Segmentation
    print_section("STAGE 3: Unsupervised K-Means Mobility Demand Segmentation (k=4)")
    segmenter = DemandSegmentationEngine(n_clusters=4, random_state=42)
    grid_gdf = segmenter.segment_neighbourhoods(grid_gdf, demand_col="relative_demand_score")
    
    # Calculate silhouette score on demand feature space
    feat_cols = [c for c in benchmark.feature_cols if c in grid_gdf.columns]
    X_sample = grid_gdf[feat_cols].fillna(0.0)
    sil_score = silhouette_score(X_sample, grid_gdf["cluster_id"])

    print(f"      K-Means Clustering Silhouette Score: {sil_score:.3f}")
    print("\n" + "-" * 70)
    print(f"{'Cluster Tier':<28} | {'Count (Cells)':<14} | {'Mean Demand Score'}")
    print("-" * 70)
    tier_summary = grid_gdf.groupby("demand_tier").agg(
        cell_count=("cluster_id", "count"),
        mean_demand=("relative_demand_score", "mean")
    ).sort_values(by="mean_demand", ascending=False).reset_index()
    for _, row in tier_summary.iterrows():
        print(f"{row['demand_tier']:<28} | {row['cell_count']:<14} | {row['mean_demand']:.2f}")
    print("-" * 70)

    # 5. Part D: SHAP Explainability & Global Feature Importance
    print_section("STAGE 4: Explainable AI - Global Feature Importance (TreeSHAP)")
    best_fitted = benchmark.fitted_models.get(best_m, list(benchmark.fitted_models.values())[0])
    explainer = DemandExplainer(best_fitted, benchmark.feature_cols)
    shap_res = explainer.explain_global(grid_gdf)
    ranking = shap_res.get("ranking", [])

    print("\n" + "-" * 70)
    print(f"{'Rank':<6} | {'Feature Attribute':<36} | {'Mean Absolute SHAP'} | {'Relative %'}")
    print("-" * 70)
    for idx, row in enumerate(ranking):
        print(f"#{idx+1:<5} | {row['feature']:<36} | {row['mean_abs_shap']:<18.4f} | {row['relative_contribution_pct']:>5.1f}%")
    print("-" * 70)

    # 6. Part E: Dynamic Transit Inequality Index (DTII) Formulation
    print_section("STAGE 5: Dynamic Transit Inequality Index (DTII = f(D, A, S, V))")
    wards_path = BASE_DIR / "data" / "features" / "mumbai_transit_deserts_wards.parquet"
    if wards_path.exists():
        wards_gdf = gpd.read_parquet(wards_path)
        dtii_calc = DTIICalculator(DTIIWeights(w_demand=0.30, w_access=0.30, w_supply=0.20, w_vulnerability=0.20))
        df_dtii = dtii_calc.calculate(wards_gdf)

        print("\nTop 5 Wards with Highest Transit Inequality (Transit Deserts Priority):")
        print("-" * 75)
        print(f"{'Rank':<6} | {'Ward':<10} | {'DTII Score':<12} | {'Category':<18} | {'Dominant Deficit'}")
        print("-" * 75)
        ranked_wards = df_dtii.sort_values(by="dtii", ascending=False).head(5)
        for i, (_, r) in enumerate(ranked_wards.iterrows()):
            print(f"#{i+1:<5} | {r['ward_name']:<10} | {r['dtii']:<12.3f} | {r['dtii_category']:<18} | {r['dominant_inequality_factor']}")
        print("-" * 75)

    elapsed = time.time() - start_total
    print_header(f"ALL DATA SCIENCE MODELS EXECUTED SUCCESSFULLY IN {elapsed:.2f} SECONDS")
    print("Note: As requested, these detailed calculations and benchmarks are kept in this terminal executable and not exposed on the client UI.")


if __name__ == "__main__":
    main()
