"""
UrbanFlow AI - Master Mobility Demand Intelligence Pipeline
Executes Parts A through E:
- Part A: Baseline Relative Demand Proxy Computation
- Part B: Spatially-Aware Regressor Benchmark (GroupKFold by Ward on Random Forest, XGBoost, GBR, Baseline)
- Part C: Neighbourhood K-Means Demand Segmentation (4 Tiers)
- Part D: SHAP Global and Local Feature Attributions
- Part E: Visualizations (Demand Heatmap, Ward Chart, Cluster Scatter, Unmet Demand Matrix, Markdown Report)

Execute via:
    python -m models.run_demand_intelligence
"""

import time
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
import geopandas as gpd

from data_ingestion.config import PROCESSED_INTEGRATED
from models.mobility_demand import (
    DemandProxyCalculator,
    SpatiallyAwareModelBenchmark,
    DemandSegmentationEngine,
    DemandExplainer,
    DEFAULT_DEMAND_WEIGHTS
)
from models.demand_visualizer import DemandVisualizer
from utils.logger import logger

REPORTS_DIR = Path("reports")
MAPS_DIR = REPORTS_DIR / "maps"


def run_demand_intelligence_pipeline() -> Dict[str, Any]:
    """Master orchestrator for mobility demand intelligence analysis."""
    start_time = time.time()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MAPS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("================================================================================")
    logger.info("   URBANFLOW AI - MOBILITY DEMAND INTELLIGENCE PIPELINE                         ")
    logger.info("================================================================================")

    # 1. Load accessibility grid
    grid_path = Path("data/features/mumbai_accessibility_grid.parquet")
    if not grid_path.exists():
        grid_path = Path("data/features/mumbai_grid_1km.parquet")
    grid_gdf = gpd.read_parquet(grid_path)

    # -------------------------------------------------------------------------
    # PART A: BASELINE RELATIVE DEMAND PROXY
    # -------------------------------------------------------------------------
    calculator = DemandProxyCalculator(DEFAULT_DEMAND_WEIGHTS)
    grid_gdf = calculator.compute_relative_demand_proxy(grid_gdf)

    # -------------------------------------------------------------------------
    # PART B: SPATIALLY-AWARE MACHINE LEARNING BENCHMARK
    # -------------------------------------------------------------------------
    benchmark = SpatiallyAwareModelBenchmark(n_splits=5, random_state=42)
    df_benchmark = benchmark.benchmark_models(
        df=grid_gdf,
        target_col="relative_demand_score",
        group_col="ward_id"
    )

    # -------------------------------------------------------------------------
    # PART C: DEMAND SEGMENTATION (K-MEANS)
    # -------------------------------------------------------------------------
    segmenter = DemandSegmentationEngine(n_clusters=4, random_state=42)
    grid_gdf = segmenter.segment_neighbourhoods(grid_gdf, demand_col="relative_demand_score")

    # -------------------------------------------------------------------------
    # PART D: SHAP EXPLAINABILITY
    # -------------------------------------------------------------------------
    best_model = benchmark.fitted_models.get("XGBoost Regressor", benchmark.fitted_models.get("Random Forest Regressor"))
    explainer = DemandExplainer(best_model, benchmark.feature_cols)
    shap_results = explainer.explain_global(grid_gdf)

    # Ward-level aggregation
    ward_agg = grid_gdf.groupby("ward_id").agg({
        "ward_name": "first",
        "population": "sum",
        "relative_demand_score": "mean",
        "multimodal_accessibility_score": "mean",
        "commercial_count": "sum",
        "total_transit_stops": "sum"
    }).reset_index()
    ward_agg["relative_demand_score"] = ward_agg["relative_demand_score"].round(1)
    ward_agg["multimodal_accessibility_score"] = ward_agg["multimodal_accessibility_score"].round(1)

    # -------------------------------------------------------------------------
    # PART E: VISUALIZATIONS & REPORTING
    # -------------------------------------------------------------------------
    visualizer = DemandVisualizer(MAPS_DIR)

    # 1. Demand Heatmap
    visualizer.create_demand_folium_heatmap(grid_gdf, "demand_heatmap.html")

    # 2. Ward-Wise Demand Bar Chart
    visualizer.create_ward_demand_barchart(ward_agg, "ward_demand_chart.html")

    # 3. Demand Clusters Scatter Plot
    visualizer.create_demand_clusters_scatter(grid_gdf, "demand_clusters_scatter.html")

    # 4. Unmet Demand Matrix (Demand vs Accessibility)
    visualizer.create_unmet_demand_matrix(grid_gdf, "unmet_demand_matrix.html")

    # Save enriched demand grid to disk
    out_parquet = Path("data/features/mumbai_demand_grid.parquet")
    grid_gdf.to_parquet(out_parquet, index=False)
    out_geojson = Path("data/features/mumbai_demand_grid.geojson")
    grid_gdf.to_file(out_geojson, driver="GeoJSON")
    logger.info(f"Saved demand grid to: {out_parquet.name} and {out_geojson.name}")

    # Save ward summary
    out_ward_summary = PROCESSED_INTEGRATED / "ward_demand_summary.parquet"
    ward_agg.to_parquet(out_ward_summary, index=False)

    # Generate Markdown Report
    md_report_path = REPORTS_DIR / "demand_intelligence_report.md"
    _generate_markdown_report(ward_agg, df_benchmark, shap_results, md_report_path)
    logger.info(f"Saved markdown report to: {md_report_path.name}")

    elapsed = round(time.time() - start_time, 2)
    logger.info("================================================================================")
    logger.info(f"   DEMAND INTELLIGENCE PIPELINE COMPLETED IN {elapsed} SECONDS                 ")
    logger.info(f"   Best Model: {df_benchmark.iloc[0]['Model']} (R²: {df_benchmark.iloc[0]['R-Squared']:.3f}, MAE: {df_benchmark.iloc[0]['MAE']})")
    logger.info("================================================================================")

    return {
        "status": "SUCCESS",
        "elapsed_seconds": elapsed,
        "best_model": str(df_benchmark.iloc[0]["Model"]),
        "best_r2": float(df_benchmark.iloc[0]["R-Squared"]),
        "best_mae": float(df_benchmark.iloc[0]["MAE"]),
        "mean_relative_demand": float(grid_gdf["relative_demand_score"].mean()),
        "demand_tiers": grid_gdf["demand_tier"].value_counts().to_dict(),
        "benchmark_table": df_benchmark.to_dict(orient="records"),
        "demand_grid_path": str(out_parquet),
        "markdown_report_path": str(md_report_path)
    }


def _generate_markdown_report(
    df_wards: pd.DataFrame,
    df_benchmark: pd.DataFrame,
    shap_info: Dict[str, Any],
    out_path: Path
) -> None:
    """Generates human-readable markdown demand intelligence report."""
    md_lines = [
        "# UrbanFlow AI - Mobility Demand Intelligence Report",
        "",
        "**Geographical Scope:** Greater Mumbai (MCGM), Maharashtra, India  ",
        "**Problem Statement:** DataNova 2026, DN-DS-01  ",
        "**Evaluation Integrity:** GroupKFold Spatially-Aware Cross-Validation by Ward (Zero Location Leakage)  ",
        "**Measurement Specification:** Estimated Relative Demand Score (0 to 100)  ",
        "",
        "---",
        "",
        "## 1. Machine Learning Benchmark Comparison",
        "",
        "| Model | MAE | RMSE | R-Squared | Validation Strategy |",
        "|---|---|---|---|---|"
    ]

    for _, r in df_benchmark.iterrows():
        md_lines.append(f"| **{r['Model']}** | {r['MAE']:.2f} | {r['RMSE']:.2f} | **{r['R-Squared']:.3f}** | {r['Spatial CV Strategy']} |")

    md_lines.extend([
        "",
        "---",
        "",
        "## 2. SHAP Global Feature Importance Ranking",
        "",
        "| Rank | Feature | Mean Absolute SHAP | Relative Contribution (%) |",
        "|---|---|---|---|"
    ])

    for rank, entry in enumerate(shap_info["ranking"], start=1):
        md_lines.append(f"| {rank} | `{entry['feature']}` | {entry['mean_abs_shap']:.4f} | **{float(entry['relative_contribution_pct']):.1f}%** |")

    md_lines.extend([
        "",
        "---",
        "",
        "## 3. Ward Relative Demand Ranking & Unmet Need Diagnostics",
        "",
        "| Rank | Ward ID | Ward Name | Population | Relative Demand (0-100) | Accessibility Score (0-100) | Priority Intervention Status |",
        "|---|---|---|---|---|---|---|"
    ])

    df_sorted = df_wards.sort_values(by="relative_demand_score", ascending=False).reset_index(drop=True)
    for rank, (_, r) in enumerate(df_sorted.iterrows(), start=1):
        d_val = r["relative_demand_score"]
        a_val = r["multimodal_accessibility_score"]
        if d_val >= 25.0 and a_val < 25.0:
            status = "🚨 **Acute Unmet Need (High Demand, Low Access)**"
        elif d_val >= 25.0 and a_val >= 25.0:
            status = "🟢 Well-Serviced Hub"
        elif d_val < 25.0 and a_val < 25.0:
            status = "⚪ Low Density Buffer"
        else:
            status = "🔵 Transit Surplus Node"

        md_lines.append(
            f"| {rank} | `{r['ward_id']}` | {r['ward_name']} | {int(r['population']):,} | **{d_val:.1f}** | {a_val:.1f} | {status} |"
        )

    md_lines.extend([
        "",
        "---",
        "",
        "## 4. Methodological & Ethical Notes",
        "- **Data Integrity Standard:** Actual passenger tap-in ridership is not publicly published for Mumbai; all outputs represent *Estimated Relative Demand* derived from verified spatial attractors and demographic mass.",
        "- **Spatial Leakage Prevention:** Machine learning regressors were cross-validated using GroupKFold grouped strictly by administrative ward boundaries to guarantee that models generalize across unseen geography.",
        "- **Actionable Transit Optimization:** Identified acute unmet demand clusters directly feed candidate stop selection for Genetic Algorithm (GA) and Particle Swarm Optimization (PSO) modules."
    ])

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))


def main():
    """CLI Entrypoint."""
    res = run_demand_intelligence_pipeline()
    print("\n=== DEMAND PIPELINE EXECUTION RESULTS ===")
    for k, v in res.items():
        if k != "benchmark_table":
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
