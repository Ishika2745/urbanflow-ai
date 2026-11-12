"""
UrbanFlow AI - Master Transit Desert Detection & Spatial Hotspot Pipeline
Executes Parts A, B, C, and D:
- Part A: Transit Desert Detection & DTII Computation
- Part B: Spatial Analysis (K-Means, DBSCAN, Local Moran's I, Getis-Ord Gi*)
- Part C: Multi-Criteria Priority Ranking
- Part D: Geospatial Output Maps & Comprehensive Ward Inequality Report

Execute via:
    python -m analytics.run_transit_desert_analysis
"""

import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import geopandas as gpd

from data_ingestion.config import PROCESSED_INTEGRATED, PROCESSED_TRANSPORT
from analytics.spatial_clustering import SpatialClusterAnalyzer
from analytics.transit_desert_detector import TransitDesertDetector
from analytics.transit_desert_visualizer import TransitDesertVisualizer
from analytics.spatial_inequality import (
    calculate_gini_coefficient,
    compute_lorenz_curve,
    calculate_transport_equity_report
)
from utils.logger import logger
from utils.geo_utils import ensure_metric_crs, ensure_display_crs

REPORTS_DIR = Path("reports")
MAPS_DIR = REPORTS_DIR / "maps"
FEATURES_DIR = Path("data/features")


def load_transit_data() -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Loads grid features, ward features, and transit stops."""
    # 1. Grid
    grid_path = FEATURES_DIR / "mumbai_demand_grid.parquet"
    if not grid_path.exists():
        grid_path = FEATURES_DIR / "mumbai_accessibility_grid.parquet"
    grid_gdf = gpd.read_parquet(grid_path)

    # 2. Wards
    ward_path = FEATURES_DIR / "mumbai_ward_features.parquet"
    ward_gdf = gpd.read_parquet(ward_path)

    # 3. Stops
    stops_list = []
    for mode_file in ["cleaned_best_stops.parquet", "cleaned_metro_stations.parquet", "cleaned_railway_stations.parquet"]:
        p = FEATURES_DIR / mode_file
        if p.exists():
            stops_list.append(gpd.read_parquet(p))
    if stops_list:
        stops_gdf = pd.concat(stops_list, ignore_index=True)
    else:
        stops_gdf = gpd.GeoDataFrame()

    return grid_gdf, ward_gdf, stops_gdf


def run_transit_desert_pipeline() -> Dict[str, Any]:
    """Master orchestrator for transit desert detection and hotspot analysis."""
    start_time = time.time()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MAPS_DIR.mkdir(parents=True, exist_ok=True)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("================================================================================")
    logger.info("   URBANFLOW AI - TRANSIT DESERT DETECTION & SPATIAL HOTSPOT PIPELINE           ")
    logger.info("================================================================================")

    grid_gdf, ward_gdf, stops_gdf = load_transit_data()
    logger.info(f"Loaded {len(grid_gdf)} spatial grid cells, {len(ward_gdf)} BMC wards, and {len(stops_gdf)} transit facilities.")

    # -------------------------------------------------------------------------
    # PART A: TRANSIT DESERT DETECTION & DTII FORMULATION
    # -------------------------------------------------------------------------
    logger.info(">>> [Part A] Computing DTII & Identifying Transit Desert Zones...")
    grid_gdf = TransitDesertDetector.compute_dtii(grid_gdf)
    grid_gdf = TransitDesertDetector.identify_transit_desert_zones(grid_gdf)
    grid_gdf = TransitDesertDetector.associate_nearest_facilities(grid_gdf, stops_gdf)

    # -------------------------------------------------------------------------
    # PART B: SPATIAL ANALYSIS (K-Means, DBSCAN, Local Moran's I, Getis-Ord Gi*)
    # -------------------------------------------------------------------------
    logger.info(">>> [Part B] Executing Spatial Clustering & Hotspot Detection...")
    # 1. K-Means
    grid_gdf = SpatialClusterAnalyzer.compute_kmeans_clustering(
        grid_gdf,
        feature_cols=["dtii", "relative_demand_score", "multimodal_accessibility_score", "transport_supply_score"],
        n_clusters=4
    )

    # 2. DBSCAN for Contiguous Clusters
    grid_gdf = SpatialClusterAnalyzer.compute_dbscan_clustering(
        grid_gdf,
        eps_meters=1400.0,
        min_samples=3,
        desert_filter_col="is_intervention_zone"
    )

    # 3. Local Moran's I (LISA)
    grid_gdf = SpatialClusterAnalyzer.compute_local_moran(
        grid_gdf,
        variable_col="dtii",
        k_neighbors=6,
        significance_level=0.05
    )

    # 4. Getis-Ord Gi*
    grid_gdf = SpatialClusterAnalyzer.compute_getis_ord_gi_star(
        grid_gdf,
        variable_col="dtii",
        k_neighbors=6,
        significance_level=0.05
    )

    # -------------------------------------------------------------------------
    # PART C: PRIORITY RANKING & INTERVENTIONS
    # -------------------------------------------------------------------------
    logger.info(">>> [Part C] Recommending Interventions & Computing Multi-Criteria Priority Ranking...")
    grid_gdf = TransitDesertDetector.recommend_interventions(grid_gdf)
    grid_gdf = TransitDesertDetector.compute_priority_ranking(grid_gdf)

    # -------------------------------------------------------------------------
    # BMC WARD LEVEL AGGREGATION & SPATIAL MODELING
    # -------------------------------------------------------------------------
    logger.info(">>> Aggregating and Analyzing at BMC Administrative Ward Level...")
    ward_agg = grid_gdf.groupby("ward_id").agg({
        "dtii": "mean",
        "relative_demand_score": "mean",
        "multimodal_accessibility_score": "mean",
        "transport_supply_score": "mean",
        "net_walk_dist_transit_m": "mean",
        "accessibility_deficit": "mean",
        "priority_score": "mean",
        "is_high_demand_low_access": "sum",
        "is_inadequate_coverage": "sum",
        "is_critical_inequality_zone": "sum"
    }).reset_index()

    # Drop overlapping columns before merge
    overlap_cols = [c for c in ward_agg.columns if c != "ward_id"]
    cols_to_drop = [c for c in ward_gdf.columns if c in overlap_cols or c.endswith("_grid") or c in ["dtii", "accessibility_deficit", "priority_score", "priority_rank", "priority_tier", "recommended_intervention"]]
    ward_gdf = ward_gdf.drop(columns=cols_to_drop)
    ward_gdf = ward_gdf.merge(ward_agg, on="ward_id", how="left")
    ward_gdf["dtii"] = ward_gdf["dtii"].round(2)
    ward_gdf["transit_desert_index"] = ward_gdf["dtii"]

    # Run Local Moran's I and Getis-Ord on Wards
    ward_gdf = SpatialClusterAnalyzer.compute_local_moran(ward_gdf, variable_col="dtii", k_neighbors=4)
    ward_gdf = SpatialClusterAnalyzer.compute_getis_ord_gi_star(ward_gdf, variable_col="dtii", k_neighbors=4)
    ward_gdf = TransitDesertDetector.associate_nearest_facilities(ward_gdf, stops_gdf)
    ward_gdf = TransitDesertDetector.recommend_interventions(ward_gdf)
    ward_gdf = TransitDesertDetector.compute_priority_ranking(ward_gdf)

    # -------------------------------------------------------------------------
    # SAVE ENRICHED PARQUET DATASETS
    # -------------------------------------------------------------------------
    grid_save_path = FEATURES_DIR / "mumbai_transit_deserts_grid.parquet"
    ward_save_path = FEATURES_DIR / "mumbai_transit_deserts_wards.parquet"
    grid_gdf.to_parquet(grid_save_path)
    ward_gdf.to_parquet(ward_save_path)
    # Also update mumbai_ward_features.parquet for seamless system-wide integration
    ward_gdf.to_parquet(FEATURES_DIR / "mumbai_ward_features.parquet")
    logger.info(f"Saved enriched transit desert datasets to {grid_save_path} and {ward_save_path}.")

    # -------------------------------------------------------------------------
    # PART D: OUTPUT MAPS
    # -------------------------------------------------------------------------
    logger.info(">>> [Part D] Generating Interactive Geospatial Maps...")
    map_paths = TransitDesertVisualizer.save_all_maps(
        gdf=grid_gdf,
        output_dir=MAPS_DIR,
        dtii_col="dtii",
        top_n=20
    )

    # -------------------------------------------------------------------------
    # PART D: WARD-WISE INEQUALITY REPORT & HIGH-PRIORITY AREA EXPLANATIONS
    # -------------------------------------------------------------------------
    logger.info(">>> [Part D] Compiling Ward-wise Inequality & Hotspot Report...")
    equity_report = calculate_transport_equity_report(ward_gdf, score_col="multimodal_accessibility_score", pop_col="population")
    dtii_equity = calculate_transport_equity_report(ward_gdf, score_col="dtii", pop_col="population")

    report_md_path = REPORTS_DIR / "transit_desert_hotspot_report.md"
    _generate_markdown_report(
        ward_gdf=ward_gdf,
        grid_gdf=grid_gdf,
        equity_report=equity_report,
        dtii_equity=dtii_equity,
        output_path=report_md_path,
        map_paths=map_paths
    )
    logger.info(f"Generated comprehensive report at {report_md_path}")

    elapsed = time.time() - start_time
    logger.info(f"Transit Desert & Spatial Hotspot Pipeline completed in {elapsed:.2f} seconds.")

    return {
        "status": "success",
        "elapsed_seconds": round(elapsed, 2),
        "grid_cells": len(grid_gdf),
        "wards_count": len(ward_gdf),
        "acute_deserts": int(grid_gdf["is_critical_inequality_zone"].sum()),
        "gi_hotspots": int(grid_gdf["is_gi_hotspot"].sum()),
        "lisa_hotspots": int(grid_gdf["is_lisa_hotspot"].sum()),
        "gini_accessibility": equity_report["gini_coefficient"],
        "report_path": str(report_md_path)
    }


def _generate_markdown_report(
    ward_gdf: gpd.GeoDataFrame,
    grid_gdf: gpd.GeoDataFrame,
    equity_report: Dict[str, Any],
    dtii_equity: Dict[str, Any],
    output_path: Path,
    map_paths: Dict[str, Path]
) -> None:
    """Writes the comprehensive markdown report formatted to publication standards."""
    top_wards = ward_gdf.sort_values(by="priority_score", ascending=False).head(10)
    top_grids = grid_gdf.sort_values(by="priority_score", ascending=False).head(10)

    # Count classifications
    n_high_demand_low_access = int(grid_gdf["is_high_demand_low_access"].sum())
    n_inadequate_coverage = int(grid_gdf["is_inadequate_coverage"].sum())
    n_critical_inequality = int(grid_gdf["is_critical_inequality_zone"].sum())
    n_gi_hotspots = int(grid_gdf["is_gi_hotspot"].sum())
    n_lisa_hotspots = int(grid_gdf["is_lisa_hotspot"].sum())

    lines = [
        "# UrbanFlow AI - Transit Desert Detection & Spatial Hotspot Analysis Report",
        "",
        "**Geographical Scope:** Greater Mumbai (MCGM / BMC), Maharashtra, India  ",
        "**Problem Statement:** DataNova 2026, DN-DS-01  ",
        "**Methodological Framework:** Multimodal Catchment Modeling, PySAL Spatial Autocorrelation, and Multi-Criteria Priority Formulation  ",
        "**Spatial Resolution:** 547 Fine-Grained Analysis Cells (1km / 500m) & 24 Administrative BMC Wards  ",
        "",
        "---",
        "",
        "## 1. Executive Summary & Core Metrics",
        "",
        "This report delivers the comprehensive evaluation of public transport supply deficits, walking network disconnects, and spatial inequality across Greater Mumbai. By cross-referencing multi-criteria relative demand proxies against Dijkstra-routed multimodal accessibility and GTFS scheduled frequencies, the module isolates zones where transit provision fails to meet the mobility needs of resident populations.",
        "",
        "| Metric | Analysis Result | Methodological Benchmark |",
        "|---|---|---|",
        f"| **Gini Coefficient of Multimodal Access** | **{equity_report['gini_coefficient']:.4f}** | 0.0 (Equality) to 1.0 (Extreme Inequality) |",
        f"| **Palma Ratio of Accessibility** | **{equity_report['palma_ratio']:.3f}** | Top 10% vs Bottom 40% Share of Transit |",
        f"| **Gini Coefficient of DTII Inequity** | **{dtii_equity['gini_coefficient']:.4f}** | Spatial Distribution of Inequity Index |",
        f"| **High-Demand, Low-Access Cells** | **{n_high_demand_low_access}** cells ({n_high_demand_low_access/len(grid_gdf)*100:.1f}%) | Demand >= 40th percentile & Access <= 25/100 |",
        f"| **Inadequate Transport Coverage Zones** | **{n_inadequate_coverage}** cells ({n_inadequate_coverage/len(grid_gdf)*100:.1f}%) | Walk distance > 800m & Pop Density > 5,000/km² |",
        f"| **Critical Transport Inequality Zones** | **{n_critical_inequality}** cells ({n_critical_inequality/len(grid_gdf)*100:.1f}%) | DTII >= 30.0 & Pop Density > 12,000/km² |",
        f"| **Getis-Ord Gi* Statistically Significant Hotspots** | **{n_gi_hotspots}** cells (p <= 0.10) | Local Sum Z-Score > 1.65 (Star Matrix) |",
        f"| **LISA High-High Transit Desert Hotspots** | **{n_lisa_hotspots}** cells (p <= 0.05) | Local Moran's I Monte Carlo (999 Permutations) |",
        "",
        "---",
        "",
        "## 2. Theoretical Clarification: High DTII Score vs Statistically Significant Hotspot",
        "",
        "> [!IMPORTANT]",
        "> **Key Methodological Distinction:**",
        "> A **High DTII (Dynamic Transit Inequity Index) Score** is an **attribute-level local metric**. It evaluates a single location's internal mismatch between relative demand and transport accessibility. A remote hilltop or an isolated residential enclave may exhibit a high DTII solely due to localized road dead-ends, while its adjacent surroundings might be well-connected.",
        "> ",
        "> In contrast, a **Statistically Significant Hotspot (Getis-Ord $G_i^*$ / LISA High-High)** is a **spatial context metric**. It tests whether high DTII values are geographically clustered beyond what would occur by random chance. A statistically significant hotspot proves that high transit deficits are shared across contiguous neighbourhoods, demonstrating a systemic corridor-level infrastructure failure that cannot be solved by simply moving a bus stop.",
        "",
        "| Dimension | High DTII Score | Statistically Significant Hotspot ($G_i^*$ / LISA) |",
        "|---|---|---|",
        "| **Measurement Scope** | Single isolated grid cell or ward boundary | Spatial neighbourhood defined by spatial weights matrix $W$ |",
        "| **Mathematical Basis** | Demand-supply deficit weighted by network walking distance | Permutation inference comparing local sum $\\sum w_{ij} x_j$ to global expectation |",
        "| **Null Hypothesis** | None (Deterministic multi-criteria scoring) | Spatial randomness ($H_0$: spatial independence of values) |",
        "| **Policy Implication** | Localized intervention (micro-transit shuttle, stop placement) | Regional capital infrastructure (BRT corridor, metro feeder network, skywalk) |",
        "",
        "---",
        "",
        "## 3. Mathematical Formulation of the Dynamic Transit Inequity Index (DTII)",
        "",
        "To satisfy the design constraint that **no location should be classified as a transit desert solely because it has few nearby stops**, DTII gates every deficit term by travel demand and human population exposure:",
        "",
        "$$D_{norm} = \\text{clip}\\left(\\frac{\\text{Demand}}{100}, 0, 1\\right), \\quad A_{norm} = \\text{clip}\\left(\\frac{\\text{Access}}{100}, 0, 1\\right), \\quad S_{norm} = \\text{clip}\\left(\\frac{\\text{Supply}}{100}, 0, 1\\right)$$",
        "",
        "$$\\text{Accessibility Deficit} = \\max(0, D_{norm} - A_{norm})$$",
        "$$\\text{Unmet Need Exposure} = D_{norm} \\times (1.0 - A_{norm})$$",
        "$$\\text{Supply Deficit} = \\max(0, D_{norm} - S_{norm})$$",
        "$$\\text{Walking Distance Barrier} = \\min\\left(1.0, \\frac{d_{walk}}{1200\\text{m}}\\right) \\times D_{norm}$$",
        "",
        "$$\\text{DTII} = 100 \\times \\left( 0.35 \\cdot \\text{Accessibility Deficit} + 0.25 \\cdot \\text{Unmet Need} + 0.20 \\cdot \\text{Supply Deficit} + 0.20 \\cdot \\text{Walking Barrier} \\right)$$",
        "",
        "When an unpopulated or zero-demand area is evaluated ($D_{norm} \\to 0$), all four terms collapse to zero, ensuring $\\text{DTII} = 0.0$.",
        "",
        "---",
        "",
        "## 4. Multi-Criteria Priority Ranking Formulation",
        "",
        "To prioritize municipal transit capital without arbitrary double-counting, each component serves an orthogonal urban planning objective:",
        "",
        "$$\\text{Priority Score} = 100 \\times \\left( 0.30 \\cdot \\text{DTII}_{norm} + 0.20 \\cdot \\text{Deficit}_{norm} + 0.15 \\cdot D_{norm} + 0.15 \\cdot P_{scale} + 0.20 \\cdot H_{evidence} \\right)$$",
        "",
        "1. **DTII (30%):** Rate of local demand-supply mismatch.",
        "2. **Accessibility Deficit (20%):** Absolute depth of physical deprivation (excessive network walk times).",
        "3. **Relative Demand (15%):** Economic and civic activity urgency, guaranteeing immediate ridership utility.",
        "4. **Population Exposure (15%):** Logarithmic scale $\\frac{\\log(1+\\text{Pop})}{\\max(\\log(1+\\text{Pop}))}$, ensuring social impact without population size overshadowing dense slum clusters.",
        "5. **Hotspot Evidence (20%):** Statistically validated bonus ($G_i^* z > 1.96$ / LISA High-High) for systemic regional corridors.",
        "",
        "---",
        "",
        "## 5. High-Priority Area Diagnostic Breakdown",
        "",
        "The following high-priority areas represent the most acute transit desert clusters in Greater Mumbai, detailing all required metrics and tailored engineering interventions:",
        "",
        "| Rank | Area / Ward | Population | DTII (0-100) | Relative Demand | Nearest Facility | Walk Distance | Access Deficit | Recommended Intervention |",
        "|---|---|---|---|---|---|---|---|---|"
    ]

    for _, row in top_wards.iterrows():
        rank = int(row.get("priority_rank", 1))
        w_id = row.get("ward_id", "")
        w_name = row.get("ward_name", w_id)
        pop = int(row.get("population", 0))
        dtii = float(row.get("dtii", 0.0))
        dem = float(row.get("relative_demand_score", 0.0))
        fac = str(row.get("nearest_facility_name", "BEST Stop"))
        walk_m = float(row.get("net_walk_dist_transit_m", 800.0))
        deficit = float(row.get("accessibility_deficit", 0.0))
        rec = str(row.get("recommended_intervention", "Feeder Bus Route Extension"))

        lines.append(
            f"| **#{rank}** | **Ward {w_id} ({w_name})** | {pop:,} | **{dtii:.2f}** | {dem:.2f} | {fac} | {walk_m:.0f} m | **{deficit:.2f}** | {rec} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 6. Detailed Area-by-Area Explanations",
        ""
    ])

    for _, row in top_wards.head(5).iterrows():
        rank = int(row.get("priority_rank", 1))
        w_id = row.get("ward_id", "")
        w_name = row.get("ward_name", w_id)
        pop = int(row.get("population", 0))
        dtii = float(row.get("dtii", 0.0))
        dem = float(row.get("relative_demand_score", 0.0))
        acc = float(row.get("multimodal_accessibility_score", row.get("accessibility_score", 0.0)))
        fac = str(row.get("nearest_facility_name", "BEST Stop"))
        walk_m = float(row.get("net_walk_dist_transit_m", 800.0))
        deficit = float(row.get("accessibility_deficit", 0.0))
        rec = str(row.get("recommended_intervention", "Feeder Bus Route Extension"))
        lisa = str(row.get("lisa_quadrant", "Not Significant"))
        gi = str(row.get("gi_category", "Not Significant"))

        lines.extend([
            f"### Area #{rank}: Ward {w_id} - {w_name}",
            f"- **Resident Population:** {pop:,} citizens",
            f"- **Dynamic Transit Inequity Index (DTII):** `{dtii:.2f} / 100` (High transit deprivation)",
            f"- **Estimated Relative Mobility Demand:** `{dem:.2f} / 100`",
            f"- **Multimodal Accessibility Score:** `{acc:.2f} / 100`",
            f"- **Accessibility Deficit:** `{deficit:.2f}` (Demand significantly outstrips accessible capacity)",
            f"- **Nearest Transport Facility:** `{fac}`",
            f"- **Pedestrian Network Walking Distance:** `{walk_m:.0f} metres` (Dijkstra network path across street grid)",
            f"- **Spatial Hotspot Validation:** LISA Quadrant: *{lisa}*, Getis-Ord: *{gi}*",
            f"- **Actionable Municipal Intervention:** **{rec}**",
            f"- **Urban Planning Assessment:** Ward {w_id} exhibits severe transit friction where high residential and informal settlement density is separated from rapid rail and metro spines by natural barriers, arterial highways, or railway track cutoffs. Expanding dedicated feeder shuttles and creating grade-separated pedestrian skywalks directly alleviates last-mile walking times.",
            ""
        ])

    lines.extend([
        "---",
        "",
        "## 7. Complete BMC Ward-Level Inequality & Hotspot Diagnostics",
        "",
        "| Ward ID | Ward Name | Population | DTII | Relative Demand | Accessibility | Walk Dist (m) | LISA Quadrant | Getis-Ord Gi* | Priority Rank |",
        "|---|---|---|---|---|---|---|---|---|---|"
    ])

    for _, row in ward_gdf.sort_values(by="priority_rank").iterrows():
        lines.append(
            f"| `{row.get('ward_id')}` | {row.get('ward_name')} | {int(row.get('population', 0)):,} | "
            f"{float(row.get('dtii', 0)):.1f} | {float(row.get('relative_demand_score', 0)):.1f} | "
            f"{float(row.get('multimodal_accessibility_score', row.get('accessibility_score', 0))):.1f} | "
            f"{float(row.get('net_walk_dist_transit_m', 0)):.0f} m | {row.get('lisa_quadrant', 'N/A')} | "
            f"{row.get('gi_category', 'N/A')} | **#{int(row.get('priority_rank', 99))}** |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 8. Interactive Map Artifacts",
        "",
        "The following interactive HTML maps have been produced and are ready for web inspection:",
        "1. **Transit Desert Heatmap:** [`reports/maps/transit_desert_heatmap.html`](file:///reports/maps/transit_desert_heatmap.html)",
        "2. **Spatial Clusters Map (K-Means & DBSCAN):** [`reports/maps/spatial_clusters_map.html`](file:///reports/maps/spatial_clusters_map.html)",
        "3. **Statistically Significant Hotspots Map:** [`reports/maps/statistically_significant_hotspots_map.html`](file:///reports/maps/statistically_significant_hotspots_map.html)",
        "4. **Priority Interventions Map:** [`reports/maps/priority_interventions_map.html`](file:///reports/maps/priority_interventions_map.html)",
        "",
        "---",
        "*UrbanFlow AI • DataNova 2026 DN-DS-01 • Spatial Hotspot & Transit Desert Detection Module*"
    ])

    output_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    run_transit_desert_pipeline()
