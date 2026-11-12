"""
UrbanFlow AI - Master Conventional Optimization & Infrastructure Recommendation Pipeline
Executes Parts A, B, C, and D:
- Part A: Feasible, infrastructure-connected candidate generation
- Part B: Conventional Optimization (Greedy Maximum Coverage, MCLP, P-Median)
- Part C: Population-weighted demand & DTII coverage maximization
- Part D: Comparative benchmark evaluation & artifact generation

Execute via:
    python -m optimization.run_conventional_optimization
"""

import time
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import geopandas as gpd
import folium

from config.settings import settings
from optimization.candidate_generation import CandidateGenerator, CANDIDATE_DISCLAIMER
from optimization.fitness_function import MultiObjectiveFitnessEvaluator, FITNESS_DISCLAIMER
from optimization.greedy_optimizer import GreedyOptimizer
from optimization.mclp_optimizer import MCLPOptimizer
from optimization.pmedian_optimizer import PMedianOptimizer
from optimization.optimization_evaluation import OptimizationBenchmarkSuite
from utils.geo_utils import ensure_display_crs
from utils.logger import logger

REPORTS_DIR = Path("reports")
MAPS_DIR = REPORTS_DIR / "maps"
PROCESSED_DIR = Path("data/processed")
FEATURES_DIR = Path("data/features")


def run_conventional_optimization_pipeline(
    budget_stops: int = 15,
    service_radius_m: float = 800.0,
    min_separation_m: float = 350.0
) -> Dict[str, Any]:
    """Master orchestrator for conventional infrastructure optimization."""
    start_time = time.time()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MAPS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("================================================================================")
    logger.info("   URBANFLOW AI - CONVENTIONAL TRANSPORT OPTIMIZATION ENGINE                    ")
    logger.info("================================================================================")

    # 1. Part A: Feasible Candidate Generation
    logger.info(">>> [Part A] Generating Feasible Infrastructure-Connected Candidates...")
    cands_gdf = CandidateGenerator.generate_feasible_candidates(
        min_dist_to_existing_m=min_separation_m,
        min_separation_candidates_m=400.0,
        target_candidate_count=60
    )

    # 2. Load demand wards and existing stops
    ward_path = FEATURES_DIR / "mumbai_transit_deserts_wards.parquet"
    if not ward_path.exists():
        ward_path = FEATURES_DIR / "mumbai_ward_features.parquet"
    wards_gdf = gpd.read_parquet(ward_path)

    stops_list = []
    for f in ["cleaned_best_stops.parquet", "cleaned_metro_stations.parquet", "cleaned_railway_stations.parquet"]:
        p = FEATURES_DIR / f
        if p.exists():
            stops_list.append(gpd.read_parquet(p))
    stops_gdf = pd.concat(stops_list, ignore_index=True) if stops_list else gpd.GeoDataFrame()

    # 3. Part B & C: Multi-Objective Evaluator Initialization
    logger.info(f">>> [Part B & C] Initializing Spatial Evaluator (Radius: {service_radius_m}m)...")
    evaluator = MultiObjectiveFitnessEvaluator(
        demand_wards_gdf=wards_gdf,
        candidate_stops_gdf=cands_gdf,
        existing_stops_gdf=stops_gdf,
        service_radius_meters=service_radius_m
    )

    # 4. Execute Conventional Benchmark Suite
    logger.info(f">>> [Part B] Solving Greedy, MCLP, and P-Median Optimizers (K={budget_stops})...")
    suite = OptimizationBenchmarkSuite(evaluator)
    
    # Run individual algorithms to collect full run details for mapping
    greedy = GreedyOptimizer(evaluator)
    r_greedy = greedy.optimize(budget_stops=budget_stops, min_separation_m=min_separation_m)

    mclp = MCLPOptimizer(evaluator)
    r_mclp = mclp.optimize(budget_stops=budget_stops, min_separation_m=min_separation_m)

    pmed = PMedianOptimizer(evaluator)
    r_pmed = pmed.optimize(budget_stops=budget_stops)

    raw_results = [r_greedy, r_mclp, r_pmed]
    df_summary = suite._compile_summary(raw_results, budget_stops)

    # 5. Store Selected Stop Locations & Artifacts
    logger.info(">>> [Part D] Persisting Selected Locations and Benchmark Results...")
    df_summary.to_parquet(PROCESSED_DIR / "conventional_optimization_results.parquet")
    df_summary.to_csv(PROCESSED_DIR / "conventional_optimization_results.csv", index=False)

    # Export selected stops per algorithm
    selected_stops_export = []
    for r in raw_results:
        algo = r["algorithm"]
        for stop_id in r["selected_stop_ids"]:
            cand_row = cands_gdf[cands_gdf["candidate_id"] == stop_id].iloc[0]
            selected_stops_export.append({
                "algorithm": algo,
                "candidate_id": stop_id,
                "candidate_name": cand_row.get("candidate_name"),
                "ward_id": cand_row.get("ward_id"),
                "latitude": cand_row.geometry.y,
                "longitude": cand_row.geometry.x,
                "estimated_cost_lakhs": cand_row.get("estimated_cost_lakhs"),
                "disclaimer": CANDIDATE_DISCLAIMER
            })
    pd.DataFrame(selected_stops_export).to_csv(PROCESSED_DIR / "selected_candidate_stops.csv", index=False)

    # 6. Generate Comparison Map
    logger.info(">>> [Part D] Generating Interactive Comparison Geospatial Map...")
    map_path = MAPS_DIR / "conventional_optimization_map.html"
    _generate_comparison_map(wards_gdf, cands_gdf, raw_results, map_path)

    # 7. Generate Evaluation Markdown Report
    logger.info(">>> [Part D] Compiling Markdown Evaluation Report...")
    report_path = REPORTS_DIR / "conventional_optimization_report.md"
    _generate_markdown_report(df_summary, raw_results, cands_gdf, report_path, budget_stops, service_radius_m)

    elapsed = time.time() - start_time
    logger.info(f"Conventional Optimization Pipeline completed in {elapsed:.2f} seconds.")

    return {
        "status": "success",
        "elapsed_seconds": round(elapsed, 2),
        "candidates_evaluated": len(cands_gdf),
        "budget_k": budget_stops,
        "summary": df_summary,
        "report_path": str(report_path),
        "map_path": str(map_path)
    }


def _generate_comparison_map(
    wards_gdf: gpd.GeoDataFrame,
    cands_gdf: gpd.GeoDataFrame,
    results: List[Dict[str, Any]],
    output_path: Path
) -> None:
    """Generates an interactive Folium map showing selected candidate locations per algorithm."""
    wards_disp = ensure_display_crs(wards_gdf.copy())
    cands_disp = ensure_display_crs(cands_gdf.copy())

    m = folium.Map(location=[19.0760, 72.8777], zoom_start=11, tiles="CartoDB positron")

    # Wards layer
    folium.GeoJson(
        wards_disp,
        name="BMC Administrative Wards",
        style_function=lambda x: {
            "fillColor": "#F0F0F0", "color": "#666666", "weight": 1.0, "fillOpacity": 0.25
        },
        tooltip=folium.GeoJsonTooltip(fields=["ward_id", "ward_name", "population"], aliases=["Ward:", "Name:", "Pop:"])
    ).add_to(m)

    # Candidate pool background layer
    pool_group = folium.FeatureGroup(name="Feasible Candidate Pool (Unselected)", show=False)
    for _, r in cands_disp.iterrows():
        folium.CircleMarker(
            location=[r.geometry.y, r.geometry.x],
            radius=4,
            color="#888888",
            fill=True,
            fill_color="#CCCCCC",
            fill_opacity=0.6,
            tooltip=f"{r.get('candidate_name')} (Cost: ₹{r.get('estimated_cost_lakhs')}L)"
        ).add_to(pool_group)
    pool_group.add_to(m)

    # Color palette for conventional algorithms
    algo_colors = {
        "Greedy Maximum Coverage": "#FF7F00",      # Orange
        "MCLP (OR-Tools Exact MILP)": "#E41A1C",    # Crimson Red
        "P-Median (OR-Tools Exact MILP)": "#377EB8" # Royal Blue
    }

    for r in results:
        algo = r["algorithm"]
        color = algo_colors.get(algo, "#984EA3")
        grp = folium.FeatureGroup(name=f"📍 {algo} (K={len(r['selected_stop_ids'])})")

        for stop_id in r["selected_stop_ids"]:
            cand = cands_disp[cands_disp["candidate_id"] == stop_id].iloc[0]
            c_lat, c_lon = cand.geometry.y, cand.geometry.x
            name = cand.get("candidate_name")
            cost = cand.get("estimated_cost_lakhs")

            popup_html = f"""
            <div style="font-family: Arial; min-width: 200px;">
                <h4 style="margin: 0; color: {color};">{name}</h4>
                <p style="margin: 4px 0; font-size: 12px;"><b>Algorithm:</b> {algo}</p>
                <p style="margin: 4px 0; font-size: 12px;"><b>Estimated Cost:</b> ₹{cost} Lakhs</p>
                <div style="background: #FFF3CD; padding: 4px; font-size: 10px; margin-top: 6px; border-left: 2px solid #856404;">
                    {CANDIDATE_DISCLAIMER}
                </div>
            </div>
            """

            folium.Marker(
                location=[c_lat, c_lon],
                icon=folium.Icon(color="red" if "MCLP" in algo else ("orange" if "Greedy" in algo else "blue"), icon="bus", prefix="fa"),
                popup=folium.Popup(popup_html, max_width=280),
                tooltip=f"{algo}: {name}"
            ).add_to(grp)

            # 800m catchment buffer
            folium.Circle(
                location=[c_lat, c_lon],
                radius=800,
                color=color,
                weight=1,
                fill=True,
                fill_color=color,
                fill_opacity=0.08
            ).add_to(grp)

        grp.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.save(str(output_path))


def _generate_markdown_report(
    df_summary: pd.DataFrame,
    results: List[Dict[str, Any]],
    cands_gdf: gpd.GeoDataFrame,
    output_path: Path,
    budget_stops: int,
    service_radius_m: float
) -> None:
    """Generates comparative evaluation markdown report."""
    summary_header = "| " + " | ".join(df_summary.columns) + " |"
    summary_sep = "|---" * len(df_summary.columns) + "|"
    summary_lines = [summary_header, summary_sep]
    for _, s_row in df_summary.iterrows():
        summary_lines.append("| " + " | ".join(str(val) for val in s_row.values) + " |")
    summary_md = "\n".join(summary_lines)

    lines = [
        "# UrbanFlow AI - Infrastructure Recommendation & Conventional Optimization Report",
        "",
        "**Problem Statement:** DataNova 2026, DN-DS-01 (Urban Mobility Pattern Analysis from Transit Data)  ",
        "**Geographical Scope:** Greater Mumbai (BMC / MCGM)  ",
        f"**Optimization Parameters:** Facility Budget K = {budget_stops}, Service Catchment Radius d_0 = {service_radius_m:.0f} meters  ",
        "",
        "> [!IMPORTANT]",
        f"> **Official Planning Disclaimer:** {CANDIDATE_DISCLAIMER}",
        "",
        "---",
        "",
        "## 1. Executive Summary & Comparative Benchmark",
        "",
        "This evaluation benchmarks the three canonical conventional facility location algorithms implemented in UrbanFlow AI:",
        "1. **Greedy Maximum Coverage:** Forward-selection heuristic maximizing marginal population-weighted demand.",
        "2. **Maximal Covering Location Problem (MCLP):** Exact Mixed-Integer Linear Program (MILP) solved via Google OR-Tools.",
        "3. **P-Median Location Optimization:** Exact MILP minimizing population-weighted pedestrian travel distance to nearest transit.",
        "",
        "### Part D: Standardized Metric Comparison Table",
        "",
        summary_md,
        "",
        "---",
        "",
        "## 2. In-Depth Algorithm Performance Analysis",
        ""
    ]

    for r in results:
        algo = r["algorithm"]
        feas = r.get("feasibility", {})
        lines.extend([
            f"### 📍 {algo}",
            f"- **Population Covered:** **{r['population_covered']:,}** ({r['coverage_pct']:.2f}% of city population)",
            f"- **Additional Population Benefiting:** **{r['additional_population_benefiting']:,}** previously unserved residents",
            f"- **Average Walking Distance:** **{r['avg_walking_distance_m']:.1f} m** (vs baseline {r['baseline_avg_walk_dist_m']:.1f} m)",
            f"- **DTII Inequity Reduction:** **{r['dtii_reduction_pct']:.2f}%** (Baseline DTII: {r['baseline_dtii']:.3f} $\\to$ {r['dtii_after']:.3f})",
            f"- **Gini Inequality Improvement:** **{r['gini_improvement_pct']:.2f}%**",
            f"- **Total Capital Cost:** **₹{r['total_cost_lakhs']:.1f} Lakhs**",
            f"- **Computational Runtime:** **{r['execution_time_sec']:.4f} seconds**",
            f"- **Feasibility Audit:** Connected to mapped road network: `{feas.get('connected_to_road_network')}`, Minimum separation rules satisfied: `{feas.get('is_feasible')}`",
            f"- **Selected Feeder Candidates:** `{', '.join(r['selected_stop_ids'])}`",
            ""
        ])

    lines.extend([
        "---",
        "",
        "## 3. Mathematical Formulations",
        "",
        "### Maximal Covering Location Problem (MCLP)",
        "$$\\text{Maximize } \\sum_{i=1}^I \\left( \\text{demand}_i \\cdot \\text{pop}_i \\cdot (1 + \\text{DTII}_i) \\right) \\cdot y_i$$",
        "$$\\text{subject to: } \\sum_{j \\in N_i} x_j \\ge y_i \\quad \\forall i, \\quad \\sum_{j=1}^J x_j \\le K, \\quad x_j, y_i \\in \\{0, 1\\}$$",
        "",
        "### P-Median Distance Minimization",
        "$$\\text{Minimize } \\sum_{i=1}^I \\sum_{j=1}^J \\text{pop}_i \\cdot d_{ij} \\cdot z_{ij}$$",
        "$$\\text{subject to: } \\sum_{j=1}^J z_{ij} = 1 \\quad \\forall i, \\quad z_{ij} \\le x_j \\quad \\forall i, j, \\quad \\sum_{j=1}^J x_j \\le K$$",
        "",
        "---",
        f"*{CANDIDATE_DISCLAIMER}*"
    ])

    output_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    run_conventional_optimization_pipeline()
