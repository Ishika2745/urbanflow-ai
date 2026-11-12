"""
UrbanFlow AI - Master Multimodal Transportation & Accessibility Analysis Pipeline
Executes Parts A through E:
- Part A: Pedestrian Network Graph Construction & Connectivity Validation
- Part B: Multimodal Transportation Graph (BEST Bus, Suburban Rail, Metro Rail, Verified Transfers)
- Part C: Dijkstra Network Walking Distance & Travel Time Calculations (400m, 800m, 1000m)
- Part D: Normalized Multimodal Accessibility Scoring (Proximity, Opportunities, Diversity, Frequency)
- Part E: Output Generation (Heatmaps, Transport Maps, Walking Catchments, Nearest-Stop Charts, Ward Report)

Execute via:
    python -m analytics.run_accessibility
"""

import time
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import geopandas as gpd

from data_ingestion.config import (
    PROCESSED_TRANSPORT,
    PROCESSED_BOUNDARIES,
    PROCESSED_INTEGRATED,
    DISPLAY_CRS
)
from preprocessing.coordinate_manager import CoordinateManager
from analytics.pedestrian_network import PedestrianNetworkBuilder
from analytics.multimodal_network import MultimodalNetworkBuilder
from analytics.accessibility_engine import MultimodalAccessibilityEngine
from analytics.network_visualizer import NetworkVisualizer
from utils.logger import logger

REPORTS_DIR = Path("reports")
MAPS_DIR = REPORTS_DIR / "maps"


def run_full_accessibility_pipeline() -> Dict[str, Any]:
    """Runs the full multimodal transportation network and accessibility analysis engine."""
    start_time = time.time()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MAPS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("================================================================================")
    logger.info("   URBANFLOW AI - MULTIMODAL NETWORK & ACCESSIBILITY ANALYSIS PIPELINE          ")
    logger.info("================================================================================")

    # -------------------------------------------------------------------------
    # PART A: PEDESTRIAN NETWORK
    # -------------------------------------------------------------------------
    ped_builder = PedestrianNetworkBuilder()
    G_ped, ped_connectivity = ped_builder.build_or_load_pedestrian_network(force_rebuild=False)

    # -------------------------------------------------------------------------
    # PART B: MULTIMODAL TRANSPORT NETWORK
    # -------------------------------------------------------------------------
    multi_builder = MultimodalNetworkBuilder()
    G_multi, multi_summary = multi_builder.build_multimodal_graph(G_ped, max_snap_distance_m=1200.0)

    # -------------------------------------------------------------------------
    # PART C & D: ACCESSIBILITY ROUTING & SCORING
    # -------------------------------------------------------------------------
    engine = MultimodalAccessibilityEngine(G_multi)

    # Load 1km grid feature set
    grid_path = Path("data/features/mumbai_grid_1km.parquet")
    if not grid_path.exists():
        grid_path = Path("data/features/mumbai_grid_1km.geojson")
    grid_1km = gpd.read_parquet(grid_path) if grid_path.suffix == ".parquet" else gpd.read_file(grid_path)

    # Load wards
    wards_path = Path("data/features/mumbai_ward_features.geojson")
    wards_gdf = gpd.read_file(wards_path)

    # Compute network accessibility across 1km grid
    grid_accessibility = engine.compute_network_accessibility(
        origins_gdf=grid_1km,
        thresholds=[400.0, 800.0, 1000.0],
        max_search_dist_m=2500.0
    )

    # Generate Ward-Level Accessibility Report
    df_ward_report = engine.generate_ward_accessibility_report(grid_accessibility, wards_gdf)

    # -------------------------------------------------------------------------
    # PART E: OUTPUT GENERATION & VISUALIZATION
    # -------------------------------------------------------------------------
    logger.info("Generating publication-quality maps and analytical diagnostic artifacts...")

    # Load transit stops for mapping
    best_stops = gpd.read_file(PROCESSED_TRANSPORT / "best_stops.geojson")
    rail_stations = gpd.read_file(PROCESSED_TRANSPORT / "railway_stations.geojson")
    metro_stations = gpd.read_file(PROCESSED_TRANSPORT / "metro_stations.geojson")

    visualizer = NetworkVisualizer(MAPS_DIR)

    # 1. Accessibility Heatmap
    visualizer.create_accessibility_folium_heatmap(grid_accessibility, "accessibility_heatmap.html")

    # 2. Transport Network Map
    visualizer.create_multimodal_network_map(rail_stations, metro_stations, best_stops, "transport_network_map.html")

    # 3. Walking Catchment Map
    major_interchanges = rail_stations[rail_stations["is_interchange"] == True].copy()
    if major_interchanges.empty:
        major_interchanges = rail_stations.head(10)
    visualizer.create_walking_catchment_map(major_interchanges, "walking_catchments_map.html")

    # 4. Nearest-Stop Analysis
    visualizer.create_nearest_stop_distribution_chart(grid_accessibility, "nearest_stop_analysis.html")

    # 5. Export Reports and Feature Datasets
    out_parquet = Path("data/features/mumbai_accessibility_grid.parquet")
    grid_accessibility.to_parquet(out_parquet, index=False)
    logger.info(f"Saved accessibility grid to: {out_parquet.name}")

    out_ward_report = PROCESSED_INTEGRATED / "ward_accessibility_report.parquet"
    df_ward_report.to_parquet(out_ward_report, index=False)
    logger.info(f"Saved ward accessibility report to: {out_ward_report.name}")

    # Generate Markdown Report
    md_report_path = REPORTS_DIR / "accessibility_report.md"
    _generate_markdown_report(df_ward_report, ped_connectivity, multi_summary, md_report_path)
    logger.info(f"Saved markdown accessibility report to: {md_report_path.name}")

    elapsed = round(time.time() - start_time, 2)
    logger.info("================================================================================")
    logger.info(f"   ACCESSIBILITY PIPELINE COMPLETED IN {elapsed} SECONDS                       ")
    logger.info(f"   Mean Multimodal Accessibility Score: {grid_accessibility['multimodal_accessibility_score'].mean():.2f}/100")
    logger.info("================================================================================")

    return {
        "status": "SUCCESS",
        "elapsed_seconds": elapsed,
        "pedestrian_nodes": ped_connectivity["total_pedestrian_nodes"],
        "multimodal_nodes": multi_summary["total_multimodal_nodes"],
        "multimodal_edges": multi_summary["total_multimodal_edges"],
        "mean_multimodal_accessibility_score": float(grid_accessibility["multimodal_accessibility_score"].mean()),
        "mean_walk_time_transit_mins": float(grid_accessibility["walk_time_transit_mins"].mean()),
        "ward_report_path": str(out_ward_report),
        "markdown_report_path": str(md_report_path)
    }


def _generate_markdown_report(
    df_report: pd.DataFrame,
    ped_info: Dict[str, Any],
    multi_info: Dict[str, Any],
    out_path: Path
) -> None:
    """Generates human-readable markdown accessibility audit report."""
    md_lines = [
        "# UrbanFlow AI - Multimodal Transportation & Accessibility Intelligence Report",
        "",
        "**Geographical Scope:** Greater Mumbai (MCGM), Maharashtra, India  ",
        "**Problem Statement:** DataNova 2026, DN-DS-01  ",
        f"**Walking Speed Standard:** {ped_info['walking_speed_assumed_kmh']} km/h ({ped_info['walking_speed_mps']} m/s)  ",
        f"**Pedestrian Connectivity:** {ped_info['largest_component_coverage_pct']}% of nodes in main component ({ped_info['total_connected_components']} components)  ",
        f"**Integrated Network Infrastructure:** {multi_info['total_multimodal_nodes']} nodes, {multi_info['total_multimodal_edges']} edges ({multi_info['verified_transfer_edges']} verified transfers)  ",
        "",
        "---",
        "",
        "## Ward Accessibility Ranking & Diagnostics",
        "",
        "| Rank | Ward | Name | Population | Grade | Multimodal Score | Mean Transit Walk (mins) | Bus Walk (m) | Rail Walk (m) | Metro Walk (m) | Pop within 800m (%) |",
        "|---|---|---|---|---|---|---|---|---|---|---|"
    ]

    for rank, (_, r) in enumerate(df_report.iterrows(), start=1):
        md_lines.append(
            f"| {rank} | `{r['ward_id']}` | {r['ward_name']} | {r['population']:,} | {r['accessibility_grade'].split(' ')[0]} | "
            f"**{r['mean_multimodal_score']:.1f}** | {r['mean_walk_time_transit_mins']}m | {r['mean_walk_dist_bus_m']:.0f}m | "
            f"{r['mean_walk_dist_rail_m']:.0f}m | {r['mean_walk_dist_metro_m']:.0f}m | {r['pop_within_800m_walk_pct']}% |"
        )

    md_lines.extend([
        "",
        "---",
        "",
        "## Methodological & Behavioral Notes",
        "- **Walking-Speed Standard:** Evaluated strictly using 4.5 km/h (1.25 m/s) urban standard.",
        "- **Transfer Integrity:** Transfers modeled solely between verified physical interchanges (Dadar, Kurla, Andheri, Ghatkopar, CSMT, Churchgate, Bandra, etc.) with 3-minute platform transfer penalty.",
        "- **Scheduled Frequency Verification:** High-frequency service claims require verified GTFS timetable data (>120 daily scheduled trips); unverified services receive no frequency bonus.",
        "- **Honest Network Reporting:** Disconnected coastal or peripheral nodes are flagged with `disconnected_pedestrian_flag` rather than assuming artificial connections."
    ])

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))


def main():
    """CLI Entrypoint."""
    res = run_full_accessibility_pipeline()
    print("\n=== PIPELINE EXECUTION COMPLETED ===")
    for k, v in res.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

