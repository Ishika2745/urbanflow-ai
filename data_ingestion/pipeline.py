"""
UrbanFlow AI - Central Automated Real-World Data Acquisition Pipeline
Command-line and programmatic orchestrator for Mumbai transit, population, and infrastructure ingestion.

Usage:
    python -m data_ingestion.pipeline
    python -m data_ingestion.pipeline --force
    python -m data_ingestion.pipeline --dataset best
"""

import sys
import argparse
import time
from typing import Dict, Any, Optional

from data_ingestion.config import (
    DATA_SOURCES,
    PROCESSED_DIR,
    METADATA_DIR
)
from data_ingestion.best_loader import BESTDataLoader
from data_ingestion.railway_loader import RailwayDataLoader
from data_ingestion.gtfs_loader import GTFSDataLoader
from data_ingestion.worldpop_loader import WorldPopDataLoader
from data_ingestion.osm_loader import OSMInfrastructureLoader
from data_ingestion.boundary_loader import BoundaryDataLoader
from data_ingestion.supplementary_loader import SupplementaryDataLoader
from data_ingestion.spatial_processor import SpatialIntegrator
from data_ingestion.catalogue import DataCatalogueManager
from data_ingestion.logging_config import logger


class DataAcquisitionPipeline:
    """Master acquisition and processing pipeline for Greater Mumbai."""

    def __init__(self, force_download: bool = False):
        self.force_download = force_download
        self.catalogue_mgr = DataCatalogueManager()
        self.summaries: Dict[str, Any] = {}

    def run_all(self) -> Dict[str, Any]:
        """Executes acquisition and validation across all primary and supplementary datasets."""
        start_time = time.time()
        logger.info("================================================================================")
        logger.info("  🚀 STARTING URBANFLOW AI REAL-WORLD DATA ACQUISITION & INGESTION PIPELINE")
        logger.info("  Scope: Greater Mumbai, Maharashtra, India (EPSG:32643 / EPSG:4326)")
        logger.info("================================================================================")

        # 1. Dataset F: Boundaries (Wards & Greater Mumbai Perimeter)
        logger.info("\n--- Phase 1: Acquiring Administrative Boundaries ---")
        b_loader = BoundaryDataLoader()
        valid_wards_gdf, boundary_gdf, b_summary = b_loader.acquire_and_process(self.force_download)
        self.summaries["boundaries"] = b_summary

        # 2. Dataset A: BEST Bus Stops & Depots
        logger.info("\n--- Phase 2: Acquiring BEST Bus Stops & Depots ---")
        best_loader = BESTDataLoader()
        best_gdf, best_summary = best_loader.acquire_and_process(self.force_download)
        self.summaries["best"] = best_summary

        # 3. Dataset B: Mumbai Suburban Railway Network 2025
        logger.info("\n--- Phase 3: Acquiring Mumbai Suburban Railway Network 2025 ---")
        rail_loader = RailwayDataLoader()
        rail_stations_gdf, rail_lines_gdf, rail_summary = rail_loader.acquire_and_process(self.force_download)
        self.summaries["railway"] = rail_summary

        # 4. Dataset C: Mumbai GTFS Bus Feeds (BEST, TMT, KDMT)
        logger.info("\n--- Phase 4: Ingesting Mumbai GTFS Bus Schedules ---")
        gtfs_loader = GTFSDataLoader()
        gtfs_stops_gdf, gtfs_routes_gdf, gtfs_summary = gtfs_loader.acquire_and_process(self.force_download)
        self.summaries["gtfs"] = gtfs_summary

        # 5. Dataset D: WorldPop Population Count Raster
        logger.info("\n--- Phase 5: Ingesting & Clipping WorldPop Population Raster ---")
        wpop_loader = WorldPopDataLoader()
        clipped_tif, pop_grid_gdf, wpop_summary = wpop_loader.acquire_and_process(
            use_1km=True,
            force_download=self.force_download,
            boundary_gdf=valid_wards_gdf
        )
        self.summaries["worldpop"] = wpop_summary

        # 6. Dataset E: OpenStreetMap Road & Walkable Pedestrian Infrastructure
        logger.info("\n--- Phase 6: Ingesting OSM Road & Pedestrian Network ---")
        osm_loader = OSMInfrastructureLoader()
        walk_graph, roads_gdf, facilities_gdf, osm_summary = osm_loader.acquire_and_process(force_query=self.force_download)
        self.summaries["osm"] = osm_summary

        # 7. Supplementary Datasets: Census 2011 PCA Demographics
        logger.info("\n--- Phase 7: Ingesting Supplementary Census Demographics ---")
        supp_loader = SupplementaryDataLoader()
        census_df, census_summary = supp_loader.acquire_and_process(self.force_download)
        self.summaries["census"] = census_summary

        # 8. Spatial Integration & Metric Projections
        logger.info("\n--- Phase 8: Performing Spatial Integration & Metric Projections ---")
        integrator = SpatialIntegrator()
        integrated_stops, integrated_wards, integ_summary = integrator.integrate_all(
            best_stops_gdf=best_gdf,
            railway_stations_gdf=rail_stations_gdf,
            wards_gdf=valid_wards_gdf,
            gtfs_stops_gdf=gtfs_stops_gdf,
            population_grid_gdf=pop_grid_gdf
        )
        self.summaries["integration"] = integ_summary

        # 9. Update Metadata Catalogue
        logger.info("\n--- Phase 9: Updating Central Data Catalogue & Audit Logs ---")
        self.catalogue_mgr.generate_full_catalogue(self.summaries)

        elapsed = time.time() - start_time
        logger.info("================================================================================")
        logger.info(f"  ✅ DATA ACQUISITION & INGESTION COMPLETED SUCCESSFULLY IN {elapsed:.1f} SECONDS")
        logger.info(f"  - Central Catalogue: {METADATA_DIR / 'data_catalogue.json'}")
        logger.info(f"  - Processed Datasets: {PROCESSED_DIR}")
        logger.info("================================================================================")
        return self.summaries


def main():
    """Command-line interface entry point."""
    parser = argparse.ArgumentParser(description="UrbanFlow AI - Automated Dataset Ingestion Pipeline")
    parser.add_argument("--force", action="store_true", help="Force redownload of all datasets even if cached")
    parser.add_argument("--dataset", type=str, choices=["best", "railway", "gtfs", "worldpop", "osm", "boundaries", "supplementary", "all"], default="all")
    args = parser.parse_args()

    pipeline = DataAcquisitionPipeline(force_download=args.force)
    pipeline.run_all()


if __name__ == "__main__":
    main()

