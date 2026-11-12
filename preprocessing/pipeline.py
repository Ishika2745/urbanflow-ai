"""
UrbanFlow AI - Master Preprocessing & Geospatial Feature Engineering Pipeline
Executes Parts A through E:
- Part A: Data Cleaning, Deduplication, Topological Repair, GTFS Validation
- Part B: Dual-CRS Metric Management (EPSG:32643 Planar & EPSG:4326 Display)
- Part C: Configurable Multi-Resolution Spatial Grids (500m, 1km, Ward-level)
- Part D: Geospatial Feature Engineering (Transit proximity, 400m/800m walksheds, GTFS frequency, densities, deficit scores)
- Part E: Data Quality & Health Reporting + Multi-Format Spatial Export (GeoParquet, GeoPackage, GeoJSON)

Can be executed directly via:
    python -m preprocessing.pipeline
"""

import time
from typing import Dict, Any
import numpy as np
import geopandas as gpd

from data_ingestion.config import (
    PROCESSED_TRANSPORT,
    PROCESSED_INFRASTRUCTURE,
    PROCESSED_BOUNDARIES,
    PROCESSED_POPULATION,
    PROCESSED_INTEGRATED,
    METADATA_DIR
)
from preprocessing.coordinate_manager import CoordinateManager
from preprocessing.data_cleaner import DataCleaner
from preprocessing.spatial_grid import SpatialGridGenerator
from preprocessing.feature_engineer import FeatureEngineer
from preprocessing.quality_reporter import QualityReporter
from preprocessing.exporter import SpatialExporter, FEATURES_DIR
from utils.logger import logger


class PreprocessingPipeline:
    """Master orchestrator for spatial data cleaning, grid generation, and feature engineering."""

    def __init__(self):
        self.cleaner = DataCleaner()
        self.grid_gen = SpatialGridGenerator()
        self.engineer = FeatureEngineer()
        self.reporter = QualityReporter()
        self.exporter = SpatialExporter(FEATURES_DIR)

    def run_pipeline(self) -> Dict[str, Any]:
        """Executes the full end-to-end preprocessing sequence."""
        start_time = time.time()
        logger.info("================================================================================")
        logger.info("   URBANFLOW AI - PREPROCESSING & GEOSPATIAL FEATURE ENGINEERING PIPELINE       ")
        logger.info("================================================================================")

        # -------------------------------------------------------------------------
        # PART A: DATA CLEANING & VALIDATION
        # -------------------------------------------------------------------------
        clean_summary = self.cleaner.clean_all()

        # Load cleaned layers
        wards_gdf = gpd.read_file(self.cleaner.interim_dir / "cleaned_ward_boundaries.geojson")
        perim_gdf = gpd.read_file(self.cleaner.interim_dir / "cleaned_greater_mumbai_boundary.geojson")
        best_stops = gpd.read_file(self.cleaner.interim_dir / "cleaned_best_stops.geojson")
        rail_stations = gpd.read_file(self.cleaner.interim_dir / "cleaned_railway_stations.geojson")
        metro_stations = gpd.read_file(self.cleaner.interim_dir / "cleaned_metro_stations.geojson")
        gtfs_stops = gpd.read_file(self.cleaner.interim_dir / "cleaned_gtfs_stops.geojson")
        facilities = gpd.read_file(self.cleaner.interim_dir / "cleaned_urban_facilities.geojson")
        roads = gpd.read_file(self.cleaner.interim_dir / "cleaned_roads.geojson")
        intersections = gpd.read_file(self.cleaner.interim_dir / "cleaned_road_intersections.geojson")

        # -------------------------------------------------------------------------
        # PART C: SPATIAL GRID GENERATION & CONSERVATIVE POPULATION INTERPOLATION
        # -------------------------------------------------------------------------
        # 1. 1-Kilometre Grid
        grid_1km = self.grid_gen.create_grid(resolution_meters=1000, perimeter_gdf=perim_gdf, prefix="GRID_1KM")
        grid_1km = self.grid_gen.allocate_population_areal(grid_1km, id_col="cell_id")
        grid_1km = self.grid_gen.assign_ward_membership(grid_1km, wards_gdf)

        # 2. 500-Metre Grid
        grid_500m = self.grid_gen.create_grid(resolution_meters=500, perimeter_gdf=perim_gdf, prefix="GRID_500M")
        grid_500m = self.grid_gen.allocate_population_areal(grid_500m, id_col="cell_id")
        grid_500m = self.grid_gen.assign_ward_membership(grid_500m, wards_gdf)

        # 3. Ward-Level Analytical Boundaries
        wards_analytical = wards_gdf.copy()
        wards_analytical = self.grid_gen.allocate_population_areal(wards_analytical, id_col="ward_id")

        # -------------------------------------------------------------------------
        # PART D: FEATURE ENGINEERING
        # -------------------------------------------------------------------------
        logger.info("Computing rich geospatial feature matrices across 1km grid, 500m grid, and wards...")

        # Feature engineering on 1km grid
        grid_1km_features = self.engineer.engineer_all_features(
            target_gdf=grid_1km,
            best_stops=best_stops,
            rail_stations=rail_stations,
            metro_stations=metro_stations,
            gtfs_stops=gtfs_stops,
            roads=roads,
            intersections=intersections,
            facilities=facilities,
            id_col="cell_id"
        )

        # Feature engineering on 500m grid
        grid_500m_features = self.engineer.engineer_all_features(
            target_gdf=grid_500m,
            best_stops=best_stops,
            rail_stations=rail_stations,
            metro_stations=metro_stations,
            gtfs_stops=gtfs_stops,
            roads=roads,
            intersections=intersections,
            facilities=facilities,
            id_col="cell_id"
        )

        # Feature engineering on Ward Level
        ward_features = self.engineer.engineer_all_features(
            target_gdf=wards_analytical,
            best_stops=best_stops,
            rail_stations=rail_stations,
            metro_stations=metro_stations,
            gtfs_stops=gtfs_stops,
            roads=roads,
            intersections=intersections,
            facilities=facilities,
            id_col="ward_id"
        )

        # Add backward-compatible ward fields for existing analytics
        ward_features["total_stops"] = ward_features["total_transit_stops"]
        ward_features["rail_metro_stops"] = ward_features["rail_stations_count"] + ward_features["metro_stations_count"]
        ward_features["bus_stops"] = ward_features["bus_stops_count"]
        ward_features["stop_density_per_sq_km"] = (ward_features["total_stops"] / ward_features["area_sq_km"]).round(2)
        ward_features["stops_per_10k_population"] = (
            (ward_features["total_stops"] / np.maximum(ward_features["population"], 1.0)) * 10000.0
        ).round(2)
        ward_features["accessibility_score"] = ward_features["transport_supply_score"]
        ward_features["vulnerability_index"] = (ward_features["transit_desert_index"] / 100.0).round(3)
        ward_features["slum_population_pct"] = np.clip(ward_features["vulnerability_index"] * 0.7, 0.1, 0.75).round(3)

        # Apply PySAL Local Moran's I (LISA) spatial autocorrelation
        from analytics.spatial_clustering import SpatialClusterAnalyzer
        ward_features = SpatialClusterAnalyzer.detect_transit_deserts(ward_features)

        # -------------------------------------------------------------------------
        # PART E: QUALITY REPORTING & MULTI-FORMAT EXPORT
        # -------------------------------------------------------------------------
        logger.info("Exporting multi-resolution features to GeoParquet, GeoPackage, and GeoJSON...")

        # Export 1km grid
        self.exporter.export_layer(grid_1km_features, "mumbai_grid_1km")

        # Export 500m grid
        self.exporter.export_layer(grid_500m_features, "mumbai_grid_500m")

        # Export Ward Features (both to features/ and processed/integrated/)
        self.exporter.export_layer(ward_features, "mumbai_ward_features")
        ward_features_clean = CoordinateManager.to_display(CoordinateManager.strip_z(ward_features))
        ward_features_clean.to_file(PROCESSED_INTEGRATED / "ward_accessibility_profiles.geojson", driver="GeoJSON")

        # Export Cleaned Transit Stops & Facilities to processed
        self.exporter.export_layer(best_stops, "cleaned_best_stops")
        self.exporter.export_layer(rail_stations, "cleaned_railway_stations")
        self.exporter.export_layer(metro_stations, "cleaned_metro_stations")
        self.exporter.export_layer(facilities, "cleaned_urban_facilities")

        # Compile Quality Report
        quality_report = self.reporter.generate_quality_report(
            datasets={
                "mumbai_grid_500m": grid_500m_features,
                "mumbai_grid_1km": grid_1km_features,
                "mumbai_ward_features": ward_features,
                "best_bus_stops": best_stops,
                "railway_stations": rail_stations,
                "metro_stations": metro_stations,
                "gtfs_stops": gtfs_stops,
                "urban_facilities": facilities,
                "road_network": roads
            },
            cleaning_audit=clean_summary["audit_records"]
        )

        elapsed = round(time.time() - start_time, 2)
        logger.info("================================================================================")
        logger.info(f"   PREPROCESSING & FEATURE ENGINEERING FINISHED IN {elapsed} SECONDS          ")
        logger.info(f"   500m Cells: {len(grid_500m_features)} | 1km Cells: {len(grid_1km_features)} | Wards: {len(ward_features)}")
        logger.info("================================================================================")

        return {
            "status": "SUCCESS",
            "elapsed_seconds": elapsed,
            "grid_500m_cells": len(grid_500m_features),
            "grid_1km_cells": len(grid_1km_features),
            "wards_count": len(ward_features),
            "total_population_allocated": float(grid_1km_features["population"].sum()),
            "quality_report_path": str(METADATA_DIR / "data_quality_report.json")
        }


def main():
    """CLI execution entrypoint."""
    pipeline = PreprocessingPipeline()
    res = pipeline.run_pipeline()
    print("\n=== PIPELINE EXECUTION RESULT ===")
    for k, v in res.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
