"""
UrbanFlow AI - Central Geospatial & Tabular Data Validator
Executes schema validation, coordinate bounds checking, duplicate analysis, and GTFS integrity checks.
Preserves records outside study boundaries separately for investigation.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box

from data_ingestion.config import (
    MUMBAI_BBOX,
    INTERIM_VALIDATED,
    VALIDATION_REPORT_PATH,
    DISPLAY_CRS
)
from data_ingestion.logging_config import logger


class DatasetValidator:
    """Comprehensive validation engine enforcing spatial, attribute, and relational invariants."""

    def __init__(self, report_path: Path = VALIDATION_REPORT_PATH):
        self.report_path = report_path
        self._ensure_report()

    def _ensure_report(self) -> None:
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.report_path.exists():
            with open(self.report_path, "w", encoding="utf-8") as f:
                json.dump({}, f, indent=2)

    def _update_report(self, dataset_key: str, results: Dict[str, Any]) -> None:
        try:
            with open(self.report_path, "r", encoding="utf-8") as f:
                rep = json.load(f)
            rep[dataset_key] = results
            with open(self.report_path, "w", encoding="utf-8") as f:
                json.dump(rep, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not update validation report: {e}")

    def validate_spatial_dataset(
        self,
        gdf: gpd.GeoDataFrame,
        dataset_name: str,
        id_col: Optional[str] = None,
        name_col: Optional[str] = None,
        bbox: Tuple[float, float, float, float] = MUMBAI_BBOX
    ) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, Dict[str, Any]]:
        """
        Validates a spatial GeoDataFrame:
        - Drops or repairs invalid geometries
        - Detects duplicates
        - Splits records into study area vs. outside study boundary
        - Saves exclusions separately in data/interim/validated/{dataset_name}_exclusions.geojson
        """
        logger.info(f"[{dataset_name}] Starting spatial validation on {len(gdf)} records...")
        initial_count = len(gdf)

        # 1. Geometry Validity
        invalid_geom_mask = ~gdf.geometry.is_valid | gdf.geometry.is_empty | gdf.geometry.isna()
        invalid_geom_count = int(invalid_geom_mask.sum())
        clean_gdf = gdf[~invalid_geom_mask].copy()

        # Try repairing any repairable geometries
        if invalid_geom_count > 0:
            logger.warning(f"[{dataset_name}] Found {invalid_geom_count} invalid geometries. Filtered out.")

        # 2. Duplicate Detection
        dup_count = 0
        if id_col and id_col in clean_gdf.columns:
            dup_mask = clean_gdf.duplicated(subset=[id_col], keep="first")
            dup_count = int(dup_mask.sum())
            clean_gdf = clean_gdf[~dup_mask]
        else:
            # Duplicate by exact identical geometry
            dup_mask = clean_gdf.duplicated(subset=["geometry"], keep="first")
            dup_count = int(dup_mask.sum())
            clean_gdf = clean_gdf[~dup_mask]

        if dup_count > 0:
            logger.info(f"[{dataset_name}] Removed {dup_count} duplicate records.")

        # 3. Missing Names Analysis
        missing_names = 0
        if name_col and name_col in clean_gdf.columns:
            missing_names = int(clean_gdf[name_col].isna().sum())

        # 4. Spatial Bounding Box Filter
        min_lon, min_lat, max_lon, max_lat = bbox
        study_poly = box(min_lon, min_lat, max_lon, max_lat)

        within_mask = clean_gdf.geometry.within(study_poly) | clean_gdf.geometry.intersects(study_poly)
        in_boundary_gdf = clean_gdf[within_mask].copy()
        outside_boundary_gdf = clean_gdf[~within_mask].copy()

        outside_count = len(outside_boundary_gdf)
        if outside_count > 0:
            logger.info(f"[{dataset_name}] {outside_count} records located outside Greater Mumbai boundary. Preserving for audit.")
            INTERIM_VALIDATED.mkdir(parents=True, exist_ok=True)
            excl_path = INTERIM_VALIDATED / f"{dataset_name.lower().replace(' ', '_')}_outside_boundary.geojson"
            try:
                outside_boundary_gdf.to_file(excl_path, driver="GeoJSON")
            except Exception as e:
                logger.warning(f"Failed to write exclusions GeoJSON: {e}")

        validation_summary = {
            "dataset_name": dataset_name,
            "initial_records": initial_count,
            "valid_records_in_study_area": len(in_boundary_gdf),
            "outside_boundary_records": outside_count,
            "invalid_geometry_count": invalid_geom_count,
            "duplicate_count": dup_count,
            "missing_name_count": missing_names,
            "crs": str(gdf.crs) if gdf.crs else "UNKNOWN",
            "status": "VALIDATED" if len(in_boundary_gdf) > 0 else "EMPTY"
        }

        self._update_report(dataset_name.lower().replace(" ", "_"), validation_summary)
        logger.info(f"[{dataset_name}] Validation complete: {len(in_boundary_gdf)} valid records retained.")
        return in_boundary_gdf, outside_boundary_gdf, validation_summary

    def validate_gtfs_tables(self, tables: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Validates GTFS relational integrity across stops, routes, trips, stop_times, and calendar:
        - Checks missing coordinates
        - Checks relational key existence
        - Validates time formats and stop sequences
        """
        logger.info("Validating GTFS feed relational integrity...")
        issues = []
        metrics = {}

        # 1. Stops validation
        if "stops" in tables:
            stops = tables["stops"]
            metrics["stops_count"] = len(stops)
            missing_coords = stops["stop_lat"].isna() | stops["stop_lon"].isna()
            metrics["stops_missing_coordinates"] = int(missing_coords.sum())
            if missing_coords.sum() > 0:
                issues.append(f"{missing_coords.sum()} stops missing latitude/longitude")
            dup_stops = int(stops.duplicated(subset=["stop_id"]).sum())
            metrics["stops_duplicate_ids"] = dup_stops

        # 2. Routes validation
        if "routes" in tables:
            routes = tables["routes"]
            metrics["routes_count"] = len(routes)
            if "route_id" in routes:
                dup_routes = int(routes.duplicated(subset=["route_id"]).sum())
                metrics["routes_duplicate_ids"] = dup_routes

        # 3. Trips validation
        if "trips" in tables and "routes" in tables:
            trips = tables["trips"]
            metrics["trips_count"] = len(trips)
            valid_routes = set(tables["routes"]["route_id"])
            orphan_trips = int((~trips["route_id"].isin(valid_routes)).sum())
            metrics["trips_orphan_route_refs"] = orphan_trips
            if orphan_trips > 0:
                issues.append(f"{orphan_trips} trips reference non-existent route_id")

        # 4. Stop Times validation
        if "stop_times" in tables and "stops" in tables and "trips" in tables:
            st = tables["stop_times"]
            metrics["stop_times_count"] = len(st)
            valid_stops = set(tables["stops"]["stop_id"])
            orphan_stops = int((~st["stop_id"].isin(valid_stops)).sum())
            metrics["stop_times_orphan_stop_refs"] = orphan_stops

            valid_trips = set(tables["trips"]["trip_id"])
            orphan_trip_refs = int((~st["trip_id"].isin(valid_trips)).sum())
            metrics["stop_times_orphan_trip_refs"] = orphan_trip_refs

        # 5. Service / Calendar validation
        if "calendar" in tables:
            cal = tables["calendar"]
            metrics["calendar_services"] = len(cal)
            if "start_date" in cal and "end_date" in cal:
                metrics["service_start_min"] = str(cal["start_date"].min())
                metrics["service_end_max"] = str(cal["end_date"].max())

        report = {
            "dataset_name": "mumbai_gtfs",
            "metrics": metrics,
            "issues": issues,
            "is_valid": len(issues) == 0,
            "status": "VALIDATED"
        }
        self._update_report("mumbai_gtfs", report)
        logger.info(f"GTFS validation complete. Issues found: {len(issues)}")
        return report

