"""
UrbanFlow AI - Master Pipeline Controller & Orchestrator
Connects all 9 analytical, optimization, and simulation stages into a single unified execution pipeline.
DataNova 2026: Problem Statement DN-DS-01.

Stages Orchestrated:
1. Ingestion & Preprocessing (data cleaning, EPSG:32643 transformation, spatial grid generation)
2. Multimodal Transportation Network & Accessibility (Dijkstra network walk catchments)
3. Mobility Demand Intelligence (Spatially-aware regression & TreeSHAP attributions)
4. Dynamic Transit Inequality Index (DTII calculation & sensitivity analysis)
5. Transit Desert Detection & Spatial Hotspots (LISA, Getis-Ord Gi*, K-Means, DBSCAN)
6. Infrastructure Candidate Generation & Conventional Optimization (Greedy, MCLP, P-Median)
7. Nature-Inspired Transport Equity Optimization (Equity-GA & RK-DPSO multi-seed benchmarks)
8. Before-and-After Scenario Simulation (Catchment recalculation & socioeconomic ROI)
9. Database Synchronization (SQLite relational tables population)
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd

from utils.logger import logger


class MasterPipelineController:
    """Central orchestrator for the complete UrbanFlow AI analytical workflow."""

    STAGE_NAMES = [
        "preprocessing",
        "accessibility",
        "demand",
        "dtii",
        "transit_deserts",
        "conventional_opt",
        "nature_opt",
        "simulation",
        "database"
    ]

    STAGE_DESCRIPTIONS = {
        "preprocessing": "Data Cleaning, EPSG:32643 Metric CRS, and Spatial Grid Generation",
        "accessibility": "Multimodal Pedestrian Graph & Dijkstra Accessibility Scoring",
        "demand": "Mobility Demand Intelligence, XGBoost Benchmarking & TreeSHAP",
        "dtii": "Dynamic Transit Inequality Index (DTII) & Multi-Weight Sensitivity",
        "transit_deserts": "Transit Desert Hotspots, Local Moran's I & Spatial Clustering",
        "conventional_opt": "Candidate Stop Generation, Greedy, MCLP & P-Median Solvers",
        "nature_opt": "Nature-Inspired Solvers (Equity-GA & RK-DPSO) Benchmarks",
        "simulation": "Before-and-After Policy Impact Sandbox & Dynamic Catchment Recalculation",
        "database": "SQLite Relational Database Population & Data Synchronization"
    }

    def __init__(self, workspace_root: Optional[Path] = None):
        self.workspace_root = workspace_root or Path.cwd()
        self.reports_dir = self.workspace_root / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.reports_dir / "pipeline_execution_manifest.json"

    def run_stage(self, stage_name: str, **kwargs) -> Dict[str, Any]:
        """Executes a single specified pipeline stage."""
        stage_name = stage_name.lower().strip()
        if stage_name not in self.STAGE_NAMES:
            raise ValueError(f"Unknown pipeline stage '{stage_name}'. Valid stages: {self.STAGE_NAMES}")

        logger.info(f">>> [PIPELINE] Starting Stage: {stage_name.upper()} - {self.STAGE_DESCRIPTIONS[stage_name]}")
        start_time = time.time()
        result: Dict[str, Any] = {"stage": stage_name, "status": "PENDING"}

        try:
            if stage_name == "preprocessing":
                from preprocessing.pipeline import PreprocessingPipeline
                p = PreprocessingPipeline()
                out = p.run_pipeline()
                result["details"] = {"records": len(out) if isinstance(out, dict) else "SUCCESS"}

            elif stage_name == "accessibility":
                from analytics.run_accessibility import run_full_accessibility_pipeline
                out = run_full_accessibility_pipeline()
                result["details"] = out

            elif stage_name == "demand":
                from models.run_demand_intelligence import run_demand_intelligence_pipeline
                out = run_demand_intelligence_pipeline()
                result["details"] = out

            elif stage_name == "dtii":
                from analytics.run_dtii_pipeline import run_dtii_pipeline
                out = run_dtii_pipeline()
                result["details"] = out

            elif stage_name == "transit_deserts":
                from analytics.run_transit_desert_analysis import run_transit_desert_pipeline
                out = run_transit_desert_pipeline()
                result["details"] = out

            elif stage_name == "conventional_opt":
                from optimization.run_conventional_optimization import run_conventional_optimization_pipeline
                budget = kwargs.get("budget_stops", 15)
                radius = kwargs.get("service_radius_m", 800.0)
                sep = kwargs.get("min_separation_m", 350.0)
                out = run_conventional_optimization_pipeline(
                    budget_stops=budget,
                    service_radius_m=radius,
                    min_separation_m=sep
                )
                result["details"] = out

            elif stage_name == "nature_opt":
                from optimization.run_nature_inspired_optimization import run_nature_inspired_optimization_pipeline
                budget = kwargs.get("budget_stops", 15)
                radius = kwargs.get("service_radius_m", 800.0)
                sep = kwargs.get("min_separation_m", 300.0)
                seeds = kwargs.get("seeds", [42, 101, 777, 2024, 9999])
                out = run_nature_inspired_optimization_pipeline(
                    budget_stops=budget,
                    service_radius_m=radius,
                    min_separation_m=sep,
                    seeds=seeds
                )
                result["details"] = out

            elif stage_name == "simulation":
                from simulation.run_impact_simulator import run_impact_simulator_pipeline
                budget = kwargs.get("budget_stops", 15)
                radius = kwargs.get("service_radius_m", 800.0)
                out = run_impact_simulator_pipeline(
                    budget_stops=budget,
                    service_radius_m=radius
                )
                result["details"] = out

            elif stage_name == "database":
                from database.populate_db import populate_all_tables
                populate_all_tables()
                result["details"] = {"status": "ALL_TABLES_SYNCHRONIZED"}

            elapsed = round(time.time() - start_time, 2)
            result["status"] = "SUCCESS"
            result["elapsed_sec"] = elapsed
            logger.info(f">>> [PIPELINE] Stage {stage_name.upper()} completed in {elapsed}s.")

        except Exception as e:
            elapsed = round(time.time() - start_time, 2)
            result["status"] = "FAILED"
            result["elapsed_sec"] = elapsed
            result["error"] = str(e)
            logger.error(f">>> [PIPELINE] Stage {stage_name.upper()} FAILED after {elapsed}s: {e}", exc_info=True)
            raise

        return result

    def run_all(
        self,
        budget_stops: int = 15,
        service_radius_m: float = 800.0,
        seeds: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Executes all 9 stages in chronological end-to-end dependency order.
        Saves run summary manifest to reports/pipeline_execution_manifest.json.
        """
        total_start = time.time()
        timestamp = datetime.now().isoformat()
        seeds = seeds or [42, 101, 777, 2024, 9999]

        logger.info("================================================================================")
        logger.info("   URBANFLOW AI - MASTER END-TO-END PIPELINE EXECUTION STARTED                 ")
        logger.info(f"   Timestamp: {timestamp} | Budget Stops: {budget_stops} | Radius: {service_radius_m}m")
        logger.info("================================================================================")

        manifest: Dict[str, Any] = {
            "execution_id": f"pipeline_run_{int(total_start)}",
            "timestamp": timestamp,
            "configuration": {
                "budget_stops": budget_stops,
                "service_radius_m": service_radius_m,
                "seeds": seeds,
                "city": "Greater Mumbai",
                "crs": "EPSG:32643"
            },
            "stages": {},
            "total_runtime_sec": 0.0,
            "overall_status": "PENDING"
        }

        try:
            for stage in self.STAGE_NAMES:
                stage_kwargs = {}
                if stage in ("conventional_opt", "nature_opt", "simulation"):
                    stage_kwargs["budget_stops"] = budget_stops
                    stage_kwargs["service_radius_m"] = service_radius_m
                if stage == "nature_opt":
                    stage_kwargs["seeds"] = seeds

                res = self.run_stage(stage, **stage_kwargs)
                manifest["stages"][stage] = {
                    "status": res["status"],
                    "elapsed_sec": res["elapsed_sec"],
                    "description": self.STAGE_DESCRIPTIONS[stage]
                }

            manifest["overall_status"] = "SUCCESS"
            logger.info("================================================================================")
            logger.info("   MASTER PIPELINE EXECUTION COMPLETED SUCCESSFULLY (100% PASS RATE)           ")
            logger.info("================================================================================")

        except Exception as e:
            manifest["overall_status"] = "FAILED"
            manifest["fatal_error"] = str(e)
            logger.error(f"Master pipeline execution halted: {e}")
            raise
        finally:
            total_elapsed = round(time.time() - total_start, 2)
            manifest["total_runtime_sec"] = total_elapsed
            with open(self.manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
            logger.info(f"Execution manifest persisted to: {self.manifest_path} (Total Runtime: {total_elapsed}s)")

        return manifest


def run_full_pipeline(budget_stops: int = 15, service_radius_m: float = 800.0) -> Dict[str, Any]:
    """Convenience functional wrapper for executing the master pipeline."""
    controller = MasterPipelineController()
    return controller.run_all(budget_stops=budget_stops, service_radius_m=service_radius_m)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="UrbanFlow AI - Master Analytical Pipeline")
    parser.add_argument("--all", action="store_true", help="Execute complete 9-stage end-to-end pipeline")
    parser.add_argument("--stage", type=str, default=None, choices=MasterPipelineController.STAGE_NAMES, help="Execute a single stage")
    parser.add_argument("--budget", type=int, default=15, help="Budget stops (default: 15)")
    parser.add_argument("--radius", type=float, default=800.0, help="Walking service radius in meters (default: 800m)")
    args = parser.parse_args()

    controller = MasterPipelineController()
    if args.stage:
        controller.run_stage(args.stage, budget_stops=args.budget, service_radius_m=args.radius)
    else:
        controller.run_all(budget_stops=args.budget, service_radius_m=args.radius)

