"""
UrbanFlow AI - Master Pipeline CLI Executable
DataNova 2026: Problem Statement DN-DS-01.

Usage:
    python run_pipeline.py --all
    python run_pipeline.py --stage [preprocessing|accessibility|demand|dtii|transit_deserts|conventional_opt|nature_opt|simulation|database]
    python run_pipeline.py --budget 20 --radius 800
"""

import sys
import argparse
from pipeline.master_pipeline import MasterPipelineController


def main():
    parser = argparse.ArgumentParser(
        description="UrbanFlow AI - Master Analytical Pipeline (Greater Mumbai, DN-DS-01)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        default=True,
        help="Execute complete 9-stage end-to-end pipeline (default: True)"
    )
    parser.add_argument(
        "--stage",
        type=str,
        default=None,
        choices=MasterPipelineController.STAGE_NAMES,
        help="Execute a specific stage only"
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=15,
        help="Proposed candidate stop budget K (default: 15)"
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=800.0,
        help="Pedestrian catchment radius in meters (default: 800.0m)"
    )

    args = parser.parse_args()
    controller = MasterPipelineController()

    if args.stage:
        print(f"Executing UrbanFlow AI Pipeline Stage: {args.stage}...")
        res = controller.run_stage(args.stage, budget_stops=args.budget, service_radius_m=args.radius)
        print(f"Stage {args.stage} finished with status: {res.get('status')} in {res.get('elapsed_sec')}s")
    else:
        print("Executing UrbanFlow AI Master End-to-End Pipeline (All 9 Stages)...")
        manifest = controller.run_all(budget_stops=args.budget, service_radius_m=args.radius)
        print(f"Master Pipeline Completed with Status: {manifest.get('overall_status')} in {manifest.get('total_runtime_sec')}s")


if __name__ == "__main__":
    main()

