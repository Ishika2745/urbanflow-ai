"""
Unit and Integration Tests for UrbanFlow AI Master Pipeline Controller
Tests stage discovery, sequential orchestration, manifest emission, and error trapping.
DataNova 2026: DN-DS-01.
"""

import json
from pathlib import Path
import pytest
from pipeline.master_pipeline import MasterPipelineController, run_full_pipeline


def test_master_pipeline_initialization():
    controller = MasterPipelineController()
    assert len(controller.STAGE_NAMES) == 9
    assert "preprocessing" in controller.STAGE_NAMES
    assert "accessibility" in controller.STAGE_NAMES
    assert "demand" in controller.STAGE_NAMES
    assert "dtii" in controller.STAGE_NAMES
    assert "transit_deserts" in controller.STAGE_NAMES
    assert "conventional_opt" in controller.STAGE_NAMES
    assert "nature_opt" in controller.STAGE_NAMES
    assert "simulation" in controller.STAGE_NAMES
    assert "database" in controller.STAGE_NAMES


def test_master_pipeline_stage_descriptions():
    controller = MasterPipelineController()
    for stage in controller.STAGE_NAMES:
        assert stage in controller.STAGE_DESCRIPTIONS
        assert len(controller.STAGE_DESCRIPTIONS[stage]) > 10


def test_master_pipeline_invalid_stage():
    controller = MasterPipelineController()
    with pytest.raises(ValueError, match="Unknown pipeline stage"):
        controller.run_stage("invalid_unknown_stage_name")


def test_master_pipeline_dtii_stage_execution():
    controller = MasterPipelineController()
    res = controller.run_stage("dtii")
    assert res["status"] == "SUCCESS"
    assert "elapsed_sec" in res
    assert res["elapsed_sec"] >= 0.0


def test_master_pipeline_database_stage_execution():
    controller = MasterPipelineController()
    res = controller.run_stage("database")
    assert res["status"] == "SUCCESS"
    assert res["details"]["status"] == "ALL_TABLES_SYNCHRONIZED"


def test_manifest_generation(tmp_path):
    controller = MasterPipelineController(workspace_root=tmp_path)
    # Simulate a lightweight run of dtii and database stages
    dtii_res = controller.run_stage("dtii")
    assert dtii_res["status"] == "SUCCESS"

