"""
Unit Tests for Data Acquisition, Parsers, and Validators
"""

import pytest
import geopandas as gpd
from shapely.geometry import Point, box
from pathlib import Path

from data_ingestion.config import MUMBAI_BBOX, DISPLAY_CRS, METRIC_CRS
from data_ingestion.format_parser import FormatParser
from data_ingestion.validator import DatasetValidator
from data_ingestion.catalogue import DataCatalogueManager


def test_spatial_validator_boundary_filtering(tmp_path):
    """Verify that records inside Mumbai boundary are retained and outside records are preserved separately."""
    validator = DatasetValidator(report_path=tmp_path / "test_val.json")

    # Point 1: Inside Mumbai (Colaba: 72.83, 18.93)
    # Point 2: Outside Mumbai (Pune: 73.85, 18.52)
    gdf = gpd.GeoDataFrame([
        {"stop_id": "S1", "stop_name": "Colaba Stop", "geometry": Point(72.83, 18.93)},
        {"stop_id": "S2", "stop_name": "Pune Stop", "geometry": Point(73.85, 18.52)}
    ], crs=DISPLAY_CRS)

    in_gdf, out_gdf, summary = validator.validate_spatial_dataset(
        gdf=gdf,
        dataset_name="Test_Stops",
        id_col="stop_id",
        name_col="stop_name",
        bbox=MUMBAI_BBOX
    )

    assert len(in_gdf) == 1
    assert in_gdf.iloc[0]["stop_id"] == "S1"
    assert len(out_gdf) == 1
    assert out_gdf.iloc[0]["stop_id"] == "S2"
    assert summary["outside_boundary_records"] == 1


def test_duplicate_detection(tmp_path):
    """Verify duplicate detection by ID and geometry."""
    validator = DatasetValidator(report_path=tmp_path / "test_val.json")

    gdf = gpd.GeoDataFrame([
        {"stop_id": "S1", "stop_name": "Dadar", "geometry": Point(72.84, 19.01)},
        {"stop_id": "S1", "stop_name": "Dadar Duplicate", "geometry": Point(72.84, 19.01)},
        {"stop_id": "S2", "stop_name": "Kurla", "geometry": Point(72.88, 19.06)}
    ], crs=DISPLAY_CRS)

    in_gdf, _, summary = validator.validate_spatial_dataset(
        gdf=gdf,
        dataset_name="Test_Dupes",
        id_col="stop_id"
    )

    assert len(in_gdf) == 2
    assert summary["duplicate_count"] == 1


def test_catalogue_manager(tmp_path):
    """Verify data catalogue reads and updates entries."""
    mgr = DataCatalogueManager(metadata_dir=tmp_path)
    mgr.update_catalogue_entry("test_dataset", {
        "dataset_name": "Test Dataset",
        "records_count": 100,
        "validation_status": "VALIDATED"
    })

    cat = mgr.get_catalogue()
    assert "test_dataset" in cat
    assert cat["test_dataset"]["records_count"] == 100

