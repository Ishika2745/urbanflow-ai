"""
Unit Tests for Configuration & CRS Management
"""

import pytest
from config.settings import settings
from config import get_city_config, get_mumbai_config


def test_mumbai_config_crs():
    """Verify Mumbai uses UTM Zone 43N (EPSG:32643) for metric operations and EPSG:4326 for display."""
    cfg = get_mumbai_config()
    assert cfg.projected_crs == "EPSG:32643"
    assert cfg.display_crs == "EPSG:4326"


def test_mumbai_wards_count():
    """Verify Mumbai configuration registers all 24 BMC administrative wards."""
    cfg = get_mumbai_config()
    assert len(cfg.wards) == 24
    assert any("M/E" in w for w in cfg.wards)


def test_city_config_factory():
    """Verify city factory correctly routes supported cities and handles unsupported."""
    cfg = get_city_config("mumbai")
    assert cfg.city_name == "Greater Mumbai"

    with pytest.raises(ValueError):
        get_city_config("non_existent_city")

