"""
UrbanFlow AI Configuration Package
Provides centralized settings, CRS specifications, and city definitions.
"""

from config.settings import settings
from config.city_config import CityConfig, BoundingBox, TransitModeConfig
from config.mumbai_config import get_mumbai_config, MUMBAI_BMC_WARDS, MUMBAI_TRANSIT_MODES


def get_city_config(city_name: str = "mumbai") -> CityConfig:
    """Factory function to retrieve CityConfig for a given city."""
    normalized = city_name.strip().lower()
    if normalized in ["mumbai", "greater mumbai", "bmc"]:
        return get_mumbai_config()
    else:
        raise ValueError(f"Unsupported city '{city_name}'. Currently configured: 'mumbai'.")


__all__ = [
    "settings",
    "CityConfig",
    "BoundingBox",
    "TransitModeConfig",
    "get_city_config",
    "get_mumbai_config",
    "MUMBAI_BMC_WARDS",
    "MUMBAI_TRANSIT_MODES"
]

