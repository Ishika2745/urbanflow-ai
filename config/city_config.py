"""
UrbanFlow AI - City Configuration Base Abstraction
Enables city-agnostic configuration so the system can be adapted to any metropolis.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


@dataclass(frozen=True)
class BoundingBox:
    """Geographic Bounding Box in WGS84 (min_lat, min_lon, max_lat, max_lon)."""
    min_lat: float
    min_lon: float
    max_lat: float
    max_lon: float

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """Returns (min_lon, min_lat, max_lon, max_lat) for Shapely/GeoPandas standard."""
        return (self.min_lon, self.min_lat, self.max_lon, self.max_lat)

    @property
    def center(self) -> Tuple[float, float]:
        """Returns (center_lat, center_lon)."""
        return ((self.min_lat + self.max_lat) / 2.0, (self.min_lon + self.max_lon) / 2.0)


@dataclass
class TransitModeConfig:
    """Configuration parameters for a distinct urban transit mode."""
    mode_id: str
    display_name: str
    color_hex: str
    average_speed_kmh: float
    stop_spacing_meters: float
    headway_peak_mins: float
    headway_offpeak_mins: float
    capacity_per_vehicle: int
    cost_per_km_inr: float
    is_rail: bool = False


@dataclass
class CityConfig:
    """
    Abstract City Configuration Specification.
    Any city (Mumbai, Bengaluru, Delhi, London, etc.) implements this schema.
    """
    city_name: str
    state_or_region: str
    country: str
    projected_crs: str      # e.g., 'EPSG:32643' for UTM Zone 43N (Metric operations)
    display_crs: str        # Standard WGS84 'EPSG:4326' (Map visualization)
    bbox: BoundingBox
    center_coords: Tuple[float, float]
    admin_level_name: str   # e.g., 'Ward', 'Borough', 'Zone'
    wards: List[str]
    transit_modes: Dict[str, TransitModeConfig]
    standard_walking_speed_kmh: float = 4.5
    feeder_service_radius_meters: float = 800.0
    metro_service_radius_meters: float = 1200.0
    bus_service_radius_meters: float = 500.0
    baseline_population: int = 12500000

