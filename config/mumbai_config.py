"""
UrbanFlow AI - Greater Mumbai (BMC) Configuration
Specialized instance of CityConfig for Mumbai's multimodal transit network.
"""

from typing import Dict, List
from config.city_config import CityConfig, BoundingBox, TransitModeConfig


MUMBAI_BMC_WARDS: List[str] = [
    "A (Colaba/Fort)",
    "B (Sandhurst Road)",
    "C (Marine Lines)",
    "D (Grant Road/Malabar Hill)",
    "E (Byculla)",
    "F/North (Matunga/Sion)",
    "F/South (Parel)",
    "G/North (Dharavi/Dadar West)",
    "G/South (Worli/Elphinstone)",
    "H/East (Bandra East/Santacruz)",
    "H/West (Bandra West/Khar)",
    "K/East (Andheri East/Jogeshwari)",
    "K/West (Andheri West/Versova)",
    "L (Kurla)",
    "M/East (Govandi/Mankhurd/Shivaji Nagar)",
    "M/West (Chembur)",
    "N (Ghatkopar)",
    "P/North (Malad)",
    "P/South (Goregaon)",
    "R/Central (Borivali)",
    "R/North (Dahisar)",
    "R/South (Kandivali)",
    "S (Bhandup/Powai)",
    "T (Mulund)"
]

MUMBAI_TRANSIT_MODES: Dict[str, TransitModeConfig] = {
    "suburban_western": TransitModeConfig(
        mode_id="suburban_western",
        display_name="Western Railway (Churchgate - Dahanu)",
        color_hex="#00529B",
        average_speed_kmh=42.0,
        stop_spacing_meters=1800.0,
        headway_peak_mins=3.5,
        headway_offpeak_mins=7.0,
        capacity_per_vehicle=4500,
        cost_per_km_inr=1.2,
        is_rail=True
    ),
    "suburban_central": TransitModeConfig(
        mode_id="suburban_central",
        display_name="Central Railway (CSMT - Kalyan)",
        color_hex="#C8102E",
        average_speed_kmh=38.0,
        stop_spacing_meters=2100.0,
        headway_peak_mins=4.0,
        headway_offpeak_mins=8.0,
        capacity_per_vehicle=4500,
        cost_per_km_inr=1.2,
        is_rail=True
    ),
    "suburban_harbour": TransitModeConfig(
        mode_id="suburban_harbour",
        display_name="Harbour Line (CSMT - Panvel / Goregaon)",
        color_hex="#28813F",
        average_speed_kmh=34.0,
        stop_spacing_meters=1600.0,
        headway_peak_mins=5.0,
        headway_offpeak_mins=10.0,
        capacity_per_vehicle=3600,
        cost_per_km_inr=1.2,
        is_rail=True
    ),
    "mumbai_metro": TransitModeConfig(
        mode_id="mumbai_metro",
        display_name="Mumbai Metro (Lines 1, 2A, 7, 3)",
        color_hex="#00A3E0",
        average_speed_kmh=33.0,
        stop_spacing_meters=1100.0,
        headway_peak_mins=4.0,
        headway_offpeak_mins=8.0,
        capacity_per_vehicle=1800,
        cost_per_km_inr=3.5,
        is_rail=True
    ),
    "best_bus": TransitModeConfig(
        mode_id="best_bus",
        display_name="BEST City Bus (Electric & Diesel Fleet)",
        color_hex="#E84A27",
        average_speed_kmh=16.5,
        stop_spacing_meters=450.0,
        headway_peak_mins=8.0,
        headway_offpeak_mins=18.0,
        capacity_per_vehicle=65,
        cost_per_km_inr=1.8,
        is_rail=False
    ),
    "micro_transit_feeder": TransitModeConfig(
        mode_id="micro_transit_feeder",
        display_name="Proposed Electric Mini-Feeder",
        color_hex="#00B140",
        average_speed_kmh=20.0,
        stop_spacing_meters=350.0,
        headway_peak_mins=5.0,
        headway_offpeak_mins=12.0,
        capacity_per_vehicle=25,
        cost_per_km_inr=2.2,
        is_rail=False
    )
}


def get_mumbai_config() -> CityConfig:
    """Returns the preconfigured Greater Mumbai CityConfig."""
    return CityConfig(
        city_name="Greater Mumbai",
        state_or_region="Maharashtra",
        country="India",
        # EPSG:32643 is UTM Zone 43N covering Mumbai's meridian (72-78 deg E)
        projected_crs="EPSG:32643",
        display_crs="EPSG:4326",
        bbox=BoundingBox(
            min_lat=18.8900,
            min_lon=72.7750,
            max_lat=19.3200,
            max_lon=72.9980
        ),
        center_coords=(19.0760, 72.8777),
        admin_level_name="BMC Ward",
        wards=MUMBAI_BMC_WARDS,
        transit_modes=MUMBAI_TRANSIT_MODES,
        standard_walking_speed_kmh=4.5,
        feeder_service_radius_meters=800.0,
        metro_service_radius_meters=1200.0,
        bus_service_radius_meters=500.0,
        baseline_population=12442373
    )

