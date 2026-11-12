"""
UrbanFlow AI - Data Ingestion Configuration
Defines directory hierarchies, CRS specifications, Mumbai study boundaries, and public data source registries.
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Tuple

# Base Project Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Standard Data Subdirectories (as required by Prompt 2)
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
METADATA_DIR = DATA_DIR / "metadata"
TEMPORARY_DIR = DATA_DIR / "temporary"

# Raw category folders
RAW_BEST = RAW_DIR / "best"
RAW_RAILWAY = RAW_DIR / "railway"
RAW_GTFS = RAW_DIR / "gtfs"
RAW_POPULATION = RAW_DIR / "population"
RAW_OSM = RAW_DIR / "osm"
RAW_BOUNDARIES = RAW_DIR / "boundaries"
RAW_SUPPLEMENTARY = RAW_DIR / "supplementary"

# Interim folders
INTERIM_CLEANED = INTERIM_DIR / "cleaned"
INTERIM_VALIDATED = INTERIM_DIR / "validated"
INTERIM_TRANSFORMED = INTERIM_DIR / "transformed"

# Processed output folders
PROCESSED_TRANSPORT = PROCESSED_DIR / "transport"
PROCESSED_POPULATION = PROCESSED_DIR / "population"
PROCESSED_INFRASTRUCTURE = PROCESSED_DIR / "infrastructure"
PROCESSED_BOUNDARIES = PROCESSED_DIR / "boundaries"
PROCESSED_INTEGRATED = PROCESSED_DIR / "integrated"

# Metadata tracking paths
DATA_CATALOGUE_PATH = METADATA_DIR / "data_catalogue.json"
DOWNLOAD_LOG_PATH = METADATA_DIR / "download_log.json"
VALIDATION_REPORT_PATH = METADATA_DIR / "validation_report.json"
SOURCE_MANIFEST_PATH = METADATA_DIR / "source_manifest.json"

# Geographical Bounding Box: Greater Mumbai (WGS84 decimal degrees)
MUMBAI_BBOX: Tuple[float, float, float, float] = (72.7750, 18.8900, 72.9980, 19.3200) # (min_lon, min_lat, max_lon, max_lat)
MUMBAI_CENTER: Tuple[float, float] = (19.0760, 72.8777) # (lat, lon)

# CRS Specifications
METRIC_CRS: str = "EPSG:32643" # UTM Zone 43N (Metric distances and areas)
DISPLAY_CRS: str = "EPSG:4326"  # WGS84 Geographic (Web maps and GeoJSON)

# Network & Request Configuration
DEFAULT_TIMEOUT_SEC: int = 30
MAX_DOWNLOAD_RETRIES: int = 3
BACKOFF_FACTOR: float = 1.5
HTTP_HEADERS: Dict[str, str] = {
    "User-Agent": "UrbanFlow-AI-DataNova-2026/1.0 (+https://github.com/urbanflow-ai)"
}

# Primary Public Data Sources Registry
DATA_SOURCES = {
    "best_bus": {
        "name": "BEST Bus Stops and Depots Data",
        "category": "Bus Transit",
        "primary_url": "https://data.opencity.in/dataset/mumbai-best-bus-stops-and-depots-data",
        "staging_url": "https://staging.opencity.in/dataset/mumbai-best-bus-stops-and-depots-data",
        "direct_download_url": "https://data.opencity.in/dataset/c2a2de1e-a017-4fb9-9e46-de6fbfbb393b/resource/514eadc2-7fa8-4749-87e5-ebe486487f09/download/b34111da-5b73-45e9-82d0-78da9d153b1c.kml",
        "format": "KML",
        "organization": "OpenCity / BEST",
        "license": "Open Data Commons / CC-BY"
    },
    "suburban_railway": {
        "name": "Mumbai Suburban Network 2025",
        "category": "Rail Transit",
        "primary_url": "https://data.opencity.in/dataset/mumbai-suburban-network-2025",
        "stations_kml_url": "https://data.opencity.in/dataset/50851056-1954-459f-a747-09366dcd416d/resource/142fbd25-9ceb-41b2-900d-72479cdc179e/download/5baa7e83-2855-4d17-86a9-9954b044ec08.kml",
        "lines_kml_url": "https://data.opencity.in/dataset/50851056-1954-459f-a747-09366dcd416d/resource/21a0809e-b279-42af-9bac-6ee73ec4e711/download/46f53a20-aebb-4096-bf33-c6d9d87afaca.kml",
        "format": "KML",
        "organization": "OpenCity / Indian Railways",
        "license": "Open Data Commons / CC-BY"
    },
    "mumbai_gtfs": {
        "name": "Mumbai GTFS Transit Feeds (BEST, TMT, KDMT)",
        "category": "Scheduled GTFS",
        "repository_url": "https://github.com/croyla/mumbai-gtfs",
        "master_zip_url": "https://github.com/croyla/mumbai-gtfs/archive/refs/heads/master.zip",
        "format": "ZIP",
        "organization": "croyla / Transit Open Data Contributors",
        "license": "MIT"
    },
    "worldpop": {
        "name": "WorldPop Population Count Raster",
        "category": "Demographics",
        "api_documentation": "https://api.worldpop.org/v3/documentation",
        "raster_1km_url": "https://data.worldpop.org/GIS/Population/Global_2000_2020_1km/2020/IND/ind_ppp_2020_1km_Aggregated.tif",
        "raster_100m_url": "https://data.worldpop.org/GIS/Population/Global_2000_2020/2020/IND/ind_ppp_2020.tif",
        "format": "GeoTIFF",
        "reference_year": "2020",
        "organization": "WorldPop / University of Southampton",
        "license": "CC-BY 4.0"
    },
    "mumbai_wards": {
        "name": "Mumbai BMC Ward Boundaries Map",
        "category": "Administrative Boundaries",
        "primary_url": "https://data.opencity.in/dataset/mumbai-wards-map",
        "wards_kml_url": "https://data.opencity.in/dataset/60f27371-717e-40b6-8554-fdc7e6c10437/resource/0318c3e8-1530-4bf4-b29b-7281573dee8a/download/e7a671e2-1f71-4219-a83c-556334bc9021.kml",
        "ward_offices_kml_url": "https://data.opencity.in/dataset/60f27371-717e-40b6-8554-fdc7e6c10437/resource/e3156762-a378-445f-85ad-1f9afb6cf31d/download/ec63d011-7ea2-4e54-8451-7ef93773d92e.kml",
        "format": "KML",
        "organization": "OpenCity / Municipal Corporation of Greater Mumbai (MCGM)",
        "license": "Open Data Commons / CC-BY"
    },
    "supplementary_census": {
        "name": "Mumbai Ward-Wise Primary Census Abstract",
        "category": "Demographics",
        "primary_url": "https://data.opencity.in/dataset/mumbai-ward-wise-census-data",
        "csv_url": "https://data.opencity.in/dataset/1a81291d-369d-43e6-97b1-e62680434437/resource/fcfeacdc-aa80-4028-a468-f0267fa05d23/download/95e22d97-7f59-4214-b244-2abbf52e6027.csv",
        "format": "CSV",
        "organization": "OpenCity / Census of India 2011",
        "license": "Open Government Data (OGD) India"
    }
}

