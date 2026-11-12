"""
UrbanFlow AI - Automated Real-World Dataset Acquisition & Ingestion Package
"""

from data_ingestion.config import (
    DATA_DIR,
    RAW_DIR,
    INTERIM_DIR,
    PROCESSED_DIR,
    METADATA_DIR,
    MUMBAI_BBOX,
    METRIC_CRS,
    DISPLAY_CRS,
    DATA_SOURCES
)
from data_ingestion.downloader import ResilientDownloader
from data_ingestion.source_discovery import DatasetSourceDiscovery
from data_ingestion.file_extractor import ArchiveExtractor
from data_ingestion.format_parser import FormatParser
from data_ingestion.validator import DatasetValidator
from data_ingestion.best_loader import BESTDataLoader
from data_ingestion.railway_loader import RailwayDataLoader
from data_ingestion.gtfs_loader import GTFSDataLoader
from data_ingestion.worldpop_loader import WorldPopDataLoader
from data_ingestion.osm_loader import OSMInfrastructureLoader
from data_ingestion.boundary_loader import BoundaryDataLoader
from data_ingestion.supplementary_loader import SupplementaryDataLoader
from data_ingestion.spatial_processor import SpatialIntegrator
from data_ingestion.catalogue import DataCatalogueManager
from data_ingestion.pipeline import DataAcquisitionPipeline

__all__ = [
    "DATA_DIR",
    "RAW_DIR",
    "INTERIM_DIR",
    "PROCESSED_DIR",
    "METADATA_DIR",
    "MUMBAI_BBOX",
    "METRIC_CRS",
    "DISPLAY_CRS",
    "DATA_SOURCES",
    "ResilientDownloader",
    "DatasetSourceDiscovery",
    "ArchiveExtractor",
    "FormatParser",
    "DatasetValidator",
    "BESTDataLoader",
    "RailwayDataLoader",
    "GTFSDataLoader",
    "WorldPopDataLoader",
    "OSMInfrastructureLoader",
    "BoundaryDataLoader",
    "SupplementaryDataLoader",
    "SpatialIntegrator",
    "DataCatalogueManager",
    "DataAcquisitionPipeline"
]

