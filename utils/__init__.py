"""
UrbanFlow AI Utilities Package
Provides logging, exception handling, geospatial transforms, and validators.
"""

from utils.logger import logger, setup_logger
from utils.exceptions import (
    UrbanFlowError,
    ConfigurationError,
    SpatialDataError,
    CRSMismatchError,
    DataIngestionError,
    OptimizationError,
    InfeasibleSolutionError,
    ConvergenceError,
    SimulationError,
    DatabaseError
)
from utils.geo_utils import (
    ensure_metric_crs,
    ensure_display_crs,
    create_catchment_buffers,
    calculate_euclidean_distance_matrix,
    bbox_to_polygon
)
from utils.validators import (
    validate_dataframe_columns,
    validate_coordinates,
    validate_geodataframe_crs
)

__all__ = [
    "logger",
    "setup_logger",
    "UrbanFlowError",
    "ConfigurationError",
    "SpatialDataError",
    "CRSMismatchError",
    "DataIngestionError",
    "OptimizationError",
    "InfeasibleSolutionError",
    "ConvergenceError",
    "SimulationError",
    "DatabaseError",
    "ensure_metric_crs",
    "ensure_display_crs",
    "create_catchment_buffers",
    "calculate_euclidean_distance_matrix",
    "bbox_to_polygon",
    "validate_dataframe_columns",
    "validate_coordinates",
    "validate_geodataframe_crs"
]

