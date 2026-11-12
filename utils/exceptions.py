"""
UrbanFlow AI - Domain-Specific Exception Hierarchy
Structured exception classes for data ingestion, spatial analysis, optimization, and simulation.
"""


class UrbanFlowError(Exception):
    """Base exception for all UrbanFlow AI errors."""
    pass


class ConfigurationError(UrbanFlowError):
    """Raised when application or city configuration is invalid or missing."""
    pass


class SpatialDataError(UrbanFlowError):
    """Base exception for geospatial data handling errors."""
    pass


class CRSMismatchError(SpatialDataError):
    """Raised when spatial datasets have incompatible or unprojected Coordinate Reference Systems."""
    def __init__(self, expected: str, actual: str, operation: str = "metric distance calculation"):
        message = (
            f"CRS Mismatch during {operation}: Expected projected CRS '{expected}' "
            f"for metric calculations, but received '{actual}'. "
            f"Use GeoDataFrame.to_crs('{expected}') before performing distance-based spatial operations."
        )
        super().__init__(message)


class DataIngestionError(UrbanFlowError):
    """Raised when external data sources (GTFS, OSM, Census) fail to download or parse."""
    pass


class OptimizationError(UrbanFlowError):
    """Base exception for algorithmic optimization failures."""
    pass


class InfeasibleSolutionError(OptimizationError):
    """Raised when an optimization model cannot satisfy its specified constraints."""
    pass


class ConvergenceError(OptimizationError):
    """Raised when a metaheuristic (GA, PSO) fails to converge within specified iterations."""
    pass


class SimulationError(UrbanFlowError):
    """Raised when what-if scenario simulations encounter invalid parameters or state."""
    pass


class DatabaseError(UrbanFlowError):
    """Raised during SQLite / PostGIS query or persistence failures."""
    pass

