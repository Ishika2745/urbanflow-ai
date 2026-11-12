"""
UrbanFlow AI - Central Application Settings
Handles environment management, data directories, CRS definitions, and active city routing.
"""

import os
from pathlib import Path
from typing import Optional
# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"


class AppSettings:
    """Central Application and Pipeline Settings."""
    
    def __init__(self):
        # Environment & Logging
        self.ENV: str = os.getenv("URBANFLOW_ENV", "development")
        self.LOG_LEVEL: str = os.getenv("URBANFLOW_LOG_LEVEL", "INFO")
        self.CITY: str = os.getenv("URBANFLOW_CITY", "mumbai").lower()

        # Filesystem Directories
        self.BASE_DIR: Path = PROJECT_ROOT
        self.DATA_DIR: Path = Path(os.getenv("DATA_DIR", str(DEFAULT_DATA_DIR)))
        self.RAW_DATA_DIR: Path = self.DATA_DIR / "raw"
        self.PROCESSED_DATA_DIR: Path = self.DATA_DIR / "processed"
        self.FEATURES_DATA_DIR: Path = self.DATA_DIR / "features"
        self.REPORTS_DIR: Path = self.BASE_DIR / "reports"
        self.ARTIFACTS_DIR: Path = self.BASE_DIR / "artifacts"
        self.DATABASE_PATH: Path = self.DATA_DIR / "urbanflow.db"

        # Spatial CRS System
        self.PROJECTED_CRS: str = os.getenv("PROJECTED_CRS", "EPSG:32643")
        self.DISPLAY_CRS: str = os.getenv("DISPLAY_CRS", "EPSG:4326")

        # Optimization & Analysis Defaults
        self.DEFAULT_SERVICE_RADIUS_METERS: float = float(os.getenv("SERVICE_RADIUS_METERS", "800.0"))
        self.DEFAULT_BUDGET: int = int(os.getenv("DEFAULT_BUDGET", "50"))
        self.RANDOM_SEED: int = int(os.getenv("RANDOM_SEED", "42"))

        # Weights for Multi-Objective Fitness Evaluation
        self.POPULATION_COVERAGE_WEIGHT: float = float(os.getenv("POPULATION_COVERAGE_WEIGHT", "0.45"))
        self.EQUITY_WEIGHT: float = float(os.getenv("EQUITY_WEIGHT", "0.35"))
        self.COST_WEIGHT: float = float(os.getenv("COST_WEIGHT", "0.20"))

        # Ensure essential directories exist
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Create necessary data and storage directories if missing."""
        for path in [
            self.DATA_DIR,
            self.RAW_DATA_DIR,
            self.PROCESSED_DATA_DIR,
            self.FEATURES_DATA_DIR,
            self.REPORTS_DIR,
            self.ARTIFACTS_DIR,
        ]:
            path.mkdir(parents=True, exist_ok=True)


# Global singleton settings instance
settings = AppSettings()
