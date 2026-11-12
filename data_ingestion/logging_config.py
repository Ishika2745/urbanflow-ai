"""
UrbanFlow AI - Ingestion Logging Configuration
Provides structured console and rotating file logging for data downloads and processing.
"""

import sys
import logging
from pathlib import Path
from data_ingestion.config import METADATA_DIR, PROJECT_ROOT

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | [%(name)s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_ingestion_logger(name: str = "data_ingestion") -> logging.Logger:
    """Configures and returns a dedicated logger for data acquisition."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Console output
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.INFO)
        console.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
        logger.addHandler(console)

        # File output to metadata/acquisition.log
        try:
            log_file = METADATA_DIR / "acquisition.log"
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
            logger.addHandler(file_handler)
        except Exception:
            pass

    return logger


logger = get_ingestion_logger()

