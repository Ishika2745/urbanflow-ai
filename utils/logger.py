"""
UrbanFlow AI - Centralized Logging Utility
Provides structured console and rotating file logging.
"""

import sys
import logging
from pathlib import Path
from typing import Optional
from config.settings import settings

# Global format
DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logger(
    name: str = "urbanflow",
    log_level: Optional[str] = None,
    log_file: Optional[Path] = None
) -> logging.Logger:
    """
    Configures and returns a named logger with console and optional file logging.
    
    Args:
        name: Name of the logger module.
        log_level: DEBUG, INFO, WARNING, ERROR, CRITICAL. Defaults to settings.LOG_LEVEL.
        log_file: Optional path for file logging.
        
    Returns:
        logging.Logger instance.
    """
    level_str = (log_level or settings.LOG_LEVEL).upper()
    level = getattr(logging, level_str, logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers if logger was already created
    if not logger.handlers:
        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        formatter = logging.Formatter(fmt=DEFAULT_LOG_FORMAT, datefmt=DATE_FORMAT)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File Handler (Optional)
        target_log_file = log_file or (settings.BASE_DIR / "urbanflow.log")
        try:
            file_handler = logging.FileHandler(target_log_file, encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception:
            # If log file cannot be created, proceed with console logging
            pass

    return logger


# Pre-instantiated root logger for quick imports
logger = setup_logger("urbanflow")

