"""
UrbanFlow AI - Database Connection Manager
Provides unified connectivity to SQLite initially, with zero-friction upgrade path to PostgreSQL/PostGIS.
"""

import sqlite3
from typing import Generator
from contextlib import contextmanager
from pathlib import Path

from config.settings import settings
from utils.logger import logger
from utils.exceptions import DatabaseError


class DatabaseManager:
    """Manages SQLite database connections and transactions."""

    def __init__(self, db_path: Path = settings.DATABASE_PATH):
        self.db_path = db_path

    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager yielding an active SQLite connection with row factory."""
        conn = None
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            yield conn
            conn.commit()
        except sqlite3.Error as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error on {self.db_path}: {e}")
            raise DatabaseError(f"Database operation failed: {e}") from e
        finally:
            if conn:
                conn.close()


db_manager = DatabaseManager()

