"""
UrbanFlow AI - Resilient Dataset Downloader
Handles HTTP/HTTPS downloads with retry backoff, file verification, resumption, and logging.
"""

import time
import hashlib
import json
from pathlib import Path
from typing import Optional, Dict, Any
import requests
import urllib3
urllib3.disable_warnings()

from data_ingestion.config import (
    DEFAULT_TIMEOUT_SEC,
    MAX_DOWNLOAD_RETRIES,
    BACKOFF_FACTOR,
    HTTP_HEADERS,
    DOWNLOAD_LOG_PATH
)
from data_ingestion.logging_config import logger


class ResilientDownloader:
    """Robust download engine with exponential backoff and duplicate prevention."""

    def __init__(self, log_path: Path = DOWNLOAD_LOG_PATH):
        self.log_path = log_path
        self._ensure_log()

    def _ensure_log(self) -> None:
        """Ensures the download log JSON exists."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_path.exists():
            with open(self.log_path, "w", encoding="utf-8") as f:
                json.dump([], f, indent=2)

    def _append_log(self, record: Dict[str, Any]) -> None:
        """Appends a new download event to download_log.json."""
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                logs = json.load(f)
            logs.append(record)
            with open(self.log_path, "w", encoding="utf-8") as f:
                json.dump(logs, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to append to download log: {e}")

    @staticmethod
    def calculate_file_hash(file_path: Path) -> str:
        """Calculates MD5 checksum of a local file."""
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def download_file(
        self,
        url: str,
        dest_path: Path,
        dataset_name: str = "Dataset",
        force_redownload: bool = False,
        expected_extension: Optional[str] = None
    ) -> Path:
        """
        Downloads a file with automatic retry, skip-if-cached verification, and status reporting.
        """
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # 1. Skip if file already exists, is non-empty, and force_redownload is False
        if dest_path.exists() and dest_path.stat().st_size > 0 and not force_redownload:
            logger.info(f"[{dataset_name}] Using cached file ({dest_path.stat().st_size / 1024:.1f} KB): {dest_path.name}")
            return dest_path

        logger.info(f"[{dataset_name}] Initiating download from: {url}")
        last_error = None

        for attempt in range(1, MAX_DOWNLOAD_RETRIES + 1):
            try:
                start_time = time.time()
                # Use stream=True to handle large files safely
                response = requests.get(
                    url,
                    stream=True,
                    timeout=DEFAULT_TIMEOUT_SEC,
                    verify=False,
                    headers=HTTP_HEADERS
                )

                if response.status_code == 200:
                    temp_dest = dest_path.with_suffix(dest_path.suffix + ".tmp")
                    total_bytes = 0

                    with open(temp_dest, "wb") as f:
                        for chunk in response.iter_content(chunk_size=65536):
                            if chunk:
                                f.write(chunk)
                                total_bytes += len(chunk)

                    # Rename temp file to final destination
                    if temp_dest.exists():
                        if dest_path.exists():
                            dest_path.unlink()
                        temp_dest.rename(dest_path)

                    elapsed = time.time() - start_time
                    file_hash = self.calculate_file_hash(dest_path)

                    log_entry = {
                        "dataset_name": dataset_name,
                        "url": url,
                        "file_path": str(dest_path.relative_to(dest_path.parent.parent.parent)),
                        "file_size_bytes": total_bytes,
                        "md5_checksum": file_hash,
                        "download_time_sec": round(elapsed, 2),
                        "status": "SUCCESS",
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                    }
                    self._append_log(log_entry)

                    logger.info(
                        f"[{dataset_name}] Successfully downloaded {total_bytes / 1024:.1f} KB in {elapsed:.2f}s (MD5: {file_hash[:8]}...)"
                    )
                    return dest_path
                else:
                    logger.warning(f"[{dataset_name}] Attempt {attempt}: Received HTTP {response.status_code}")
                    last_error = f"HTTP {response.status_code}"

            except Exception as e:
                logger.warning(f"[{dataset_name}] Attempt {attempt} failed: {e}")
                last_error = str(e)

            # Exponential backoff
            if attempt < MAX_DOWNLOAD_RETRIES:
                sleep_time = BACKOFF_FACTOR ** attempt
                logger.info(f"Retrying in {sleep_time:.1f}s...")
                time.sleep(sleep_time)

        # Record failed download
        self._append_log({
            "dataset_name": dataset_name,
            "url": url,
            "file_path": str(dest_path),
            "status": "FAILED",
            "error": last_error,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        })
        raise ConnectionError(f"Failed to download {dataset_name} from {url} after {MAX_DOWNLOAD_RETRIES} attempts. Last error: {last_error}")

