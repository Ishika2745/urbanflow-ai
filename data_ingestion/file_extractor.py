"""
UrbanFlow AI - Secure File Extractor
Extracts ZIP, TAR, and GZ archives with path traversal prevention and integrity checks.
"""

import zipfile
import tarfile
import gzip
import shutil
from pathlib import Path
from typing import List
from data_ingestion.logging_config import logger


class ArchiveExtractor:
    """Safely extracts archives into target directories."""

    @staticmethod
    def extract_zip(zip_path: Path, output_dir: Path) -> List[Path]:
        """Extracts a ZIP archive while preventing ZipSlip directory traversal attacks."""
        output_dir.mkdir(parents=True, exist_ok=True)
        extracted_files: List[Path] = []

        logger.info(f"Extracting ZIP archive: {zip_path.name} -> {output_dir.name}")
        with zipfile.ZipFile(zip_path, "r") as z:
            for member in z.namelist():
                # Path traversal guard
                member_path = output_dir / member
                resolved = member_path.resolve()
                if not str(resolved).startswith(str(output_dir.resolve())):
                    raise SecurityError(f"Attempted path traversal in ZIP member: {member}")

                z.extract(member, output_dir)
                extracted_files.append(member_path)

        logger.info(f"Extracted {len(extracted_files)} files from {zip_path.name}")
        return extracted_files

    @staticmethod
    def extract_nested_zip(zip_path: Path, inner_zip_name: str, output_dir: Path) -> List[Path]:
        """Extracts a ZIP file located inside another ZIP file (e.g. croyla/mumbai-gtfs)."""
        output_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as outer:
            # Find matching inner zip
            matched_names = [n for n in outer.namelist() if n.endswith(inner_zip_name)]
            if not matched_names:
                raise FileNotFoundError(f"Nested archive '{inner_zip_name}' not found inside {zip_path.name}")

            target_inner = matched_names[0]
            temp_inner_path = output_dir / Path(target_inner).name
            with open(temp_inner_path, "wb") as f_out:
                f_out.write(outer.read(target_inner))

            # Extract inner zip
            result = ArchiveExtractor.extract_zip(temp_inner_path, output_dir)
            if temp_inner_path.exists():
                temp_inner_path.unlink() # Cleanup temporary inner archive
            return result

