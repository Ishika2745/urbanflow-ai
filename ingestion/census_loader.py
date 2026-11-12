"""
UrbanFlow AI - Demographic & Census Loader
Loads ward-level socioeconomic, slum percentage, and employment density statistics.
"""

from pathlib import Path
from typing import Optional
import pandas as pd
from config.settings import settings
from utils.logger import logger
from utils.exceptions import DataIngestionError


class CensusDataLoader:
    """Loads demographic profiles and social vulnerability indicators."""

    @staticmethod
    def load_ward_demographics(file_path: Optional[Path] = None) -> pd.DataFrame:
        """Loads ward socioeconomic table with population, density, and vulnerability indices."""
        target_path = file_path or (settings.RAW_DATA_DIR / "mumbai_census_ward_profiles.csv")
        if not target_path.exists():
            logger.info("Census CSV not found; generating from Mumbai demographic defaults.")
            from ingestion.synthetic_generator import MUMBAI_WARD_PROFILES
            records = [
                {
                    "ward_id": code,
                    "ward_name": name,
                    "population": pop,
                    "area_sq_km": area,
                    "slum_pct": slum_pct,
                    "vulnerability_index": vul_idx,
                    "population_density_per_sq_km": pop / area
                }
                for code, name, pop, area, _, _, slum_pct, vul_idx in MUMBAI_WARD_PROFILES
            ]
            df = pd.DataFrame(records)
            df.to_csv(target_path, index=False)
            return df

        return pd.read_csv(target_path)

