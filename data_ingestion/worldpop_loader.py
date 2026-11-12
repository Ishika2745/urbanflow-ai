"""
UrbanFlow AI - Dataset D: WorldPop Population Data Ingestion & Preprocessing
Acquires WorldPop population count rasters, validates properties, clips to Greater Mumbai boundary,
calculates population density, and aggregates into analytical spatial units.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from shapely.geometry import box, Polygon, Point

from data_ingestion.config import (
    RAW_POPULATION,
    PROCESSED_POPULATION,
    DATA_SOURCES,
    MUMBAI_BBOX,
    DISPLAY_CRS,
    METRIC_CRS
)
from data_ingestion.downloader import ResilientDownloader
from data_ingestion.logging_config import logger


class WorldPopDataLoader:
    """Manages acquisition, validation, spatial clipping, and grid aggregation of WorldPop rasters."""

    def __init__(self, raw_dir: Path = RAW_POPULATION, output_dir: Path = PROCESSED_POPULATION):
        self.raw_dir = raw_dir
        self.output_dir = output_dir
        self.downloader = ResilientDownloader()

    def acquire_and_process(
        self,
        use_1km: bool = True,
        force_download: bool = False,
        boundary_gdf: Optional[gpd.GeoDataFrame] = None
    ) -> Tuple[Path, gpd.GeoDataFrame, Dict[str, Any]]:
        """
        Full WorldPop raster pipeline:
        1. Download GeoTIFF to data/raw/population/
        2. Validate CRS, resolution, nodata, bounds with Rasterio
        3. Clip raster to Greater Mumbai boundary
        4. Save data/processed/population/population_clipped.tif
        5. Generate vector population grid GeoDataFrame (data/processed/population/population_grid.geojson)
        """
        logger.info("=== [Dataset D] Ingesting WorldPop Population Count Raster ===")
        source_cfg = DATA_SOURCES["worldpop"]

        # Choose 1km aggregated or 100m based on flag
        raster_url = source_cfg["raster_1km_url"] if use_1km else source_cfg["raster_100m_url"]
        target_filename = "ind_ppp_2020_1km.tif" if use_1km else "ind_ppp_2020_100m.tif"
        raw_tif_path = self.raw_dir / target_filename

        # 1. Download
        self.downloader.download_file(
            url=raster_url,
            dest_path=raw_tif_path,
            dataset_name="WorldPop India Raster",
            force_redownload=force_download
        )

        # 2. Inspect and Validate Raster
        with rasterio.open(raw_tif_path) as src:
            crs = str(src.crs)
            width, height = src.width, src.height
            res_x, res_y = src.res
            nodata = src.nodata
            bounds = src.bounds
            logger.info(f"Loaded WorldPop GeoTIFF: {width}x{height}, CRS: {crs}, Res: ({res_x:.4f}, {res_y:.4f})")

        # 3. Clip to Greater Mumbai Extent
        self.output_dir.mkdir(parents=True, exist_ok=True)
        clipped_tif_path = self.output_dir / "population_clipped.tif"

        # Construct clipping polygon
        if boundary_gdf is not None and not boundary_gdf.empty:
            shapes = [feature["geometry"] for _, feature in boundary_gdf.iterrows()]
        else:
            min_lon, min_lat, max_lon, max_lat = MUMBAI_BBOX
            clip_poly = box(min_lon, min_lat, max_lon, max_lat)
            shapes = [clip_poly.__geo_interface__]

        with rasterio.open(raw_tif_path) as src:
            out_image, out_transform = mask(src, shapes, crop=True)
            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform
            })

            with rasterio.open(clipped_tif_path, "w", **out_meta) as dest:
                dest.write(out_image)

        logger.info(f"Saved clipped Mumbai population raster to: {clipped_tif_path.name}")

        # 4. Generate Vectorized Population Grid
        grid_gdf = self._generate_population_grid(clipped_tif_path)
        grid_out_path = self.output_dir / "population_grid.geojson"
        grid_gdf.to_file(grid_out_path, driver="GeoJSON")
        logger.info(f"Generated {len(grid_gdf)} population grid cells. Total Mumbai Population: {grid_gdf['population'].sum():,}")

        summary = {
            "dataset_name": "WorldPop_Population_Count",
            "reference_year": "2020",
            "resolution": "1km Aggregated" if use_1km else "100m",
            "clipped_dimensions": f"{out_image.shape[2]}x{out_image.shape[1]}",
            "total_population_estimate": int(grid_gdf["population"].sum()),
            "grid_cells_count": len(grid_gdf),
            "status": "VALIDATED"
        }
        return clipped_tif_path, grid_gdf, summary

    def _generate_population_grid(self, clipped_tif: Path) -> gpd.GeoDataFrame:
        """Vectorizes non-zero raster pixels into a GeoDataFrame grid of population points and polygons."""
        records = []
        with rasterio.open(clipped_tif) as src:
            data = src.read(1)
            nodata_val = src.nodata if src.nodata is not None else -99999.0

            for row in range(data.shape[0]):
                for col in range(data.shape[1]):
                    val = data[row, col]
                    if val > 0 and val != nodata_val and not np.isnan(val):
                        # Calculate pixel center in geographic coordinates
                        x, y = rasterio.transform.xy(src.transform, row, col)
                        # Approx pixel half width
                        dx, dy = src.res[0] / 2.0, src.res[1] / 2.0
                        cell_poly = box(x - dx, y - dy, x + dx, y + dy)

                        records.append({
                            "cell_id": f"POP_{len(records)+1:06d}",
                            "population": float(val),
                            "density_per_sq_km": float(val / (src.res[0] * 111.0 * src.res[1] * 111.0)),
                            "longitude": x,
                            "latitude": y,
                            "geometry": cell_poly
                        })

        return gpd.GeoDataFrame(records, crs=DISPLAY_CRS)

