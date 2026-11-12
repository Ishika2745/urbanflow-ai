"""
UrbanFlow AI - Multi-Format Geospatial & Tabular Parser
Parses KML, KMZ, GeoJSON, Shapefiles, GTFS TXT files, CSVs, and GeoTIFFs into GeoPandas & Pandas.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List
import xml.etree.ElementTree as ET
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon, LineString, MultiPolygon
from shapely import wkt
import fiona

from data_ingestion.config import DISPLAY_CRS
from data_ingestion.logging_config import logger

# Enable KML drivers in fiona
fiona.drvsupport.supported_drivers["KML"] = "rw"
fiona.drvsupport.supported_drivers["LIBKML"] = "rw"


class FormatParser:
    """Robust parser for diverse geospatial and tabular data formats."""

    @staticmethod
    def parse_kml(kml_path: Path) -> gpd.GeoDataFrame:
        """
        Parses a KML file into a GeoDataFrame.
        Tries Fiona driver first; if Fiona fails or yields partial attributes,
        falls back to custom ElementTree XML parser to extract all Placemark details.
        """
        logger.info(f"Parsing KML file: {kml_path.name}...")
        try:
            gdf = gpd.read_file(kml_path, driver="KML")
            if not gdf.empty and len(gdf) > 5:
                logger.info(f"Fiona successfully parsed {len(gdf)} records from {kml_path.name}")
                if gdf.crs is None:
                    gdf = gdf.set_crs(DISPLAY_CRS)
                return gdf
        except Exception as e:
            logger.debug(f"Fiona KML parse warning: {e}. Falling back to XML ElementTree parser.")

        # Fallback: Robust XML ElementTree Placemark Parser
        records = []
        tree = ET.parse(kml_path)
        root = tree.getroot()

        # Handle KML XML namespace
        ns = {"kml": "http://www.opengis.net/kml/2.2"}
        # If no namespace prefix matches, try without namespace
        placemarks = root.findall(".//kml:Placemark", ns) or root.findall(".//Placemark")

        for pm in placemarks:
            name_el = pm.find("kml:name", ns) if pm.find("kml:name", ns) is not None else pm.find("name")
            name = name_el.text.strip() if name_el is not None and name_el.text else "Unnamed"

            desc_el = pm.find("kml:description", ns) if pm.find("kml:description", ns) is not None else pm.find("description")
            desc = desc_el.text.strip() if desc_el is not None and desc_el.text else ""

            # Check for Point coordinates
            point_el = pm.find(".//kml:Point/kml:coordinates", ns) or pm.find(".//Point/coordinates")
            poly_el = pm.find(".//kml:Polygon//kml:coordinates", ns) or pm.find(".//Polygon//coordinates")
            line_el = pm.find(".//kml:LineString/kml:coordinates", ns) or pm.find(".//LineString/coordinates")

            geom = None
            if point_el is not None and point_el.text:
                parts = point_el.text.strip().split(",")
                if len(parts) >= 2:
                    try:
                        lon, lat = float(parts[0]), float(parts[1])
                        geom = Point(lon, lat)
                    except ValueError:
                        pass

            elif line_el is not None and line_el.text:
                coords = []
                for pt_str in line_el.text.strip().split():
                    parts = pt_str.split(",")
                    if len(parts) >= 2:
                        try:
                            coords.append((float(parts[0]), float(parts[1])))
                        except ValueError:
                            pass
                if len(coords) >= 2:
                    geom = LineString(coords)

            elif poly_el is not None and poly_el.text:
                coords = []
                for pt_str in poly_el.text.strip().split():
                    parts = pt_str.split(",")
                    if len(parts) >= 2:
                        try:
                            coords.append((float(parts[0]), float(parts[1])))
                        except ValueError:
                            pass
                if len(coords) >= 3:
                    geom = Polygon(coords)

            if geom is not None:
                records.append({
                    "Name": name,
                    "Description": desc,
                    "geometry": geom
                })

        gdf = gpd.GeoDataFrame(records, crs=DISPLAY_CRS)
        logger.info(f"XML ElementTree parsed {len(gdf)} placemark records from {kml_path.name}")
        return gdf

    @staticmethod
    def parse_csv_coordinates(
        csv_path: Path,
        lat_col: str = "latitude",
        lon_col: str = "longitude"
    ) -> gpd.GeoDataFrame:
        """Parses a CSV file containing latitude/longitude coordinates into a GeoDataFrame."""
        df = pd.read_csv(csv_path)
        # Case-insensitive column matching
        col_map = {col.lower(): col for col in df.columns}
        actual_lat = col_map.get(lat_col.lower()) or col_map.get("lat") or col_map.get("stop_lat")
        actual_lon = col_map.get(lon_col.lower()) or col_map.get("lon") or col_map.get("stop_lon") or col_map.get("lng")

        if not actual_lat or not actual_lon:
            raise ValueError(f"Could not locate coordinate columns in {csv_path.name}. Found: {list(df.columns)}")

        # Convert to numeric, dropping NaNs
        df[actual_lat] = pd.to_numeric(df[actual_lat], errors="coerce")
        df[actual_lon] = pd.to_numeric(df[actual_lon], errors="coerce")
        df = df.dropna(subset=[actual_lat, actual_lon])

        geometry = [Point(xy) for xy in zip(df[actual_lon], df[actual_lat])]
        return gpd.GeoDataFrame(df, geometry=geometry, crs=DISPLAY_CRS)

    @staticmethod
    def parse_geojson(geojson_path: Path) -> gpd.GeoDataFrame:
        """Parses standard GeoJSON file."""
        gdf = gpd.read_file(geojson_path)
        if gdf.crs is None:
            gdf = gdf.set_crs(DISPLAY_CRS)
        return gdf

