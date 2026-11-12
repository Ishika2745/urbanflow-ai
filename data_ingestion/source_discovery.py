"""
UrbanFlow AI - Automated Dataset Source Discovery Engine
Queries CKAN open data portals, GitHub releases, and WorldPop APIs dynamically.
"""

from typing import Dict, Any, List, Optional
import requests
import re
import urllib3
urllib3.disable_warnings()

from data_ingestion.config import HTTP_HEADERS
from data_ingestion.logging_config import logger


class DatasetSourceDiscovery:
    """Discovers exact download URLs dynamically from OpenCity, GitHub, and WorldPop."""

    @staticmethod
    def discover_opencity_package(package_id_or_query: str) -> Optional[Dict[str, Any]]:
        """
        Queries OpenCity CKAN API (`/api/3/action/package_show` or `package_search`)
        to retrieve package metadata and downloadable resources.
        """
        headers = HTTP_HEADERS
        # Try direct package show first
        show_url = f"https://data.opencity.in/api/3/action/package_show?id={package_id_or_query}"
        try:
            r = requests.get(show_url, timeout=12, headers=headers, verify=False)
            if r.status_code == 200:
                result = r.json().get("result", {})
                logger.info(f"Discovered OpenCity package: '{result.get('title')}'")
                return result
        except Exception as e:
            logger.debug(f"Direct package_show failed for '{package_id_or_query}': {e}")

        # Fallback to search query
        search_url = f"https://data.opencity.in/api/3/action/package_search?q={package_id_or_query}"
        try:
            r = requests.get(search_url, timeout=12, headers=headers, verify=False)
            if r.status_code == 200:
                results = r.json().get("result", {}).get("results", [])
                if results:
                    logger.info(f"Discovered {len(results)} matching OpenCity packages for '{package_id_or_query}'")
                    return results[0]
        except Exception as e:
            logger.warning(f"Package search error on OpenCity: {e}")

        return None

    @staticmethod
    def scrape_dataset_page_for_kml(page_url: str) -> List[str]:
        """Scrapes an OpenCity dataset webpage directly to extract downloadable KML/GeoJSON URLs."""
        try:
            r = requests.get(page_url, timeout=12, headers=HTTP_HEADERS, verify=False)
            if r.status_code == 200:
                links = re.findall(
                    r'href=[\'"](https?://[^\'"]+\.(?:kml|kmz|csv|zip|geojson|json))[\'"]',
                    r.text,
                    re.IGNORECASE
                )
                return list(set(links))
        except Exception as e:
            logger.warning(f"Error scraping {page_url}: {e}")
        return []

    @staticmethod
    def discover_worldpop_latest(iso3: str = "IND") -> Optional[Dict[str, Any]]:
        """Queries WorldPop REST API to retrieve the latest population raster product."""
        api_url = f"https://hub.worldpop.org/rest/data/pop/wpgp?iso3={iso3}"
        try:
            r = requests.get(api_url, timeout=15, headers=HTTP_HEADERS)
            if r.status_code == 200:
                items = r.json().get("data", [])
                if items:
                    # Sort by popyear descending to get latest
                    items_sorted = sorted(items, key=lambda x: int(x.get("popyear", 0)), reverse=True)
                    latest = items_sorted[0]
                    logger.info(f"Discovered WorldPop latest product: Year {latest.get('popyear')} - '{latest.get('title')}'")
                    return latest
        except Exception as e:
            logger.warning(f"WorldPop API discovery error: {e}")
        return None

