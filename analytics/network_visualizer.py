"""
UrbanFlow AI - Network & Accessibility Visualizer
Implements Part E:
Generates interactive spatial visualizations and audit maps:
- Accessibility Heatmap (Plotly Mapbox & Folium)
- Multimodal Transport Network Map (Separate symbology for Rail, Metro, Bus, Transfers, Skywalks)
- Walking Catchment Map (400m, 800m, 1000m walkshed isochrones)
- Nearest-Stop Walking Distance & Time Analysis Charts
- Saves self-contained HTML maps and PNG/CSV reports.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import HeatMap
import plotly.express as px
import plotly.graph_objects as go

from data_ingestion.config import DISPLAY_CRS, METRIC_CRS
from preprocessing.coordinate_manager import CoordinateManager
from utils.logger import logger

REPORTS_DIR = Path("reports")
MAPS_DIR = REPORTS_DIR / "maps"


class NetworkVisualizer:
    """Generates publication-quality interactive maps and diagnostic charts for accessibility."""

    def __init__(self, output_dir: Path = MAPS_DIR):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_accessibility_folium_heatmap(
        self,
        grid_accessibility_gdf: gpd.GeoDataFrame,
        output_filename: str = "accessibility_heatmap.html"
    ) -> folium.Map:
        """
        Creates an interactive Folium map rendering continuous multimodal accessibility scores.
        """
        logger.info("Generating Folium Accessibility Heatmap...")
        gdf = CoordinateManager.to_display(grid_accessibility_gdf)

        # Center map on Greater Mumbai
        mumbai_center = [19.0760, 72.8777]
        m = folium.Map(location=mumbai_center, zoom_start=11, tiles="CartoDB positron")

        # Add Choropleth/GeoJson grid cells styled by accessibility score
        def style_function(feature):
            score = feature["properties"].get("multimodal_accessibility_score", 50.0)
            # Viridis-like color gradient
            if score >= 75:
                color = "#2ca02c"  # High Access Green
            elif score >= 55:
                color = "#1f77b4"  # Moderate Blue
            elif score >= 35:
                color = "#ff7f0e"  # Underserved Orange
            else:
                color = "#d62728"  # Transit Desert Red
            return {
                "fillColor": color,
                "color": "#ffffff",
                "weight": 0.5,
                "fillOpacity": 0.65
            }

        folium.GeoJson(
            gdf[[
                "cell_id", "ward_id", "multimodal_accessibility_score",
                "net_walk_dist_transit_m", "walk_time_transit_mins",
                "transit_options_800m", "geometry"
            ]],
            style_function=style_function,
            tooltip=folium.GeoJsonTooltip(
                fields=[
                    "cell_id", "ward_id", "multimodal_accessibility_score",
                    "net_walk_dist_transit_m", "walk_time_transit_mins", "transit_options_800m"
                ],
                aliases=[
                    "Cell ID:", "Ward:", "Accessibility Score (0-100):",
                    "Walk Dist to Transit (m):", "Walk Time (mins):", "Transit Options (800m):"
                ],
                localize=True
            ),
            name="Accessibility Grid"
        ).add_to(m)

        # Add Layer Control
        folium.LayerControl().add_to(m)

        out_path = self.output_dir / output_filename
        m.save(str(out_path))
        logger.info(f"Saved interactive Accessibility Heatmap to: {out_path}")
        return m

    def create_multimodal_network_map(
        self,
        rail_stations: gpd.GeoDataFrame,
        metro_stations: gpd.GeoDataFrame,
        best_stops_sample: gpd.GeoDataFrame,
        output_filename: str = "transport_network_map.html"
    ) -> folium.Map:
        """
        Creates an interactive Folium map displaying the multimodal transportation infrastructure
        with separate layers and symbology for Suburban Rail, Metro, Bus, and Transfer Hubs.
        """
        logger.info("Generating Multimodal Transport Network Map...")
        m = folium.Map(location=[19.0760, 72.8777], zoom_start=11, tiles="CartoDB dark_matter")

        # 1. Suburban Railway Stations (Maroon circles)
        rail_group = folium.FeatureGroup(name="Suburban Railway Stations")
        for _, r in rail_stations.iterrows():
            folium.CircleMarker(
                location=[r.geometry.y, r.geometry.x],
                radius=6,
                color="#b2182b",
                fill=True,
                fill_color="#d6604d",
                fill_opacity=0.9,
                tooltip=f"🚆 Suburban Rail: {r.get('station_name', 'Station')}"
            ).add_to(rail_group)
        rail_group.add_to(m)

        # 2. Metro Stations (Cyan circles)
        metro_group = folium.FeatureGroup(name="Operational Metro Stations")
        for _, r in metro_stations.iterrows():
            folium.CircleMarker(
                location=[r.geometry.y, r.geometry.x],
                radius=5,
                color="#00ffff",
                fill=True,
                fill_color="#2b83ba",
                fill_opacity=0.9,
                tooltip=f"🚇 Metro Rail: {r.get('station_name', 'Station')} ({r.get('line', 'Metro')})"
            ).add_to(metro_group)
        metro_group.add_to(m)

        # 3. BEST Bus Stops sample (Green dots)
        bus_group = folium.FeatureGroup(name="BEST Bus Stops", show=False)
        sample_bus = best_stops_sample.sample(min(1500, len(best_stops_sample)), random_state=42)
        for _, r in sample_bus.iterrows():
            folium.CircleMarker(
                location=[r.geometry.y, r.geometry.x],
                radius=2,
                color="#78c679",
                fill=True,
                fill_color="#31a354",
                fill_opacity=0.7,
                tooltip=f"🚏 BEST Stop: {r.get('stop_name', 'Stop')}"
            ).add_to(bus_group)
        bus_group.add_to(m)

        # 4. Key Verified Multimodal Interchange Hubs
        interchange_hubs = [
            ("Dadar Central/Western Interchange", 19.0180, 72.8430),
            ("Kurla Central/Harbour Interchange", 19.0655, 72.8795),
            ("Andheri Rail/Metro L1 Interchange", 19.1200, 72.8465),
            ("Ghatkopar Rail/Metro L1 Interchange", 19.0860, 72.9080),
            ("Churchgate Western Rail Terminal", 18.9350, 72.8270),
            ("CSMT Central/Harbour Rail Terminal", 18.9400, 72.8355),
            ("Bandra Rail/Bus Interchange", 19.0550, 72.8410),
            ("Borivali Rail/Metro L2A Hub", 19.2270, 72.8450)
        ]
        hub_group = folium.FeatureGroup(name="⭐ Verified Multimodal Interchanges")
        for name, lat, lon in interchange_hubs:
            folium.Marker(
                location=[lat, lon],
                icon=folium.Icon(color="orange", icon="star", prefix="fa"),
                tooltip=f"⭐ Verified Transfer Hub: {name}"
            ).add_to(hub_group)
        hub_group.add_to(m)

        folium.LayerControl().add_to(m)
        out_path = self.output_dir / output_filename
        m.save(str(out_path))
        logger.info(f"Saved Multimodal Transport Network Map to: {out_path}")
        return m

    def create_walking_catchment_map(
        self,
        major_stations: gpd.GeoDataFrame,
        output_filename: str = "walking_catchments_map.html"
    ) -> folium.Map:
        """
        Creates an interactive Folium map illustrating 400m (5-min), 800m (10-min),
        and 1000m (13.3-min) walking isochrone catchments around major transit hubs.
        """
        logger.info("Generating Walking Catchment Isochrones Map...")
        m = folium.Map(location=[19.0760, 72.8777], zoom_start=12, tiles="CartoDB positron")

        stat_metric = CoordinateManager.to_metric(major_stations)

        # Buffers: 1000m, 800m, 400m (drawn in descending order of size)
        b1000 = stat_metric.geometry.buffer(1000.0).union_all()
        b800 = stat_metric.geometry.buffer(800.0).union_all()
        b400 = stat_metric.geometry.buffer(400.0).union_all()

        g1000 = CoordinateManager.to_display(gpd.GeoDataFrame([{"geometry": b1000}], crs=METRIC_CRS))
        g800 = CoordinateManager.to_display(gpd.GeoDataFrame([{"geometry": b800}], crs=METRIC_CRS))
        g400 = CoordinateManager.to_display(gpd.GeoDataFrame([{"geometry": b400}], crs=METRIC_CRS))

        folium.GeoJson(
            g1000,
            style_function=lambda x: {"fillColor": "#fee08b", "color": "#d73027", "weight": 1, "fillOpacity": 0.25},
            name="1000m Catchment (~13.3 min walk)"
        ).add_to(m)

        folium.GeoJson(
            g800,
            style_function=lambda x: {"fillColor": "#d9ef8b", "color": "#1a9850", "weight": 1.5, "fillOpacity": 0.35},
            name="800m Catchment (~10 min walk)"
        ).add_to(m)

        folium.GeoJson(
            g400,
            style_function=lambda x: {"fillColor": "#91cf60", "color": "#006837", "weight": 2, "fillOpacity": 0.5},
            name="400m Catchment (~5 min walk)"
        ).add_to(m)

        folium.LayerControl().add_to(m)
        out_path = self.output_dir / output_filename
        m.save(str(out_path))
        logger.info(f"Saved Walking Catchment Map to: {out_path}")
        return m

    def create_nearest_stop_distribution_chart(
        self,
        grid_accessibility_gdf: gpd.GeoDataFrame,
        output_filename: str = "nearest_stop_analysis.html"
    ) -> go.Figure:
        """
        Creates an interactive Plotly box & distribution comparison of walking distances
        across modes: Bus vs Suburban Rail vs Metro.
        """
        logger.info("Generating Nearest-Stop Walking Distance Analysis Charts...")
        df = grid_accessibility_gdf[[
            "net_walk_dist_bus_m", "net_walk_dist_rail_m", "net_walk_dist_metro_m"
        ]].copy()

        df_melt = df.melt(
            var_name="Transit Mode",
            value_name="Walking Network Distance (meters)"
        )
        mode_names = {
            "net_walk_dist_bus_m": "BEST Bus",
            "net_walk_dist_rail_m": "Suburban Railway",
            "net_walk_dist_metro_m": "Metro Rail"
        }
        df_melt["Transit Mode"] = df_melt["Transit Mode"].map(mode_names)

        fig = px.box(
            df_melt,
            x="Transit Mode",
            y="Walking Network Distance (meters)",
            color="Transit Mode",
            color_discrete_map={"BEST Bus": "#31a354", "Suburban Railway": "#b2182b", "Metro Rail": "#00a0dc"},
            points="outliers",
            title="Greater Mumbai: Pedestrian Network Distance Distribution by Transit Mode"
        )
        fig.add_hline(y=400, line_dash="dash", line_color="green", annotation_text="5-min walk (400m)")
        fig.add_hline(y=800, line_dash="dash", line_color="orange", annotation_text="10-min walk (800m)")
        fig.add_hline(y=1000, line_dash="dash", line_color="red", annotation_text="13.3-min walk (1000m)")

        fig.update_layout(
            template="plotly_white",
            yaxis_title="Network Walking Distance (meters)",
            xaxis_title="Transportation Mode",
            showlegend=False
        )

        out_path = self.output_dir / output_filename
        fig.write_html(str(out_path))
        logger.info(f"Saved Nearest-Stop Analysis Chart to: {out_path}")
        return fig

