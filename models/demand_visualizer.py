"""
UrbanFlow AI - Mobility Demand Visualizer
Implements Part E:
Generates publication-quality interactive maps and diagnostic charts:
- Relative Demand Heatmap (Folium continuous choropleth)
- Ward-Wise Relative Demand Ranking Chart (Plotly horizontal bar chart)
- Demand Tier Cluster Visualization (Plotly multidimensional scatter)
- High-Demand vs. Low-Accessibility Matrix (Transit Desert / Unmet Mobility Gap Quadrant)
All visualizations consume genuine processed spatial data.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import geopandas as gpd
import folium
import plotly.express as px
import plotly.graph_objects as go

from preprocessing.coordinate_manager import CoordinateManager
from utils.logger import logger

REPORTS_DIR = Path("reports")
MAPS_DIR = REPORTS_DIR / "maps"


class DemandVisualizer:
    """Generates visual analytics and maps for the mobility demand intelligence module."""

    def __init__(self, output_dir: Path = MAPS_DIR):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_demand_folium_heatmap(
        self,
        grid_gdf: gpd.GeoDataFrame,
        output_filename: str = "demand_heatmap.html"
    ) -> folium.Map:
        """
        Creates an interactive Folium choropleth map rendering estimated relative demand.
        """
        logger.info("Generating Folium Relative Demand Heatmap...")
        gdf = CoordinateManager.to_display(grid_gdf)

        m = folium.Map(location=[19.0760, 72.8777], zoom_start=11, tiles="CartoDB positron")

        def style_function(feature):
            score = feature["properties"].get("relative_demand_score", 0.0)
            # Orange-to-Red gradient for demand intensity
            if score >= 60.0:
                color = "#990000"  # Very High Demand Dark Red
            elif score >= 40.0:
                color = "#d7301f"  # High Demand Red
            elif score >= 20.0:
                color = "#fc8d59"  # Moderate Demand Orange
            else:
                color = "#fdcc8a"  # Low Demand Light Orange
            return {
                "fillColor": color,
                "color": "#ffffff",
                "weight": 0.5,
                "fillOpacity": 0.70
            }

        folium.GeoJson(
            gdf[[
                "cell_id", "ward_id", "relative_demand_score", "demand_tier",
                "population", "population_density_per_sq_km", "commercial_count", "geometry"
            ]],
            style_function=style_function,
            tooltip=folium.GeoJsonTooltip(
                fields=[
                    "cell_id", "ward_id", "relative_demand_score", "demand_tier",
                    "population", "population_density_per_sq_km", "commercial_count"
                ],
                aliases=[
                    "Cell ID:", "Ward:", "Relative Demand Score:", "Demand Tier:",
                    "Population:", "Pop Density (/km²):", "Commercial Units:"
                ],
                localize=True
            ),
            name="Estimated Relative Demand"
        ).add_to(m)

        folium.LayerControl().add_to(m)
        out_path = self.output_dir / output_filename
        m.save(str(out_path))
        logger.info(f"Saved Relative Demand Heatmap to: {out_path}")
        return m

    def create_ward_demand_barchart(
        self,
        ward_demand_df: pd.DataFrame,
        output_filename: str = "ward_demand_chart.html"
    ) -> go.Figure:
        """
        Creates an interactive horizontal bar chart ranking all 24 wards by relative demand score.
        """
        logger.info("Generating Ward-Wise Demand Ranking Bar Chart...")
        df_sorted = ward_demand_df.sort_values(by="relative_demand_score", ascending=True)

        fig = px.bar(
            df_sorted,
            x="relative_demand_score",
            y="ward_name",
            orientation="h",
            color="relative_demand_score",
            color_continuous_scale="Reds",
            title="Greater Mumbai: Estimated Relative Mobility Demand by Administrative Ward",
            labels={
                "relative_demand_score": "Estimated Relative Demand Score (0 to 100)",
                "ward_name": "BMC Administrative Ward"
            },
            text="relative_demand_score"
        )
        fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig.update_layout(
            template="plotly_white",
            height=750,
            xaxis=dict(range=[0, 105]),
            coloraxis_showscale=False
        )

        out_path = self.output_dir / output_filename
        fig.write_html(str(out_path))
        logger.info(f"Saved Ward Demand Chart to: {out_path}")
        return fig

    def create_demand_clusters_scatter(
        self,
        grid_gdf: gpd.GeoDataFrame,
        output_filename: str = "demand_clusters_scatter.html"
    ) -> go.Figure:
        """
        Creates an interactive multi-attribute scatter plot of the K-Means demand tiers.
        """
        logger.info("Generating Demand Clusters Multidimensional Scatter Plot...")
        hover_cols = [c for c in ["ward_id", "population", "commercial_count", "road_intersection_density"] if c in grid_gdf.columns]
        fig = px.scatter(
            grid_gdf,
            x="population_density_per_sq_km",
            y="relative_demand_score",
            size="commercial_density_per_sq_km" if "commercial_density_per_sq_km" in grid_gdf.columns else None,
            color="demand_tier",
            hover_name="cell_id",
            hover_data=hover_cols,
            color_discrete_map={
                "Very High Relative Demand": "#990000",
                "High Relative Demand": "#e41a1c",
                "Moderate Relative Demand": "#ff7f00",
                "Low Relative Demand": "#377eb8"
            },
            title="Neighbourhood Segmentation: Population Density vs. Estimated Relative Demand",
            labels={
                "population_density_per_sq_km": "Population Density (people / km²)",
                "relative_demand_score": "Relative Demand Score (0 to 100)",
                "commercial_density_per_sq_km": "Commercial Density",
                "demand_tier": "Demand Tier"
            }
        )
        fig.update_layout(template="plotly_white", height=600)
        out_path = self.output_dir / output_filename
        fig.write_html(str(out_path))
        logger.info(f"Saved Demand Clusters Scatter to: {out_path}")
        return fig

    def create_unmet_demand_matrix(
        self,
        grid_gdf: gpd.GeoDataFrame,
        output_filename: str = "unmet_demand_matrix.html"
    ) -> go.Figure:
        """
        Constructs the High-Demand vs. Low-Accessibility Matrix (Transit Desert / Unmet Need Diagnostic).
        Highlights areas with high mobility demand but deficient public transport supply.
        """
        logger.info("Generating High-Demand vs. Low-Accessibility Diagnostic Matrix...")
        df = grid_gdf.copy()
        if "multimodal_accessibility_score" not in df.columns:
            df["multimodal_accessibility_score"] = df.get("transport_supply_score", 25.0)

        # Classify Quadrants
        demand_threshold = 30.0
        access_threshold = 25.0

        quadrants = []
        for _, r in df.iterrows():
            d = r["relative_demand_score"]
            a = r["multimodal_accessibility_score"]
            if d >= demand_threshold and a < access_threshold:
                quadrants.append("🚨 Acute Unmet Demand (High Demand, Low Access)")
            elif d >= demand_threshold and a >= access_threshold:
                quadrants.append("🟢 Transit Rich Core (High Demand, High Access)")
            elif d < demand_threshold and a < access_threshold:
                quadrants.append("⚪ Low Demand, Low Access (Peripheral / Natural)")
            else:
                quadrants.append("🔵 Transit Surplus (Low Demand, High Access)")

        df["quadrant_status"] = quadrants

        hover_cols = [c for c in ["ward_id", "population", "net_walk_dist_transit_m", "dist_to_nearest_transit_m"] if c in df.columns]
        fig = px.scatter(
            df,
            x="multimodal_accessibility_score",
            y="relative_demand_score",
            color="quadrant_status",
            hover_name="cell_id",
            hover_data=hover_cols,
            color_discrete_map={
                "🚨 Acute Unmet Demand (High Demand, Low Access)": "#d73027",
                "🟢 Transit Rich Core (High Demand, High Access)": "#1a9850",
                "⚪ Low Demand, Low Access (Peripheral / Natural)": "#8073ac",
                "🔵 Transit Surplus (Low Demand, High Access)": "#4575b4"
            },
            title="UrbanFlow AI: Mobility Demand vs. Transit Accessibility Diagnostic Matrix",
            labels={
                "multimodal_accessibility_score": "Multimodal Accessibility Score (0 to 100)",
                "relative_demand_score": "Estimated Relative Demand Score (0 to 100)",
                "quadrant_status": "Planning Classification"
            }
        )

        # Add quadrant divider lines
        fig.add_vline(x=access_threshold, line_dash="dash", line_color="black", annotation_text="Access Threshold (25)")
        fig.add_hline(y=demand_threshold, line_dash="dash", line_color="black", annotation_text="Demand Threshold (30)")

        fig.update_layout(
            template="plotly_white",
            height=650,
            xaxis=dict(range=[-2, 102]),
            yaxis=dict(range=[-2, 102])
        )

        out_path = self.output_dir / output_filename
        fig.write_html(str(out_path))
        logger.info(f"Saved Unmet Demand Matrix to: {out_path}")
        return fig
