"""
UrbanFlow AI - Transit Desert & Spatial Hotspot Visualizer
Generates interactive Folium geospatial maps for Part D:
1. Transit Desert Heatmap (DTII gradient & density).
2. Spatial Cluster Map (K-Means typologies & DBSCAN contiguous clusters).
3. Statistically Significant Hotspots Map (Getis-Ord Gi* & LISA quadrants).
4. Priority Area Ranking Map with detailed intervention popups.
"""

from typing import Optional, List, Dict, Any
from pathlib import Path
import folium
from folium.plugins import HeatMap, FeatureGroupSubGroup
import geopandas as gpd
import numpy as np
import pandas as pd

from utils.geo_utils import ensure_display_crs
from utils.logger import logger

MUMBAI_CENTER = [19.0760, 72.8777]


class TransitDesertVisualizer:
    """Builds interactive web maps for transit desert detection and hotspot analysis."""

    @staticmethod
    def create_transit_desert_heatmap(
        gdf: gpd.GeoDataFrame,
        dtii_col: str = "dtii",
        center_coords: List[float] = MUMBAI_CENTER,
        zoom_start: int = 11
    ) -> folium.Map:
        """
        Builds Folium interactive Transit Desert Heatmap.
        Features:
        - Density heatmap layer weighted by DTII.
        - Polygons colored by DTII severity (Green -> Yellow -> Orange -> Crimson).
        - Tooltips showing DTII, Relative Demand, and Accessibility.
        """
        disp = ensure_display_crs(gdf.copy())
        m = folium.Map(location=center_coords, zoom_start=zoom_start, tiles="CartoDB positron")

        # 1. HeatMap density layer
        heat_data = []
        for _, row in disp.iterrows():
            geom = row.geometry
            if geom is not None and not geom.is_empty:
                c = geom.centroid
                val = float(row.get(dtii_col, row.get("transit_desert_index", 20.0)))
                heat_data.append([c.y, c.x, val / 100.0])

        if heat_data:
            HeatMap(
                heat_data,
                name="🔥 DTII Intensity Heatmap",
                min_opacity=0.3,
                radius=18,
                blur=15,
                max_zoom=13
            ).add_to(m)

        # 2. Polygon Choropleth styling
        def style_fn(feat):
            props = feat["properties"]
            score = props.get(dtii_col, props.get("transit_desert_index", 25.0))
            if score >= 50.0:
                color = "#BD0026"  # Crimson (Severe Desert)
            elif score >= 35.0:
                color = "#FC4E2A"  # Red-Orange
            elif score >= 20.0:
                color = "#FEB24C"  # Amber
            else:
                color = "#31A354"  # Green (Well-served)

            return {
                "fillColor": color,
                "color": "#444444",
                "weight": 1.0,
                "fillOpacity": 0.45
            }

        tooltip_fields = ["ward_name", dtii_col, "relative_demand_score", "multimodal_accessibility_score", "population"]
        tooltip_aliases = ["Area:", "DTII Score:", "Relative Demand:", "Accessibility Score:", "Population:"]
        
        # Filter available fields
        avail_fields = [f for f in tooltip_fields if f in disp.columns]
        avail_aliases = [tooltip_aliases[i] for i, f in enumerate(tooltip_fields) if f in disp.columns]

        folium.GeoJson(
            disp,
            name="📊 Transit Desert Index (DTII)",
            style_function=style_fn,
            tooltip=folium.GeoJsonTooltip(fields=avail_fields, aliases=avail_aliases, localize=True)
        ).add_to(m)

        folium.LayerControl(collapsed=False).add_to(m)
        return m

    @staticmethod
    def create_spatial_cluster_map(
        gdf: gpd.GeoDataFrame,
        center_coords: List[float] = MUMBAI_CENTER,
        zoom_start: int = 11
    ) -> folium.Map:
        """
        Builds interactive map displaying:
        - K-Means operational typologies (Acute Deserts, Well-Serviced Hubs, etc.)
        - DBSCAN contiguous spatial cluster outlines.
        """
        disp = ensure_display_crs(gdf.copy())
        m = folium.Map(location=center_coords, zoom_start=zoom_start, tiles="CartoDB positron")

        kmeans_colors = {
            "Acute Transit Desert": "#E41A1C",           # Bright Red
            "Developing Transit Corridor": "#FF7F00",    # Orange
            "Well-Serviced Multimodal Hub": "#377EB8",   # Blue
            "Low-Density / Buffer Zone": "#4DAF4A"       # Green
        }

        # K-Means Layer
        def kmeans_style(feat):
            label = feat["properties"].get("kmeans_label", "Cluster")
            color = kmeans_colors.get(label, "#984EA3")
            return {
                "fillColor": color,
                "color": "#333333",
                "weight": 1.0,
                "fillOpacity": 0.55
            }

        folium.GeoJson(
            disp,
            name="🔷 K-Means Mobility Typologies",
            style_function=kmeans_style,
            tooltip=folium.GeoJsonTooltip(
                fields=[c for c in ["ward_name", "kmeans_label", "dtii", "relative_demand_score", "multimodal_accessibility_score"] if c in disp.columns],
                aliases=[a for i, a in enumerate(["Area:", "K-Means Typology:", "DTII:", "Demand:", "Access:"]) if ["ward_name", "kmeans_label", "dtii", "relative_demand_score", "multimodal_accessibility_score"][i] in disp.columns],
                localize=True
            )
        ).add_to(m)

        # DBSCAN Contiguous Cluster Outlines Layer
        if "dbscan_cluster" in disp.columns:
            dbscan_layer = folium.FeatureGroup(name="🔗 DBSCAN Contiguous Deficit Belts")
            palette = ["#984EA3", "#FF7F00", "#FFFF33", "#A65628", "#F781BF"]

            for _, row in disp[disp["dbscan_cluster"] >= 0].iterrows():
                c_id = row["dbscan_cluster"]
                color = palette[c_id % len(palette)]
                c = row.geometry.centroid
                folium.CircleMarker(
                    location=[c.y, c.x],
                    radius=5,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.85,
                    tooltip=f"DBSCAN Contiguous Cluster #{c_id} ({row.get('ward_name', 'Zone')})"
                ).add_to(dbscan_layer)
            dbscan_layer.add_to(m)

        folium.LayerControl(collapsed=False).add_to(m)
        return m

    @staticmethod
    def create_hotspot_significance_map(
        gdf: gpd.GeoDataFrame,
        center_coords: List[float] = MUMBAI_CENTER,
        zoom_start: int = 11
    ) -> folium.Map:
        """
        Builds interactive map displaying:
        - Getis-Ord Gi* Statistically Significant Hotspots & Coldspots.
        - Local Moran's I (LISA) Quadrants.
        """
        disp = ensure_display_crs(gdf.copy())
        m = folium.Map(location=center_coords, zoom_start=zoom_start, tiles="CartoDB positron")

        gi_colors = {
            "Hotspot - 99% Confidence": "#B30000",
            "Hotspot - 95% Confidence": "#E34A33",
            "Hotspot - 90% Confidence": "#FDBB84",
            "Not Significant": "#E0E0E0",
            "Coldspot - 90% Confidence": "#BDC9E1",
            "Coldspot - 95% Confidence": "#74A9CF",
            "Coldspot - 99% Confidence": "#0570B0"
        }

        # 1. Getis-Ord Gi* Layer
        def gi_style(feat):
            cat = feat["properties"].get("gi_category", "Not Significant")
            color = gi_colors.get(cat, "#E0E0E0")
            opacity = 0.25 if cat == "Not Significant" else 0.70
            return {
                "fillColor": color,
                "color": "#555555",
                "weight": 1.0,
                "fillOpacity": opacity
            }

        folium.GeoJson(
            disp,
            name="🔥 Getis-Ord Gi* Hotspots (Statistical Confidence)",
            style_function=gi_style,
            tooltip=folium.GeoJsonTooltip(
                fields=[c for c in ["ward_name", "gi_category", "getis_ord_z", "dtii", "relative_demand_score"] if c in disp.columns],
                aliases=[a for i, a in enumerate(["Area:", "Gi* Category:", "Z-Score:", "DTII:", "Demand:"]) if ["ward_name", "gi_category", "getis_ord_z", "dtii", "relative_demand_score"][i] in disp.columns],
                localize=True
            )
        ).add_to(m)

        # 2. Local Moran's I (LISA) Quadrants Layer
        lisa_colors = {
            "High-High (Transit Desert Hotspot)": "#E41A1C",
            "Low-Low (Transit Oasis)": "#377EB8",
            "High-Low (Spatial Outlier)": "#FF7F00",
            "Low-High (Spatial Outlier)": "#984EA3",
            "Not Significant": "#CCCCCC"
        }

        def lisa_style(feat):
            quad = feat["properties"].get("lisa_quadrant", "Not Significant")
            color = lisa_colors.get(quad, "#CCCCCC")
            opacity = 0.20 if quad == "Not Significant" else 0.65
            return {
                "fillColor": color,
                "color": "#333333",
                "weight": 1.0,
                "fillOpacity": opacity
            }

        folium.GeoJson(
            disp,
            name="📍 Local Moran's I (LISA) Quadrants",
            style_function=lisa_style,
            tooltip=folium.GeoJsonTooltip(
                fields=[c for c in ["ward_name", "lisa_quadrant", "local_moran_i", "lisa_p_value", "dtii"] if c in disp.columns],
                aliases=[a for i, a in enumerate(["Area:", "LISA Quadrant:", "Moran's I:", "p-value:", "DTII:"]) if ["ward_name", "lisa_quadrant", "local_moran_i", "lisa_p_value", "dtii"][i] in disp.columns],
                localize=True
            ),
            show=False
        ).add_to(m)

        folium.LayerControl(collapsed=False).add_to(m)
        return m

    @staticmethod
    def create_priority_ranking_map(
        gdf: gpd.GeoDataFrame,
        top_n: int = 15,
        center_coords: List[float] = MUMBAI_CENTER,
        zoom_start: int = 11
    ) -> folium.Map:
        """
        Builds interactive map displaying Top Priority Municipal Intervention Areas.
        Each area popup displays:
        - Population
        - DTII
        - Relative Demand
        - Nearest Transport Facility
        - Walking-network distance
        - Accessibility deficit
        - Recommended intervention
        """
        disp = ensure_display_crs(gdf.copy())
        m = folium.Map(location=center_coords, zoom_start=zoom_start, tiles="CartoDB positron")

        # Base choropleth of priority scores
        def prio_style(feat):
            score = feat["properties"].get("priority_score", 30.0)
            if score >= 60.0:
                color = "#990000"
            elif score >= 45.0:
                color = "#D7301F"
            elif score >= 30.0:
                color = "#EF6548"
            else:
                color = "#FEE8C8"
            return {
                "fillColor": color,
                "color": "#444444",
                "weight": 1.0,
                "fillOpacity": 0.45
            }

        folium.GeoJson(
            disp,
            name="🎯 Priority Area Intensity",
            style_function=prio_style
        ).add_to(m)

        # Top N Pinpoints with Detailed Popups
        top_df = disp.sort_values(by="priority_score", ascending=False).head(top_n)
        marker_group = folium.FeatureGroup(name=f"🚨 Top {top_n} Priority Intervention Hubs")

        for _, row in top_df.iterrows():
            c = row.geometry.centroid
            rank = int(row.get("priority_rank", 1))
            name = row.get("ward_name", row.get("cell_id", f"Zone #{rank}"))
            pop = int(row.get("population", 0))
            dtii = float(row.get("dtii", 0.0))
            dem = float(row.get("relative_demand_score", 0.0))
            acc = float(row.get("multimodal_accessibility_score", row.get("accessibility_score", 0.0)))
            deficit = float(row.get("accessibility_deficit", max(0.0, dem - acc)))
            dist_walk = float(row.get("net_walk_dist_transit_m", 800.0))
            facility = str(row.get("nearest_facility_name", "Nearest Transit Hub"))
            intervention = str(row.get("recommended_intervention", "Feeder Bus Expansion"))
            tier = str(row.get("priority_tier", "Tier 1"))

            popup_html = f"""
            <div style="font-family: Arial, sans-serif; min-width: 250px; font-size: 13px;">
                <h4 style="margin: 0 0 6px 0; color: #BD0026;">Rank #{rank}: {name}</h4>
                <div style="background: #FFF3CD; padding: 4px 8px; border-radius: 4px; margin-bottom: 8px; font-weight: bold; color: #856404;">
                    {tier}
                </div>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr><td>👥 <b>Population:</b></td><td style="text-align: right;">{pop:,}</td></tr>
                    <tr><td>⚠️ <b>DTII Inequity Index:</b></td><td style="text-align: right; color: #BD0026; font-weight: bold;">{dtii:.2f} / 100</td></tr>
                    <tr><td>📈 <b>Relative Demand:</b></td><td style="text-align: right;">{dem:.2f} / 100</td></tr>
                    <tr><td>⭐ <b>Accessibility:</b></td><td style="text-align: right;">{acc:.2f} / 100</td></tr>
                    <tr><td>📉 <b>Accessibility Deficit:</b></td><td style="text-align: right; color: #C00;"><b>{deficit:.2f}</b></td></tr>
                    <tr><td>🚏 <b>Nearest Facility:</b></td><td style="text-align: right;">{facility}</td></tr>
                    <tr><td>🚶 <b>Network Walk Dist:</b></td><td style="text-align: right;">{dist_walk:.0f} m</td></tr>
                </table>
                <hr style="margin: 8px 0; border: 0; border-top: 1px solid #ddd;" />
                <div style="background: #E8F4F8; padding: 6px 8px; border-left: 3px solid #007BFF; font-size: 12px;">
                    <b>Recommended Intervention:</b><br/>{intervention}
                </div>
            </div>
            """

            folium.Marker(
                location=[c.y, c.x],
                icon=folium.Icon(color="red" if rank <= 5 else "orange", icon="info-sign"),
                popup=folium.Popup(popup_html, max_width=320),
                tooltip=f"Rank #{rank}: {name} (DTII: {dtii:.1f})"
            ).add_to(marker_group)

        marker_group.add_to(m)
        folium.LayerControl(collapsed=False).add_to(m)
        return m

    @staticmethod
    def save_all_maps(
        gdf: gpd.GeoDataFrame,
        output_dir: Path,
        dtii_col: str = "dtii",
        top_n: int = 15
    ) -> Dict[str, Path]:
        """Saves all 4 interactive maps to HTML files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = {}

        # 1. Heatmap
        m_heat = TransitDesertVisualizer.create_transit_desert_heatmap(gdf, dtii_col=dtii_col)
        p_heat = output_dir / "transit_desert_heatmap.html"
        m_heat.save(str(p_heat))
        paths["heatmap"] = p_heat
        logger.info(f"Saved Transit Desert Heatmap to {p_heat}")

        # 2. Spatial Clusters
        m_clust = TransitDesertVisualizer.create_spatial_cluster_map(gdf)
        p_clust = output_dir / "spatial_clusters_map.html"
        m_clust.save(str(p_clust))
        paths["clusters"] = p_clust
        logger.info(f"Saved Spatial Clusters Map to {p_clust}")

        # 3. Hotspots
        m_hot = TransitDesertVisualizer.create_hotspot_significance_map(gdf)
        p_hot = output_dir / "statistically_significant_hotspots_map.html"
        m_hot.save(str(p_hot))
        paths["hotspots"] = p_hot
        logger.info(f"Saved Statistically Significant Hotspots Map to {p_hot}")

        # 4. Priority Ranking
        m_prio = TransitDesertVisualizer.create_priority_ranking_map(gdf, top_n=top_n)
        p_prio = output_dir / "priority_interventions_map.html"
        m_prio.save(str(p_prio))
        paths["priority"] = p_prio
        logger.info(f"Saved Priority Interventions Map to {p_prio}")

        return paths

