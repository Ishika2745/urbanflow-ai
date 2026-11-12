"""
UrbanFlow AI - Interactive Geospatial Visualizations
Builds PyDeck and Folium maps for Mumbai transit networks, desert clusters, and candidate locations.
"""

from typing import Optional, List, Dict, Any
import folium
from folium.plugins import MarkerCluster
import pydeck as pdk
import pandas as pd
import geopandas as gpd

from config.settings import settings
from utils.geo_utils import ensure_display_crs
from utils.logger import logger


def create_transit_desert_folium_map(
    wards_gdf: gpd.GeoDataFrame,
    stops_gdf: Optional[gpd.GeoDataFrame] = None,
    candidate_stops_gdf: Optional[gpd.GeoDataFrame] = None,
    selected_candidate_ids: Optional[List[str]] = None,
    center_coords: List[float] = [19.0760, 72.8777],
    zoom_start: int = 11,
    color_column: str = "transit_desert_index",
    heatmap_mode: Optional[str] = None,
    **kwargs
) -> folium.Map:
    """
    Creates an interactive Folium leaflet map highlighting:
    - Mumbai BMC Ward boundaries colored dynamically by DTII / Transit Desert Index / Accessibility
    - Existing Multimodal transit stops
    - Proposed candidate stops (green if selected by optimizer, amber if unselected candidate)
    - Floating responsive map legend.
    """
    wards = ensure_display_crs(wards_gdf.copy())

    # Map column alias if needed
    col_target = color_column
    if col_target not in wards.columns:
        if "dtii" in wards.columns and ("dtii" in col_target.lower() or "desert" in col_target.lower()):
            col_target = "dtii"
        elif "transit_desert_index" in wards.columns:
            col_target = "transit_desert_index"
        elif "accessibility_score" in wards.columns and "access" in col_target.lower():
            col_target = "accessibility_score"
        elif "population_density_per_sq_km" in wards.columns and "density" in col_target.lower():
            col_target = "population_density_per_sq_km"

    m = folium.Map(
        location=center_coords,
        zoom_start=zoom_start,
        tiles="OpenStreetMap"
    )

    # 1. Ward Choropleth
    def style_function(feature):
        props = feature["properties"]
        val = props.get(col_target, 0.5)

        if "access" in col_target.lower():
            # Accessibility: High is green, Low is red
            if val >= 70.0:
                color = "#1A9850"
            elif val >= 50.0:
                color = "#91CF60"
            elif val >= 35.0:
                color = "#FDAE61"
            else:
                color = "#D73027"
        elif "density" in col_target.lower():
            # Density: High is dark blue/purple
            if val >= 45000:
                color = "#08519C"
            elif val >= 30000:
                color = "#3182BD"
            elif val >= 15000:
                color = "#6BAED6"
            else:
                color = "#BDD7E7"
        else:
            # DTII / Transit Desert Index: High is red (Critical Inequity), Low is green
            if val > 0.60:
                color = "#D73027"  # Critical Inequity
            elif val > 0.45:
                color = "#FDAE61"  # High Inequity
            elif val > 0.30:
                color = "#FEE08B"  # Moderate Inequity
            else:
                color = "#1A9850"  # Low Inequity / Well-Served

        return {
            "fillColor": color,
            "color": "#333333",
            "weight": 1.2,
            "fillOpacity": 0.45
        }

    # Available fields for tooltip
    avail_cols = [c for c in ["ward_id", "ward_name", "population", "accessibility_score", "dtii", "transit_desert_index"] if c in wards.columns]
    aliases = [c.replace("_", " ").title() + ":" for c in avail_cols]

    folium.GeoJson(
        wards,
        name=f"BMC Wards ({col_target.replace('_', ' ').title()})",
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(fields=avail_cols, aliases=aliases, localize=True)
    ).add_to(m)

    # 2. Add Existing Transit Stops
    if stops_gdf is not None and not stops_gdf.empty:
        stops = ensure_display_crs(stops_gdf.copy())
        from folium.plugins import FastMarkerCluster

        # Separate rail/metro for distinct high-visibility stations
        rail_metro_mask = stops["mode_id"].str.contains("rail|suburban|metro", case=False, na=False) if "mode_id" in stops.columns else False
        rail_metro_stops = stops[rail_metro_mask] if isinstance(rail_metro_mask, pd.Series) and rail_metro_mask.any() else gpd.GeoDataFrame()
        bus_stops = stops[~rail_metro_mask] if isinstance(rail_metro_mask, pd.Series) and rail_metro_mask.any() else stops

        # Fast cluster for bus stops (prevents browser lag)
        if not bus_stops.empty:
            bus_locs = [[r.geometry.y, r.geometry.x] for _, r in bus_stops.iterrows() if r.geometry is not None]
            FastMarkerCluster(
                bus_locs,
                name="BEST Bus Stops (5,753 stops clustered)",
                options={"disableClusteringAtZoom": 16}
            ).add_to(m)

        # Distinct station markers for Suburban Rail and Metro
        if not rail_metro_stops.empty:
            station_fg = folium.FeatureGroup(name="Rail & Metro Stations (105 stations)")
            for _, row in rail_metro_stops.iterrows():
                if row.geometry is None:
                    continue
                mode = str(row.get("mode_id", "")).lower()
                is_metro = "metro" in mode
                s_color = "#9333EA" if is_metro else "#2563EB"
                s_name = row.get("station_name", row.get("stop_name", "Transit Station"))
                m_type = "Metro Station" if is_metro else "Suburban Rail Station"
                folium.CircleMarker(
                    location=[row.geometry.y, row.geometry.x],
                    radius=6,
                    color=s_color,
                    fill=True,
                    fill_color=s_color,
                    fill_opacity=0.9,
                    tooltip=f"{s_name} ({m_type})"
                ).add_to(station_fg)
            station_fg.add_to(m)

    # 2b. Add HeatMap Layer if requested
    if heatmap_mode is not None:
        from folium.plugins import HeatMap
        if "density" in heatmap_mode.lower() or "supply" in heatmap_mode.lower() or "bus" in heatmap_mode.lower():
            if stops_gdf is not None and not stops_gdf.empty:
                heat_stops = ensure_display_crs(stops_gdf.copy())
                heat_data = [[r.geometry.y, r.geometry.x, 1.0] for _, r in heat_stops.iterrows() if r.geometry is not None]
                HeatMap(heat_data, radius=18, blur=14, max_zoom=14, name="Transit Supply Heatmap").add_to(m)
        else:
            col_weight = "dtii" if "dtii" in wards.columns else ("dtii_score" if "dtii_score" in wards.columns else col_target)
            heat_data = []
            for _, r in wards.iterrows():
                if r.geometry is not None:
                    c = r.geometry.centroid
                    val = float(r.get(col_weight, 0.5))
                    heat_data.append([c.y, c.x, max(0.1, val) * 10.0])
            HeatMap(heat_data, radius=42, blur=28, max_zoom=13, name="Transit Inequality Gap Heatmap").add_to(m)

    # 3. Add Candidate Stops
    if candidate_stops_gdf is not None:
        cand_layer = folium.FeatureGroup(name="Proposed Facility Candidates")
        selected_set = set(selected_candidate_ids or [])
        cands = ensure_display_crs(candidate_stops_gdf.copy())

        for _, row in cands.iterrows():
            cid = row.get("candidate_id")
            is_selected = cid in selected_set
            color = "#00B140" if is_selected else "#999999"
            radius = 7 if is_selected else 4

            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=radius,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.9,
                tooltip=f"Candidate: {cid} | Cost: ₹{row.get('estimated_cost_lakhs', 25)}L | {'SELECTED' if is_selected else 'Available'}"
            ).add_to(cand_layer)
        cand_layer.add_to(m)

    # 4. Floating Map Legend
    legend_html = f"""
    <div style="
        position: fixed; 
        bottom: 25px; left: 25px; width: 230px;
        background-color: rgba(255, 255, 255, 0.94);
        z-index: 9999; font-size: 11px;
        border: 1px solid #CBD5E1; border-radius: 6px;
        padding: 10px 14px; box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    ">
        <b style="font-size: 12px; color: #0B1B3D;">Legend: {col_target.replace('_', ' ').title()}</b><br>
        <div style="margin-top: 6px; line-height: 1.6;">
            <i style="background: #D73027; width: 14px; height: 14px; display: inline-block; vertical-align: middle; margin-right: 6px; border-radius: 2px;"></i>
            <span style="color: #334155;">Critical Inequity / Low Access</span><br>
            <i style="background: #FDAE61; width: 14px; height: 14px; display: inline-block; vertical-align: middle; margin-right: 6px; border-radius: 2px;"></i>
            <span style="color: #334155;">High Inequity / Below Avg</span><br>
            <i style="background: #FEE08B; width: 14px; height: 14px; display: inline-block; vertical-align: middle; margin-right: 6px; border-radius: 2px;"></i>
            <span style="color: #334155;">Moderate Inequity</span><br>
            <i style="background: #1A9850; width: 14px; height: 14px; display: inline-block; vertical-align: middle; margin-right: 6px; border-radius: 2px;"></i>
            <span style="color: #334155;">Low Inequity / Well-Served</span>
        </div>
        <hr style="margin: 6px 0; border: none; border-top: 1px solid #E2E8F0;">
        <span style="font-size: 10px; color: #64748B;">● Blue/Red: Existing Stops | ● Green: Proposed</span>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    folium.LayerControl().add_to(m)
    return m


def create_before_after_simulator_map(
    wards_gdf: gpd.GeoDataFrame,
    existing_stops_gdf: Optional[gpd.GeoDataFrame] = None,
    proposed_stops_gdf: Optional[gpd.GeoDataFrame] = None,
    service_radius_m: float = 800.0,
    mode: str = "after",  # "before" or "after"
    center_coords: List[float] = [19.0760, 72.8777],
    zoom_start: int = 11
) -> folium.Map:
    """
    Renders either the Baseline (Before) or Simulated (After) transit network map
    with pedestrian catchment buffers and infrastructure details.
    """
    wards = ensure_display_crs(wards_gdf.copy())
    m = folium.Map(location=center_coords, zoom_start=zoom_start, tiles="OpenStreetMap")

    dtii_col = "simulated_dtii" if (mode == "after" and "simulated_dtii" in wards.columns) else "baseline_dtii" if "baseline_dtii" in wards.columns else "dtii" if "dtii" in wards.columns else "transit_desert_index"
    access_col = "simulated_accessibility" if (mode == "after" and "simulated_accessibility" in wards.columns) else "baseline_accessibility" if "baseline_accessibility" in wards.columns else "accessibility_score"

    def ward_style(feature):
        val = feature["properties"].get(dtii_col, 0.40)
        if val > 0.55:
            color = "#D73027"  # High inequality / transit desert
        elif val > 0.40:
            color = "#FDAE61"  # Moderate inequality
        else:
            color = "#1A9850"  # Low inequality / well connected
        return {
            "fillColor": color,
            "color": "#444444",
            "weight": 1.2,
            "fillOpacity": 0.40
        }

    # Ward choropleth layer
    tooltip_fields = ["ward_id", "ward_name", "population"]
    tooltip_aliases = ["Ward:", "Name:", "Population:"]
    if dtii_col in wards.columns:
        tooltip_fields.append(dtii_col)
        tooltip_aliases.append(f"DTII ({mode.title()}):")
    if access_col in wards.columns:
        tooltip_fields.append(access_col)
        tooltip_aliases.append(f"Access ({mode.title()}):")

    folium.GeoJson(
        wards,
        name=f"Wards ({'Simulated Post-Intervention' if mode == 'after' else 'Baseline Current'})",
        style_function=ward_style,
        tooltip=folium.GeoJsonTooltip(fields=tooltip_fields, aliases=tooltip_aliases, localize=True)
    ).add_to(m)

    # Existing stops layer
    if existing_stops_gdf is not None:
        stops = ensure_display_crs(existing_stops_gdf.copy())
        stops_layer = folium.FeatureGroup(name="Existing Multimodal Stops (BEST/Rail/Metro)")
        for _, r in stops.iterrows():
            folium.CircleMarker(
                location=[r.geometry.y, r.geometry.x],
                radius=3,
                color="#00529B",
                fill=True,
                fill_color="#00529B",
                fill_opacity=0.6,
                tooltip=f"{r.get('stop_name', 'Stop')} (Existing)"
            ).add_to(stops_layer)
        stops_layer.add_to(m)

    # Proposed candidate stops (only in "after" mode)
    if mode == "after" and proposed_stops_gdf is not None and not proposed_stops_gdf.empty:
        cands = ensure_display_crs(proposed_stops_gdf.copy())
        prop_layer = folium.FeatureGroup(name=f"Proposed Feeder Stops ({len(cands)})")

        for _, r in cands.iterrows():
            geom = r.geometry
            folium.CircleMarker(
                location=[geom.y, geom.x],
                radius=7,
                color="#00B140",
                fill=True,
                fill_color="#00FF66",
                fill_opacity=0.9,
                popup=folium.Popup(
                    f"<b>Proposed Feeder Stop</b><br>"
                    f"<b>ID:</b> {r.get('candidate_id', 'CAND')}<br>"
                    f"<b>Infrastructure:</b> {r.get('connected_infrastructure', r.get('candidate_name', 'Feeder Candidate'))}<br>"
                    f"<b>Ward:</b> {r.get('ward_id', 'M/E')}<br>"
                    f"<b>Estimated Cost:</b> ₹{r.get('estimated_cost_lakhs', 25.0)} Lakhs<br>"
                    f"<i>Catchment Radius: {service_radius_m:.0f}m</i><br>"
                    f"<small>Simulated Proposal (UrbanFlow AI)</small>",
                    max_width=280
                )
            ).add_to(prop_layer)

            # Pedestrian Catchment Circle
            folium.Circle(
                location=[geom.y, geom.x],
                radius=service_radius_m,
                color="#00B140",
                weight=1.5,
                fill=True,
                fill_color="#00B140",
                fill_opacity=0.10
            ).add_to(prop_layer)

        prop_layer.add_to(m)

    # Floating Simulation Legend
    sim_legend_html = f"""
    <div style="
        position: fixed; 
        bottom: 25px; left: 25px; width: 230px;
        background-color: rgba(255, 255, 255, 0.94);
        z-index: 9999; font-size: 11px;
        border: 1px solid #CBD5E1; border-radius: 6px;
        padding: 10px 14px; box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    ">
        <b style="font-size: 12px; color: #0B1B3D;">{mode.title()} Intervention View</b><br>
        <div style="margin-top: 6px; line-height: 1.6;">
            <i style="background: #D73027; width: 14px; height: 14px; display: inline-block; vertical-align: middle; margin-right: 6px; border-radius: 2px;"></i>
            <span style="color: #334155;">Critical Transit Desert</span><br>
            <i style="background: #FDAE61; width: 14px; height: 14px; display: inline-block; vertical-align: middle; margin-right: 6px; border-radius: 2px;"></i>
            <span style="color: #334155;">Moderate Transit Desert</span><br>
            <i style="background: #1A9850; width: 14px; height: 14px; display: inline-block; vertical-align: middle; margin-right: 6px; border-radius: 2px;"></i>
            <span style="color: #334155;">Well-Served Corridor</span>
        </div>
        <hr style="margin: 6px 0; border: none; border-top: 1px solid #E2E8F0;">
        <span style="font-size: 10px; color: #64748B;">
            ● Blue: Existing Stops<br>
            ● Green: Proposed Feeder ({service_radius_m:.0f}m Buffer)
        </span>
    </div>
    """
    m.get_root().html.add_child(folium.Element(sim_legend_html))

    folium.LayerControl(collapsed=False).add_to(m)
    return m

