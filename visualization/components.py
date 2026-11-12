"""
UrbanFlow AI - Clean UI Components
Minimal, map-first layout for Greater Mumbai transit planning.
"""

import streamlit as st
from typing import Optional


def inject_custom_theme() -> None:
    """Injects clean, minimal product CSS (no PPT banners or heavy boxes)."""
    st.markdown(
        """
        <style>
        html, body, [class*="css"] {
            font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            color: #111827;
        }
        .stApp {
            background-color: #FFFFFF;
        }
        .block-container {
            padding-top: 1.5rem !important;
            padding-bottom: 2rem !important;
            max-width: 1400px;
        }

        /* Clean light-neutral or dark-slate sidebar */
        section[data-testid="stSidebar"] {
            background-color: #111827 !important;
            border-right: 1px solid #1F2937;
        }
        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span,
        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] label {
            color: #D1D5DB !important;
            font-size: 0.86rem !important;
        }
        section[data-testid="stSidebar"] a {
            color: #F3F4F6 !important;
            font-size: 0.88rem !important;
            padding: 4px 6px !important;
            border-radius: 4px;
        }
        section[data-testid="stSidebar"] a:hover {
            background-color: #1F2937 !important;
            color: #FFFFFF !important;
        }

        /* Clean flat metric cards */
        div[data-testid="stMetric"] {
            background-color: #F9FAFB;
            border: 1px solid #E5E7EB;
            border-radius: 6px;
            padding: 12px 14px;
        }
        div[data-testid="stMetricLabel"] {
            font-size: 0.78rem !important;
            font-weight: 500 !important;
            color: #6B7280 !important;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.45rem !important;
            font-weight: 700 !important;
            color: #111827 !important;
            font-variant-numeric: tabular-nums;
        }

        /* Clean tabs */
        button[data-baseweb="tab"] {
            font-size: 0.88rem !important;
            font-weight: 500 !important;
            color: #6B7280 !important;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            color: #111827 !important;
            border-bottom-color: #2563EB !important;
            font-weight: 600 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header_banner(
    title: str,
    subtitle: str,
    badge: str = "",
    tag: str = "",
) -> None:
    """Renders a clean, compact 1-line page header (no PPT-style boxes)."""
    st.markdown(
        f"""
        <div style="border-bottom: 1px solid #E5E7EB; padding-bottom: 10px; margin-bottom: 16px;">
            <h2 style="margin: 0; font-size: 1.5rem; font-weight: 700; color: #111827; letter-spacing: -0.02em;">
                {title}
            </h2>
            <p style="margin: 2px 0 0 0; font-size: 0.88rem; color: #6B7280;">
                {subtitle}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_card(
    label: str,
    value: str,
    delta: Optional[str] = None,
    delta_color: str = "normal",
    color_border: str = "#2563EB",
) -> None:
    """Standardized clean KPI metric."""
    st.metric(label=label, value=value, delta=delta, delta_color=delta_color)


def render_sidebar_header():
    """Renders minimal sidebar title and city selector."""
    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 4px 0 10px 0; border-bottom: 1px solid #1F2937; margin-bottom: 10px;">
                <div style="font-size: 1.1rem; font-weight: 700; color: #FFFFFF;">UrbanFlow</div>
                <div style="font-size: 0.75rem; color: #9CA3AF;">Mumbai Transit Planner</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        city = st.selectbox(
            "City",
            options=["Mumbai (BMC)", "Navi Mumbai (Planned)", "Thane (Planned)"],
            index=0,
        )
        return city


def render_sidebar_navigation(current_page: Optional[str] = None) -> None:
    """Renders clean, plain-English navigation links."""
    with st.sidebar:
        st.page_link("app.py", label="Home")
        st.page_link("pages/01_Mobility_Overview.py", label="City Overview")
        st.page_link("pages/02_Demand_Intelligence.py", label="Travel Demand")
        st.page_link("pages/03_Multimodal_Accessibility.py", label="Walk to Transit")
        st.page_link("pages/04_Transit_Desert_Detection.py", label="Underserved Areas")
        st.page_link("pages/05_DTII_Analysis.py", label="Transit Gap Score (DTII)")
        st.page_link("pages/06_Infrastructure_Optimization.py", label="Stop Placement")
        st.page_link("pages/07_GA_and_PSO_Comparison.py", label="Compare Solvers (GA vs PSO)")
        st.page_link("pages/08_Scenario_Simulator.py", label="Test New Stops")
        st.page_link("pages/09_Model_Performance.py", label="Model Accuracy")
        st.page_link("pages/10_Data_Quality_and_Sources.py", label="Data Sources")
