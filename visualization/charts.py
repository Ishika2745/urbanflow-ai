"""
UrbanFlow AI - Interactive Analytics Charts
Builds Plotly figures for Lorenz inequality curves, convergence histories, and benchmark comparisons.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


def create_lorenz_curve_figure(
    p_baseline: List[float],
    l_baseline: List[float],
    baseline_gini: float,
    p_simulated: Optional[List[float]] = None,
    l_simulated: Optional[List[float]] = None,
    simulated_gini: Optional[float] = None
) -> go.Figure:
    """
    Renders comparative Lorenz Curve displaying cumulative population vs cumulative transit accessibility.
    """
    fig = go.Figure()

    # Line of Perfect Equality
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(color="#888888", dash="dash"),
        name="Line of Perfect Equality (Gini = 0.0)"
    ))

    # Baseline Lorenz Curve
    fig.add_trace(go.Scatter(
        x=p_baseline, y=l_baseline,
        mode="lines",
        line=dict(color="#E84A27", width=2.5),
        name=f"Current Baseline (Gini = {baseline_gini:.3f})"
    ))

    # Simulated Post-Intervention Curve
    if p_simulated and l_simulated and simulated_gini is not None:
        fig.add_trace(go.Scatter(
            x=p_simulated, y=l_simulated,
            mode="lines",
            line=dict(color="#00B140", width=2.5),
            name=f"Optimized Scenario (Gini = {simulated_gini:.3f})"
        ))

    fig.update_layout(
        title="<b>Lorenz Curve: Spatial Transport Accessibility Inequality</b>",
        xaxis_title="Cumulative Share of Population",
        yaxis_title="Cumulative Share of Transit Opportunity",
        template="plotly_white",
        legend=dict(x=0.05, y=0.95),
        hovermode="x unified",
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig


def create_benchmark_comparison_bar(df_benchmark: pd.DataFrame) -> go.Figure:
    """
    Produces multi-metric comparative bar chart across all 5 benchmarked algorithms.
    """
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df_benchmark["Algorithm"],
        y=df_benchmark["Coverage %"],
        name="Population Coverage %",
        marker_color="#00A3E0"
    ))

    fig.add_trace(go.Bar(
        x=df_benchmark["Algorithm"],
        y=df_benchmark["Gini Improvement %"],
        name="Gini Equity Improvement %",
        marker_color="#00B140"
    ))

    fig.add_trace(go.Bar(
        x=df_benchmark["Algorithm"],
        y=df_benchmark["DTII Reduction %"],
        name="DTII Inequality Reduction %",
        marker_color="#FF8C00"
    ))

    fig.update_layout(
        title="<b>Algorithm Benchmarking: Coverage vs. Equity & DTII Gains</b>",
        barmode="group",
        yaxis_title="Percentage (%)",
        template="plotly_white",
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig


def create_pareto_tradeoff_chart(eval_results: Any) -> go.Figure:
    """
    Produces a 2D Pareto trade-off scatter plot showing Coverage Improvement vs Equity/DTII Improvement.
    Uses clean interactive markers with hovertooltips to prevent text collisions.
    """
    fig = go.Figure()

    rows = []
    if eval_results is not None:
        if hasattr(eval_results, "to_dict"):
            try:
                rows = eval_results.to_dict(orient="records")
            except Exception:
                rows = []
        elif isinstance(eval_results, (list, tuple)):
            rows = list(eval_results)

    # Distinct color palette for clear algorithm identification
    algo_colors = {
        "Greedy Maximum Coverage": "#3B82F6",
        "MCLP (OR-Tools Exact MILP)": "#EF4444",
        "P-Median (OR-Tools Exact MILP)": "#10B981",
        "Genetic Algorithm (Equity-GA)": "#8B5CF6",
        "Particle Swarm Optimization (Discrete PSO)": "#F59E0B"
    }

    for i, res in enumerate(rows):
        if not isinstance(res, dict):
            continue
        algo = res.get("algorithm", res.get("Algorithm", f"Solver {i+1}"))
        cov = float(res.get("coverage_pct", res.get("Coverage %", 0.0)))
        dtii = float(res.get("dtii_reduction_pct", res.get("DTII Reduction %", 0.0)))
        runtime = float(res.get("execution_time_sec", res.get("Runtime (sec)", 0.1)))
        gini = float(res.get("gini_improvement_pct", res.get("Gini Improvement %", 0.0)))
        color = algo_colors.get(algo, "#2563EB")

        fig.add_trace(go.Scatter(
            x=[cov],
            y=[dtii],
            mode="markers",
            name=algo,
            hovertemplate=(
                f"<b>{algo}</b><br><br>"
                f"• Population Coverage: <b>{cov:.1f}%</b><br>"
                f"• DTII Inequality Reduction: <b>{dtii:.2f}%</b><br>"
                f"• Gini Equity Gain: <b>{gini:.1f}%</b><br>"
                f"• Execution Runtime: <b>{runtime:.3f}s</b>"
                "<extra></extra>"
            ),
            marker=dict(
                size=22,
                color=color,
                line=dict(width=2.5, color="#1E293B"),
                opacity=0.92
            )
        ))

    fig.update_layout(
        title="<b>Trade-Off: Population Coverage vs. Helping Neglected Areas</b>",
        xaxis=dict(
            title="Population Coverage (%)",
            range=[96.0, 104.0],
            dtick=2,
            gridcolor="#E2E8F0"
        ),
        yaxis=dict(
            title="Inequality (DTII) Reduction (%)",
            range=[-0.15, 1.65],
            gridcolor="#E2E8F0"
        ),
        template="plotly_white",
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=1.02,
            font=dict(size=11),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#CBD5E1",
            borderwidth=1
        ),
        hoverlabel=dict(
            bgcolor="#1E293B",
            font_size=12,
            font_color="#FFFFFF"
        ),
        margin=dict(l=50, r=40, t=50, b=50)
    )
    return fig


def create_stability_boxplot(df_raw: pd.DataFrame) -> go.Figure:
    """
    Renders box plot comparing multi-seed stability (distribution of fitness scores) for GA and PSO.
    """
    fig = px.box(
        df_raw,
        x="Algorithm",
        y="Fitness",
        color="Algorithm",
        points="all",
        title="<b>Run-to-Run Stability Analysis (Multi-Seed Distribution)</b>",
        labels={"Fitness": "Multi-Objective Fitness Score", "Algorithm": "Optimization Algorithm"}
    )
    fig.update_layout(
        template="plotly_white",
        showlegend=False,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig


def create_convergence_chart(history_dict: Dict[str, List[float]]) -> go.Figure:
    """
    Plots generation/iteration fitness progression for iterative optimizers (GA, PSO, Greedy).
    """
    fig = go.Figure()
    colors = {"GA": "#E84A27", "PSO": "#00529B", "Greedy": "#28813F"}

    for algo_name, curve in history_dict.items():
        fig.add_trace(go.Scatter(
            x=list(range(len(curve))),
            y=curve,
            mode="lines+markers",
            name=algo_name,
            line=dict(width=2, color=colors.get(algo_name))
        ))

    fig.update_layout(
        title="<b>Optimization Convergence Trajectory</b>",
        xaxis_title="Generation / Iteration / Step",
        yaxis_title="Objective Fitness Value",
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig


def create_shap_summary_bar(importance_ranking: List[Dict[str, Any]]) -> go.Figure:
    """
    Plots horizontal bar chart of SHAP feature importance contributions.
    """
    df = pd.DataFrame(importance_ranking).sort_values(by="importance", ascending=True)

    fig = go.Figure(go.Bar(
        x=df["importance"],
        y=df["feature"],
        orientation="h",
        marker_color="#4361EE"
    ))

    fig.update_layout(
        title="<b>SHAP Feature Attribution: Predictors of Transit Deprivation</b>",
        xaxis_title="Mean |SHAP Value| (Impact on Mobility Demand)",
        yaxis_title="Engineered Spatial Feature",
        template="plotly_white",
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig


def create_dtii_comparison_bar(df_sim_wards: pd.DataFrame) -> go.Figure:
    """
    Renders grouped bar chart comparing Baseline DTII vs Simulated DTII for top affected wards.
    """
    df_sorted = df_sim_wards.sort_values(by="dtii_delta", ascending=False).head(14)
    ward_labels = [f"Ward {w} ({n})" for w, n in zip(df_sorted["ward_id"], df_sorted.get("ward_name", df_sorted["ward_id"]))]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=ward_labels,
        y=df_sorted["baseline_dtii"],
        name="Baseline DTII (Before)",
        marker_color="#E63946"
    ))

    fig.add_trace(go.Bar(
        x=ward_labels,
        y=df_sorted["simulated_dtii"],
        name="Simulated DTII (After)",
        marker_color="#2A9D8F"
    ))

    fig.update_layout(
        title="<b>Dynamic Transit Inequality Index (DTII): Before vs. After by Ward</b>",
        barmode="group",
        yaxis_title="DTII Score (Higher = Greater Inequality)",
        template="plotly_white",
        legend=dict(x=0.75, y=0.98),
        margin=dict(l=40, r=40, t=50, b=90),
        xaxis_tickangle=-35
    )
    return fig


def create_ward_improvement_bar(df_sim_wards: pd.DataFrame) -> go.Figure:
    """
    Produces horizontal bar chart of accessibility boost and walking distance savings across wards.
    """
    df_sorted = df_sim_wards.sort_values(by="accessibility_delta", ascending=True).tail(10)
    ward_labels = [f"Ward {w} ({n})" for w, n in zip(df_sorted["ward_id"], df_sorted.get("ward_name", df_sorted["ward_id"]))]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=ward_labels,
        x=df_sorted["accessibility_delta"],
        orientation="h",
        name="Accessibility Score Boost (+pts)",
        marker_color="#00A3E0"
    ))

    fig.update_layout(
        title="<b>Top Beneficiary Wards: Net Accessibility Score Improvement</b>",
        xaxis_title="Accessibility Score Increase (Points)",
        yaxis_title="Municipal Ward",
        template="plotly_white",
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig

