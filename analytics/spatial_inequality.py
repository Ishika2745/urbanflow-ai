"""
UrbanFlow AI - Spatial Transport Inequality Analytics
Computes Gini Coefficient, Lorenz Curve Coordinates, Palma Ratio, and Theil Index for urban mobility.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from utils.logger import logger


def calculate_gini_coefficient(values: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    """
    Computes the Gini coefficient of inequality for transit accessibility or trip supply.
    
    Args:
        values: 1D array of accessibility scores or transit supplies per zone.
        weights: Optional population weights for each zone.
        
    Returns:
        float between 0.0 (perfect spatial equality) and 1.0 (maximal spatial inequality).
    """
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return 0.0

    if weights is None:
        weights = np.ones_like(v)
    else:
        weights = np.asarray(weights, dtype=float)

    # Sort values and align weights
    sorted_indices = np.argsort(v)
    v_sorted = v[sorted_indices]
    w_sorted = weights[sorted_indices]

    cum_w = np.cumsum(w_sorted)
    cum_vw = np.cumsum(v_sorted * w_sorted)

    total_w = cum_w[-1]
    total_vw = cum_vw[-1]

    if total_vw == 0 or total_w == 0:
        return 0.0

    # Normalized cumulative shares
    p = cum_w / total_w
    L = cum_vw / total_vw

    # Gini formula using trapezoidal integration
    # G = 1 - sum_{i=1}^n (p_i - p_{i-1}) * (L_i + L_{i-1})
    p_prev = np.insert(p[:-1], 0, 0.0)
    L_prev = np.insert(L[:-1], 0, 0.0)

    area_under_lorenz = np.sum((p - p_prev) * (L + L_prev) * 0.5)
    gini = 1.0 - 2.0 * area_under_lorenz
    return float(np.clip(gini, 0.0, 1.0))


def compute_lorenz_curve(values: np.ndarray, weights: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates (p, L) cumulative population vs cumulative transit opportunity coordinates
    for rendering interactive Plotly Lorenz curves.
    """
    v = np.asarray(values, dtype=float)
    if weights is None:
        weights = np.ones_like(v)
    else:
        weights = np.asarray(weights, dtype=float)

    sorted_indices = np.argsort(v)
    v_sorted = v[sorted_indices]
    w_sorted = weights[sorted_indices]

    cum_w = np.cumsum(w_sorted)
    cum_vw = np.cumsum(v_sorted * w_sorted)

    total_w = cum_w[-1]
    total_vw = cum_vw[-1]

    if total_vw == 0:
        p = np.linspace(0, 1, len(v) + 1)
        return p, p

    p = np.insert(cum_w / total_w, 0, 0.0)
    L = np.insert(cum_vw / total_vw, 0, 0.0)
    return p, L


def calculate_transport_equity_report(df: pd.DataFrame, score_col: str = "accessibility_score", pop_col: str = "population") -> Dict[str, Any]:
    """
    Generates comprehensive equity metrics across city wards.
    """
    scores = df[score_col].values
    pops = df[pop_col].values

    gini = calculate_gini_coefficient(scores, weights=pops)
    p_curve, l_curve = compute_lorenz_curve(scores, weights=pops)

    # Palma ratio: Share of top 10% accessible population / share of bottom 40%
    sorted_df = df.sort_values(by=score_col).reset_index(drop=True)
    sorted_df["cum_pop"] = sorted_df[pop_col].cumsum()
    total_pop = sorted_df[pop_col].sum()
    
    bottom_40 = sorted_df[sorted_df["cum_pop"] <= 0.40 * total_pop]
    top_10 = sorted_df[sorted_df["cum_pop"] >= 0.90 * total_pop]

    bottom_40_share = (bottom_40[score_col] * bottom_40[pop_col]).sum() / max(1e-6, (sorted_df[score_col] * sorted_df[pop_col]).sum())
    top_10_share = (top_10[score_col] * top_10[pop_col]).sum() / max(1e-6, (sorted_df[score_col] * sorted_df[pop_col]).sum())

    palma_ratio = float(top_10_share / max(1e-6, bottom_40_share))

    return {
        "gini_coefficient": round(gini, 4),
        "palma_ratio": round(palma_ratio, 3),
        "lorenz_p": p_curve.tolist(),
        "lorenz_l": l_curve.tolist(),
        "mean_score": round(float(np.average(scores, weights=pops)), 2),
        "min_ward_score": round(float(np.min(scores)), 2),
        "max_ward_score": round(float(np.max(scores)), 2)
    }
