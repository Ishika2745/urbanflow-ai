"""
Unit Tests for Transport Inequality Analytics
"""

import numpy as np
from analytics.spatial_inequality import (
    calculate_gini_coefficient,
    compute_lorenz_curve
)


def test_gini_coefficient_extremes():
    """Verify Gini = 0 for perfect equality and Gini > 0 for inequality."""
    # Perfect equality
    equal_vals = np.array([50.0, 50.0, 50.0, 50.0])
    gini_equal = calculate_gini_coefficient(equal_vals)
    assert abs(gini_equal) < 0.05

    # High inequality
    unequal_vals = np.array([5.0, 10.0, 15.0, 150.0])
    gini_unequal = calculate_gini_coefficient(unequal_vals)
    assert gini_unequal > 0.40


def test_lorenz_curve_monotonicity():
    """Verify Lorenz curve starts at 0, ends at 1, and is monotonically non-decreasing."""
    vals = np.array([10.0, 20.0, 40.0, 80.0])
    p, L = compute_lorenz_curve(vals)

    assert p[0] == 0.0 and p[-1] == 1.0
    assert L[0] == 0.0 and L[-1] == 1.0
    assert np.all(np.diff(L) >= -1e-6)

