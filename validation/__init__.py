"""
UrbanFlow AI - Validation, Benchmarking & Decision Auditing Suite
DataNova 2026: Problem Statement DN-DS-01.

Implements:
- PART D: Analytical Validations (monotonicity, bounds, network routing, non-overlapping pop, missing data, candidate validity, constraints, reproducibility)
- PART E: Fair Algorithm Comparison (Greedy, MCLP, P-Median, GA, PSO under identical inputs, multi-seed stability, convergence curves)
- PART F: Baseline Comparisons (Euclidean vs Network, Fixed buffer vs Network, Pop-only vs DTII, Greedy vs MCLP, Conventional vs GA/PSO, Existing vs Simulated)
- PART G: Final Test & Validation Reporting
"""

from validation.analytical_validator import AnalyticalValidator, ValidationResult
from validation.baseline_comparisons import BaselineComparator

__all__ = [
    "AnalyticalValidator",
    "ValidationResult",
    "BaselineComparator"
]

