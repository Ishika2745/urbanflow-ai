"""
UrbanFlow AI - Explainability and Intelligent Decision-Support Module
DataNova 2026: Problem Statement DN-DS-01.

Provides transparent, human-interpretable decision-support for municipal transit planners,
translating complex spatial statistics, nature-inspired optimization outputs, and machine learning
forecasts into clear, actionable natural-language explanations and quantitative trade-off audits.
"""

from explainability.decision_support import (
    TransitDecisionSupport,
    NeighbourhoodExplainer,
    OptimizationExplainer,
    AlgorithmComparisonExplainer,
    NeighbourhoodExplanation,
    OptimizationExplanation,
    AlgorithmTradeoffExplanation,
    get_decision_support_engine
)

__all__ = [
    "TransitDecisionSupport",
    "NeighbourhoodExplainer",
    "OptimizationExplainer",
    "AlgorithmComparisonExplainer",
    "NeighbourhoodExplanation",
    "OptimizationExplanation",
    "AlgorithmTradeoffExplanation",
    "get_decision_support_engine"
]

