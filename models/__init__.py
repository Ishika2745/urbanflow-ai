"""
UrbanFlow AI - Machine Learning & Demand Intelligence Package
Provides relative demand proxies, spatially-aware regressors, K-Means segmentation, and SHAP explainability.
"""

from models.mobility_demand import (
    DemandProxyCalculator,
    SpatiallyAwareModelBenchmark,
    DemandSegmentationEngine,
    DemandExplainer,
    DEFAULT_DEMAND_WEIGHTS
)
from models.demand_visualizer import DemandVisualizer
from models.run_demand_intelligence import run_demand_intelligence_pipeline

__all__ = [
    "DemandProxyCalculator",
    "SpatiallyAwareModelBenchmark",
    "DemandSegmentationEngine",
    "DemandExplainer",
    "DemandVisualizer",
    "DEFAULT_DEMAND_WEIGHTS",
    "run_demand_intelligence_pipeline"
]
