"""
UrbanFlow AI - Model Explainability with SHAP
Computes feature attributions and Shapley value summaries for demand and equity models.
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
import shap
from utils.logger import logger


class ModelExplainer:
    """Provides SHAP-based feature importance and local attribution explanations."""

    def __init__(self, model_obj, feature_names: List[str]):
        self.model = model_obj
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None

    def explain(self, X_sample: pd.DataFrame) -> Dict[str, Any]:
        """Calculates mean absolute SHAP feature importance values."""
        logger.info(f"Computing SHAP explanations on {len(X_sample)} sample rows...")
        X_df = X_sample[self.feature_names]
        
        try:
            self.explainer = shap.TreeExplainer(self.model)
            self.shap_values = self.explainer.shap_values(X_df)
        except Exception:
            # Fallback for models without direct TreeExplainer hooks
            self.explainer = shap.Explainer(self.model, X_df)
            self.shap_values = self.explainer(X_df).values

        # Mean absolute SHAP values per feature
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
        importance_ranking = [
            {"feature": feat, "importance": round(float(val), 4)}
            for feat, val in sorted(zip(self.feature_names, mean_abs_shap), key=lambda x: x[1], reverse=True)
        ]

        logger.info(f"Top SHAP feature drivers: {importance_ranking}")
        return {
            "importance_ranking": importance_ranking,
            "feature_names": self.feature_names
        }

