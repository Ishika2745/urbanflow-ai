"""
UrbanFlow AI - Machine Learning Passenger Demand Forecasting
Trains and evaluates XGBoost / Random Forest regressors to predict hourly OD passenger mobility volumes.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

from utils.logger import logger


class DemandForecastModel:
    """Predicts OD passenger commute volume using spatial, temporal, and demographic features."""

    def __init__(self, model_type: str = "xgboost"):
        self.model_type = model_type
        if model_type == "xgboost":
            self.model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.08,
                random_state=42
            )
        else:
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=6,
                random_state=42
            )
        self.feature_names = []
        self.is_trained = False

    def train(self, df_flows: pd.DataFrame) -> Dict[str, float]:
        """Trains model on OD pair mobility attributes."""
        # Feature columns
        feature_cols = [
            "distance_km",
            "transit_dependency_factor"
        ]
        if "origin_pop" in df_flows.columns:
            feature_cols.extend(["origin_pop", "dest_pop", "dest_attraction"])

        target_col = "morning_peak_trips" if "morning_peak_trips" in df_flows.columns else "daily_estimated_trips"

        X = df_flows[feature_cols]
        y = df_flows[target_col]
        self.feature_names = feature_cols

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        logger.info(f"Training {self.model_type} demand forecast regressor on {len(X_train)} samples...")
        self.model.fit(X_train, y_train)
        self.is_trained = True

        preds = self.model.predict(X_test)
        metrics = {
            "r2_score": round(float(r2_score(y_test, preds)), 3),
            "mae": round(float(mean_absolute_error(y_test, preds)), 2),
            "rmse": round(float(np.sqrt(mean_squared_error(y_test, preds))), 2)
        }
        logger.info(f"Model trained successfully. Evaluation metrics: {metrics}")
        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generates demand forecasts."""
        if not self.is_trained:
            raise ValueError("Model is not trained yet.")
        return self.model.predict(X[self.feature_names])

