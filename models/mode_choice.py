"""
UrbanFlow AI - Multimodal Mode Choice Model
Simulates commuter utility and mode split (Suburban Rail, BEST Bus, Metro, Auto/Cab).
"""

from typing import Dict
import numpy as np
import pandas as pd


class ModeChoiceModel:
    """Multinomial Logit Utility Model for Mumbai transit mode split."""

    MODES = ["suburban_rail", "best_bus", "mumbai_metro", "auto_cab"]

    def __init__(
        self,
        beta_time: float = -0.05,
        beta_cost: float = -0.02,
        asc_rail: float = 1.2,
        asc_metro: float = 1.5,
        asc_bus: float = 0.5,
        asc_auto: float = 0.0
    ):
        self.beta_time = beta_time
        self.beta_cost = beta_cost
        self.asc = {
            "suburban_rail": asc_rail,
            "mumbai_metro": asc_metro,
            "best_bus": asc_bus,
            "auto_cab": asc_auto
        }

    def predict_split(
        self,
        travel_times_min: Dict[str, float],
        travel_costs_inr: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculates choice probabilities:
        V_m = ASC_m + beta_time * Time_m + beta_cost * Cost_m
        P_m = exp(V_m) / sum_k(exp(V_k))
        """
        utilities = {}
        for m in self.MODES:
            time = travel_times_min.get(m, 45.0)
            cost = travel_costs_inr.get(m, 20.0)
            utilities[m] = self.asc[m] + (self.beta_time * time) + (self.beta_cost * cost)

        max_u = max(utilities.values())
        exp_u = {m: np.exp(u - max_u) for m, u in utilities.items()}
        sum_exp = sum(exp_u.values())

        return {m: round(exp_u[m] / sum_exp, 4) for m in self.MODES}

