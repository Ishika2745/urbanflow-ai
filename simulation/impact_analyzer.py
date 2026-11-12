"""
UrbanFlow AI - Socioeconomic Impact Analyzer
Translates simulation outputs into actionable policy metrics (CO2 offset, commute minutes saved, ROI).
"""

from typing import Dict, Any
import numpy as np


class SocioeconomicImpactAnalyzer:
    """Computes socioeconomic returns, average commute time savings, and carbon emissions reduction."""

    @staticmethod
    def compute_policy_roi(
        population_gain: int,
        gini_reduction: float,
        capital_cost_lakhs: float,
        benefiting_population: int = 0,
        avg_walk_dist_saved_m: float = 0.0,
        avg_trips_per_day: float = 2.4,
        avg_minutes_saved_per_trip: float = 14.0
    ) -> Dict[str, Any]:
        """
        Calculates social return on investment:
        - Commuter hours saved annually (modal shift + pedestrian walk-time reduction)
        - Annual economic productivity unlocked (based on average Mumbai urban wage ~ INR 120/hr)
        - Estimated metric tons of CO2 avoided via private-to-public modal shift
        """
        # 1. Daily commuters newly enabled by transit access (modal shift from 2-wheelers / walking)
        new_riders = int(population_gain * 0.45)
        annual_new_trips = new_riders * avg_trips_per_day * 300  # 300 working days

        # Modal shift commute minutes saved
        annual_shift_minutes = annual_new_trips * avg_minutes_saved_per_trip

        # 2. Existing commuters gaining shortened pedestrian walking distance
        walking_commuters = int(benefiting_population * 0.40)  # ~40% transit modal share in Mumbai
        walk_mins_saved_per_trip = (avg_walk_dist_saved_m / 1.2) / 60.0  # at 1.2 m/s
        annual_walk_minutes = walking_commuters * avg_trips_per_day * 300 * walk_mins_saved_per_trip

        total_annual_minutes = annual_shift_minutes + annual_walk_minutes
        annual_hours_saved = total_annual_minutes / 60.0

        # Economic value of time saved (50% of hourly wage = INR 60/hr)
        econ_value_inr = annual_hours_saved * 60.0
        econ_value_crores = round(econ_value_inr / 1e7, 2)

        # Carbon offset (Assuming 120g CO2 saved per shifted km, avg trip 7 km)
        annual_shifted_km = annual_new_trips * 7.0
        co2_tons_saved = round((annual_shifted_km * 0.120) / 1000.0, 1)

        # If zero shifted trips, baseline pedestrian walk-shift estimate
        if co2_tons_saved == 0.0 and benefiting_population > 0:
            # Feeder bus connectivity prevents short auto-rickshaw feeder trips (~1.5 km, 80g CO2)
            auto_trips_avoided = walking_commuters * 0.08 * avg_trips_per_day * 300
            co2_tons_saved = round((auto_trips_avoided * 1.5 * 0.080) / 1000.0, 1)

        cost_crores = capital_cost_lakhs / 100.0
        roi_years = round(cost_crores / max(0.1, econ_value_crores), 2)

        return {
            "new_daily_riders": new_riders,
            "annual_hours_saved": int(annual_hours_saved),
            "annual_economic_benefit_crores": econ_value_crores,
            "co2_metric_tons_avoided": co2_tons_saved,
            "capital_cost_crores": round(cost_crores, 2),
            "payback_period_years": roi_years,
            "gini_equity_improvement": round(gini_reduction, 4)
        }

