"""
UrbanFlow AI - Dynamic Transit Inequality Index (DTII) Engine
The central analytical contribution of UrbanFlow AI.

OBJECTIVE:
Identify neighbourhoods where transport accessibility and service availability
are inadequate relative to local mobility needs.

CONCEPTUAL FORMULATION:
    DTII_i = w1 * D_i + w2 * (1 - A_i) + w3 * (1 - S_i) + w4 * V_i

Where:
    - D_i: Normalized Relative Mobility Demand (trip generation & attraction)
    - (1 - A_i): Transport Accessibility Deficit (inverted Dijkstra walking access)
    - (1 - S_i): Public Transport Supply Deficit (inverted GTFS stop & trip frequency)
    - V_i: Social / Demographic Vulnerability (slum population share / poverty)
    - All components normalized in [0, 1]
    - Weights sum to 1.0 (w1 + w2 + w3 + w4 = 1.0)
    - Higher DTII indicates higher transport inequality.

DISCLAIMER:
The Dynamic Transit Inequality Index (DTII) is the analytical formulation proposed
by UrbanFlow AI for academic and municipal decision-support benchmarking (DataNova 2026, DN-DS-01).
It is not an officially gazetted government standard.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, List, Optional, Union
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.stats import spearmanr, pearsonr

from utils.logger import logger

DTII_DISCLAIMER = (
    "Notice: The Dynamic Transit Inequality Index (DTII) is the proposed UrbanFlow AI analytical index "
    "designed for municipal decision-support and urban mobility pattern analysis (DataNova 2026, DN-DS-01). "
    "It is not an officially gazetted government standard."
)


@dataclass
class DTIIWeights:
    """Configurable weights for the DTII composite formula."""
    w_demand: float = 0.30       # w1: Relative Demand
    w_access: float = 0.30       # w2: Accessibility Deficit (1 - A)
    w_supply: float = 0.20       # w3: Supply Deficit (1 - S)
    w_vulnerability: float = 0.20 # w4: Demographic Vulnerability (V)

    def __post_init__(self):
        self.normalize()

    def normalize(self) -> None:
        """Validates and ensures weights sum exactly to 1.0."""
        total = self.w_demand + self.w_access + self.w_supply + self.w_vulnerability
        if total <= 0:
            raise ValueError("Sum of DTII weights must be greater than zero.")
        if abs(total - 1.0) > 1e-5:
            logger.debug(f"Normalizing DTII weights (sum={total:.4f}) to 1.0000.")
            self.w_demand /= total
            self.w_access /= total
            self.w_supply /= total
            self.w_vulnerability /= total

    def to_dict(self) -> Dict[str, float]:
        return {
            "w_demand": round(self.w_demand, 4),
            "w_access": round(self.w_access, 4),
            "w_supply": round(self.w_supply, 4),
            "w_vulnerability": round(self.w_vulnerability, 4)
        }

    def to_reduced_3_component(self) -> "DTIIWeights":
        """
        Creates reduced 3-component weights when social vulnerability data is unavailable.
        Redistributes w4 proportionally across demand, access deficit, and supply deficit.
        """
        total_3 = self.w_demand + self.w_access + self.w_supply
        if total_3 <= 0:
            return DTIIWeights(w_demand=0.40, w_access=0.35, w_supply=0.25, w_vulnerability=0.0)
        return DTIIWeights(
            w_demand=self.w_demand / total_3,
            w_access=self.w_access / total_3,
            w_supply=self.w_supply / total_3,
            w_vulnerability=0.0
        )


class DTIICalculator:
    """Calculates, classifies, and explains the Dynamic Transit Inequality Index."""

    DISCLAIMER = DTII_DISCLAIMER

    # Pre-defined policy-tested weight configurations for sensitivity analysis
    SENSITIVITY_PROFILES: Dict[str, Dict[str, float]] = {
        "balanced_baseline": {
            "name": "Balanced Baseline (UrbanFlow AI)",
            "w_demand": 0.30, "w_access": 0.30, "w_supply": 0.20, "w_vulnerability": 0.20,
            "description": "Standard multi-criteria weighting balancing demand, physical network access, transit supply, and social equity."
        },
        "equal_weights": {
            "name": "Equal Weights (Neutral Prior)",
            "w_demand": 0.25, "w_access": 0.25, "w_supply": 0.25, "w_vulnerability": 0.25,
            "description": "Neutral unweighted configuration allocating identical 25% weight to each dimension."
        },
        "demand_prioritized": {
            "name": "Demand-Prioritized (Commuter Activity Focus)",
            "w_demand": 0.45, "w_access": 0.25, "w_supply": 0.15, "w_vulnerability": 0.15,
            "description": "Prioritizes high trip generation corridors, commercial hubs, and educational clusters."
        },
        "accessibility_prioritized": {
            "name": "Accessibility-Prioritized (Physical Barrier Focus)",
            "w_demand": 0.20, "w_access": 0.45, "w_supply": 0.20, "w_vulnerability": 0.15,
            "description": "Focuses on network walking detours, topographic barriers, and excessive pedestrian travel times."
        },
        "supply_prioritized": {
            "name": "Supply-Prioritized (Infrastructure Deficit Focus)",
            "w_demand": 0.20, "w_access": 0.20, "w_supply": 0.45, "w_vulnerability": 0.15,
            "description": "Emphasizes scheduled GTFS frequency deficits, low bus stop density, and lack of rail/metro stations."
        },
        "vulnerability_prioritized": {
            "name": "Vulnerability-Focused (Social Equity / Pro-Poor Focus)",
            "w_demand": 0.20, "w_access": 0.25, "w_supply": 0.15, "w_vulnerability": 0.40,
            "description": "Directs municipal prioritization toward high informal settlement shares and vulnerable communities."
        }
    }

    # Documented absolute classification thresholds [0.0 to 1.0]
    DEFAULT_THRESHOLDS: Dict[str, Tuple[float, float]] = {
        "Low Inequality": (0.00, 0.35),
        "Moderate Inequality": (0.35, 0.50),
        "High Inequality": (0.50, 0.65),
        "Critical Inequality": (0.65, 1.00)
    }

    def __init__(
        self,
        weights: Optional[Union[DTIIWeights, Dict[str, float]]] = None,
        normalization_method: str = "ratio",
        classification_method: str = "threshold"
    ):
        """
        Initializes the DTII Calculator.

        Args:
            weights: DTIIWeights instance or dict of weights. If None, uses balanced baseline.
            normalization_method: 'ratio' (scale / 100), 'minmax' ((x-min)/(max-min)), or 'robust' (quantile clipped).
            classification_method: 'threshold' (fixed documented cutoffs) or 'quantile' (4 quartiles).
        """
        if weights is None:
            self.weights = DTIIWeights()
        elif isinstance(weights, DTIIWeights):
            self.weights = weights
        elif isinstance(weights, dict):
            self.weights = DTIIWeights(
                w_demand=weights.get("w_demand", 0.30),
                w_access=weights.get("w_access", 0.30),
                w_supply=weights.get("w_supply", 0.20),
                w_vulnerability=weights.get("w_vulnerability", 0.20)
            )
        else:
            raise TypeError("weights must be DTIIWeights or dict.")

        self.normalization_method = normalization_method
        self.classification_method = classification_method

    @staticmethod
    def normalize_series(
        s: pd.Series,
        method: str = "ratio",
        expected_range: Tuple[float, float] = (0.0, 100.0)
    ) -> pd.Series:
        """
        Consistently normalizes a feature series strictly into [0.0, 1.0].
        Handles constant values, NaNs, and extreme outliers gracefully.
        """
        clean_s = s.fillna(s.median() if not s.dropna().empty else 0.0)

        if method == "ratio":
            # Scale division by expected maximum (e.g. 100.0)
            scale = expected_range[1] - expected_range[0]
            if scale <= 0:
                scale = 100.0
            norm = (clean_s - expected_range[0]) / scale
            return np.clip(norm, 0.0, 1.0).astype(float)

        elif method == "minmax":
            s_min = clean_s.min()
            s_max = clean_s.max()
            if s_max - s_min <= 1e-9:
                return pd.Series(0.5, index=s.index)
            norm = (clean_s - s_min) / (s_max - s_min)
            return np.clip(norm, 0.0, 1.0).astype(float)

        elif method == "robust":
            q01 = clean_s.quantile(0.01)
            q99 = clean_s.quantile(0.99)
            clipped = np.clip(clean_s, q01, q99)
            if q99 - q01 <= 1e-9:
                return pd.Series(0.5, index=s.index)
            norm = (clipped - q01) / (q99 - q01)
            return np.clip(norm, 0.0, 1.0).astype(float)

        else:
            raise ValueError(f"Unknown normalization method: {method}")

    def extract_and_normalize_components(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.Series, pd.Series, pd.Series, Optional[pd.Series], bool]:
        """
        Extracts and normalizes the 4 conceptual DTII components:
        1. Demand (D): trips generated & attracted
        2. Accessibility (A): multimodal shortest-path access
        3. Transport Supply (S): GTFS trip frequency & stop coverage
        4. Social Vulnerability (V): informal settlement / slum population share

        Returns:
            Tuple of (D_norm, A_norm, S_norm, V_norm, has_vulnerability)
        """
        # 1. Relative Demand (D)
        if "relative_demand_score" in df.columns:
            s_demand = df["relative_demand_score"]
        elif "demand_score" in df.columns:
            s_demand = df["demand_score"]
        else:
            # Fallback demand proxy from population density
            pop_dense = df.get("population_density_per_sq_km", pd.Series(10000.0, index=df.index))
            s_demand = (pop_dense / max(pop_dense.quantile(0.95), 1000.0)) * 50.0

        D_norm = self.normalize_series(s_demand, method=self.normalization_method, expected_range=(0.0, 100.0))

        # 2. Transport Accessibility (A)
        if "multimodal_accessibility_score" in df.columns:
            s_access = df["multimodal_accessibility_score"]
        elif "accessibility_score" in df.columns:
            s_access = df["accessibility_score"]
        else:
            s_access = pd.Series(30.0, index=df.index)

        A_norm = self.normalize_series(s_access, method=self.normalization_method, expected_range=(0.0, 100.0))

        # 3. Transport Supply (S)
        if "transport_supply_score" in df.columns:
            s_supply = df["transport_supply_score"]
        elif "supply_score" in df.columns:
            s_supply = df["supply_score"]
        else:
            s_supply = pd.Series(50.0, index=df.index)

        S_norm = self.normalize_series(s_supply, method=self.normalization_method, expected_range=(0.0, 100.0))

        # 4. Social / Demographic Vulnerability (V)
        # Evaluated ONLY when defensible data is available
        has_vulnerability = False
        V_norm = None

        vuln_col = None
        for candidate in ["vulnerability_index", "slum_population_pct", "slum_pct", "poverty_rate", "socioeconomic_vulnerability"]:
            if candidate in df.columns and not df[candidate].isna().all():
                vuln_col = candidate
                break

        if vuln_col is not None:
            s_vuln = df[vuln_col]
            # If slum pct is in [0, 1], ratio with range (0, 1), else (0, 100)
            max_val = s_vuln.max()
            v_range = (0.0, 1.0) if max_val <= 1.05 else (0.0, 100.0)
            V_norm = self.normalize_series(s_vuln, method="minmax" if self.normalization_method == "minmax" else "ratio", expected_range=v_range)
            has_vulnerability = True
            logger.debug(f"Demographic Vulnerability component extracted from '{vuln_col}'.")
        else:
            logger.info("Defensible demographic vulnerability data not available. Applying reduced-component DTII.")

        return D_norm, A_norm, S_norm, V_norm, has_vulnerability

    def calculate(
        self,
        df: Union[pd.DataFrame, gpd.GeoDataFrame],
        custom_weights: Optional[DTIIWeights] = None
    ) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
        """
        Executes the DTII formula on the spatial dataframe.

        Formula:
            DTII_i = w1 * D_i + w2 * (1 - A_i) + w3 * (1 - S_i) + w4 * V_i

        Outputs:
            - 'dtii': continuous score in [0.0, 1.0]
            - 'dtii_score': scaled index in [0.0, 100.0]
            - 'norm_demand': normalized demand D_i
            - 'accessibility_deficit': (1 - A_i)
            - 'supply_deficit': (1 - S_i)
            - 'norm_vulnerability': normalized vulnerability V_i (or 0.0)
            - 'contrib_demand', 'contrib_access', 'contrib_supply', 'contrib_vuln': absolute points
            - 'pct_contrib_demand', etc.: percentage share of total DTII
            - 'dominant_inequality_factor': highest contributing component
            - 'dtii_category': inequality classification
            - 'dtii_rank': priority ranking (1 = highest inequality)
        """
        res = df.copy()
        weights = custom_weights if custom_weights is not None else self.weights

        # Extract normalized components
        D_norm, A_norm, S_norm, V_norm, has_vuln = self.extract_and_normalize_components(res)

        # Invert accessibility and supply to represent DEFICITS
        acc_deficit = 1.0 - A_norm
        supply_deficit = 1.0 - S_norm

        if not has_vuln:
            # Reduced-component calculation: redistribute w4
            active_weights = weights.to_reduced_3_component()
            w1 = active_weights.w_demand
            w2 = active_weights.w_access
            w3 = active_weights.w_supply
            w4 = 0.0
            V_clean = pd.Series(0.0, index=res.index)
            is_reduced = True
        else:
            w1 = weights.w_demand
            w2 = weights.w_access
            w3 = weights.w_supply
            w4 = weights.w_vulnerability
            V_clean = V_norm
            is_reduced = False

        # Compute Absolute Component Contributions
        c_demand = w1 * D_norm
        c_access = w2 * acc_deficit
        c_supply = w3 * supply_deficit
        c_vuln = w4 * V_clean

        # Master Formula: DTII = w1*D + w2*(1-A) + w3*(1-S) + w4*V
        dtii_raw = c_demand + c_access + c_supply + c_vuln
        dtii_clean = np.clip(dtii_raw, 0.0, 1.0).round(4)

        # Percentage contributions (handling division by zero)
        total_non_zero = np.maximum(dtii_clean.values, 1e-6)
        pct_demand = np.round((c_demand / total_non_zero) * 100.0, 2)
        pct_access = np.round((c_access / total_non_zero) * 100.0, 2)
        pct_supply = np.round((c_supply / total_non_zero) * 100.0, 2)
        pct_vuln = np.round((c_vuln / total_non_zero) * 100.0, 2)

        # Identify dominant bottleneck factor
        dominant_factors = []
        contrib_matrix = np.column_stack([c_demand.values, c_access.values, c_supply.values, c_vuln.values])
        factor_names = [
            "High Commuter Demand",
            "Severe Accessibility Deficit",
            "Inadequate Transit Supply",
            "Demographic Vulnerability"
        ]
        for row in contrib_matrix:
            max_idx = int(np.argmax(row))
            dominant_factors.append(factor_names[max_idx])

        # Assign output columns
        res["dtii"] = dtii_clean
        res["dtii_score"] = np.round(dtii_clean * 100.0, 2)
        res["norm_demand"] = np.round(D_norm, 4)
        res["norm_accessibility"] = np.round(A_norm, 4)
        res["norm_supply"] = np.round(S_norm, 4)
        res["norm_vulnerability"] = np.round(V_clean, 4)
        res["accessibility_deficit"] = np.round(acc_deficit, 4)
        res["supply_deficit"] = np.round(supply_deficit, 4)

        res["contrib_demand"] = np.round(c_demand, 4)
        res["contrib_access"] = np.round(c_access, 4)
        res["contrib_supply"] = np.round(c_supply, 4)
        res["contrib_vuln"] = np.round(c_vuln, 4)

        res["pct_contrib_demand"] = pct_demand
        res["pct_contrib_access"] = pct_access
        res["pct_contrib_supply"] = pct_supply
        res["pct_contrib_vuln"] = pct_vuln

        res["dominant_inequality_factor"] = dominant_factors
        res["is_reduced_dtii"] = is_reduced

        # Synchronize backward-compatible transit_desert_index
        res["transit_desert_index"] = res["dtii_score"]

        # Classification
        res["dtii_category"] = self.classify(res["dtii"])
        res["dtii_rank"] = res["dtii"].rank(ascending=False, method="min").astype(int)

        logger.info(
            f"Calculated DTII across {len(res)} spatial units (Mean: {res['dtii'].mean():.3f}, "
            f"Max: {res['dtii'].max():.3f}, ReducedMode: {is_reduced})."
        )
        return res

    def classify(self, s_dtii: pd.Series) -> pd.Series:
        """
        Categorizes DTII into:
        - Low Inequality
        - Moderate Inequality
        - High Inequality
        - Critical Inequality
        """
        if self.classification_method == "quantile":
            # 4 Quartiles
            labels = ["Low Inequality", "Moderate Inequality", "High Inequality", "Critical Inequality"]
            try:
                return pd.qcut(s_dtii, q=4, labels=labels, duplicates="drop")
            except Exception:
                # Fallback to threshold
                pass

        # Threshold-based classification
        categories = []
        for val in s_dtii:
            if val >= 0.65:
                categories.append("Critical Inequality")
            elif val >= 0.50:
                categories.append("High Inequality")
            elif val >= 0.35:
                categories.append("Moderate Inequality")
            else:
                categories.append("Low Inequality")
        return pd.Series(categories, index=s_dtii.index)

    def verify_monotonicity(self) -> Dict[str, bool]:
        """
        Verifies the mathematical monotonicity axioms of the active DTII formulation:
        - Higher Demand (D) -> Higher DTII (dDTII/dD > 0)
        - Higher Accessibility (A) -> Lower DTII (dDTII/dA < 0)
        - Higher Supply (S) -> Lower DTII (dDTII/dS < 0)
        - Higher Vulnerability (V) -> Higher DTII (dDTII/dV > 0)
        """
        w = self.weights

        def eval_dtii(d: float, a: float, s: float, v: float) -> float:
            return w.w_demand * d + w.w_access * (1.0 - a) + w.w_supply * (1.0 - s) + w.w_vulnerability * v

        base = eval_dtii(0.5, 0.5, 0.5, 0.5)
        return {
            "demand_monotonic": (eval_dtii(0.8, 0.5, 0.5, 0.5) > base) if w.w_demand > 0 else True,
            "access_monotonic": (eval_dtii(0.5, 0.8, 0.5, 0.5) < base) if w.w_access > 0 else True,
            "supply_monotonic": (eval_dtii(0.5, 0.5, 0.8, 0.5) < base) if w.w_supply > 0 else True,
            "vuln_monotonic": (eval_dtii(0.5, 0.5, 0.5, 0.8) > base) if w.w_vulnerability > 0 else True,
        }

    def run_sensitivity_analysis(
        self,
        df: pd.DataFrame,
        profiles: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Performs rigorous sensitivity analysis by comparing baseline rankings
        against alternative weight schemes.
        Computes Spearman's rho, Pearson's r, mean rank displacement, and max shift.
        """
        test_profiles = profiles if profiles is not None else self.SENSITIVITY_PROFILES

        # 1. Baseline calculation
        baseline_df = self.calculate(df)
        baseline_dtii = baseline_df["dtii"].values
        baseline_ranks = baseline_df["dtii_rank"].values

        profile_results = []
        rank_matrix = {baseline_df.get("ward_id", baseline_df.index).tolist()[i]: {"baseline": int(baseline_ranks[i])} for i in range(len(baseline_ranks))}

        for key, p_data in test_profiles.items():
            w = DTIIWeights(
                w_demand=p_data["w_demand"],
                w_access=p_data["w_access"],
                w_supply=p_data["w_supply"],
                w_vulnerability=p_data["w_vulnerability"]
            )
            sim_df = self.calculate(df, custom_weights=w)
            sim_dtii = sim_df["dtii"].values
            sim_ranks = sim_df["dtii_rank"].values

            # Statistical alignment metrics
            rho, p_rho = spearmanr(baseline_ranks, sim_ranks)
            r, p_r = pearsonr(baseline_dtii, sim_dtii)
            rank_diff = np.abs(baseline_ranks - sim_ranks)
            mean_shift = float(np.mean(rank_diff))
            max_shift = int(np.max(rank_diff))

            profile_results.append({
                "profile_key": key,
                "profile_name": p_data["name"],
                "weights": w.to_dict(),
                "spearman_rho": round(float(rho), 4),
                "pearson_r": round(float(r), 4),
                "mean_rank_displacement": round(mean_shift, 2),
                "max_rank_displacement": max_shift,
                "stability_grade": "High Robustness" if rho >= 0.90 else ("Moderate Robustness" if rho >= 0.75 else "Sensitive")
            })

            # Record ranks for each area
            for i, area_key in enumerate(rank_matrix.keys()):
                rank_matrix[area_key][key] = int(sim_ranks[i])

        summary_df = pd.DataFrame(profile_results)
        return {
            "summary_table": summary_df,
            "rank_matrix": pd.DataFrame.from_dict(rank_matrix, orient="index"),
            "baseline_ranks": baseline_ranks,
            "num_profiles_tested": len(test_profiles)
        }

    @staticmethod
    def calculate_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates correlation between candidate proxies to ensure no double-counting.
        """
        candidates = [
            "relative_demand_score", "multimodal_accessibility_score",
            "transport_supply_score", "vulnerability_index", "slum_population_pct",
            "population_density_per_sq_km", "net_walk_dist_transit_m"
        ]
        avail = [c for c in candidates if c in df.columns]
        if len(avail) < 2:
            return pd.DataFrame()
        return df[avail].corr(method="pearson").round(3)

    @staticmethod
    def explain_area_inequality(
        df_or_row: Union[pd.DataFrame, pd.Series],
        area_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Dynamic neighbourhood explanation answering:
        "Why does this area have a high transport inequality score?"
        Uses actual calculated values.
        """
        if isinstance(df_or_row, pd.DataFrame):
            if area_id is not None:
                matches = df_or_row[(df_or_row.get("ward_id") == area_id) | (df_or_row.get("cell_id") == area_id)]
                if matches.empty:
                    row = df_or_row.iloc[0]
                else:
                    row = matches.iloc[0]
            else:
                row = df_or_row.iloc[0]
        else:
            row = df_or_row

        # Extract values
        name = row.get("ward_name", row.get("cell_id", "Selected Neighbourhood"))
        w_id = row.get("ward_id", "")
        dtii = float(row.get("dtii", 0.0))
        dtii_score = float(row.get("dtii_score", dtii * 100.0))
        rank = int(row.get("dtii_rank", 1))
        category = str(row.get("dtii_category", "High Inequality"))
        pop = int(row.get("population", 0))

        # Normalized values
        D_norm = float(row.get("norm_demand", 0.0))
        acc_def = float(row.get("accessibility_deficit", 0.0))
        sup_def = float(row.get("supply_deficit", 0.0))
        V_norm = float(row.get("norm_vulnerability", 0.0))

        # Component contributions
        pct_D = float(row.get("pct_contrib_demand", 0.0))
        pct_A = float(row.get("pct_contrib_access", 0.0))
        pct_S = float(row.get("pct_contrib_supply", 0.0))
        pct_V = float(row.get("pct_contrib_vuln", 0.0))
        dominant = str(row.get("dominant_inequality_factor", "Severe Accessibility Deficit"))

        # Physical context
        fac = str(row.get("nearest_facility_name", "Local Transit Stop"))
        walk_m = float(row.get("net_walk_dist_transit_m", 800.0))
        rec = str(row.get("recommended_intervention", "Feeder Bus Expansion"))

        # Structured diagnostic text
        explanation_markdown = f"""
### 🔍 UrbanFlow AI Inequality Audit: {name} (Ward {w_id})

**Overall Evaluation:**  
{name} exhibits a **DTII score of {dtii:.3f}** (`{dtii_score:.1f} / 100`), ranking **#{rank}** across Greater Mumbai, classified under **{category}**.

#### 1. Why does this area have a high transport inequality score?
The primary driver of inequality in {name} is **{dominant}**, which accounts for **{max(pct_D, pct_A, pct_S, pct_V):.1f}%** of the total inequality index.

- **Commuter Mobility Demand ($D_i = {D_norm:.2f}$):** Contributes **{pct_D:.1f}%** ({row.get('contrib_demand', 0.0):.3f} pts). Represents trip generation from {pop:,} residents, commercial offices, and educational centers.
- **Accessibility Deficit ($1 - A_i = {acc_def:.2f}$):** Contributes **{pct_A:.1f}%** ({row.get('contrib_access', 0.0):.3f} pts). Residents must walk an average of **{walk_m:.0f} metres** to reach the nearest rapid transit facility (`{fac}`).
- **Public Transport Supply Deficit ($1 - S_i = {sup_def:.2f}$):** Contributes **{pct_S:.1f}%** ({row.get('contrib_supply', 0.0):.3f} pts). Reflects insufficient bus/metro stop density and off-peak headway delays.
- **Social & Demographic Vulnerability ($V_i = {V_norm:.2f}$):** Contributes **{pct_V:.1f}%** ({row.get('contrib_vuln', 0.0):.3f} pts). Captures informal settlement concentrations where affordability and transit dependency are acute.

#### 2. Actionable Engineering Prescription
**Recommended Municipal Intervention:** **{rec}**  
Targeted infrastructure investment addressing the dominant **{dominant.lower()}** will achieve the steepest reduction in transport inequality for this community.

> *Methodological Note:* {DTII_DISCLAIMER}
        """

        return {
            "area_name": name,
            "ward_id": w_id,
            "dtii": dtii,
            "dtii_score": dtii_score,
            "dtii_rank": rank,
            "category": category,
            "dominant_factor": dominant,
            "population": pop,
            "components": {
                "demand": {"normalized": D_norm, "pct_contribution": pct_D},
                "accessibility_deficit": {"normalized": acc_def, "pct_contribution": pct_A},
                "supply_deficit": {"normalized": sup_def, "pct_contribution": pct_S},
                "vulnerability": {"normalized": V_norm, "pct_contribution": pct_V}
            },
            "nearest_facility": fac,
            "walk_distance_m": walk_m,
            "recommended_intervention": rec,
            "narrative_explanation": explanation_markdown.strip()
        }

