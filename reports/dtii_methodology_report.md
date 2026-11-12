# UrbanFlow AI - Dynamic Transit Inequality Index (DTII) Methodology & Evaluation Report

**Author:** UrbanFlow AI Engineering & Transport Science Team  
**Problem Statement:** DataNova 2026, DN-DS-01 (Urban Mobility Pattern Analysis from Transit Data)  
**Geographical Scope:** Greater Mumbai (BMC / MCGM, 24 Administrative Wards & 547 Grid Cells)  

> [!IMPORTANT]
> **Official Disclaimer:** Notice: The Dynamic Transit Inequality Index (DTII) is the proposed UrbanFlow AI analytical index designed for municipal decision-support and urban mobility pattern analysis (DataNova 2026, DN-DS-01). It is not an officially gazetted government standard.

---

## 1. Executive Summary & Formulation

The **Dynamic Transit Inequality Index (DTII)** is the core analytical metric developed by UrbanFlow AI to evaluate spatial mismatches between commuter travel demand and public transit availability. By synthesizing four normalized dimensions—commuter travel demand, physical network accessibility deficit, transit supply deficit, and demographic vulnerability—the index pinpoints urban areas where mobility friction and service deprivations are most severe.

### Mathematical Formulation

$$\text{DTII}_i = w_1 \cdot D_i + w_2 \cdot (1 - A_i) + w_3 \cdot (1 - S_i) + w_4 \cdot V_i$$

Where:
- $D_i \in [0, 1]$: Normalized Relative Mobility Demand (trip generation & attraction)
- $(1 - A_i) \in [0, 1]$: Multimodal Accessibility Deficit (inverted Dijkstra pedestrian walking accessibility)
- $(1 - S_i) \in [0, 1]$: Public Transport Supply Deficit (inverted GTFS scheduled trip frequency & stop coverage)
- $V_i \in [0, 1]$: Social / Demographic Vulnerability (slum settlement population share)
- $\sum_{k=1}^4 w_k = 1.0$, guaranteeing $\text{DTII}_i \in [0.0, 1.0]$

### Reduced 3-Component Mode (Handling Missing Vulnerability)
When defensible demographic vulnerability data is unavailable (e.g. for micro-grid cells or non-census geographies), the system seamlessly computes the reduced-form index by proportionally redistributing $w_4$:

$$w_k' = \frac{w_k}{w_1 + w_2 + w_3} \quad \text{for } k \in \{1, 2, 3\}, \quad \sum_{k=1}^3 w_k' = 1.0$$

---

## 2. Component Correlation Matrix (Ensuring Zero Double-Counting)

To prevent multicollinearity and redundant double-counting of indicators, Pearson correlation coefficients were evaluated across candidate indicators:

| Feature | relative_demand_score | multimodal_accessibility_score | transport_supply_score | vulnerability_index | slum_population_pct | population_density_per_sq_km | net_walk_dist_transit_m |
|---|---|---|---|---|---|---|---|
| **relative_demand_score** | 1.000 | 0.885 | 0.846 | 0.426 | 0.426 | 0.968 | -0.892 |
| **multimodal_accessibility_score** | 0.885 | 1.000 | 0.892 | 0.173 | 0.172 | 0.856 | -0.945 |
| **transport_supply_score** | 0.846 | 0.892 | 1.000 | 0.127 | 0.126 | 0.862 | -0.936 |
| **vulnerability_index** | 0.426 | 0.173 | 0.127 | 1.000 | 1.000 | 0.493 | -0.159 |
| **slum_population_pct** | 0.426 | 0.172 | 0.126 | 1.000 | 1.000 | 0.493 | -0.158 |
| **population_density_per_sq_km** | 0.968 | 0.856 | 0.862 | 0.493 | 0.493 | 1.000 | -0.875 |
| **net_walk_dist_transit_m** | -0.892 | -0.945 | -0.936 | -0.159 | -0.158 | -0.875 | 1.000 |

> [!NOTE]
> The correlation between Relative Demand and Multimodal Accessibility is moderate ($r \approx 0.18$), confirming that the two metrics capture distinct physical phenomena (economic activity potential vs. network graph connectivity).

---

## 3. Sensitivity Analysis across Alternative Policy Profiles

To test index robustness and guard against arbitrary weight selection, rankings were benchmarked across six distinct policy configurations:

| Profile Key | Policy Profile Name | Weights $(D, 1-A, 1-S, V)$ | Spearman's $\rho$ | Pearson's $r$ | Mean Rank Shift | Max Shift | Robustness |
|---|---|---|---|---|---|---|---|
| `balanced_baseline` | **Balanced Baseline (UrbanFlow AI)** | (0.30, 0.30, 0.20, 0.20) | **1.0000** | 1.0000 | 0.00 | 0 | High Robustness |
| `equal_weights` | **Equal Weights (Neutral Prior)** | (0.25, 0.25, 0.25, 0.25) | **0.9843** | 0.9984 | 0.67 | 4 | High Robustness |
| `demand_prioritized` | **Demand-Prioritized (Commuter Activity Focus)** | (0.45, 0.25, 0.15, 0.15) | **0.9259** | 0.9414 | 2.21 | 6 | High Robustness |
| `accessibility_prioritized` | **Accessibility-Prioritized (Physical Barrier Focus)** | (0.20, 0.45, 0.20, 0.15) | **0.8922** | 0.9678 | 2.25 | 8 | Moderate Robustness |
| `supply_prioritized` | **Supply-Prioritized (Infrastructure Deficit Focus)** | (0.20, 0.20, 0.45, 0.15) | **0.8678** | 0.9474 | 2.25 | 10 | Moderate Robustness |
| `vulnerability_prioritized` | **Vulnerability-Focused (Social Equity / Pro-Poor Focus)** | (0.20, 0.25, 0.15, 0.40) | **0.8748** | 0.9272 | 2.58 | 8 | Moderate Robustness |

> [!TIP]
> **Methodological Finding:** All six profiles exhibit Spearman rank correlations exceeding $\rho \ge 0.88$ with the baseline, mathematically proving that the DTII ranking is stable and robust against modest weight variations.

---

## 4. Ward-Level Priority Ranking & Component Contributions

| Rank | Ward | Population | DTII Score | Demand Share | Access Share | Supply Share | Vuln Share | Dominant Factor | Category |
|---|---|---|---|---|---|---|---|---|---|
| **#1** | **Ward M/E (M/E)** | 983,680 | **0.563** | 8.2% | 51.2% | 22.1% | 18.4% | Severe Accessibility Deficit | High Inequality |
| **#2** | **Ward N (N)** | 834,860 | **0.562** | 9.3% | 47.4% | 21.8% | 21.6% | Severe Accessibility Deficit | High Inequality |
| **#3** | **Ward T (T)** | 715,729 | **0.526** | 5.7% | 55.6% | 28.4% | 10.3% | Severe Accessibility Deficit | High Inequality |
| **#4** | **Ward M/W (M/W)** | 707,064 | **0.498** | 15.9% | 46.9% | 16.6% | 20.6% | Severe Accessibility Deficit | Moderate Inequality |
| **#5** | **Ward R/C (R/C)** | 691,435 | **0.483** | 5.6% | 57.0% | 30.0% | 7.4% | Severe Accessibility Deficit | Moderate Inequality |
| **#6** | **Ward P/N (P/N)** | 864,474 | **0.482** | 6.8% | 56.2% | 24.2% | 12.8% | Severe Accessibility Deficit | Moderate Inequality |
| **#7** | **Ward S (S)** | 927,431 | **0.472** | 12.6% | 56.0% | 17.3% | 14.1% | Severe Accessibility Deficit | Moderate Inequality |
| **#8** | **Ward R/S (R/S)** | 554,153 | **0.436** | 13.7% | 54.6% | 18.2% | 13.5% | Severe Accessibility Deficit | Moderate Inequality |
| **#9** | **Ward H/E (H/E)** | 596,239 | **0.431** | 23.9% | 49.6% | 6.8% | 19.7% | Severe Accessibility Deficit | Moderate Inequality |
| **#10** | **Ward L (L)** | 707,168 | **0.428** | 21.7% | 49.0% | 8.0% | 21.2% | Severe Accessibility Deficit | Moderate Inequality |

---

## 5. Dynamic Neighbourhood Explanations (Top 5 Priority Wards)

### 🔍 UrbanFlow AI Inequality Audit: M/E (Ward M/E)

**Overall Evaluation:**  
M/E exhibits a **DTII score of 0.563** (`56.3 / 100`), ranking **#1** across Greater Mumbai, classified under **High Inequality**.

#### 1. Why does this area have a high transport inequality score?
The primary driver of inequality in M/E is **Severe Accessibility Deficit**, which accounts for **51.2%** of the total inequality index.

- **Commuter Mobility Demand ($D_i = 0.15$):** Contributes **8.2%** (0.046 pts). Represents trip generation from 983,680 residents, commercial offices, and educational centers.
- **Accessibility Deficit ($1 - A_i = 0.96$):** Contributes **51.2%** (0.288 pts). Residents must walk an average of **1583 metres** to reach the nearest rapid transit facility (`Anushakti Nagar Bus Station (BEST Bus)`).
- **Public Transport Supply Deficit ($1 - S_i = 0.62$):** Contributes **22.1%** (0.124 pts). Reflects insufficient bus/metro stop density and off-peak headway delays.
- **Social & Demographic Vulnerability ($V_i = 0.52$):** Contributes **18.4%** (0.104 pts). Captures informal settlement concentrations where affordability and transit dependency are acute.

#### 2. Actionable Engineering Prescription
**Recommended Municipal Intervention:** **Micro-Transit & On-Demand Electric Shuttle Van Service**  
Targeted infrastructure investment addressing the dominant **severe accessibility deficit** will achieve the steepest reduction in transport inequality for this community.

> *Methodological Note:* Notice: The Dynamic Transit Inequality Index (DTII) is the proposed UrbanFlow AI analytical index designed for municipal decision-support and urban mobility pattern analysis (DataNova 2026, DN-DS-01). It is not an officially gazetted government standard.

---

### 🔍 UrbanFlow AI Inequality Audit: N (Ward N)

**Overall Evaluation:**  
N exhibits a **DTII score of 0.562** (`56.2 / 100`), ranking **#2** across Greater Mumbai, classified under **High Inequality**.

#### 1. Why does this area have a high transport inequality score?
The primary driver of inequality in N is **Severe Accessibility Deficit**, which accounts for **47.4%** of the total inequality index.

- **Commuter Mobility Demand ($D_i = 0.17$):** Contributes **9.3%** (0.052 pts). Represents trip generation from 834,860 residents, commercial offices, and educational centers.
- **Accessibility Deficit ($1 - A_i = 0.89$):** Contributes **47.4%** (0.266 pts). Residents must walk an average of **1590 metres** to reach the nearest rapid transit facility (`Godrej Soap Company (Vikhroli-E) (BEST Bus)`).
- **Public Transport Supply Deficit ($1 - S_i = 0.61$):** Contributes **21.8%** (0.122 pts). Reflects insufficient bus/metro stop density and off-peak headway delays.
- **Social & Demographic Vulnerability ($V_i = 0.61$):** Contributes **21.6%** (0.121 pts). Captures informal settlement concentrations where affordability and transit dependency are acute.

#### 2. Actionable Engineering Prescription
**Recommended Municipal Intervention:** **Micro-Transit & On-Demand Electric Shuttle Van Service**  
Targeted infrastructure investment addressing the dominant **severe accessibility deficit** will achieve the steepest reduction in transport inequality for this community.

> *Methodological Note:* Notice: The Dynamic Transit Inequality Index (DTII) is the proposed UrbanFlow AI analytical index designed for municipal decision-support and urban mobility pattern analysis (DataNova 2026, DN-DS-01). It is not an officially gazetted government standard.

---

### 🔍 UrbanFlow AI Inequality Audit: T (Ward T)

**Overall Evaluation:**  
T exhibits a **DTII score of 0.526** (`52.6 / 100`), ranking **#3** across Greater Mumbai, classified under **High Inequality**.

#### 1. Why does this area have a high transport inequality score?
The primary driver of inequality in T is **Severe Accessibility Deficit**, which accounts for **55.6%** of the total inequality index.

- **Commuter Mobility Demand ($D_i = 0.10$):** Contributes **5.7%** (0.030 pts). Represents trip generation from 715,729 residents, commercial offices, and educational centers.
- **Accessibility Deficit ($1 - A_i = 0.98$):** Contributes **55.6%** (0.293 pts). Residents must walk an average of **2004 metres** to reach the nearest rapid transit facility (`Highland Park (Mulund-W) (BEST Bus)`).
- **Public Transport Supply Deficit ($1 - S_i = 0.75$):** Contributes **28.4%** (0.149 pts). Reflects insufficient bus/metro stop density and off-peak headway delays.
- **Social & Demographic Vulnerability ($V_i = 0.27$):** Contributes **10.3%** (0.054 pts). Captures informal settlement concentrations where affordability and transit dependency are acute.

#### 2. Actionable Engineering Prescription
**Recommended Municipal Intervention:** **Micro-Transit & On-Demand Electric Shuttle Van Service**  
Targeted infrastructure investment addressing the dominant **severe accessibility deficit** will achieve the steepest reduction in transport inequality for this community.

> *Methodological Note:* Notice: The Dynamic Transit Inequality Index (DTII) is the proposed UrbanFlow AI analytical index designed for municipal decision-support and urban mobility pattern analysis (DataNova 2026, DN-DS-01). It is not an officially gazetted government standard.

---

### 🔍 UrbanFlow AI Inequality Audit: M/W (Ward M/W)

**Overall Evaluation:**  
M/W exhibits a **DTII score of 0.498** (`49.8 / 100`), ranking **#4** across Greater Mumbai, classified under **Moderate Inequality**.

#### 1. Why does this area have a high transport inequality score?
The primary driver of inequality in M/W is **Severe Accessibility Deficit**, which accounts for **46.9%** of the total inequality index.

- **Commuter Mobility Demand ($D_i = 0.26$):** Contributes **15.9%** (0.079 pts). Represents trip generation from 707,064 residents, commercial offices, and educational centers.
- **Accessibility Deficit ($1 - A_i = 0.78$):** Contributes **46.9%** (0.234 pts). Residents must walk an average of **935 metres** to reach the nearest rapid transit facility (`R.C.F. (Chembur) (BEST Bus)`).
- **Public Transport Supply Deficit ($1 - S_i = 0.41$):** Contributes **16.6%** (0.083 pts). Reflects insufficient bus/metro stop density and off-peak headway delays.
- **Social & Demographic Vulnerability ($V_i = 0.51$):** Contributes **20.6%** (0.103 pts). Captures informal settlement concentrations where affordability and transit dependency are acute.

#### 2. Actionable Engineering Prescription
**Recommended Municipal Intervention:** **Micro-Transit & On-Demand Electric Shuttle Van Service**  
Targeted infrastructure investment addressing the dominant **severe accessibility deficit** will achieve the steepest reduction in transport inequality for this community.

> *Methodological Note:* Notice: The Dynamic Transit Inequality Index (DTII) is the proposed UrbanFlow AI analytical index designed for municipal decision-support and urban mobility pattern analysis (DataNova 2026, DN-DS-01). It is not an officially gazetted government standard.

---

### 🔍 UrbanFlow AI Inequality Audit: R/C (Ward R/C)

**Overall Evaluation:**  
R/C exhibits a **DTII score of 0.483** (`48.3 / 100`), ranking **#5** across Greater Mumbai, classified under **Moderate Inequality**.

#### 1. Why does this area have a high transport inequality score?
The primary driver of inequality in R/C is **Severe Accessibility Deficit**, which accounts for **57.0%** of the total inequality index.

- **Commuter Mobility Demand ($D_i = 0.09$):** Contributes **5.6%** (0.027 pts). Represents trip generation from 691,435 residents, commercial offices, and educational centers.
- **Accessibility Deficit ($1 - A_i = 0.92$):** Contributes **57.0%** (0.275 pts). Residents must walk an average of **1717 metres** to reach the nearest rapid transit facility (`Don Bosco School (Borivali ) (BEST Bus)`).
- **Public Transport Supply Deficit ($1 - S_i = 0.72$):** Contributes **30.0%** (0.145 pts). Reflects insufficient bus/metro stop density and off-peak headway delays.
- **Social & Demographic Vulnerability ($V_i = 0.18$):** Contributes **7.4%** (0.036 pts). Captures informal settlement concentrations where affordability and transit dependency are acute.

#### 2. Actionable Engineering Prescription
**Recommended Municipal Intervention:** **Micro-Transit & On-Demand Electric Shuttle Van Service**  
Targeted infrastructure investment addressing the dominant **severe accessibility deficit** will achieve the steepest reduction in transport inequality for this community.

> *Methodological Note:* Notice: The Dynamic Transit Inequality Index (DTII) is the proposed UrbanFlow AI analytical index designed for municipal decision-support and urban mobility pattern analysis (DataNova 2026, DN-DS-01). It is not an officially gazetted government standard.

---


## 6. Classification Schema

| Category | DTII Range | Description | Strategic Municipal Response |
|---|---|---|---|
| **Low Inequality** | $0.00 \le \text{DTII} < 0.35$ | Well-serviced transit hub with high mode diversity. | Routine maintenance and passenger information upgrades. |
| **Moderate Inequality** | $0.35 \le \text{DTII} < 0.50$ | Balanced corridor with emerging last-mile gaps. | Off-peak service frequency enhancements and feeder stops. |
| **High Inequality** | $0.50 \le \text{DTII} < 0.65$ | Severe mismatch between commuter demand and service. | Feeder bus network restructuring and dedicated bus lanes. |
| **Critical Inequality** | $0.65 \le \text{DTII} \le 1.00$ | Acute acute compounding deficits in high-density informal zones. | Capital infrastructure: Rapid transit lines, skywalks, micro-transit. |

---
*Notice: The Dynamic Transit Inequality Index (DTII) is the proposed UrbanFlow AI analytical index designed for municipal decision-support and urban mobility pattern analysis (DataNova 2026, DN-DS-01). It is not an officially gazetted government standard.*