# UrbanFlow AI - Mobility Demand Intelligence Report

**Geographical Scope:** Greater Mumbai (MCGM), Maharashtra, India  
**Problem Statement:** DataNova 2026, DN-DS-01  
**Evaluation Integrity:** GroupKFold Spatially-Aware Cross-Validation by Ward (Zero Location Leakage)  
**Measurement Specification:** Estimated Relative Demand Score (0 to 100)  

---

## 1. Machine Learning Benchmark Comparison

| Model | MAE | RMSE | R-Squared | Validation Strategy |
|---|---|---|---|---|
| **XGBoost Regressor** | 0.49 | 1.27 | **0.987** | GroupKFold (5 Ward Splits) |
| **Gradient Boosting Regressor** | 0.55 | 1.27 | **0.987** | GroupKFold (5 Ward Splits) |
| **Random Forest Regressor** | 1.11 | 2.42 | **0.960** | GroupKFold (5 Ward Splits) |
| **Baseline (Mean Predictor)** | 11.09 | 13.04 | **-0.017** | GroupKFold (5 Ward Splits) |

---

## 2. SHAP Global Feature Importance Ranking

| Rank | Feature | Mean Absolute SHAP | Relative Contribution (%) |
|---|---|---|---|
| 1 | `population_density_per_sq_km` | 10.4910 | **78.6%** |
| 2 | `road_intersection_density` | 1.1844 | **8.9%** |
| 3 | `school_density_per_sq_km` | 0.5419 | **4.1%** |
| 4 | `commercial_density_per_sq_km` | 0.5233 | **3.9%** |
| 5 | `hospital_density_per_sq_km` | 0.3588 | **2.7%** |
| 6 | `market_density_per_sq_km` | 0.2382 | **1.8%** |
| 7 | `pedestrian_network_density_km_per_sq_km` | 0.0036 | **0.0%** |

---

## 3. Ward Relative Demand Ranking & Unmet Need Diagnostics

| Rank | Ward ID | Ward Name | Population | Relative Demand (0-100) | Accessibility Score (0-100) | Priority Intervention Status |
|---|---|---|---|---|---|---|
| 1 | `H/E` | H/E | 600,155 | **34.4** | 28.8 | 🟢 Well-Serviced Hub |
| 2 | `G/N` | G/N | 290,478 | **31.9** | 34.2 | 🟢 Well-Serviced Hub |
| 3 | `L` | L | 755,297 | **31.1** | 30.0 | 🟢 Well-Serviced Hub |
| 4 | `B` | B | 151,198 | **28.8** | 23.2 | 🚨 **Acute Unmet Need (High Demand, Low Access)** |
| 5 | `F/N` | F/N | 464,108 | **27.1** | 25.1 | 🟢 Well-Serviced Hub |
| 6 | `K/E` | K/E | 924,064 | **27.1** | 28.6 | 🟢 Well-Serviced Hub |
| 7 | `M/W` | M/W | 725,213 | **26.3** | 22.2 | 🚨 **Acute Unmet Need (High Demand, Low Access)** |
| 8 | `E` | E | 241,580 | **26.1** | 27.7 | 🟢 Well-Serviced Hub |
| 9 | `C` | C | 21,049 | **25.4** | 26.5 | 🟢 Well-Serviced Hub |
| 10 | `H/W` | H/W | 384,194 | **24.8** | 27.1 | 🔵 Transit Surplus Node |
| 11 | `F/S` | F/S | 366,185 | **24.8** | 25.7 | 🔵 Transit Surplus Node |
| 12 | `G/S` | G/S | 366,286 | **22.4** | 19.6 | ⚪ Low Density Buffer |
| 13 | `A` | A | 330,721 | **21.1** | 25.5 | 🔵 Transit Surplus Node |
| 14 | `P/S` | P/S | 668,973 | **20.8** | 16.9 | ⚪ Low Density Buffer |
| 15 | `K/W` | K/W | 739,098 | **20.4** | 25.3 | 🔵 Transit Surplus Node |
| 16 | `D` | D | 303,236 | **20.1** | 20.2 | ⚪ Low Density Buffer |
| 17 | `R/S` | R/S | 581,583 | **19.9** | 20.6 | ⚪ Low Density Buffer |
| 18 | `S` | S | 949,911 | **19.8** | 12.0 | ⚪ Low Density Buffer |
| 19 | `N` | N | 764,997 | **17.4** | 11.2 | ⚪ Low Density Buffer |
| 20 | `R/N` | R/N | 392,882 | **15.8** | 15.2 | ⚪ Low Density Buffer |
| 21 | `M/E` | M/E | 980,522 | **15.4** | 3.9 | ⚪ Low Density Buffer |
| 22 | `P/N` | P/N | 850,006 | **10.9** | 9.6 | ⚪ Low Density Buffer |
| 23 | `T` | T | 710,045 | **10.0** | 2.4 | ⚪ Low Density Buffer |
| 24 | `R/C` | R/C | 677,983 | **9.0** | 8.2 | ⚪ Low Density Buffer |

---

## 4. Methodological & Ethical Notes
- **Data Integrity Standard:** Actual passenger tap-in ridership is not publicly published for Mumbai; all outputs represent *Estimated Relative Demand* derived from verified spatial attractors and demographic mass.
- **Spatial Leakage Prevention:** Machine learning regressors were cross-validated using GroupKFold grouped strictly by administrative ward boundaries to guarantee that models generalize across unseen geography.
- **Actionable Transit Optimization:** Identified acute unmet demand clusters directly feed candidate stop selection for Genetic Algorithm (GA) and Particle Swarm Optimization (PSO) modules.