# 🚇 UrbanFlow AI: Urban Mobility Intelligence & Transport Optimization

**DataNova 2026 | Problem Statement DN-DS-01:** *Urban Mobility Pattern Analysis from Transit Data*

**Project Title:** *An AI-Powered Urban Mobility Intelligence, Accessibility Analysis and Nature-Inspired Transport Optimization System for Mumbai.*

---

## 🌟 Executive Summary

**UrbanFlow AI** is a competition-level urban intelligence and decision-support system built for Greater Mumbai (Brihanmumbai Municipal Corporation - BMC). The platform analyzes multimodal public transit networks, detects underserved transit deserts using spatial autocorrelation, forecasts passenger commute demand using machine learning, computes spatial transport inequality (Gini coefficient & Lorenz curves), recommends infrastructure improvements through nature-inspired and mathematical optimization, and simulates socioeconomic policy impacts.

---

## 🏛️ System Architecture

UrbanFlow AI is built as a **modular monolith** adhering to strict separation of concerns:

```
┌────────────────────────────────────────────────────────────────────────┐
│                   STREAMLIT INTERACTIVE DASHBOARD                      │
│   (Executive Summary | Deserts & Equity | Demand | Optimization | ROI) │
└────────────────────────────────────┬───────────────────────────────────┘
                                     │
           ┌─────────────────────────┼─────────────────────────┐
           ▼                         ▼                         ▼
┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐
│  SPATIAL ANALYTICS   │  │ OPTIMIZATION ENGINE  │  │  MACHINE LEARNING    │
│  - 2SFCA Catchment   │  │  - Genetic Algorithm │  │  - XGBoost Demand    │
│  - Local Moran's I   │  │  - Discrete PSO      │  │  - Mode Choice MNL   │
│  - Gini / Lorenz     │  │  - MCLP & P-Median   │  │  - SHAP Explainer    │
│  - Centrality Graph  │  │  - Greedy Selection  │  │  - Feature Eng.      │
└──────────┬───────────┘  └──────────┬───────────┘  └──────────┬───────────┘
           │                         │                         │
           └─────────────────────────┼─────────────────────────┘
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│                 CRS DUAL-PROJECTION ENGINE                             │
│   - Calculation CRS: EPSG:32643 (UTM Zone 43N - Metric Distance/Area) │
│   - Map Display CRS: EPSG:4326 (WGS84 - Lat/Lon Folium & PyDeck)       │
└────────────────────────────────────┬───────────────────────────────────┘
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│                 DATA PERSISTENCE & INGESTION LAYER                     │
│   - SQLite (PostGIS-ready)  | GTFS Feeds | OSMnx Street Networks       │
└────────────────────────────────────────────────────────────────────────┘
```

### City-Agnostic Abstraction
The city configuration model (`config/city_config.py` and `config/mumbai_config.py`) decouples the mathematical and algorithmic pipeline from Mumbai-specific geography. Extending UrbanFlow AI to another metropolis (e.g., Bengaluru, Delhi, or London) requires only specifying a new `CityConfig` instance with its projected metric CRS, bounding box, administrative boundaries, and transit modes.

---

## 📐 Dual Coordinate Reference System (CRS) Strategy

To prevent distance distortion errors common when performing Euclidean distance operations in angular coordinates (degrees):
- **Metric Distance Calculations (`EPSG:32643`):** Universal Transverse Mercator (UTM Zone 43N) covers Mumbai (72°E to 78°E). All buffer generation (e.g. 500m, 800m, 1200m catchments), nearest-stop distances, 2SFCA evaluations, and facility location formulations operate strictly in `EPSG:32643`.
- **Map & Web Display (`EPSG:4326`):** WGS84 decimal degrees are used strictly for GeoJSON serialization and interactive rendering in Folium, PyDeck, and Streamlit.
- **Enforcement Guards:** `utils.geo_utils.ensure_metric_crs()` and `utils.geo_utils.ensure_display_crs()` enforce CRS invariants at every pipeline boundary.

---

## 🧬 Optimization Framework

The `optimization/` package provides five distinct optimization paradigms evaluated against a unified multi-objective fitness function:

$$\max J(x) = w_{\text{cov}} \cdot \text{CoveragePct}(x) + w_{\text{eq}} \cdot \Delta\text{Gini}(x) - w_{\text{cost}} \cdot \text{Cost}(x)$$

1. **Genetic Algorithm (`genetic_algorithm.py`):** Uses DEAP with uniform crossover, bit-flip mutation, and a cardinality repair operator to enforce exact budget constraints.
2. **Discrete Particle Swarm Optimization (`particle_swarm.py`):** Implements Binary PSO (BPSO) with sigmoid velocity squashing and top-$k$ thresholding.
3. **Greedy Forward Selection (`greedy_optimizer.py`):** Selects candidate facilities yielding the maximal marginal fitness gain at each step.
4. **Maximal Covering Location Problem (`mclp_optimizer.py`):** Formulated as an exact Mixed-Integer Linear Program (MILP) solved using Google OR-Tools SCIP solver.
5. **P-Median Problem (`pmedian_optimizer.py`):** Formulated as an exact MILP minimizing population-weighted distance to the nearest transit facility.
6. **Benchmark Suite (`optimization_evaluation.py`):** Compares all five solvers across fitness, coverage %, Gini improvement, cost, and execution runtime.

---

## 📁 Modular Directory Structure

```
urbanflow-ai/
├── app.py                      # Main Streamlit dashboard entrypoint
├── requirements.txt            # Pinned dependency specifications
├── README.md                   # Project documentation
├── .env.example                # Environment template
├── .gitignore                  # Git exclusions
├── config/                     # Settings & City abstraction
│   ├── settings.py             # Global directories, CRS, and weights
│   ├── city_config.py          # CityConfig dataclass abstraction
│   └── mumbai_config.py        # BMC Greater Mumbai 24-ward configuration
├── data/                       # Central data storage
│   ├── raw/                    # Raw GeoJSON and tabular feeds
│   ├── processed/              # Parquet processed OD flows
│   └── features/               # Engineered spatial features
├── database/                   # Relational persistence
│   ├── connection.py           # SQLite connection context manager
│   ├── schema.py               # Table creation scripts
│   └── repository.py           # DAO query repository
├── ingestion/                  # Data extraction
│   ├── synthetic_generator.py  # High-fidelity Mumbai multimodal generator
│   ├── gtfs_loader.py          # GTFS feed parser
│   ├── osm_loader.py           # OSMnx street network extractor
│   └── census_loader.py        # Ward demographic indicator loader
├── preprocessing/              # Data preparation
│   ├── spatial_cleaner.py      # Geometry validation & snapping
│   ├── network_builder.py      # NetworkX multimodal graph constructor
│   ├── demand_aggregator.py    # Gravity model OD commute matrix
│   └── feature_engineering.py  # Spatial lag & transit deficit indices
├── analytics/                  # Regional science & spatial statistics
│   ├── accessibility.py        # 2SFCA & catchment buffers
│   ├── spatial_inequality.py   # Gini, Lorenz curve, Palma ratio
│   ├── spatial_clustering.py   # K-Means, DBSCAN, PySAL Local Moran's I (LISA), Getis-Ord Gi*
│   ├── transit_desert_detector.py # DTII computation, deficit zoning, priority ranking
│   ├── transit_desert_visualizer.py # Interactive Folium heatmaps & cluster maps
│   ├── run_transit_desert_analysis.py # Master Transit Desert & Hotspot Analysis pipeline
│   └── connectivity_metrics.py # Centrality measures (Betweenness, PageRank)
├── models/                     # Predictive modeling & explainability
│   ├── demand_forecasting.py   # XGBoost passenger volume forecasting
│   ├── mode_choice.py          # Multinomial logit modal split
│   └── explainability.py       # SHAP TreeExplainer feature attributions
├── optimization/               # Nature-inspired & exact facility solvers
│   ├── candidate_generation.py # Desert hotspot candidate generator
│   ├── fitness_function.py     # Multi-objective fitness evaluator
│   ├── genetic_algorithm.py    # DEAP Genetic Algorithm
│   ├── particle_swarm.py       # Discrete Binary PSO
│   ├── greedy_optimizer.py     # Greedy forward heuristic
│   ├── mclp_optimizer.py       # Maximal Covering MILP (OR-Tools)
│   ├── pmedian_optimizer.py    # P-Median MILP (OR-Tools)
│   └── optimization_evaluation.py # 5-way comparative benchmark suite
├── simulation/                 # What-if policy scenarios
│   ├── transit_simulator.py    # Infrastructure expansion simulator
│   └── impact_analyzer.py      # Economic ROI, commute hours, CO2 offset
├── visualization/              # UI mapping & charts
│   ├── maps.py                 # Folium interactive spatial layers
│   ├── charts.py               # Plotly Lorenz & benchmark figures
│   └── components.py           # Streamlit metric cards & banners
├── pages/                      # Multi-page dashboard views (10 Dedicated Sections)
│   ├── 1_📊_Mobility_Overview.py
│   ├── 2_🤖_Demand_Intelligence.py
│   ├── 3_🚶_Multimodal_Accessibility.py
│   ├── 4_🏜️_Transit_Desert_Detection.py
│   ├── 5_⚖️_DTII_Analysis.py
│   ├── 6_🏗️_Infrastructure_Optimization.py
│   ├── 7_🧬_GA_and_PSO_Comparison.py
│   ├── 8_🔮_Scenario_Simulator.py
│   ├── 9_📈_Model_Performance.py
│   └── 10_🛡️_Data_Quality_and_Sources.py
├── pipeline/                   # Master pipeline orchestration
│   ├── master_pipeline.py      # Unified 9-stage pipeline controller
│   └── __init__.py
├── run_pipeline.py             # Single-command CLI master runner
├── tests/                      # Automated unit test suite (93 tests, 100% pass)
│   ├── test_config.py
│   ├── test_geo_utils.py
│   ├── test_analytics.py
│   ├── test_dtii.py
│   ├── test_transit_deserts.py
│   ├── test_optimizers.py
│   ├── test_conventional_optimization.py
│   ├── test_mobility_demand.py
│   ├── test_multimodal_accessibility.py
│   ├── test_database.py
│   ├── test_simulation.py
│   ├── test_explainability.py
│   ├── test_validation_benchmarks.py
│   └── test_master_pipeline.py
├── reports/                    # Generated analytical reports & Folium maps
│   ├── final_project_report.md # Comprehensive final submission report
│   ├── pipeline_execution_manifest.json # Master pipeline execution run record
│   └── maps/                   # Standalone interactive Leaflet HTML maps
└── data/                       # Spatial layers & SQLite database
    ├── urbanflow.db            # 15 relational tables
    ├── raw/                    # Validated open transit GeoJSONs
    └── features/               # Engineered GeoParquet grids (1km, 500m, wards)
```

---

## 🚀 Installation & Execution

### 1. Prerequisites
- Python 3.10+ (Tested and verified on Python 3.13)
- Windows / Linux / macOS

### 2. Set Up Virtual Environment

```bash
# Clone or navigate to the project directory
cd urbanflow-ai

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run Master Analytical Pipeline (Single-Command Execution)

```bash
# Execute the complete 9-stage end-to-end analytical pipeline:
python run_pipeline.py --all

# Or run a specific stage (e.g. DTII, optimization, or simulation):
python run_pipeline.py --stage dtii
python run_pipeline.py --stage nature_opt --budget 15
```

### 5. Run Automated Test Suite

```bash
# Run all 93 automated unit and integration tests (100% pass rate)
pytest tests/ -v
```

### 6. Launch Interactive Dashboard (10 Sections)

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`.

