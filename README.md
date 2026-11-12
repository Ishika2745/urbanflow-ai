# UrbanFlow AI

Urban transit accessibility planner and equity analysis system for Greater Mumbai.

## Overview

UrbanFlow AI is a data science project that analyzes public transportation accessibility across Greater Mumbai. It examines where residents live, calculates walking distances to existing transit stops along real street networks, identifies underserved areas ("transit deserts"), and uses optimization algorithms to recommend where new bus stops should be placed to improve transit equity.

The system analyzes real transit data across all 24 administrative municipal wards of Greater Mumbai, covering 13.2 million residents, 5,753 BEST bus stops, and 105 Suburban Rail and Metro stations.

## Key Features

1. Transit Accessibility Mapping
Measures on-foot walking access along real street networks rather than straight lines, calculating exact 5-minute (400m) and 10-minute (800m) walking catchment zones around bus stops and railway stations.

2. Transit Desert Detection (DTII)
Uses the Dynamic Transit Inequality Index (DTII) to identify neighbourhoods where public transport supply does not meet local commuter demand. The index combines four indicators:
- Commuter travel demand
- Walking access deficit
- Public transit service supply
- Socioeconomic vulnerability

3. AI Stop Placement Optimization
Uses nature-inspired algorithms (Genetic Algorithm and Particle Swarm Optimization) alongside conventional operations research solvers (Greedy heuristic and Maximal Covering Location Problem) to select optimal new bus stop locations that maximize population coverage and reduce transit inequality.

4. Scenario Simulation
Allows urban planners to test different stop addition budgets (such as 5, 8, or 15 stops) and walking thresholds, calculating the resulting reduction in average walk times and equity improvements before any physical construction begins.

5. Interactive Dashboard & Export
Provides a web dashboard with interactive maps, heatmaps, analytical charts, and downloadable CSV reports for municipal planning.

## System Architecture

The project is structured into modular layers:

- Data Ingestion: Processes OpenStreetMap street networks, BEST bus stop feeds, Suburban Railway and Metro station coordinates, and population density grids.
- Preprocessing: Standardizes coordinate reference systems (WGS84 EPSG:4326 for display and UTM Zone 43N EPSG:32643 for metric distances), cleans coordinates, and builds pedestrian network graphs.
- Analytical Engine: Computes shortest walking paths using Dijkstra's algorithm, calculates the DTII score for each ward, and runs spatial clustering.
- Machine Learning Models: Evaluates travel demand using supervised regressors (XGBoost, Gradient Boosting, Random Forest) with spatial GroupKFold cross-validation, and segments neighbourhoods using K-Means clustering.
- Optimization Solvers: Solves the multi-objective stop placement problem using Genetic Algorithm, Particle Swarm Optimization, Greedy heuristic, and MCLP.
- Web Dashboard: A Streamlit application displaying maps, charts, simulations, and downloadable reports.

## Project Structure

- app.py: Streamlit web dashboard application.
- evaluate_ml_models.py: Standalone terminal script to evaluate machine learning models and spatial statistics.
- run_eval.bat: Batch file to run model evaluation with one click on Windows.
- analytics/: Multimodal accessibility calculations, pedestrian network routing, and DTII calculation engine.
- models/: Demand estimation models, feature engineering, and model training pipelines.
- optimization/: Candidate stop generation, fitness evaluation, Genetic Algorithm, PSO, Greedy, and MCLP solvers.
- simulation/: Before-and-after scenario simulator and impact analyzer.
- visualization/: Folium interactive Leaflet maps and Plotly analytical charts.
- data/features/: Cleaned, processed Parquet and GeoJSON datasets.
- reports/: Generated analytical reports and summaries.

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Windows, macOS, or Linux

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Ishika2745/urbanflow-ai.git
cd urbanflow-ai
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
```

On Windows:
```powershell
.\.venv\Scripts\activate
```

On macOS / Linux:
```bash
source .venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

### Running the Dashboard

To launch the interactive web dashboard:

```bash
streamlit run app.py
```

Once running, open your web browser and go to `http://localhost:8501`.

### Running Machine Learning Model Evaluation

To evaluate the predictive demand models (XGBoost, Gradient Boosting, Random Forest, GroupKFold cross-validation, K-Means clustering, and TreeSHAP) and view results in the terminal:

```bash
python evaluate_ml_models.py
```

On Windows, you can also double-click `run_eval.bat`.

## Technologies Used

- Python: Core programming language.
- GeoPandas, Shapely, PyProj: Geospatial data processing and coordinate reference system transformations.
- OSMnx, NetworkX: Pedestrian street network modeling and shortest-path routing.
- Scikit-learn, XGBoost: Supervised demand prediction, cross-validation, and K-Means segmentation.
- SHAP: Machine learning feature attribution and explainability.
- Streamlit: Web application framework for the user interface.
- Folium, Leaflet: Interactive mapping and geospatial heatmaps.
- Plotly: Interactive analytical charts and benchmark plots.
- Apache Arrow / Parquet: Columnar data storage.

## Data Sources

- BEST (Brihanmumbai Electric Supply and Transport): Municipal bus stop locations.
- Mumbai Suburban Railway and Mumbai Metro: Station coordinates and network alignment.
- OpenStreetMap: Walkable road network, footpaths, and intersections.
- WorldPop and Census: Demographic and population density estimates for Greater Mumbai wards.

## License

This project is open-source under the MIT License.
