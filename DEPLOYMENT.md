# UrbanFlow AI — Deployment & Operations Guide
**DataNova 2026: Problem Statement DN-DS-01**  
**Urban Mobility Intelligence & Nature-Inspired Transport Optimization Platform**

---

## 1. Executive Deployment Overview

UrbanFlow AI is engineered for zero-configuration, containerized, and cloud-native deployment. It operates entirely on open geospatial data without proprietary paid API keys (such as Google Maps or Mapbox).

### Supported Platforms:
- **Streamlit Community Cloud** (Recommended for evaluation & competition demo)
- **Docker / Podman Containerization**
- **Cloud Run / AWS App Runner / Azure Container Instances**
- **Local Workstation (Windows, Linux, macOS)**

---

## 2. Streamlit Community Cloud Deployment (1-Click)

### Step 1: Repository Configuration
1. Push the repository to GitHub. Ensure the following files are present in the repository root:
   - `app.py` (Main entrypoint)
   - `pages/` (Multi-page dashboard scripts)
   - `requirements.txt` (Pinned / bounded dependencies)
   - `.streamlit/config.toml` (Theme & server configuration)
   - `data/` (Processed GeoParquet and SQLite tables)

### Step 2: Cloud Console Setup
1. Log into [share.streamlit.io](https://share.streamlit.io).
2. Select **"Create app"** > **"From existing repo"**.
3. Set:
   - **Repository:** `<your-username>/urbanflow-ai`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. In **Advanced Settings**, optionally add environment variables from `.env.example`:
   ```toml
   URBANFLOW_ENV = "production"
   URBANFLOW_LOG_LEVEL = "INFO"
   URBANFLOW_CITY = "mumbai"
   ```
5. Click **Deploy!**

---

## 3. Local Execution Guide

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13
- Git

### Commands:
```bash
# 1. Clone the repository
git clone https://github.com/<your-repo>/urbanflow-ai.git
cd urbanflow-ai

# 2. Create virtual environment
python -m venv .venv

# 3. Activate virtual environment
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# 4. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 5. Populate SQLite Database (All 15 Relational Tables)
python -m database.populate_db

# 6. Run Test Suite (Verifies 87/87 tests pass)
python -m pytest tests/ -v

# 7. Launch Interactive Dashboard
streamlit run app.py
```

The application will be accessible at: `http://localhost:8501`.

---

## 4. Docker Container Deployment

A production-ready `Dockerfile` is provided:

### Dockerfile
```dockerfile
FROM python:3.11-slim

# System geospatial libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgdal-dev \
    libgeos-dev \
    libspatialindex-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Initialize relational database and ensure permissions
RUN python -m database.populate_db

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Docker Build & Run:
```bash
docker build -t urbanflow-ai:latest .
docker run -p 8501:8501 --name urbanflow urbanflow-ai:latest
```

---

## 5. Cold-Start Dataset Initialization Strategy

UrbanFlow AI includes an automated self-healing data pipeline:
1. When deployed to a fresh container or cloud instance where SQLite has not been pre-populated, the platform executes `init_database()` and populates all 15 relational tables directly from compressed GeoParquet files.
2. GeoParquet files are located in `data/features/` and `data/processed/` and take only ~12MB of disk space.
3. If processed layers are absent, `utils/data_loader.py` dynamically synthesizes ward boundaries, bus stops, and road candidate geometries.

---

## 6. Security, Secrets & Credential Management

- **Zero Hardcoded Secrets:** UrbanFlow AI contains no API keys, tokens, or plaintext passwords.
- **Open Standards:** The platform utilizes open-source cartography (OpenStreetMap tiles via Leaflet) requiring zero external telemetry or paid map service tokens.
- **Environment Isolation:** All customizable parameters (catchment radius, weights, seed) are sourced via `os.getenv` with safe defaults defined in `config/settings.py`.
- **Disclaimer Enforcement:** Every page includes explicit planning disclaimers preventing misleading claims of operational government status.

