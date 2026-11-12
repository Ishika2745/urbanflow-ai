@echo off
cd /d "%~dp0"
echo ======================================================================
echo Launching UrbanFlow AI - Data Science Model Evaluation CLI
echo ======================================================================
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" evaluate_ml_models.py
) else (
    python evaluate_ml_models.py
)
pause

