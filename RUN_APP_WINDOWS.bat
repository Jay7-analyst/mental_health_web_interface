@echo off
cd /d "%~dp0"
echo ==========================================
echo Starting Mental Health Web Interface
echo ==========================================
echo.
echo If the browser does not open automatically, copy the local URL shown below.
echo.
python -m streamlit run app.py
pause
