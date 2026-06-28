@echo off
cd /d "%~dp0"
echo Installing Streamlit only...
python -m pip install streamlit
echo.
echo Now running the app...
python -m streamlit run app.py
pause
