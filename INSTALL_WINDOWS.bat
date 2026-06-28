@echo off
cd /d "%~dp0"
echo ==========================================
echo Installing Python libraries for the app
echo ==========================================
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
echo.
echo Installation finished.
echo.
pause
