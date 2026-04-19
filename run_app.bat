@echo off
cd /d "%~dp0"
start http://localhost:8501
timeout /t 2 /nobreak >nul
.\python\Scripts\streamlit.exe run app.py
pause