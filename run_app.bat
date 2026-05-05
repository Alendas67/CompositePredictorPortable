@echo off
cd /d "%~dp0"
echo Running from: %cd%
.\python\python.exe -m streamlit run app.py
pause