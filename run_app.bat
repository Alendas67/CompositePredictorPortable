@echo off
REM Переходим в папку, где лежит bat-файл
cd /d "%~dp0"
REM Показываем текущий путь для отладки
echo Запуск из: %cd%

REM Запускаем Streamlit через модуль Python (надёжнее)
.\python\python.exe -m streamlit run app.py

pause