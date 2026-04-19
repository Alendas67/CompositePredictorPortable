Creating README.md...
# Composite Predictor Pro

Intelligent system for predicting the strength of composite materials.

## Features
- Upload CSV (last column is target - strength)
- Neural network training (TensorFlow/Keras)
- Prediction with parameter descriptions
- Portable version for Windows

## Quick start
1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Run the app:
   ```
   streamlit run app.py
   ```

## Requirements
- Python 3.9+
- streamlit, pandas, numpy, tensorflow, scikit-learn, matplotlib

## License
MIT
## 🚀 Как пользоваться

### Вариант 1: Портативная версия (не требует Python)
1. Скачайте архив `CompositePredictorPortable.zip` из раздела [Releases](https://github.com/Alendas67/CompositePredictorPortable/releases).
2. Распакуйте в любую папку.
3. Запустите `run_app.bat`.
4. Откроется браузер с приложением.
5. Ознакомьтесь с интрукцией в разделе "Помощь"

### Вариант 2: Из исходного кода (требуется Python 3.9+)
```bash
git clone https://github.com/Alendas67/CompositePredictorPortable.git
cd CompositePredictorPortable
pip install -r requirements.txt
streamlit run app.py
📖 Как использовать приложение
Боковая панель – загрузите CSV (последний столбец – целевая переменная) и нажмите «Обучить новую модель» или используйте сохранённую.

Вкладка «Прогноз» – заполните параметры (с русскими подсказками) или выберите строку из загруженного CSV, нажмите «Рассчитать прогноз».

Вкладка «Помощь» – подробная инструкция.
