# -*- coding: utf-8 -*-
"""
Модуль веб-приложения для прогнозирования прочности композитных материалов
на основе нейронной сети. Использует Streamlit для интерактивного интерфейса.
"""

# ========================= ИМПОРТ БИБЛИОТЕК =========================

import streamlit as st           # Фреймворк для создания веб-интерфейсов
import pandas as pd              # Работа с табличными данными (DataFrame)
import numpy as np               # Математические операции и массивы
import tensorflow as tf          # Глубокое обучение, нейронные сети
import random                    # Генерация случайных чисел (для воспроизводимости)
import os                        # Взаимодействие с файловой системой
import pickle                    # Сериализация объектов Python (сохранение/загрузка)
import matplotlib.pyplot as plt  # Построение графиков (кривые обучения)
from tensorflow.keras import layers, models  # Слои и модели Keras
from sklearn.preprocessing import StandardScaler, LabelEncoder  # Нормализация и кодирование категорий
from sklearn.model_selection import train_test_split            # Разделение данных на обучающую/тестовую выборки
from sklearn.metrics import r2_score, mean_absolute_error       # Метрики качества регрессии

# ====================== ГЛОБАЛЬНЫЕ КОНСТАНТЫ =======================

# Пути для сохранения обученной модели и сопутствующих метаданных
MODEL_PATH = "saved_model.keras"      # Файл модели в формате Keras
METADATA_PATH = "model_metadata.pkl"  # Файл с масштабаторами, энкодерами и статистикой

# Словарь для отображения технических имён признаков в понятные подписи с единицами измерения
FIELD_DESCRIPTIONS = {
    "layer_count": "количество слоёв",
    "void_content_pct": "содержание пустот, %",
    "resin_type": "тип смолы",
    "curing_temperature_c": "температура отверждения, °C",
    "density_g_cm3": "плотность, г/см³",
    "fiber_volume_fraction": "объёмная доля волокна",
    "fiber_type": "тип волокна"
}

def get_display_name(col_name):
    """
    Возвращает отображаемое имя признака с пояснением в скобках,
    если оно есть в FIELD_DESCRIPTIONS. Иначе возвращает исходное имя.
    """
    if col_name in FIELD_DESCRIPTIONS:
        return f"{col_name} ({FIELD_DESCRIPTIONS[col_name]})"
    else:
        return col_name

def seed_everything(seed=42):
    """
    Фиксирует все генераторы случайных чисел для воспроизводимости результатов.
    :param seed: целое число – начальное значение (по умолчанию 42)
    """
    random.seed(seed)                     # Стандартный random
    os.environ['PYTHONHASHSEED'] = str(seed)  # Отключает хэширование в Python
    np.random.seed(seed)                  # NumPy
    tf.random.set_seed(seed)              # TensorFlow

# Устанавливаем seed перед любыми операциями с данными или моделью
seed_everything(42)

# Настройка страницы Streamlit: заголовок вкладки браузера и широкий макет
st.set_page_config(page_title="Composite Predictor Pro", layout="wide")
st.title("🔬 Интеллектуальная система прогноза прочности")

# ===================== ФУНКЦИИ РАБОТЫ С МОДЕЛЬЮ =====================

def save_model_assets(model, scaler_x, scaler_y, encoders, stats, feature_cols, metrics):
    """
    Сохраняет обученную нейронную сеть и все вспомогательные объекты в файлы.
    :param model: обученная модель Keras
    :param scaler_x: StandardScaler для нормализации входных признаков
    :param scaler_y: StandardScaler для нормализации целевой переменной (прочность)
    :param encoders: словарь LabelEncoder для категориальных признаков
    :param stats: словарь с минимальными и максимальными значениями числовых признаков
    :param feature_cols: список названий признаков, используемых моделью
    :param metrics: словарь с метриками качества (R², MAE)
    """
    model.save(MODEL_PATH)   # Сохраняем архитектуру и веса модели
    # Упаковываем метаданные в словарь
    metadata = {
        'scaler_x': scaler_x,
        'scaler_y': scaler_y,
        'encoders': encoders,
        'stats': stats,
        'feature_cols': feature_cols,
        'metrics': metrics
    }
    # Сохраняем метаданные в бинарный файл с помощью pickle
    with open(METADATA_PATH, 'wb') as f:
        pickle.dump(metadata, f)

def load_model_assets():
    """
    Загружает ранее сохранённую модель и метаданные из файлов.
    :return: (model, metadata) если файлы существуют и читаются, иначе (None, None)
    """
    if os.path.exists(MODEL_PATH) and os.path.exists(METADATA_PATH):
        try:
            model = models.load_model(MODEL_PATH)   # Загружаем модель
            with open(METADATA_PATH, 'rb') as f:
                metadata = pickle.load(f)           # Загружаем метаданные
            return model, metadata
        except Exception as e:
            st.error(f"Ошибка чтения файлов модели: {e}")
    return None, None

# ================== БОКОВАЯ ПАНЕЛЬ (УПРАВЛЕНИЕ) ===================

st.sidebar.header("⚙️ Управление моделью")

# Виджет загрузки CSV-файла (пользователь может загрузить свои данные)
uploaded_file = st.sidebar.file_uploader("Загрузите CSV для автозаполнения или обучения", type="csv")

# Если файл модели уже существует локально, предлагаем кнопку для её загрузки
if os.path.exists(MODEL_PATH):
    if st.sidebar.button("📂 Использовать сохраненную модель"):
        model, meta = load_model_assets()
        if model:
            # Сохраняем загруженные объекты в сессионное состояние Streamlit
            st.session_state.update({
                'model': model,
                'scaler_x': meta['scaler_x'],
                'scaler_y': meta['scaler_y'],
                'encoders': meta['encoders'],
                'stats': meta['stats'],
                'feature_cols': meta['feature_cols'],
                'saved_metrics': meta.get('metrics')
            })
            st.sidebar.success("✅ Модель загружена!")

st.sidebar.write("---")  # Разделитель

# ================== ОСНОВНЫЕ ВКЛАДКИ ИНТЕРФЕЙСА ===================

# Создаём две вкладки: "Прогноз" и "Помощь"
tab1, tab2 = st.tabs(["🔮 Прогноз", "❓ Помощь"])

# ------------------ ВКЛАДКА ПРОГНОЗА -----------------------------
with tab1:
    # --- БЛОК ОБУЧЕНИЯ МОДЕЛИ НА ЗАГРУЖЕННОМ ФАЙЛЕ ---
    if uploaded_file is not None:
        # Читаем CSV-файл (автоматически определяем разделитель)
        df = pd.read_csv(uploaded_file, sep=None, engine='python')
        st.session_state['raw_df'] = df   # Сохраняем исходные данные для автозаполнения
        st.write("### Предварительный просмотр данных:")
        st.dataframe(df.head(10))         # Показываем первые 10 строк

        # Кнопка обучения новой модели
        if st.sidebar.button("🚀 Обучить новую модель"):
            try:
                # Предполагаем, что последний столбец — целевая переменная (прочность)
                target_col = df.columns[-1]
                feature_cols = df.columns[:-1].tolist()   # Все остальные — признаки
                df_proc = df.copy()   # Работаем с копией, чтобы не испортить исходные данные

                encoders = {}   # Словарь для LabelEncoder (категориальные признаки)
                stats = {}      # Словарь для min/max числовых признаков (для ограничений в интерфейсе)

                # Обработка каждого столбца: приведение к числам, кодирование категорий
                for col in df_proc.columns:
                    if col in feature_cols and df[col].dtype != 'object':
                        # Для числовых признаков сохраняем допустимый диапазон
                        stats[col] = {'min': df[col].min(), 'max': df[col].max()}
                    try:
                        # Пытаемся преобразовать в число (заменяем запятые на точки)
                        df_proc[col] = pd.to_numeric(df_proc[col].astype(str).str.replace(',', '.'))
                    except ValueError:
                        # Если не получается, значит категориальный признак — применяем LabelEncoder
                        le = LabelEncoder()
                        # Убираем пробелы по краям для единообразия
                        df_proc[col] = le.fit_transform(df_proc[col].astype(str).str.strip())
                        encoders[col] = le   # Сохраняем энкодер для обратного преобразования при вводе

                # Формируем матрицу признаков X и вектор целевой переменной y
                X = df_proc[feature_cols].values.astype(np.float32)
                y = df_proc[target_col].values.reshape(-1, 1).astype(np.float32)

                # Разделяем данные на обучающую (80%) и тестовую (20%) выборки
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Нормализуем признаки и целевую переменную (z-масштабирование)
                scaler_x = StandardScaler()
                scaler_y = StandardScaler()
                X_train_scaled = scaler_x.fit_transform(X_train)
                X_test_scaled = scaler_x.transform(X_test)
                y_train_scaled = scaler_y.fit_transform(y_train)

                # Создаём архитектуру нейронной сети
                model = models.Sequential([
                    layers.Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
                    layers.BatchNormalization(),    # Ускоряет сходимость
                    layers.Dropout(0.1),            # Регуляризация для борьбы с переобучением
                    layers.Dense(64, activation='relu'),
                    layers.BatchNormalization(),
                    layers.Dense(32, activation='relu'),
                    layers.Dense(1)                  # Выходной слой без активации (регрессия)
                ])

                # Компиляция: оптимизатор Adam, функция потерь MSE, метрика MAE
                model.compile(optimizer='adam', loss='mse', metrics=['mae'])

                # Обучение модели с индикатором прогресса
                with st.spinner('Обучение...'):
                    history = model.fit(
                        X_train_scaled, y_train_scaled,
                        epochs=150,           # Количество эпох
                        batch_size=32,
                        validation_split=0.2, # 20% из обучающей выборки — на валидацию
                        verbose=0,            # Без вывода в консоль
                        shuffle=False
                    )

                # Предсказание на тестовой выборке и обратное масштабирование
                y_pred_sc = model.predict(X_test_scaled)
                y_pred = scaler_y.inverse_transform(y_pred_sc)

                # Вычисляем метрики качества
                metrics = {
                    "r2": r2_score(y_test, y_pred),
                    "mae": mean_absolute_error(y_test, y_pred)
                }

                # Сохраняем историю обучения (потери) для построения графика
                st.session_state['history'] = history.history

                # Сохраняем все объекты в сессионное состояние
                st.session_state.update({
                    'model': model,
                    'scaler_x': scaler_x,
                    'scaler_y': scaler_y,
                    'encoders': encoders,
                    'feature_cols': feature_cols,
                    'stats': stats,
                    'saved_metrics': metrics
                })

                # Сохраняем модель и метаданные на диск
                save_model_assets(model, scaler_x, scaler_y, encoders, stats, feature_cols, metrics)
                st.sidebar.success("💾 Модель сохранена!")

            except Exception as e:
                st.error(f"Ошибка при обучении: {e}")

    # --- БЛОК ОТОБРАЖЕНИЯ МЕТРИК И ГРАФИКОВ (если модель уже есть) ---
    if 'saved_metrics' in st.session_state:
        st.write("---")
        st.write("### 📊 Результаты обучения и точность")
        m = st.session_state['saved_metrics']
        col_metrics, col_plot = st.columns([1, 2])  # Две колонки: метрики и график

        with col_metrics:
            st.metric("Коэффициент детерминации (R²)", f"{m['r2']:.4f}")
            st.metric("Средняя абсолютная ошибка (MAE)", f"{m['mae']:.2f} МПа")

        with col_plot:
            if 'history' in st.session_state:
                # Рисуем график функции потерь на обучающей и валидационной выборках
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(st.session_state['history']['loss'], label='Обучающая выборка')
                ax.plot(st.session_state['history']['val_loss'], label='Валидационная выборка')
                ax.set_title("Кривые потерь (Loss History)")
                ax.set_xlabel("Эпохи")
                ax.set_ylabel("MSE")
                ax.legend()
                st.pyplot(fig)
                plt.close(fig)   # Закрываем фигуру, чтобы освободить память

    # --- БЛОК ПРОГНОЗИРОВАНИЯ (с подсказками) ---
    if 'model' in st.session_state:
        st.write("---")
        st.write("### 🔮 Прогноз прочности")

        # Если есть загруженный датасет, предлагаем выбрать строку для автозаполнения полей
        if 'raw_df' in st.session_state:
            st.subheader("📥 Автозаполнение параметров из строки")
            row_idx = st.number_input(
                "Введите индекс строки датасета",
                min_value=0,
                max_value=len(st.session_state['raw_df'])-1,
                value=0
            )
            if st.button("Загрузить данные этой строки"):
                # Сохраняем выбранную строку в сессию для предзаполнения формы
                st.session_state['current_inputs'] = st.session_state['raw_df'].iloc[row_idx].to_dict()
                st.rerun()   # Перезагружаем страницу, чтобы обновить значения полей

        # Форма ввода признаков для прогноза
        with st.form("prediction_form"):
            input_dict = {}
            cols = st.columns(3)   # Разбиваем поля ввода на три колонки для компактности
            curr = st.session_state.get('current_inputs', {})  # Предзаполненные значения (если есть)

            # Перебираем все признаки, которые использовались при обучении
            for i, col_name in enumerate(st.session_state['feature_cols']):
                with cols[i % 3]:
                    display_name = get_display_name(col_name)   # Человеко-читаемое имя

                    # Формируем текст всплывающей подсказки (tooltip)
                    if col_name in st.session_state['encoders']:
                        # Для категориального признака показываем список допустимых значений
                        le = st.session_state['encoders'][col_name]
                        help_text = f"Допустимые значения: {', '.join(le.classes_)}"
                    else:
                        # Для числового признака показываем диапазон [min, max]
                        s = st.session_state['stats'][col_name]
                        help_text = f"Диапазон: [{s['min']:.4f}, {s['max']:.4f}]"

                    # Если признак категориальный (был закодирован LabelEncoder'ом)
                    if col_name in st.session_state['encoders']:
                        le = st.session_state['encoders'][col_name]
                        # Пытаемся получить значение из предзаполнения, иначе берём первую категорию
                        val_in_row = str(curr.get(col_name, le.classes_[0])).strip()
                        idx = list(le.classes_).index(val_in_row) if val_in_row in le.classes_ else 0
                        choice = st.selectbox(
                            display_name,
                            options=le.classes_,
                            index=idx,
                            help=help_text      # ← подсказка для selectbox
                        )
                        # Преобразуем выбранную категорию в число, понятное модели
                        input_dict[col_name] = le.transform([choice])[0]
                    else:
                        # Числовой признак: используем поле ввода с ограничениями
                        s = st.session_state['stats'][col_name]
                        # По умолчанию середина диапазона (если нет предзаполнения)
                        default_f = float(curr.get(col_name, (s['min'] + s['max'])/2))
                        input_dict[col_name] = st.number_input(
                            display_name,
                            value=default_f,
                            format="%.4f",
                            help=help_text      # ← подсказка для числового поля
                        )

            # Кнопка отправки формы
            if st.form_submit_button("🧪 РАССЧИТАТЬ ПРОГНОЗ"):
                # Преобразуем введённые данные в массив numpy
                x_raw = np.array([[input_dict[c] for c in st.session_state['feature_cols']]], dtype=np.float32)
                # Нормализуем признаки с помощью обученного scaler_x
                x_in = st.session_state['scaler_x'].transform(x_raw)
                # Делаем предсказание, затем обратно масштабируем результат
                res = st.session_state['scaler_y'].inverse_transform(st.session_state['model'].predict(x_in))
                pred = res.item()   # Извлекаем скалярное значение

                st.info(f"### Прогноз нейросети: **{pred:.2f} МПа**")

                # Если данные для автозаполнения содержат реальное значение прочности,
                # показываем сравнение прогноза с фактом и ошибку
                if curr and 'raw_df' in st.session_state:
                    target_name = st.session_state['raw_df'].columns[-1]  # Имя столбца-цели
                    if target_name in curr:
                        actual = float(curr[target_name])
                        abs_err = abs(pred - actual)
                        c_a, c_b, c_c = st.columns(3)
                        c_a.write(f"📊 Реальное значение: **{actual:.2f} МПа**")
                        c_b.write(f"❌ Абсолютная ошибка: **{abs_err:.2f} МПа**")
                        c_c.write(f"📉 Отклонение: **{(abs_err/actual)*100:.2f}%**")

                st.balloons()   # Эффект для визуального подтверждения расчёта

# ------------------ ВКЛАДКА ПОМОЩИ (ИНСТРУКЦИЯ) --------------------
with tab2:
    st.header("📘 Краткая инструкция по работе")
    st.markdown("""
    **1. Загрузка данных и обучение модели**  
    - В левой боковой панели нажмите «Загрузите CSV для автозаполнения или обучения».  
    - Файл должен содержать признаки и последний столбец – прочность.  
    - Нажмите «🚀 Обучить новую модель» в боковой панели.  

    **2. Прогнозирование**  
    - Заполните поля ввода вручную. **Справа от каждого поля есть значок «?»** – при наведении на него показывается диапазон допустимых значений (для чисел) или список доступных категорий.  
    - Или выберите номер строки из загруженного CSV и нажмите «Загрузить данные этой строки».  
    - Нажмите «🧪 РАССЧИТАТЬ ПРОГНОЗ».  

    **3. Метрики**  
    - R² – коэффициент детерминации (чем ближе к 1, тем лучше).  
    - MAE – средняя абсолютная ошибка в МПа.  
    """)
    st.info("💡 Совет: для качественного прогноза используйте данные с объёмом не менее 50–100 строк.")