# -*- coding: utf-8 -*-
"""
Модуль веб-приложения для прогнозирования прочности композитных материалов
на основе нейронной сети. Использует Streamlit для интерактивного интерфейса.
"""

# ========================= ИМПОРТ БИБЛИОТЕК =========================

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import random
import os
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler, OneHotEncoder   # <-- OneHotEncoder вместо LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# ====================== ГЛОБАЛЬНЫЕ КОНСТАНТЫ =======================

MODEL_PATH = "saved_model.keras"
METADATA_PATH = "model_metadata.pkl"

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
    if col_name in FIELD_DESCRIPTIONS:
        return f"{col_name} ({FIELD_DESCRIPTIONS[col_name]})"
    else:
        return col_name

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

seed_everything(42)

st.set_page_config(page_title="Composite Predictor Pro", layout="wide")
st.title("🔬 Интеллектуальная система прогноза прочности")

# ===================== ФУНКЦИИ РАБОТЫ С МОДЕЛЬЮ =====================

def save_model_assets(model, scaler_x, scaler_y, onehot_encoder,
                      numerical_cols, categorical_cols,
                      feature_cols_encoded, stats, metrics):
    """
    Сохраняет обученную нейронную сеть и все вспомогательные объекты.
    """
    model.save(MODEL_PATH)
    metadata = {
        'scaler_x': scaler_x,
        'scaler_y': scaler_y,
        'onehot_encoder': onehot_encoder,
        'numerical_cols': numerical_cols,
        'categorical_cols': categorical_cols,
        'feature_cols_encoded': feature_cols_encoded,
        'stats': stats,
        'metrics': metrics
    }
    with open(METADATA_PATH, 'wb') as f:
        pickle.dump(metadata, f)

def load_model_assets():
    if os.path.exists(MODEL_PATH) and os.path.exists(METADATA_PATH):
        try:
            model = models.load_model(MODEL_PATH)
            with open(METADATA_PATH, 'rb') as f:
                metadata = pickle.load(f)
            return model, metadata
        except Exception as e:
            st.error(f"Ошибка чтения файлов модели: {e}")
    return None, None

# ================== БОКОВАЯ ПАНЕЛЬ (УПРАВЛЕНИЕ) ===================

st.sidebar.header("⚙️ Управление моделью")

uploaded_file = st.sidebar.file_uploader("Загрузите CSV для автозаполнения или обучения", type="csv")

if os.path.exists(MODEL_PATH):
    if st.sidebar.button("📂 Использовать сохраненную модель"):
        model, meta = load_model_assets()
        if model:
            st.session_state.update({
                'model': model,
                'scaler_x': meta['scaler_x'],
                'scaler_y': meta['scaler_y'],
                'onehot_encoder': meta['onehot_encoder'],
                'numerical_cols': meta['numerical_cols'],
                'categorical_cols': meta['categorical_cols'],
                'feature_cols_encoded': meta['feature_cols_encoded'],
                'stats': meta['stats'],
                'saved_metrics': meta.get('metrics')
            })
            st.sidebar.success("✅ Модель загружена!")

st.sidebar.write("---")

# ================== ОСНОВНЫЕ ВКЛАДКИ ИНТЕРФЕЙСА ===================

tab1, tab2 = st.tabs(["🔮 Прогноз", "❓ Помощь"])

# ------------------ ВКЛАДКА ПРОГНОЗА -----------------------------
with tab1:
    # --- БЛОК ОБУЧЕНИЯ МОДЕЛИ НА ЗАГРУЖЕННОМ ФАЙЛЕ ---
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, sep=None, engine='python')
        st.session_state['raw_df'] = df
        st.write("### Предварительный просмотр данных:")
        st.dataframe(df.head(10))

        if st.sidebar.button("🚀 Обучить новую модель"):
            try:
                target_col = df.columns[-1]
                all_feature_cols = df.columns[:-1].tolist()
                df_proc = df.copy()

                # Определяем числовые и категориальные столбцы
                numerical_cols = []
                categorical_cols = []
                stats = {}          # для числовых: min/max; для категориальных: список категорий

                for col in all_feature_cols:
                    # Пытаемся преобразовать в число
                    try:
                        pd.to_numeric(df_proc[col].astype(str).str.replace(',', '.'))
                        numerical_cols.append(col)
                        stats[col] = {'min': df[col].min(), 'max': df[col].max()}
                    except ValueError:
                        categorical_cols.append(col)
                        # Сохраняем уникальные категории (для подсказок в интерфейсе)
                        unique_cats = df_proc[col].astype(str).str.strip().unique()
                        stats[col] = list(unique_cats)

                # One-Hot кодирование категориальных признаков
                onehot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                if categorical_cols:
                    X_cat = df_proc[categorical_cols].astype(str).apply(lambda x: x.str.strip())
                    X_cat_encoded = onehot_encoder.fit_transform(X_cat)
                    # Имена новых признаков (для информативности)
                    cat_feature_names = onehot_encoder.get_feature_names_out(categorical_cols)
                else:
                    X_cat_encoded = np.empty((df_proc.shape[0], 0))
                    cat_feature_names = []

                # Числовые признаки
                if numerical_cols:
                    X_num = df_proc[numerical_cols].astype(float).values
                else:
                    X_num = np.empty((df_proc.shape[0], 0))

                # Объединяем числовые и one-hot признаки
                X = np.hstack([X_num, X_cat_encoded])
                feature_cols_encoded = numerical_cols + list(cat_feature_names)

                # Целевая переменная
                y = df_proc[target_col].values.reshape(-1, 1).astype(np.float32)

                # Разделение выборок
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Нормализация
                scaler_x = StandardScaler()
                scaler_y = StandardScaler()
                X_train_scaled = scaler_x.fit_transform(X_train)
                X_test_scaled = scaler_x.transform(X_test)
                y_train_scaled = scaler_y.fit_transform(y_train)

                # Построение модели
                model = models.Sequential([
                    layers.Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
                    layers.BatchNormalization(),
                    layers.Dropout(0.1),
                    layers.Dense(64, activation='relu'),
                    layers.BatchNormalization(),
                    layers.Dense(32, activation='relu'),
                    layers.Dense(1)
                ])
                model.compile(optimizer='adam', loss='mse', metrics=['mae'])

                with st.spinner('Обучение...'):
                    history = model.fit(
                        X_train_scaled, y_train_scaled,
                        epochs=150,
                        batch_size=32,
                        validation_split=0.2,
                        verbose=0,
                        shuffle=False
                    )

                y_pred_sc = model.predict(X_test_scaled)
                y_pred = scaler_y.inverse_transform(y_pred_sc)
                metrics = {
                    "r2": r2_score(y_test, y_pred),
                    "mae": mean_absolute_error(y_test, y_pred)
                }

                st.session_state['history'] = history.history
                st.session_state.update({
                    'model': model,
                    'scaler_x': scaler_x,
                    'scaler_y': scaler_y,
                    'onehot_encoder': onehot_encoder,
                    'numerical_cols': numerical_cols,
                    'categorical_cols': categorical_cols,
                    'feature_cols_encoded': feature_cols_encoded,
                    'stats': stats,
                    'saved_metrics': metrics
                })

                save_model_assets(model, scaler_x, scaler_y, onehot_encoder,
                                  numerical_cols, categorical_cols,
                                  feature_cols_encoded, stats, metrics)
                st.sidebar.success("💾 Модель сохранена!")

            except Exception as e:
                st.error(f"Ошибка при обучении: {e}")

    # --- БЛОК ОТОБРАЖЕНИЯ МЕТРИК И ГРАФИКОВ ---
    if 'saved_metrics' in st.session_state:
        st.write("---")
        st.write("### 📊 Результаты обучения и точность")
        m = st.session_state['saved_metrics']
        col_metrics, col_plot = st.columns([1, 2])

        with col_metrics:
            st.metric("Коэффициент детерминации (R²)", f"{m['r2']:.4f}")
            st.metric("Средняя абсолютная ошибка (MAE)", f"{m['mae']:.2f} МПа")

        with col_plot:
            if 'history' in st.session_state:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(st.session_state['history']['loss'], label='Обучающая выборка')
                ax.plot(st.session_state['history']['val_loss'], label='Валидационная выборка')
                ax.set_title("Кривые потерь (Loss History)")
                ax.set_xlabel("Эпохи")
                ax.set_ylabel("MSE")
                ax.legend()
                st.pyplot(fig)
                plt.close(fig)

    # --- БЛОК ПРОГНОЗИРОВАНИЯ ---
    if 'model' in st.session_state:
        st.write("---")
        st.write("### 🔮 Прогноз прочности")

        # Автозаполнение из строки датасета
        if 'raw_df' in st.session_state:
            st.subheader("📥 Автозаполнение параметров из строки")
            row_idx = st.number_input(
                "Введите индекс строки датасета",
                min_value=0,
                max_value=len(st.session_state['raw_df'])-1,
                value=0
            )
            if st.button("Загрузить данные этой строки"):
                st.session_state['current_inputs'] = st.session_state['raw_df'].iloc[row_idx].to_dict()
                st.rerun()

        # Форма ввода
        with st.form("prediction_form"):
            input_dict = {}
            # Собираем все исходные признаки (числовые и категориальные) в порядке из numerical_cols + categorical_cols
            all_original_cols = st.session_state['numerical_cols'] + st.session_state['categorical_cols']
            cols = st.columns(3)

            for i, col_name in enumerate(all_original_cols):
                with cols[i % 3]:
                    display_name = get_display_name(col_name)
                    if col_name in st.session_state['categorical_cols']:
                        # Категориальный признак: selectbox из stats[col_name]
                        categories = st.session_state['stats'][col_name]
                        help_text = f"Допустимые значения: {', '.join(categories)}"
                        curr_val = st.session_state.get('current_inputs', {}).get(col_name, categories[0])
                        # Приводим к строке для совпадения
                        if curr_val not in categories and str(curr_val) in categories:
                            curr_val = str(curr_val)
                        elif curr_val not in categories:
                            curr_val = categories[0]
                        choice = st.selectbox(
                            display_name,
                            options=categories,
                            index=categories.index(curr_val),
                            help=help_text
                        )
                        input_dict[col_name] = choice
                    else:
                        # Числовой признак
                        s = st.session_state['stats'][col_name]
                        help_text = f"Диапазон: [{s['min']:.4f}, {s['max']:.4f}]"
                        default_val = float(st.session_state.get('current_inputs', {}).get(col_name, (s['min'] + s['max'])/2))
                        input_dict[col_name] = st.number_input(
                            display_name,
                            value=default_val,
                            format="%.4f",
                            help=help_text
                        )

            if st.form_submit_button("🧪 РАССЧИТАТЬ ПРОГНОЗ"):
                # Формируем входной вектор для модели
                # 1) Числовые признаки
                X_num = np.array([[input_dict[col] for col in st.session_state['numerical_cols']]], dtype=np.float32)
                # 2) Категориальные признаки в виде DataFrame для onehot_encoder
                if st.session_state['categorical_cols']:
                    cat_df = pd.DataFrame([[input_dict[col] for col in st.session_state['categorical_cols']]],
                                          columns=st.session_state['categorical_cols'])
                    # Преобразуем в строки и обрезаем пробелы
                    cat_df = cat_df.astype(str).apply(lambda x: x.str.strip())
                    X_cat = st.session_state['onehot_encoder'].transform(cat_df)
                else:
                    X_cat = np.empty((1, 0))
                # Объединяем
                X_raw = np.hstack([X_num, X_cat])
                # Нормализуем и предсказываем
                X_in = st.session_state['scaler_x'].transform(X_raw)
                res = st.session_state['scaler_y'].inverse_transform(st.session_state['model'].predict(X_in))
                pred = res.item()
                st.info(f"### Прогноз нейросети: **{pred:.2f} МПа**")

                # Сравнение с реальным значением (если есть)
                if 'current_inputs' in st.session_state and 'raw_df' in st.session_state:
                    target_name = st.session_state['raw_df'].columns[-1]
                    if target_name in st.session_state['current_inputs']:
                        actual = float(st.session_state['current_inputs'][target_name])
                        abs_err = abs(pred - actual)
                        c_a, c_b, c_c = st.columns(3)
                        c_a.write(f"📊 Реальное значение: **{actual:.2f} МПа**")
                        c_b.write(f"❌ Абсолютная ошибка: **{abs_err:.2f} МПа**")
                        c_c.write(f"📉 Отклонение: **{(abs_err/actual)*100:.2f}%**")

                st.balloons()

# ------------------ ВКЛАДКА ПОМОЩИ --------------------
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