# -*- coding: utf-8 -*-
"""
Модуль веб-приложения для прогнозирования прочности композитных материалов
на основе нейронной сети с Embedding слоями, увеличенной архитектурой,
повышенным Dropout и GaussianNoise.
"""

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.layers import Concatenate, Dense, Dropout, BatchNormalization, GaussianNoise
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import random
import os
import pickle
import matplotlib.pyplot as plt
import time   # <--- ДОБАВЛЕН ДЛЯ ТАЙМЕРА

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

st.set_page_config(page_title="Composite Predictor Pro (Embedding + Advanced)", layout="wide")
st.title("🔬 Интеллектуальная система прогноза прочности (Embedding, Dropout 0.4, GaussianNoise)")

# ===================== ФУНКЦИИ РАБОТЫ С МОДЕЛЬЮ =====================

def save_model_assets(model, scaler_x, scaler_y, numerical_cols, categorical_mappings,
                      feature_cols_names, metrics):
    model.save(MODEL_PATH)
    metadata = {
        'scaler_x': scaler_x,
        'scaler_y': scaler_y,
        'numerical_cols': numerical_cols,
        'categorical_mappings': categorical_mappings,
        'feature_cols_names': feature_cols_names,
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

# ================== БОКОВАЯ ПАНЕЛЬ ===================

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
                'numerical_cols': meta['numerical_cols'],
                'categorical_mappings': meta['categorical_mappings'],
                'feature_cols_names': meta['feature_cols_names'],
                'saved_metrics': meta.get('metrics')
            })
            st.sidebar.success("✅ Модель загружена!")

st.sidebar.write("---")

# ================== ОСНОВНЫЕ ВКЛАДКИ ===================
tab1, tab2 = st.tabs(["🔮 Прогноз", "❓ Помощь"])

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

                # Разделяем на числовые и категориальные
                numerical_cols = []
                categorical_cols = []
                categorical_mappings = {}

                for col in all_feature_cols:
                    try:
                        pd.to_numeric(df[col].astype(str).str.replace(',', '.'))
                        numerical_cols.append(col)
                    except ValueError:
                        categorical_cols.append(col)
                        unique_cats = df[col].astype(str).str.strip().unique()
                        unique_cats = sorted(unique_cats)
                        vocab = {cat: idx for idx, cat in enumerate(unique_cats)}
                        categorical_mappings[col] = {
                            'classes': unique_cats,
                            'vocab': vocab,
                            'vocab_size': len(unique_cats)
                        }

                # Подготовка данных
                X_num = df[numerical_cols].astype(float).values if numerical_cols else np.empty((df.shape[0], 0))
                X_cat_list = []
                for col in categorical_cols:
                    vocab = categorical_mappings[col]['vocab']
                    col_data = df[col].astype(str).str.strip().map(vocab).fillna(0).astype(np.int32)
                    X_cat_list.append(col_data.values.reshape(-1, 1))

                y = df[target_col].values.reshape(-1, 1).astype(np.float32)

                # Разделение выборок
                X_num_train, X_num_test, y_train, y_test = train_test_split(
                    X_num, y, test_size=0.2, random_state=42
                )
                X_cat_train_list = []
                X_cat_test_list = []
                for cat_arr in X_cat_list:
                    tr, te = train_test_split(cat_arr, test_size=0.2, random_state=42)
                    X_cat_train_list.append(tr)
                    X_cat_test_list.append(te)

                # Нормализация
                scaler_x_num = StandardScaler()
                if X_num_train.shape[1] > 0:
                    X_num_train_scaled = scaler_x_num.fit_transform(X_num_train)
                    X_num_test_scaled = scaler_x_num.transform(X_num_test)
                else:
                    X_num_train_scaled = np.empty((X_num_train.shape[0], 0))
                    X_num_test_scaled = np.empty((X_num_test.shape[0], 0))

                scaler_y = StandardScaler()
                y_train_scaled = scaler_y.fit_transform(y_train)
                y_test_scaled = scaler_y.transform(y_test)

                # ---------- ПОСТРОЕНИЕ МОДЕЛИ ----------
                numerical_input = Input(shape=(len(numerical_cols),), name="numerical_input")
                categorical_inputs = []
                embeddings = []
                for col in categorical_cols:
                    vocab_size = categorical_mappings[col]['vocab_size']
                    embed_dim = 8
                    inp = Input(shape=(1,), name=f"cat_{col}", dtype=tf.int32)
                    categorical_inputs.append(inp)
                    emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, name=f"embed_{col}")(inp)
                    emb = layers.Flatten()(emb)
                    embeddings.append(emb)

                if numerical_cols:
                    all_features = Concatenate()([numerical_input] + embeddings)
                else:
                    all_features = Concatenate()(embeddings)

                x = GaussianNoise(0.05)(all_features)
                x = Dense(256, activation='relu')(x)
                x = BatchNormalization()(x)
                x = Dropout(0.4)(x)
                x = Dense(128, activation='relu')(x)
                x = BatchNormalization()(x)
                x = Dropout(0.3)(x)
                x = Dense(64, activation='relu')(x)
                x = BatchNormalization()(x)
                x = Dense(32, activation='relu')(x)
                output = Dense(1, name="output")(x)

                model = models.Model(inputs=[numerical_input] + categorical_inputs, outputs=output)
                model.compile(optimizer='adam', loss='mse', metrics=['mae'])

                train_inputs = [X_num_train_scaled] + X_cat_train_list
                test_inputs = [X_num_test_scaled] + X_cat_test_list

                # --- Обучение с прогресс-баром, таймером, ручным управлением LR и ранней остановкой ---
                total_epochs = 150
                progress_bar = st.progress(0, text="Инициализация...")
                status_text = st.empty()

                # Фиксируем валидационную выборку (20% от train)
                X_train_full, X_val, y_train_full, y_val = train_test_split(
                    train_inputs[0], y_train_scaled, test_size=0.2, random_state=42
                )
                if len(train_inputs) > 1:
                    val_cat = []
                    train_cat = []
                    for i in range(1, len(train_inputs)):
                        tr, val = train_test_split(train_inputs[i], test_size=0.2, random_state=42)
                        train_cat.append(tr)
                        val_cat.append(val)
                    train_inputs_fixed = [X_train_full] + train_cat
                    val_inputs = [X_val] + val_cat
                else:
                    train_inputs_fixed = [X_train_full]
                    val_inputs = [X_val]

                best_val_loss = np.inf
                patience_counter = 0
                current_lr = float(model.optimizer.learning_rate.numpy())
                history = {'loss': [], 'val_loss': [], 'mae': [], 'val_mae': []}

                # ЗАПУСК ТАЙМЕРА
                start_time = time.time()

                for epoch in range(total_epochs):
                    epoch_start = time.time()
                    hist = model.fit(
                        train_inputs_fixed, y_train_full,
                        epochs=1,
                        batch_size=32,
                        validation_data=(val_inputs, y_val),
                        verbose=0,
                        shuffle=False
                    )
                    for key in history.keys():
                        history[key].append(hist.history[key][0])

                    progress = (epoch + 1) / total_epochs
                    elapsed_total = time.time() - start_time
                    # Оценочное оставшееся время
                    if epoch > 0:
                        avg_time_per_epoch = elapsed_total / (epoch + 1)
                        remaining_epochs = total_epochs - (epoch + 1)
                        eta = avg_time_per_epoch * remaining_epochs
                        eta_str = f" | ETA: {eta:.1f} сек"
                    else:
                        eta_str = ""

                    progress_bar.progress(progress, text=f"Эпоха {epoch+1}/{total_epochs}")
                    status_text.info(
                        f"Эпоха {epoch+1}/{total_epochs} | val_loss: {history['val_loss'][-1]:.4f} | "
                        f"val_mae: {history['val_mae'][-1]:.4f} | Прошло: {elapsed_total:.1f} сек{eta_str}"
                    )

                    # ReduceLROnPlateau (patience=15, factor=0.5)
                    if history['val_loss'][-1] < best_val_loss - 1e-7:
                        best_val_loss = history['val_loss'][-1]
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= 15 and current_lr > 1e-6:
                            current_lr /= 2
                            model.optimizer.learning_rate.assign(current_lr)
                            patience_counter = 0
                            status_text.info(f"  🔽 Learning rate уменьшен до {current_lr:.2e}")

                    # EarlyStopping (patience=30)
                    if patience_counter >= 30:
                        status_text.warning(f"Ранняя остановка на эпохе {epoch+1}")
                        break

                # Окончание таймера
                elapsed_total = time.time() - start_time
                status_text.success(
                    f"✅ Обучение завершено за {elapsed_total:.2f} секунд ({elapsed_total/60:.2f} мин)"
                )

                st.session_state['history'] = history

                # Предсказание и метрики
                y_pred_scaled = model.predict(test_inputs)
                y_pred = scaler_y.inverse_transform(y_pred_scaled)
                metrics = {
                    "r2": r2_score(y_test, y_pred),
                    "mae": mean_absolute_error(y_test, y_pred)
                }

                st.session_state.update({
                    'model': model,
                    'scaler_x': scaler_x_num,
                    'scaler_y': scaler_y,
                    'numerical_cols': numerical_cols,
                    'categorical_mappings': categorical_mappings,
                    'feature_cols_names': numerical_cols + categorical_cols,
                    'saved_metrics': metrics
                })

                save_model_assets(model, scaler_x_num, scaler_y, numerical_cols,
                                  categorical_mappings, numerical_cols + categorical_cols, metrics)
                st.sidebar.success("💾 Модель с улучшенной архитектурой сохранена!")

            except Exception as e:
                st.error(f"Ошибка при обучении: {e}")

    # --- ОТОБРАЖЕНИЕ МЕТРИК И ГРАФИКОВ ---
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

        # Форма ввода с изменённым порядком полей
        with st.form("prediction_form"):
            input_dict = {}
            numerical_cols = st.session_state['numerical_cols']
            categorical_mappings = st.session_state['categorical_mappings']
            all_original_cols = st.session_state['feature_cols_names']

            # Приоритетные поля
            priority_fields = ['fiber_type', 'resin_type']
            ordered_fields = []
            for field in priority_fields:
                if field in all_original_cols:
                    ordered_fields.append(field)
            for field in all_original_cols:
                if field not in ordered_fields:
                    ordered_fields.append(field)

            cols = st.columns(3)
            for i, col_name in enumerate(ordered_fields):
                with cols[i % 3]:
                    display_name = get_display_name(col_name)
                    if col_name in categorical_mappings:
                        classes = categorical_mappings[col_name]['classes']
                        help_text = f"Допустимые значения: {', '.join(classes)}"
                        curr_val = st.session_state.get('current_inputs', {}).get(col_name, classes[0])
                        if curr_val not in classes:
                            curr_val = classes[0]
                        choice = st.selectbox(
                            display_name,
                            options=classes,
                            index=classes.index(curr_val),
                            help=help_text
                        )
                        input_dict[col_name] = choice
                    else:
                        help_text = "Введите числовое значение"
                        default_val = float(st.session_state.get('current_inputs', {}).get(col_name, 0.0))
                        input_dict[col_name] = st.number_input(
                            display_name,
                            value=default_val,
                            format="%.4f",
                            help=help_text
                        )

            if st.form_submit_button("🧪 РАССЧИТАТЬ ПРОГНОЗ"):
                # Подготовка входов для модели
                X_num = np.array([[input_dict[col] for col in numerical_cols]], dtype=np.float32)
                X_num_scaled = st.session_state['scaler_x'].transform(X_num) if numerical_cols else np.empty((1,0))
                X_cat = []
                for col in categorical_mappings.keys():
                    vocab = categorical_mappings[col]['vocab']
                    cat_str = input_dict[col].strip()
                    idx = vocab.get(cat_str, 0)
                    X_cat.append(np.array([[idx]], dtype=np.int32))
                model_inputs = [X_num_scaled] + X_cat
                pred_scaled = st.session_state['model'].predict(model_inputs)
                pred = st.session_state['scaler_y'].inverse_transform(pred_scaled).item()
                st.info(f"### Прогноз нейросети (Embedding + улучшенная архитектура): **{pred:.2f} МПа**")

                # Проверка наличия комбинации признаков в датасете
                show_comparison = False
                actual_value = None
                if 'raw_df' in st.session_state:
                    df_raw = st.session_state['raw_df']
                    target_col_name = df_raw.columns[-1]
                    feature_cols_for_match = st.session_state['feature_cols_names']
                    for _, row in df_raw.iterrows():
                        match = True
                        for col in feature_cols_for_match:
                            val_from_row = row[col]
                            val_from_input = input_dict[col]
                            if col in categorical_mappings:
                                if str(val_from_row).strip() != str(val_from_input).strip():
                                    match = False
                                    break
                            else:
                                try:
                                    if abs(float(val_from_row) - float(val_from_input)) > 1e-5:
                                        match = False
                                        break
                                except:
                                    match = False
                                    break
                        if match:
                            show_comparison = True
                            actual_value = float(row[target_col_name])
                            break

                if show_comparison and actual_value is not None:
                    abs_err = abs(pred - actual_value)
                    c_a, c_b, c_c = st.columns(3)
                    c_a.write(f"📊 Реальное значение: **{actual_value:.2f} МПа**")
                    c_b.write(f"❌ Абсолютная ошибка: **{abs_err:.2f} МПа**")
                    c_c.write(f"📉 Отклонение: **{(abs_err/actual_value)*100:.2f}%**")
                else:
                    st.warning("⚠️ Для данной комбинации значений признаков отсутствует экспериментальное значение Прочности. Метрики качества прогноза не могут быть определены!")

                st.balloons()

# ------------------ ВКЛАДКА ПОМОЩИ --------------------
with tab2:
    st.header("📘 Инструкция (Embedding + улучшенная архитектура)")
    st.markdown("""
    **1. Загрузка данных и обучение модели**  
    - В левой боковой панели загрузите CSV (последний столбец – прочность).  
    - Нажмите «🚀 Обучить новую модель».  

    **2. Особенности модели**  
    - Категориальные признаки преобразуются в плотные векторы (Embedding размерности 8).  
    - Добавлен слой 256 нейронов, Dropout (0.4 и 0.3), GaussianNoise для устойчивости.  
    - Используются ReduceLROnPlateau и EarlyStopping.  

    **3. Прогнозирование**  
    - Заполните поля ввода (для категорий – выпадающие списки).  
    - Нажмите «РАССЧИТАТЬ ПРОГНОЗ».  

    **4. Метрики**  
    - R² (чем ближе к 1, тем лучше), MAE – средняя абсолютная ошибка в МПа.  
    """)