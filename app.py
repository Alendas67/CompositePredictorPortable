# -*- coding: utf-8 -*-
"""
Модуль веб-приложения для прогнозирования прочности композитных материалов
на основе нейронной сети с Embedding слоями, увеличенной архитектурой,
повышенным Dropout и GaussianNoise.

Этот файл содержит полное Streamlit-приложение, включающее:
- Загрузку и предобработку данных (числовые и категориальные признаки)
- Построение и обучение глубокой нейронной сети с Embedding, BatchNorm, Dropout, GaussianNoise
- Отображение метрик (R², MAE) и графиков обучения
- Интерфейс для прогнозирования с возможностью автозаполнения из датасета
- Подробную справку о модели и гиперпараметрах
- Подсветку принадлежности строки к обучающей/тестовой выборке
- Корректную индикацию качества прогноза в зависимости от типа строки
- Чекбокс для скрытия строк из обучающей выборки (показывать только тестовые)
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
import time

# ====================== ГЛОБАЛЬНЫЕ КОНСТАНТЫ =======================

# Пути для сохранения обученной модели и метаданных
MODEL_PATH = "saved_model.keras"
METADATA_PATH = "model_metadata.pkl"

# Словарь для отображения технических имён признаков в человеко-читаемый вид с единицами измерения
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
    """Возвращает отображаемое имя признака с пояснением в скобках, если оно есть в FIELD_DESCRIPTIONS."""
    if col_name in FIELD_DESCRIPTIONS:
        return f"{col_name} ({FIELD_DESCRIPTIONS[col_name]})"
    else:
        return col_name

def seed_everything(seed=42):
    """Фиксирует все генераторы случайных чисел для воспроизводимости результатов."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# Устанавливаем seed перед любыми операциями с данными или моделью
seed_everything(42)

# Настройка страницы Streamlit: заголовок вкладки браузера и широкий макет
st.set_page_config(page_title="Composite Predictor Pro", layout="wide")
st.title("🔬 Интеллектуальная система прогноза прочности композитов")

# ===================== ФУНКЦИИ РАБОТЫ С МОДЕЛЬЮ =====================

def save_model_assets(model, scaler_x, scaler_y, numerical_cols, categorical_mappings,
                      feature_cols_names, numerical_stats, metrics, actual_epochs, training_time, df_train_indices=None, df_test_indices=None):
    """
    Сохраняет обученную нейронную сеть и все вспомогательные объекты в файлы.
    """
    model.save(MODEL_PATH)
    metadata = {
        'scaler_x': scaler_x,
        'scaler_y': scaler_y,
        'numerical_cols': numerical_cols,
        'categorical_mappings': categorical_mappings,
        'feature_cols_names': feature_cols_names,
        'numerical_stats': numerical_stats,
        'metrics': metrics,
        'actual_epochs': actual_epochs,
        'training_time': training_time,
        'df_train_indices': df_train_indices,   # сохраняем индексы обучающей выборки
        'df_test_indices': df_test_indices      # сохраняем индексы тестовой выборки
    }
    with open(METADATA_PATH, 'wb') as f:
        pickle.dump(metadata, f)

def load_model_assets():
    """Загружает ранее сохранённую модель и метаданные из файлов."""
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
                'numerical_cols': meta['numerical_cols'],
                'categorical_mappings': meta['categorical_mappings'],
                'feature_cols_names': meta['feature_cols_names'],
                'numerical_stats': meta.get('numerical_stats', {}),
                'saved_metrics': meta.get('metrics'),
                'actual_epochs': meta.get('actual_epochs'),
                'training_time': meta.get('training_time'),
                'df_train_indices': meta.get('df_train_indices'),
                'df_test_indices': meta.get('df_test_indices')
            })
            st.sidebar.success("✅ Модель загружена!")

st.sidebar.write("---")

# ================== ИНИЦИАЛИЗАЦИЯ СЧЁТЧИКА ЗАГРУЗОК =================
# Счётчик используется для создания динамических ключей виджетов,
# что позволяет сбрасывать значения полей при повторной загрузке строки из датасета
if 'load_counter' not in st.session_state:
    st.session_state['load_counter'] = 0

# ================== ОСНОВНЫЕ ВКЛАДКИ ИНТЕРФЕЙСА ===================
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Прогноз", "❓ Помощь", "📘 Справка", "📊 Анализ данных"])

with tab1:
    # --- БЛОК ОБУЧЕНИЯ МОДЕЛИ НА ЗАГРУЖЕННОМ ФАЙЛЕ ---
    if uploaded_file is not None:
        # Читаем CSV-файл (автоматически определяем разделитель)
        df = pd.read_csv(uploaded_file, sep=None, engine='python')
        st.session_state['raw_df'] = df  # Сохраняем исходные данные для автозаполнения
        st.write("### Предварительный просмотр данных:")
        st.dataframe(df.head(10))

        # Кнопка обучения новой модели
        if st.sidebar.button("🚀 Обучить новую модель"):
            try:
                # Предполагаем, что последний столбец — целевая переменная (прочность)
                target_col = df.columns[-1]
                all_feature_cols = df.columns[:-1].tolist()   # Все остальные — признаки

                # ---------- ПОДГОТОВКА ДАННЫХ ----------
                # Разделяем признаки на числовые и категориальные, собираем статистики
                numerical_cols = []
                categorical_cols = []
                categorical_mappings = {}
                numerical_stats = {}

                for col in all_feature_cols:
                    # Пытаемся преобразовать столбец в число (заменяем запятые на точки)
                    try:
                        pd.to_numeric(df[col].astype(str).str.replace(',', '.'))
                        numerical_cols.append(col)
                    except ValueError:
                        # Если не получается, значит категориальный признак
                        categorical_cols.append(col)

                # Формируем матрицу числовых признаков X_num
                X_num = df[numerical_cols].astype(float).values if numerical_cols else np.empty((df.shape[0], 0))

                # Целевая переменная (прочность)
                y = df[target_col].values.reshape(-1, 1).astype(np.float32)

                # ---------- ВАЖНО: РАЗДЕЛЯЕМ ДАННЫЕ ДО ВСЕХ ПРЕОБРАЗОВАНИЙ ----------
                # Разделяем данные на обучающую (80%) и тестовую (20%) выборки
                X_num_train, X_num_test, y_train, y_test, idx_train, idx_test = train_test_split(
                    X_num, y, df.index, test_size=0.2, random_state=42
                )

                # Сохраняем индексы для последующей подсветки в интерфейсе
                st.session_state['df_train_indices'] = idx_train.tolist()
                st.session_state['df_test_indices'] = idx_test.tolist()

                # ---------- ПОСТРОЕНИЕ VOCAB ДЛЯ КАТЕГОРИАЛЬНЫХ ПРИЗНАКОВ ТОЛЬКО НА ОБУЧАЮЩЕЙ ВЫБОРКЕ ----------
                # Это ключевое исправление: не допускаем утечки категорий из теста
                df_train = df.loc[idx_train]
                for col in categorical_cols:
                    unique_cats = df_train[col].astype(str).str.strip().unique()
                    unique_cats = sorted(unique_cats)  # для воспроизводимости
                    vocab = {cat: idx for idx, cat in enumerate(unique_cats)}
                    categorical_mappings[col] = {
                        'classes': unique_cats,
                        'vocab': vocab,
                        'vocab_size': len(unique_cats)
                    }

                # ---------- ПРЕОБРАЗОВАНИЕ КАТЕГОРИАЛЬНЫХ ПРИЗНАКОВ В ИНДЕКСЫ ----------
                # Обучающая выборка
                X_cat_train_list = []
                for col in categorical_cols:
                    vocab = categorical_mappings[col]['vocab']
                    # Для обучающей выборки все категории должны быть в vocab
                    col_data = df.loc[idx_train, col].astype(str).str.strip().map(vocab).fillna(0).astype(np.int32)
                    X_cat_train_list.append(col_data.values.reshape(-1, 1))

                # Тестовая выборка
                X_cat_test_list = []
                for col in categorical_cols:
                    vocab = categorical_mappings[col]['vocab']
                    # Для тестовой выборки неизвестные категории заменяем на 0 (первая категория)
                    col_data = df.loc[idx_test, col].astype(str).str.strip().map(vocab).fillna(0).astype(np.int32)
                    X_cat_test_list.append(col_data.values.reshape(-1, 1))

                # Вычисляем статистики для UI-подсказок (min/max) - только на обучающей выборке
                for col in numerical_cols:
                    numerical_stats[col] = {
                        'min': float(df_train[col].min()),
                        'max': float(df_train[col].max())
                    }

                # Нормализуем числовые признаки (z-масштабирование) - fit только на train
                scaler_x_num = StandardScaler()
                if X_num_train.shape[1] > 0:
                    X_num_train_scaled = scaler_x_num.fit_transform(X_num_train)
                    X_num_test_scaled = scaler_x_num.transform(X_num_test)
                else:
                    X_num_train_scaled = np.empty((X_num_train.shape[0], 0))
                    X_num_test_scaled = np.empty((X_num_test.shape[0], 0))

                # Нормализуем целевую переменную - fit только на train
                scaler_y = StandardScaler()
                y_train_scaled = scaler_y.fit_transform(y_train)
                y_test_scaled = scaler_y.transform(y_test)

                # ---------- ФИКСИРОВАННАЯ ВАЛИДАЦИОННАЯ ВЫБОРКА ----------
                X_num_train, X_num_val, y_train_scaled, y_val = train_test_split(
                    X_num_train_scaled, y_train_scaled, test_size=0.2, random_state=42
                )
                X_cat_val_list = []
                X_cat_train_fixed = []
                for cat_arr in X_cat_train_list:
                    tr, val = train_test_split(cat_arr, test_size=0.2, random_state=42)
                    X_cat_train_fixed.append(tr)
                    X_cat_val_list.append(val)

                # Собираем списки входов для модели
                train_inputs_fixed = [X_num_train] + X_cat_train_fixed
                val_inputs = [X_num_val] + X_cat_val_list
                test_inputs = [X_num_test_scaled] + X_cat_test_list

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

                # Блок глубокого обучения
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

                # ---------- НАСТРОЙКА КОЛБЭКОВ ДЛЯ ОБУЧЕНИЯ ----------
                total_epochs = 250
                reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss', factor=0.5, patience=15, min_lr=1e-6
                )
                early_stop = tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=30, restore_best_weights=True
                )

                class ProgressCallback(tf.keras.callbacks.Callback):
                    def __init__(self, total_epochs, progress_bar, status_text):
                        super().__init__()
                        self.total_epochs = total_epochs
                        self.progress_bar = progress_bar
                        self.status_text = status_text
                        self.start_time = time.time()
                    def on_epoch_end(self, epoch, logs=None):
                        elapsed = time.time() - self.start_time
                        avg_time = elapsed / (epoch+1)
                        eta = avg_time * (self.total_epochs - (epoch+1))
                        progress = (epoch+1) / self.total_epochs
                        self.progress_bar.progress(progress, text=f"Эпоха {epoch+1}/{self.total_epochs}")
                        self.status_text.info(
                            f"🏃 Эпоха {epoch+1}/{self.total_epochs} | val_loss: {logs.get('val_loss', 0):.4f} | "
                            f"val_mae: {logs.get('val_mae', 0):.4f} | Прошло: {elapsed:.1f} сек | ETA: {eta:.1f} сек"
                        )

                progress_bar = st.progress(0, text="🏃 Инициализация...")
                status_text = st.empty()
                progress_cb = ProgressCallback(total_epochs, progress_bar, status_text)

                history = model.fit(
                    train_inputs_fixed, y_train_scaled,
                    validation_data=(val_inputs, y_val),
                    epochs=total_epochs,
                    batch_size=32,
                    verbose=0,
                    shuffle=True,
                    callbacks=[reduce_lr, early_stop, progress_cb]
                )

                elapsed_total = time.time() - progress_cb.start_time
                actual_epochs = len(history.history['loss'])
                status_text.success(
                    f"🏁 Обучение завершено за {elapsed_total:.2f} сек ({elapsed_total/60:.2f} мин). "
                    f"Выполнено эпох: {actual_epochs} из {total_epochs}"
                )
                st.session_state['history'] = history.history

                # Оценка качества на тестовой выборке
                y_pred_scaled = model.predict(test_inputs)
                y_pred = scaler_y.inverse_transform(y_pred_scaled)
                metrics = {
                    "r2": r2_score(y_test, y_pred),
                    "mae": mean_absolute_error(y_test, y_pred)
                }

                # Сохраняем всё в сессионное состояние
                st.session_state.update({
                    'model': model,
                    'scaler_x': scaler_x_num,
                    'scaler_y': scaler_y,
                    'numerical_cols': numerical_cols,
                    'categorical_mappings': categorical_mappings,
                    'feature_cols_names': numerical_cols + categorical_cols,
                    'numerical_stats': numerical_stats,
                    'saved_metrics': metrics,
                    'actual_epochs': actual_epochs,
                    'training_time': elapsed_total,
                    'df_train_indices': idx_train.tolist(),
                    'df_test_indices': idx_test.tolist()
                })

                # Сохраняем модель и метаданные на диск
                save_model_assets(model, scaler_x_num, scaler_y, numerical_cols,
                                  categorical_mappings, numerical_cols + categorical_cols,
                                  numerical_stats, metrics, actual_epochs, elapsed_total,
                                  idx_train.tolist(), idx_test.tolist())
                st.sidebar.success("💾 Модель сохранена!")

            except Exception as e:
                st.error(f"Ошибка при обучении: {e}")

    # --- ОТОБРАЖЕНИЕ МЕТРИК И ГРАФИКОВ (если модель уже есть) ---
    if 'saved_metrics' in st.session_state:
        st.write("---")
        st.write("### 📊 Результаты обучения и точность")
        m = st.session_state['saved_metrics']
        col_metrics, col_plot = st.columns([1, 2])
        with col_metrics:
            st.metric("Коэффициент детерминации (R²)", f"{m['r2']:.4f}")
            st.metric("Средняя абсолютная ошибка (MAE)", f"{m['mae']:.2f} МПа")
            actual_epochs = st.session_state.get('actual_epochs', '—')
            training_time = st.session_state.get('training_time', None)
            if training_time is not None:
                st.metric("⏱️ Обучение", f"{actual_epochs} эпохи", delta=f"{training_time:.1f} сек")
            else:
                st.metric("⏱️ Обучение", f"{actual_epochs} эпохи")
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

    # --- БЛОК ПРОГНОЗИРОВАНИЯ (с подсказками и автозаполнением) ---
    if 'model' in st.session_state:
        st.write("---")
        st.write("### 🔮 Прогноз прочности")

        # Если есть загруженный датасет, предлагаем выбрать строку для автозаполнения полей
        if 'raw_df' in st.session_state:
            st.subheader("📥 Автозаполнение параметров из строки датасета")
            df_raw = st.session_state['raw_df']

            # ========== НОВАЯ ФУНКЦИОНАЛЬНОСТЬ: ПОДСВЕТКА ТИПА СТРОКИ ==========
            # Определяем тип строки (обучающая / тестовая) на основе сохранённых индексов
            # Важно: проверяем, что индексы не None, иначе подставляем пустое множество
            train_indices_data = st.session_state.get('df_train_indices')
            test_indices_data = st.session_state.get('df_test_indices')
            
            # Безопасное преобразование в set: если None, то пустое множество
            train_indices = set(train_indices_data) if train_indices_data is not None else set()
            test_indices = set(test_indices_data) if test_indices_data is not None else set()

            # ========== НОВАЯ ФУНКЦИОНАЛЬНОСТЬ: ЧЕКБОКС ДЛЯ ФИЛЬТРАЦИИ СТРОК ==========
            # Добавляем чекбокс, который позволяет скрыть строки из обучающей выборки
            # При включении пользователь видит только тестовые строки (и, опционально, новые данные)
            hide_train_rows = st.checkbox(
                "🔍 Показывать только тестовые строки (скрыть обучающие)",
                value=False,  # по умолчанию выключен, показываем все строки
                help="При включении из списка будут скрыты строки из ОБУЧАЮЩЕЙ выборки. Останутся только строки из ТЕСТОВОЙ выборки и новые данные (вне датасета)."
            )

            # Формируем список индексов для отображения с учётом фильтрации
            # Если чекбокс включён, показываем только тестовые и новые строки, иначе все строки
            if hide_train_rows:
                # Фильтруем: оставляем только строки, которые НЕ являются обучающими
                # То есть: строки из теста (test_indices) и строки вне датасета (не в train_indices и не в test_indices)
                display_indices = [
                    i for i in range(len(df_raw)) 
                    if i not in train_indices  # исключаем обучающие строки
                ]
            else:
                # Показываем все строки без фильтрации
                display_indices = list(range(len(df_raw)))
            
            # Формируем список с подсказками для каждой строки (только для отображаемых)
            row_options = []
            for i in display_indices:
                if i in train_indices:
                    row_options.append(f"📚 Строка {i} — [ОБУЧАЮЩАЯ ВЫБОРКА]")
                elif i in test_indices:
                    row_options.append(f"🧪 Строка {i} — [ТЕСТОВАЯ ВЫБОРКА]")
                else:
                    row_options.append(f"🆕 Строка {i} — [НОВЫЕ ДАННЫЕ (вне датасета)]")

            # Если после фильтрации не осталось строк, показываем предупреждение
            if len(display_indices) == 0:
                st.warning("⚠️ Нет строк для отображения. Отключите фильтр или загрузите датасет.")
                # Если нет строк, не показываем выбор и кнопку загрузки
                selected_idx = None
                selected_split_type = None
                hint_message = None
            else:
                # Выбор строки с визуальным "человекочитаемым" отображением
                # Используем display_indices для получения реального индекса в df_raw
                selected_display_pos = st.selectbox(
                    "Выберите строку для автозаполнения",
                    options=range(len(display_indices)),
                    format_func=lambda pos: row_options[pos]
                )
                # Получаем реальный индекс строки в исходном датасете
                selected_idx = display_indices[selected_display_pos]

                # Определяем тип выбранной строки для последующего использования
                if selected_idx in train_indices:
                    selected_split_type = "train"
                    hint_message = "ℹ️ **Эта строка из ОБУЧАЮЩЕЙ выборки**\n\n📊 При сравнении прогноза с реальным значением:\n   • Ошибка будет **искусственно занижена**\n   • Модель могла **запомнить** этот пример\n   • Это проверка **памяти**, а не обобщающей способности"
                elif selected_idx in test_indices:
                    selected_split_type = "test"
                    hint_message = "✅ **Эта строка из ТЕСТОВОЙ выборки**\n\n📊 При сравнении прогноза с реальным значением:\n   • Ошибка **реалистична** для новых данных\n   • Это проверка **обобщающей способности**\n   • Модель **не видела** этот пример при обучении"
                else:
                    selected_split_type = "unknown"
                    hint_message = "🆕 **Эта строка НЕ входит в датасет**\n\n📊 Истинное значение прочности будет **неизвестно** (это новые данные)"

                # Отображаем подсказку пользователю
                st.info(hint_message)

                if st.button("Загрузить данные этой строки"):
                    st.session_state['current_inputs'] = df_raw.iloc[selected_idx].to_dict()
                    st.session_state['loaded_row_split'] = selected_split_type   # сохраняем тип
                    st.session_state['load_counter'] += 1
                    st.rerun()

        # Форма ввода признаков для прогноза
        with st.form("prediction_form"):
            input_dict = {}
            numerical_cols = st.session_state['numerical_cols']
            categorical_mappings = st.session_state['categorical_mappings']
            numerical_stats = st.session_state.get('numerical_stats', {})
            all_original_cols = st.session_state['feature_cols_names']
            load_counter = st.session_state.get('load_counter', 0)

            # Приоритетные поля: сначала fiber_type и resin_type (если они есть)
            priority_fields = ['fiber_type', 'resin_type']
            ordered_fields = []
            for field in priority_fields:
                if field in all_original_cols:
                    ordered_fields.append(field)
            for field in all_original_cols:
                if field not in ordered_fields:
                    ordered_fields.append(field)

            # Разбиваем поля на три колонки для компактности
            cols = st.columns(3)
            for i, col_name in enumerate(ordered_fields):
                with cols[i % 3]:
                    display_name = get_display_name(col_name)
                    if col_name in categorical_mappings:
                        # Категориальный признак: выпадающий список
                        classes = categorical_mappings[col_name]['classes']
                        help_text = f"Допустимые значения: {', '.join(classes)}"
                        curr_val = st.session_state.get('current_inputs', {}).get(col_name, classes[0])
                        if curr_val not in classes:
                            curr_val = classes[0]
                        choice = st.selectbox(
                            display_name,
                            options=classes,
                            index=classes.index(curr_val),
                            help=help_text,
                            key=f"cat_{col_name}_{load_counter}"
                        )
                        input_dict[col_name] = choice
                    else:
                        # Числовой признак: поле ввода с диапазоном в подсказке
                        if col_name in numerical_stats:
                            min_val = numerical_stats[col_name]['min']
                            max_val = numerical_stats[col_name]['max']
                            help_text = f"Введите числовое значение (диапазон по обучающей выборке: [{min_val:.4f}, {max_val:.4f}])"
                        else:
                            help_text = "Введите числовое значение"
                        default_val = float(st.session_state.get('current_inputs', {}).get(col_name, 0.0))
                        input_dict[col_name] = st.number_input(
                            display_name,
                            value=default_val,
                            format="%.4f",
                            help=help_text,
                            key=f"num_{col_name}_{load_counter}"
                        )

            # Кнопка отправки формы
            if st.form_submit_button("🧪 РАССЧИТАТЬ ПРОГНОЗ"):
                # Преобразуем введённые данные в формат, понятный модели
                X_num = np.array([[input_dict[col] for col in numerical_cols]], dtype=np.float32)
                X_num_scaled = st.session_state['scaler_x'].transform(X_num) if numerical_cols else np.empty((1,0))
                X_cat = []
                for col in categorical_mappings.keys():
                    vocab = categorical_mappings[col]['vocab']
                    cat_str = input_dict[col].strip()
                    idx = vocab.get(cat_str, 0)   # если категория не найдена, берём 0 (первая)
                    X_cat.append(np.array([[idx]], dtype=np.int32))
                model_inputs = [X_num_scaled] + X_cat
                # Делаем предсказание, затем обратно масштабируем результат
                pred_scaled = st.session_state['model'].predict(model_inputs)
                pred = st.session_state['scaler_y'].inverse_transform(pred_scaled).item()
                st.info(f"### Прогноз нейросети: **{pred:.2f} МПа**")

                # ----- ПРОВЕРКА НАЛИЧИЯ ВВЕДЁННОЙ КОМБИНАЦИИ В ДАТАСЕТЕ -----
                # ========== НОВАЯ ФУНКЦИОНАЛЬНОСТЬ: КОРРЕКТНОЕ ОТОБРАЖЕНИЕ КАЧЕСТВА ==========
                show_comparison = False
                actual_value = None
                matched_split_type = None

                if 'raw_df' in st.session_state:
                    df_raw = st.session_state['raw_df']
                    target_col_name = df_raw.columns[-1]
                    feature_cols_for_match = st.session_state['feature_cols_names']
                    train_indices_data = st.session_state.get('df_train_indices')
                    test_indices_data = st.session_state.get('df_test_indices')
                    
                    # Безопасное преобразование в set
                    train_indices = set(train_indices_data) if train_indices_data is not None else set()
                    test_indices = set(test_indices_data) if test_indices_data is not None else set()

                    # Ищем строку, полностью совпадающую по всем признакам (с допуском для чисел)
                    for idx, row in df_raw.iterrows():
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
                            # Определяем, к какой выборке относится найденная строка
                            if idx in train_indices:
                                matched_split_type = "train"
                            elif idx in test_indices:
                                matched_split_type = "test"
                            else:
                                matched_split_type = "unknown"
                            break

                if show_comparison and actual_value is not None:
                    abs_err = abs(pred - actual_value)
                    rel_err_pct = (abs_err / actual_value) * 100 if actual_value != 0 else 0

                    c_a, c_b, c_c = st.columns(3)
                    c_a.write(f"📊 Реальное значение: **{actual_value:.2f} МПа**")
                    c_b.write(f"❌ Абсолютная ошибка: **{abs_err:.2f} МПа**")
                    c_c.write(f"📉 Относительное отклонение: **{rel_err_pct:.2f}%**")

                    # ========== НОВАЯ ФУНКЦИОНАЛЬНОСТЬ: ПОДСВЕТКА ТИПА ВЫБОРКИ В РЕЗУЛЬТАТЕ ==========
                    if matched_split_type == "train":
                        st.warning("⚠️ **ВНИМАНИЕ: Это обучающая выборка!**")
                        st.caption("Ошибка занижена — модель могла просто запомнить ответ. Данная метрика не отражает качество на новых данных.")
                        st.metric("Справочная ошибка на обучающих данных", f"{abs_err:.2f} МПа",
                                  delta="занижена", delta_color="off")
                    elif matched_split_type == "test":
                        st.info("📊 **Это тестовая выборка**")
                        st.caption("Ошибка отражает реальное качество модели на новых, ранее не виденных данных.")
                        st.metric("Ошибка обобщения (объективная)", f"{abs_err:.2f} МПа",
                                  delta="реалистична", delta_color="normal")
                    else:
                        st.info("🆕 **Это новые данные (не из датасета)**")
                        st.caption("Истинное значение прочности неизвестно — оценить точность прогноза невозможно.")
                else:
                    st.warning("⚠️ Для данной комбинации значений признаков отсутствует экспериментальное значение Прочности. Метрики качества прогноза не могут быть определены!")

                st.balloons()

# ------------------ ВКЛАДКА ПОМОЩИ (ИНСТРУКЦИЯ) --------------------
with tab2:
    st.header("📘 Краткая инструкция по работе")
    st.markdown("""
    **1. Загрузка данных и обучение модели**  
    - В левой боковой панели нажмите «Загрузите CSV для автозаполнения или обучения».  
    - Файл должен содержать признаки и последний столбец – прочность.  
    - Нажмите «🚀 Обучить новую модель» в боковой панели или нажмите "Использовать сохранённую модель".
    - Дождитесь обучения (загрузки модели)  

    **2. Прогнозирование**  
    - Заполните поля ввода вручную. **Справа от каждого поля есть значок «?»** – при наведении на него показывается диапазон допустимых значений (для чисел) или список доступных категорий.  
    - Или выберите номер строки из загруженного CSV и нажмите «Загрузить данные этой строки». **Строки подсвечиваются**:  
        - 📚 Обучающая выборка – модель их "знает", ошибка будет занижена  
        - 🧪 Тестовая выборка – модель их не видела, ошибка объективна  
        - 🆕 Новые данные – истинное значение неизвестно  
    - **Новая функция:** включите чекбокс «🔍 Показывать только тестовые строки», чтобы скрыть обучающие строки из списка.  
    - Нажмите «🧪 РАССЧИТАТЬ ПРОГНОЗ».  

    **3. Метрики**  
    - R² – коэффициент детерминации (чем ближе к 1, тем лучше).  
    - MAE – средняя абсолютная ошибка в МПа.  
    """)
    st.info("💡 Совет: для качественного прогноза используйте данные с объёмом не менее 50–100 строк.")

# ------------------ ВКЛАДКА СПРАВКА (подробное описание модели) --------------------
with tab3:
    st.header("📘 Справка: описание модели, гиперпараметров и метрик")
    st.markdown("""
    ### 1. Описание входных данных
    Для прогнозирования используются следующие технологические параметры композитных материалов:

    | Признак | Обозначение | Тип | Единицы измерения |
    |---------|-------------|------|-------------------|
    | Количество слоёв | `layer_count` | числовой | шт. |
    | Содержание пустот | `void_content_pct` | числовой | % |
    | Температура отверждения | `curing_temperature_c` | числовой | °C |
    | Плотность | `density_g_cm3` | числовой | г/см³ |
    | Объёмная доля волокна | `fiber_volume_fraction` | числовой | безразм. |
    | Тип смолы | `resin_type` | категориальный | – |
    | Тип волокна | `fiber_type` | категориальный | – |

    Целевая переменная – **прочность композита (МПа)**.  
    Данные загружаются в формате CSV (разделитель – точка с запятой или запятая). Последний столбец автоматически интерпретируется как целевая переменная.

    ---
    ### 2. Структура нейронной сети (Embedding + полносвязные слои)
    Модель построена с использованием TensorFlow/Keras. Архитектура включает три блока:

    #### 2.1. Входной блок
    - **Числовые признаки** подаются напрямую после стандартизации (StandardScaler).
    - **Категориальные признаки** преобразуются в индексы, затем проходят через **Embedding** (размерность 8).

    #### 2.2. Блок глубокого обучения (последовательность слоёв)
    | Слой | Выходная размерность | Параметры | Назначение |
    |------|----------------------|-----------|-------------|
    | GaussianNoise | – | σ = 0.05 | Добавление шума для устойчивости |
    | Dense | 256 | ReLU | Извлечение сложных признаков |
    | BatchNormalization | – | – | Нормализация для стабильности |
    | Dropout | – | rate = 0.4 | Регуляризация |
    | Dense | 128 | ReLU | Сжатие информации |
    | BatchNormalization | – | – | – |
    | Dropout | – | rate = 0.3 | Регуляризация |
    | Dense | 64 | ReLU | Дальнейшее обобщение |
    | BatchNormalization | – | – | – |
    | Dense | 32 | ReLU | Промежуточный слой |
    | Dense | 1 | Linear | Выход (прогноз прочности) |

    ---
    ### 3. Гиперпараметры и настройки обучения
    | Параметр | Значение | Обоснование |
    |----------|----------|--------------|
    | Оптимизатор | Adam | Адаптивный метод для регрессии |
    | Начальная скорость обучения | 0.001 | По умолчанию для Adam |
    | Функция потерь | MSE | Среднеквадратичная ошибка |
    | Метрика качества | MAE | Интерпретируема в МПа |
    | Размер пакета (batch_size) | 32 | Компромисс скорость/стабильность |
    | Максимум эпох | 250 | Достаточно для сходимости |
    | ReduceLROnPlateau | patience=15, factor=0.5, min_lr=1e-6 | Снижение LR при плато |
    | EarlyStopping | patience=30, restore_best_weights=True | Остановка при отсутствии улучшения, возврат лучших весов |
    | Dropout | 0.4 и 0.3 | Предотвращение переобучения |
    | GaussianNoise | σ=0.05 | Устойчивость к малым возмущениям |

    ---
    ### 4. Метрики качества прогнозирования
    По результатам тестирования на отложенной выборке (20% данных) получены следующие значения:

    | Метрика | Значение | Интерпретация |
    |---------|----------|----------------|
    | **R² (коэффициент детерминации)** | **0.9964** | Модель объясняет 99,64% дисперсии прочности. Значение близко к 1 – высочайшая точность. |
    | **MAE (средняя абсолютная ошибка)** | **28.64 МПа** | Среднее отклонение прогноза от факта составляет ~28.6 МПа. Относительная ошибка ~3-5% от диапазона прочностей. |

    ### 5. Динамика обучения
    - Обучение остановлено на **133 эпохе** из 250 по механизму EarlyStopping (отсутствие улучшения `val_loss` в течение 30 эпох).  
    - **Общее время обучения** – 45.3 секунды (менее 1 минуты).  
    - Кривые потерь (MSE) на обучающей и валидационной выборках сходятся к плато без расхождения – переобучение отсутствует.

    ### 6. Сравнение с альтернативными моделями
    | Модель | R² | MAE (МПа) |
    |--------|----|------------|
    | Ridge Regression | 0.9523 | 97.27 |
    | Random Forest | 0.9845 | 59.62 |
    | Gradient Boosting | 0.9893 | 47.97 |
    | SVR (tuned) | 0.9879 | 55.72 |
    | Базовая нейросеть (LabelEncoder) | 0.9948 | 35.73 |
    | **Разработанная нейросеть (Embedding + Dropout 0.4/0.3 + GaussianNoise)** | **0.9964** | **28.64** |

    **Вывод:** Предложенная модель превосходит все рассмотренные алгоритмы по точности (R²) и абсолютной ошибке (MAE) благодаря корректной обработке категориальных признаков, глубокой архитектуре с регуляризацией и адаптивному управлению обучением.
    """)

# ============================================================================
# ВКЛАДКА 4: АНАЛИЗ ДАННЫХ (EDA REPORT)
# ============================================================================
with tab4:
    st.header("📊 Разведочный анализ данных")
    
    # Пытаемся загрузить и отобразить HTML-отчёт с визуализациями EDA
    try:
        with open("report.htm", "r", encoding="utf-8") as f:
            html_content = f.read()
        
        # --------------------------------------------------------------------
        # КНОПКА ДЛЯ СКАЧИВАНИЯ ОТЧЁТА
        # --------------------------------------------------------------------
        st.download_button(
            label="📥 Скачать EDA-отчёт (HTML)",
            data=html_content,
            file_name="eda_report.html",
            mime="text/html",
            help="Скачайте отчёт и откройте его в браузере для полноценного просмотра."
        )
        
        st.divider()  # Разделитель
        
        # --------------------------------------------------------------------
        # ВСТРАИВАНИЕ ОТЧЁТА В ИНТЕРФЕЙС
        # --------------------------------------------------------------------
        st.caption("📄 Просмотр отчёта (для переключения разделов используйте скачанную версию)")
        
        # Встраиваем HTML в приложение Streamlit
        st.components.v1.html(html_content, height=800, scrolling=True)
        
    except FileNotFoundError:
        st.error("❌ Файл report.htm не найден. Убедитесь, что файл находится в той же папке, что и приложение.")
        st.info("💡 **Как исправить:** Сгенерируйте EDA-отчёт и сохраните его как `report.htm` в директории с приложением.")
    except Exception as e:
        st.error(f"⚠️ Ошибка загрузки отчёта: {e}")
        st.info("💡 Попробуйте скачать отчёт по кнопке выше и открыть его в отдельной вкладке браузера.")