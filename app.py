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

    # ----- ПРОВЕРКА НАЛИЧИЯ КОМБИНАЦИИ ПРИЗНАКОВ В ДАТАСЕТЕ -----
    show_comparison = False
    actual_value = None

    if 'raw_df' in st.session_state:
        df_raw = st.session_state['raw_df']
        target_col_name = df_raw.columns[-1]
        feature_cols_names = st.session_state['feature_cols_names']  # все признаки, кроме целевого

        # Перебираем строки датасета в поисках полного совпадения по всем признакам
        for idx, row in df_raw.iterrows():
            match = True
            for col in feature_cols_names:
                val_from_row = row[col]
                val_from_input = input_dict[col]

                # Приводим к строке (обрезаем пробелы) для категориальных
                if col in categorical_mappings:
                    if str(val_from_row).strip() != str(val_from_input).strip():
                        match = False
                        break
                else:
                    # Числовое сравнение с допуском 1e-5
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

    # Вывод метрик сравнения только если найдено точное совпадение
    if show_comparison and actual_value is not None:
        abs_err = abs(pred - actual_value)
        c_a, c_b, c_c = st.columns(3)
        c_a.write(f"📊 Реальное значение: **{actual_value:.2f} МПа**")
        c_b.write(f"❌ Абсолютная ошибка: **{abs_err:.2f} МПа**")
        c_c.write(f"📉 Отклонение: **{(abs_err/actual_value)*100:.2f}%**")
    else:
        st.warning("⚠️ Для данной комбинации значений признаков отсутствует экспериментальное значение Прочности. Метрики качества прогноза не могут быть определены!")

    st.balloons()