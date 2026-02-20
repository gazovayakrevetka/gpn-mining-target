import io
from pathlib import Path
from typing import Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent

NUMERIC_FEATURES_BASE = [
    "X",
    "Y",
    "Z",
    "Ore_Grade (%)",
    "Tonnage",
    "Ore_Value (USD/tonne)",
    "Mining_Cost (USD)",
    "Processing_Cost (USD)",
    "Waste_Flag",
]
DERIVED_FEATURES = ["Total_Cost_per_tonne"]
CATEGORICAL_FEATURES = ["Rock_Type"]
TARGET_COLUMN = "Target"


@st.cache_resource
def load_artifacts():
    """Загружает модель, кодировщик, скейлер и метрики из папки GPN2."""
    best_model = joblib.load(BASE_DIR / "best_model.pkl")
    encoder = joblib.load(BASE_DIR / "encoder.pkl")
    scaler = joblib.load(BASE_DIR / "scaler.pkl")

    metrics_text = ""
    metrics_json = BASE_DIR / "model_performance.json"
    metrics_txt = BASE_DIR / "model_performance.txt"
    if metrics_json.exists():
        metrics_text = metrics_json.read_text(encoding="utf-8")
    elif metrics_txt.exists():
        metrics_text = metrics_txt.read_text(encoding="utf-8")

    metadata_path = BASE_DIR / "preprocessing_metadata.pkl"
    metadata = joblib.load(metadata_path) if metadata_path.exists() else {}

    return best_model, encoder, scaler, metrics_text, metadata


def preprocess_dataframe(
    df: pd.DataFrame, encoder, scaler
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Приводит DataFrame к формату, ожидаемому моделью:
    - приводим типы,
    - заполняем пропуски,
    - создаём Total_Cost_per_tonne,
    - применяем scaler и encoder.
    """
    df = df.copy()

    if "Waste_Flag" in df.columns and df["Waste_Flag"].dtype == bool:
        df["Waste_Flag"] = df["Waste_Flag"].astype(int)

    for col in NUMERIC_FEATURES_BASE:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = 0.0

    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str)
        else:
            df[col] = "Unknown"

    df["Total_Cost_per_tonne"] = df["Mining_Cost (USD)"] + df["Processing_Cost (USD)"]

    numeric_features = NUMERIC_FEATURES_BASE + DERIVED_FEATURES
    cat_features = CATEGORICAL_FEATURES

    X_num = scaler.transform(df[numeric_features])
    X_cat = encoder.transform(df[cat_features]) if cat_features else np.empty(
        (X_num.shape[0], 0)
    )

    X = np.hstack([X_num, X_cat])
    return X, df


def single_block_input() -> pd.DataFrame:
    """Форма ввода одного блока в сайдбаре."""
    st.sidebar.subheader("Параметры блока")

    x = st.sidebar.number_input("X", value=0.0)
    y = st.sidebar.number_input("Y", value=0.0)
    z = st.sidebar.number_input("Z", value=0.0)

    ore_grade = st.sidebar.number_input("Ore_Grade (%)", min_value=0.0, value=1.0)
    tonnage = st.sidebar.number_input("Tonnage", min_value=0.0, value=1000.0)
    ore_value = st.sidebar.number_input(
        "Ore_Value (USD/tonne)", min_value=0.0, value=50.0
    )
    mining_cost = st.sidebar.number_input(
        "Mining_Cost (USD)", min_value=0.0, value=20.0
    )
    processing_cost = st.sidebar.number_input(
        "Processing_Cost (USD)", min_value=0.0, value=10.0
    )
    waste_flag = st.sidebar.checkbox("Waste_Flag (1 = waste, 0 = ore?)", value=False)

    _, encoder, _, _, _ = load_artifacts()
    if CATEGORICAL_FEATURES and hasattr(encoder, "categories_"):
        rock_categories = list(encoder.categories_[0])
    else:
        rock_categories = ["Type_A", "Type_B", "Type_C"]
    rock_type = st.sidebar.selectbox("Rock_Type", rock_categories)

    data = {
        "X": x,
        "Y": y,
        "Z": z,
        "Ore_Grade (%)": ore_grade,
        "Tonnage": tonnage,
        "Ore_Value (USD/tonne)": ore_value,
        "Mining_Cost (USD)": mining_cost,
        "Processing_Cost (USD)": processing_cost,
        "Waste_Flag": int(waste_flag),
        "Rock_Type": rock_type,
    }
    return pd.DataFrame([data])


def batch_input() -> pd.DataFrame:
    """Загрузка CSV/XLSX с несколькими блоками."""
    st.subheader("Загрузка файла с блоками")
    uploaded_file = st.file_uploader(
        "Загрузите CSV или Excel (XLSX) файл с данными блоков", type=["csv", "xlsx"]
    )
    if not uploaded_file:
        return pd.DataFrame()

    if uploaded_file.name.lower().endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.write("Первые строки загруженного файла:")
    st.dataframe(df.head())
    return df


def plot_prediction_vs_actual(df: pd.DataFrame, y_pred: np.ndarray) -> None:
    """Строит scatter plot предсказаний против факта при наличии Target."""
    if TARGET_COLUMN not in df.columns:
        st.info("В загруженном файле нет столбца Target – график не построен.")
        return

    valid_mask = df[TARGET_COLUMN].notna()
    if valid_mask.sum() == 0:
        st.info("Во всех строках Target = NaN – график не построен.")
        return

    y_true = df.loc[valid_mask, TARGET_COLUMN]
    y_pred_valid = y_pred[valid_mask.values]

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred_valid, alpha=0.6)
    max_val = max(y_true.max(), y_pred_valid.max())
    min_val = min(y_true.min(), y_pred_valid.min())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", label="y = x")
    plt.xlabel("Фактический Target")
    plt.ylabel("Предсказанный Target")
    plt.title("Предсказание vs Факт")
    plt.legend()
    st.pyplot(plt.gcf())
    plt.close()


def add_css() -> None:
    """Небольшой CSS для улучшения внешнего вида."""
    st.markdown(
        """
        <style>
        .main {
            background-color: #f7f7f7;
        }
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
        h1, h2, h3 {
            color: #003366;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(
        page_title="Прогноз Target для горных блоков",
        layout="wide",
    )
    add_css()
    st.title("Прогноз целевой переменной в данных горной добычи")

    st.markdown(
        """
        Это приложение использует обученную модель для прогноза целевой переменной **Target**
        по данным блока (координаты, тип породы, содержание руды, стоимости и пр.).
        """
    )

    try:
        model, encoder, scaler, metrics_text, metadata = load_artifacts()
    except Exception as e:
        st.error(
            "Не удалось загрузить модель или преобразователи. "
            "Убедитесь, что вы запустили 2_preprocessing.py и 3_model_training.py.\n\n"
            f"Ошибка: {e}"
        )
        return

    col_left, col_right = st.columns([1, 2])

    with col_left:
        with st.expander("О модели и пайплайне", expanded=True):
            st.write(
                """
                1. **1_eda.py** – разведочный анализ данных и графики.
                2. **2_preprocessing.py** – предобработка, масштабирование и кодирование.
                3. **3_model_training.py** – обучение нескольких моделей и выбор лучшей.
                4. **app.py** – это приложение для интерактивного использования модели.
                """
            )

        with st.expander("Метрики модели", expanded=True):
            if metrics_text:
                st.text(metrics_text)
            else:
                st.info("Файл model_performance.txt не найден.")

        with st.expander("Пояснения по признакам"):
            st.markdown(
                """
                - **X, Y, Z** – координаты блока в пространстве.
                - **Rock_Type** – тип породы (категориальный признак).
                - **Ore_Grade (%)** – содержание полезного компонента в руде.
                - **Tonnage** – тоннаж блока.
                - **Ore_Value (USD/tonne)** – стоимость руды за тонну.
                - **Mining_Cost (USD)** – затраты на добычу за тонну.
                - **Processing_Cost (USD)** – затраты на переработку за тонну.
                - **Waste_Flag** – флаг, указывающий на пустую породу.
                - **Total_Cost_per_tonne** – суммарные затраты на тонну (созданный признак).
                """
            )

    with col_right:
        mode = st.radio(
            "Режим работы",
            ["Один блок", "Файл с несколькими блоками"],
            horizontal=True,
        )

        if mode == "Один блок":
            df_input = single_block_input()
            if st.button("Предсказать", type="primary"):
                X, df_proc = preprocess_dataframe(df_input, encoder, scaler)
                y_pred = model.predict(X)[0]
                st.success(f"Предсказанное значение Target: **{y_pred:.4f}**")
                st.write("Использованные входные данные:")
                st.dataframe(df_proc)
        else:
            df_batch = batch_input()
            if not df_batch.empty and st.button("Предсказать для файла", type="primary"):
                X, df_proc = preprocess_dataframe(df_batch, encoder, scaler)
                y_pred = model.predict(X)
                df_result = df_proc.copy()
                df_result["Predicted_Target"] = y_pred

                st.subheader("Результаты прогнозирования")
                st.dataframe(df_result.head(50))

                csv_buf = io.StringIO()
                df_result.to_csv(csv_buf, index=False)
                st.download_button(
                    label="Скачать результаты в CSV",
                    data=csv_buf.getvalue().encode("utf-8-sig"),
                    file_name="predictions.csv",
                    mime="text/csv",
                )

                plot_prediction_vs_actual(df_batch, y_pred)


if __name__ == "__main__":
    main()

