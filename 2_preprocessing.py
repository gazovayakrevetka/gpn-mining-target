from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib


RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parent
DATA_XLSX = BASE_DIR / "mining_block_model.xlsx"
DATA_CSV = BASE_DIR / "mining_block_model.csv"

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

CATEGORICAL_FEATURES = ["Rock_Type"]
DERIVED_FEATURES = ["Total_Cost_per_tonne"]
TARGET_COLUMN = "Target"


def load_data() -> pd.DataFrame:
    """Загружает данные из Excel или CSV в папке GPN2."""
    if DATA_XLSX.exists():
        df = pd.read_excel(DATA_XLSX)
        print(f"Загружен Excel: {DATA_XLSX}")
    elif DATA_CSV.exists():
        df = pd.read_csv(DATA_CSV)
        print(f"Загружен CSV: {DATA_CSV}")
    else:
        raise FileNotFoundError(
            "Не найден файл данных. "
            "Положите mining_block_model.xlsx или mining_block_model.csv в папку GPN2."
        )
    return df


def clean_and_engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Простая очистка и создание признаков.
    - Удаляем строки без Target.
    - Числовые пропуски -> медиана.
    - Rock_Type -> мода.
    - Добавляем Total_Cost_per_tonne.
    """
    df = df.copy()

    df = df.dropna(subset=[TARGET_COLUMN])

    if "Waste_Flag" in df.columns and df["Waste_Flag"].dtype == bool:
        df["Waste_Flag"] = df["Waste_Flag"].astype(int)

    numeric_cols = [c for c in NUMERIC_FEATURES_BASE if c in df.columns]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    if "Rock_Type" in df.columns:
        mode_value = df["Rock_Type"].mode(dropna=True)
        fill_value = mode_value.iloc[0] if not mode_value.empty else "Unknown"
        df["Rock_Type"] = df["Rock_Type"].fillna(fill_value).astype(str)

    if "Mining_Cost (USD)" in df.columns and "Processing_Cost (USD)" in df.columns:
        df["Total_Cost_per_tonne"] = (
            df["Mining_Cost (USD)"] + df["Processing_Cost (USD)"]
        )
    else:
        df["Total_Cost_per_tonne"] = np.nan

    df["Total_Cost_per_tonne"] = df["Total_Cost_per_tonne"].fillna(
        df["Total_Cost_per_tonne"].median()
    )

    return df


def preprocess_and_split(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, OneHotEncoder, StandardScaler]:
    """Выполняет предобработку и разбиение на train/test."""
    df = clean_and_engineer_features(df)

    numeric_features = [c for c in NUMERIC_FEATURES_BASE + DERIVED_FEATURES if c in df.columns]
    categorical_features = [c for c in CATEGORICAL_FEATURES if c in df.columns]

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"В данных нет столбца '{TARGET_COLUMN}'.")

    X = df[numeric_features + categorical_features]
    y = df[TARGET_COLUMN].values

    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(X_train_df[numeric_features])
    X_test_num = scaler.transform(X_test_df[numeric_features])

    if categorical_features:
        # Для новых версий scikit-learn используем sparse_output=False
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        X_train_cat = encoder.fit_transform(X_train_df[categorical_features])
        X_test_cat = encoder.transform(X_test_df[categorical_features])
    else:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        X_train_cat = np.empty((X_train_num.shape[0], 0))
        X_test_cat = np.empty((X_test_num.shape[0], 0))

    X_train = np.hstack([X_train_num, X_train_cat])
    X_test = np.hstack([X_test_num, X_test_cat])

    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

    return X_train, X_test, y_train, y_test, encoder, scaler


def save_artifacts(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    encoder: OneHotEncoder,
    scaler: StandardScaler,
) -> None:
    """Сохраняет предобработанные данные и преобразователи."""
    joblib.dump(X_train, BASE_DIR / "X_train.pkl")
    joblib.dump(X_test, BASE_DIR / "X_test.pkl")
    joblib.dump(y_train, BASE_DIR / "y_train.pkl")
    joblib.dump(y_test, BASE_DIR / "y_test.pkl")
    joblib.dump(encoder, BASE_DIR / "encoder.pkl")
    joblib.dump(scaler, BASE_DIR / "scaler.pkl")

    metadata = {
        "numeric_features_base": NUMERIC_FEATURES_BASE,
        "derived_features": DERIVED_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target_column": TARGET_COLUMN,
    }
    joblib.dump(metadata, BASE_DIR / "preprocessing_metadata.pkl")

    print(
        "Сохранены: X_train.pkl, X_test.pkl, y_train.pkl, y_test.pkl, "
        "encoder.pkl, scaler.pkl, preprocessing_metadata.pkl"
    )


def main() -> None:
    df = load_data()
    X_train, X_test, y_train, y_test, encoder, scaler = preprocess_and_split(df)
    save_artifacts(X_train, X_test, y_train, y_test, encoder, scaler)


if __name__ == "__main__":
    main()

