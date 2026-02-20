import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


sns.set(style="whitegrid", context="notebook")

BASE_DIR = Path(__file__).resolve().parent
DATA_XLSX = BASE_DIR / "mining_block_model.xlsx"
DATA_CSV = BASE_DIR / "mining_block_model.csv"
IMAGES_DIR = BASE_DIR / "images"


def load_data() -> pd.DataFrame:
    """
    Загружает данные из Excel (mining_block_model.xlsx) в папке GPN2.
    Если Excel нет, пробует CSV (mining_block_model.csv) в той же папке.
    """
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


def ensure_images_dir() -> None:
    """Создаёт папку images/, если её нет."""
    os.makedirs(IMAGES_DIR, exist_ok=True)


def basic_info(df: pd.DataFrame) -> None:
    """Выводит базовую информацию о данных."""
    print("\n=== Первые строки ===")
    print(df.head())

    print("\n=== Общая информация ===")
    print(df.info())

    print("\n=== Описательная статистика (числовые) ===")
    print(df.describe())

    print("\n=== Пропуски ===")
    print(df.isna().sum())


def plot_numeric_distributions(df: pd.DataFrame) -> None:
    """Строит гистограммы и boxplot'ы для всех числовых признаков."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in numeric_cols:
        safe_name = col.replace(" ", "_").replace("/", "_")

        plt.figure(figsize=(6, 4))
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f"Histogram of {col}")
        plt.tight_layout()
        plt.savefig(IMAGES_DIR / f"hist_{safe_name}.png", dpi=150)
        plt.close()

        plt.figure(figsize=(4, 4))
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot of {col}")
        plt.tight_layout()
        plt.savefig(IMAGES_DIR / f"box_{safe_name}.png", dpi=150)
        plt.close()


def plot_correlations(df: pd.DataFrame) -> None:
    """Строит корреляционную матрицу и pairplot для ключевых признаков."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        print("Нет числовых признаков для корреляции.")
        return

    plt.figure(figsize=(10, 8))
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=False, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation matrix (numeric features)")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()

    candidate_cols = [
        "Ore_Grade (%)",
        "Tonnage",
        "Ore_Value (USD/tonne)",
        "Mining_Cost (USD)",
        "Processing_Cost (USD)",
        "Target",
    ]
    pair_cols = [c for c in candidate_cols if c in df.columns]
    if len(pair_cols) >= 2:
        sns.pairplot(df[pair_cols].dropna())
        plt.savefig(IMAGES_DIR / "pairplot_key_features.png", dpi=150)
        plt.close()
    else:
        print("Недостаточно признаков для pairplot.")


def analyze_rock_type_relation(df: pd.DataFrame) -> None:
    """Анализирует связь категориального признака Rock_Type с целевой переменной."""
    if "Rock_Type" not in df.columns or "Target" not in df.columns:
        print("Для анализа Rock_Type нужны столбцы 'Rock_Type' и 'Target'.")
        return

    tmp = df[["Rock_Type", "Target"]].dropna()

    plt.figure(figsize=(8, 4))
    order = tmp.groupby("Rock_Type")["Target"].mean().sort_values().index
    sns.boxplot(data=tmp, x="Rock_Type", y="Target", order=order)
    plt.xticks(rotation=45, ha="right")
    plt.title("Target distribution by Rock_Type (boxplot)")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "rock_type_target_boxplot.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4))
    mean_target = tmp.groupby("Rock_Type")["Target"].mean().reset_index()
    sns.barplot(data=mean_target, x="Rock_Type", y="Target", order=order)
    plt.xticks(rotation=45, ha="right")
    plt.title("Mean Target by Rock_Type")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "rock_type_target_mean_barplot.png", dpi=150)
    plt.close()


def main() -> None:
    ensure_images_dir()
    df = load_data()
    basic_info(df)
    plot_numeric_distributions(df)
    plot_correlations(df)
    analyze_rock_type_relation(df)
    print(f"\nВсе графики сохранены в: {IMAGES_DIR}")


if __name__ == "__main__":
    main()

