import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold, cross_val_score, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor


RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parent


def load_preprocessed_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Загружает предобработанные train/test множества из GPN2."""
    X_train = joblib.load(BASE_DIR / "X_train.pkl")
    X_test = joblib.load(BASE_DIR / "X_test.pkl")
    y_train = joblib.load(BASE_DIR / "y_train.pkl")
    y_test = joblib.load(BASE_DIR / "y_test.pkl")
    return X_train, X_test, y_train, y_test


def get_base_models() -> Dict[str, object]:
    """Базовые модели для первичного сравнения."""
    return {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            random_state=RANDOM_STATE,
            n_estimators=200,
            max_depth=None,
        ),
        "XGBRegressor": XGBRegressor(
            random_state=RANDOM_STATE,
            objective="reg:squarederror",
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=1,  # чтобы избежать проблем с параллелизмом
        ),
        "MLPRegressor": MLPRegressor(
            random_state=RANDOM_STATE,
            hidden_layer_sizes=(64, 32),
            max_iter=500,
        ),
    }


def evaluate_with_cv(models, X_train, y_train) -> Dict[str, Dict[str, float]]:
    """Оценивает модели по 5‑fold CV и возвращает метрики."""
    results = {}
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for name, model in models.items():
        rmse_scores = cross_val_score(
            model,
            X_train,
            y_train,
            scoring="neg_root_mean_squared_error",
            cv=cv,
            n_jobs=1,  # без многопроцессорности, чтобы не падать
        )
        r2_scores = cross_val_score(
            model,
            X_train,
            y_train,
            scoring="r2",
            cv=cv,
            n_jobs=1,
        )
        results[name] = {
            "rmse_mean": -rmse_scores.mean(),
            "rmse_std": rmse_scores.std(),
            "r2_mean": r2_scores.mean(),
            "r2_std": r2_scores.std(),
        }
    return results


def get_param_distributions(model_name: str) -> Dict:
    """Небольшие пространства гиперпараметров для RandomizedSearchCV."""
    if model_name == "RandomForest":
        return {
            "n_estimators": [100, 200, 300, 400],
            "max_depth": [None, 4, 6, 8, 10],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }
    if model_name == "XGBRegressor":
        return {
            "n_estimators": [200, 300, 400],
            "max_depth": [3, 4, 5, 6],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.7, 0.8, 1.0],
            "colsample_bytree": [0.7, 0.8, 1.0],
        }
    if model_name == "MLPRegressor":
        return {
            "hidden_layer_sizes": [(64,), (64, 32), (128, 64)],
            "alpha": [0.0001, 0.001, 0.01],
        }
    if model_name == "LinearRegression":
        return {}
    return {}


def tune_model(model_name: str, base_model, X_train, y_train) -> object:
    """Подбор гиперпараметров для выбранной модели (если есть что подбирать)."""
    param_dist = get_param_distributions(model_name)
    if not param_dist:
        print(f"{model_name}: без подбора гиперпараметров.")
        base_model.fit(X_train, y_train)
        return base_model

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=10,
        scoring="neg_root_mean_squared_error",
        cv=3,
        random_state=RANDOM_STATE,
        n_jobs=1,  # тоже без параллелизма
        verbose=1,
    )
    search.fit(X_train, y_train)
    print(f"{model_name}: лучшие параметры: {search.best_params_}")
    return search.best_estimator_


def evaluate_on_test(model, X_test, y_test) -> Dict[str, float]:
    """Считает RMSE, MAE и R² на тестовой выборке."""
    y_pred = model.predict(X_test)
    # Без параметра squared для совместимости с разными версиями sklearn
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return {"rmse": rmse, "mae": mae, "r2": r2}


def save_best_model_and_metrics(
    best_model_name: str,
    best_model,
    cv_results: Dict[str, Dict[str, float]],
    test_metrics: Dict[str, Dict[str, float]],
) -> None:
    """Сохраняет лучшую модель и метрики на диск."""
    joblib.dump(best_model, BASE_DIR / "best_model.pkl")

    summary = {
        "best_model_name": best_model_name,
        "cv_results": cv_results,
        "test_metrics": test_metrics,
    }

    with open(BASE_DIR / "model_performance.json", "w", encoding="utf-8") as f_json:
        json.dump(summary, f_json, ensure_ascii=False, indent=2)

    lines = []
    lines.append(f"Best model: {best_model_name}")
    lines.append("\n=== CV results (5-fold, train) ===")
    for name, res in cv_results.items():
        lines.append(
            f"{name}: RMSE={res['rmse_mean']:.4f}±{res['rmse_std']:.4f}, "
            f"R2={res['r2_mean']:.4f}±{res['r2_std']:.4f}"
        )

    lines.append("\n=== Test metrics ===")
    for name, m in test_metrics.items():
        lines.append(
            f"{name}: RMSE={m['rmse']:.4f}, MAE={m['mae']:.4f}, R2={m['r2']:.4f}"
        )

    with open(BASE_DIR / "model_performance.txt", "w", encoding="utf-8") as f_txt:
        f_txt.write("\n".join(lines))

    print("Сохранены best_model.pkl, model_performance.json и model_performance.txt")


def main() -> None:
    X_train, X_test, y_train, y_test = load_preprocessed_data()

    base_models = get_base_models()
    cv_results = evaluate_with_cv(base_models, X_train, y_train)

    print("\n=== Результаты CV ===")
    for name, res in cv_results.items():
        print(
            f"{name}: RMSE={res['rmse_mean']:.4f}±{res['rmse_std']:.4f}, "
            f"R2={res['r2_mean']:.4f}±{res['r2_std']:.4f}"
        )

    sorted_models = sorted(cv_results.items(), key=lambda x: x[1]["rmse_mean"])
    top2_names = [sorted_models[0][0], sorted_models[1][0]]
    print(f"\nТоп-2 модели для подбора гиперпараметров: {top2_names}")

    tuned_models = {}
    for name in top2_names:
        tuned_models[name] = tune_model(name, base_models[name], X_train, y_train)

    test_metrics = {}
    for name, model in tuned_models.items():
        test_metrics[name] = evaluate_on_test(model, X_test, y_test)
        print(
            f"\n{name} на тесте: "
            f"RMSE={test_metrics[name]['rmse']:.4f}, "
            f"MAE={test_metrics[name]['mae']:.4f}, "
            f"R2={test_metrics[name]['r2']:.4f}"
        )

    best_name = min(test_metrics.items(), key=lambda x: x[1]["rmse"])[0]
    best_model = tuned_models[best_name]

    save_best_model_and_metrics(best_name, best_model, cv_results, test_metrics)


if __name__ == "__main__":
    main()

