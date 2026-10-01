import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LassoCV, LinearRegression, Ridge, RidgeCV
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score

from analysis import RANDOM_STATE, TARGET, load_prepared

ROOT = Path(__file__).resolve().parent
WEIGHTS_CSV = ROOT / "weights.csv"
WEIGHTS_JSON = ROOT / "model_weights.json"


def _metrics(model, x_train, y_train, x_test, y_test, kf):
    cv_scores = cross_val_score(model, x_train, y_train, cv=kf, scoring="r2")
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    return {
        "cv_r2_mean": float(cv_scores.mean()),
        "cv_r2_std": float(cv_scores.std()),
        "r2_test": float(r2_score(y_test, pred)),
        "rmse_test": float(np.sqrt(mean_squared_error(y_test, pred))),
        "mape_test": float(mean_absolute_percentage_error(y_test, pred) * 100),
        "intercept": float(model.intercept_),
        "coefficients": [float(v) for v in model.coef_],
        "prediction": pred,
    }


def train_and_export():
    data = load_prepared()
    feature_names = data["feature_names"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    x_train = data["X_train_scaled"]
    x_test = data["X_test_scaled"]
    k_opt = data["k_opt"]
    pca_names = [f"ГК {i + 1}" for i in range(k_opt)]

    alphas = np.logspace(-3, 3, 25)
    ridge_alpha = float(RidgeCV(alphas=alphas, cv=5).fit(x_train, y_train).alpha_)
    lasso_alpha = float(
        LassoCV(alphas=alphas, cv=5, random_state=RANDOM_STATE, max_iter=20000)
        .fit(x_train, y_train)
        .alpha_
    )
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    specs = [
        ("linear_regression", "Линейная регрессия", LinearRegression()),
        ("ridge_regression", "Гребневая регрессия", Ridge(alpha=ridge_alpha)),
        ("lasso_regression", "Лассо", Lasso(alpha=lasso_alpha, max_iter=20000)),
    ]
    original = {}
    for key, title, model in specs:
        fitted = _metrics(model, x_train, y_train, x_test, y_test, kf)
        fitted.pop("prediction")
        original[key] = {
            "title": title,
            "weights": dict(zip(feature_names, fitted.pop("coefficients"))),
            **fitted,
        }

    pca_models = {}
    pca_specs = [
        ("linear_regression", "Линейная (PCA)", LinearRegression()),
        ("ridge_regression", "Ridge PCA", Ridge(alpha=ridge_alpha)),
        ("lasso_regression", "Lasso PCA", Lasso(alpha=lasso_alpha, max_iter=20000)),
    ]
    for key, title, model in pca_specs:
        fitted = _metrics(
            model, data["X_train_pca"], y_train, data["X_test_pca"], y_test, kf
        )
        fitted.pop("prediction")
        pca_models[key] = {
            "title": title,
            "weights": dict(zip(pca_names, fitted.pop("coefficients"))),
            **fitted,
        }

    weights_csv = pd.DataFrame({
        "Признак": feature_names,
        "Линейная регрессия (OLS)": list(original["linear_regression"]["weights"].values()),
        "Ridge (L2-регуляризация)": list(original["ridge_regression"]["weights"].values()),
        "Lasso (L1-регуляризация)": list(original["lasso_regression"]["weights"].values()),
    })
    weights_csv = pd.concat([
        weights_csv,
        pd.DataFrame([{
            "Признак": "Свободный член",
            "Линейная регрессия (OLS)": original["linear_regression"]["intercept"],
            "Ridge (L2-регуляризация)": original["ridge_regression"]["intercept"],
            "Lasso (L1-регуляризация)": original["lasso_regression"]["intercept"],
        }]),
    ], ignore_index=True)
    weights_csv.to_csv(WEIGHTS_CSV, index=False, encoding="utf-8-sig")

    loadings = {
        feature: {
            col: float(data["loadings"].loc[feature, col])
            for col in data["loadings"].columns
        }
        for feature in feature_names
    }
    payload = {
        "dataset": "data/boston_housing.csv (Boston Housing, Harrison & Rubinfeld, 1978)",
        "source": "https://www.kaggle.com/datasets/arunjangir245/boston-housing-dataset",
        "target": f"{TARGET} (medv), тыс. $",
        "n_rows": int(data["df"].shape[0]),
        "n_features": len(feature_names),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "test_size": 0.2,
        "random_state": RANDOM_STATE,
        "standardization": "StandardScaler, fit только на train",
        "ridge_alpha": ridge_alpha,
        "lasso_alpha": lasso_alpha,
        "vif": {
            row["Признак"]: float(row["VIF"])
            for _, row in data["vif"].iterrows()
        },
        "pca_n_components": int(k_opt),
        "pca_variance_explained": float(
            np.cumsum(data["pca_full"].explained_variance_ratio_)[k_opt - 1]
        ),
        "kaiser_components": int((data["pca_full"].explained_variance_ >= 1).sum()),
        "models_original_features": original,
        "models_pca": pca_models,
        "pca_loadings": loadings,
    }
    WEIGHTS_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Ridge alpha = {ridge_alpha:.4f}, Lasso alpha = {lasso_alpha:.4f}, компонент PCA = {k_opt}")
    print("\nИсходные признаки:")
    for item in original.values():
        print(
            f"  {item['title']}: R2={item['r2_test']:.4f}, "
            f"RMSE={item['rmse_test']:.4f}, MAPE={item['mape_test']:.2f}%"
        )
    print("\nГлавные компоненты:")
    for item in pca_models.values():
        print(
            f"  {item['title']}: R2={item['r2_test']:.4f}, "
            f"RMSE={item['rmse_test']:.4f}, MAPE={item['mape_test']:.2f}%"
        )
    print(f"\nВеса: {WEIGHTS_CSV.name}")
    print(f"Полный разбор: {WEIGHTS_JSON.name}")
    return payload


if __name__ == "__main__":
    train_and_export()
