from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "boston_housing.csv"

RU_NAMES = {
    "crim": "Преступность",
    "zn": "Крупные участки",
    "indus": "Доля промзон",
    "chas": "Река Чарльз",
    "nox": "Оксиды азота",
    "rm": "Число комнат",
    "age": "Старый жилфонд",
    "dis": "Удалённость",
    "tax": "Налог",
    "ptratio": "Ученики/учитель",
    "b": "Индекс B",
    "lstat": "Низкий статус",
    "medv": "Цена жилья",
}

TARGET = "Цена жилья"
RANDOM_STATE = 42
TEST_SIZE = 0.20
PCA_VARIANCE = 0.85


def load_frame():
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=["rad"])
    return df.rename(columns=RU_NAMES)


def descriptive_table(df):
    numeric_cols = [c for c in df.columns if c != "Река Чарльз"]
    stats = df[numeric_cols].describe().T
    stats["Медиана"] = df[numeric_cols].median()
    stats["Межквартильный размах (IQR)"] = stats["75%"] - stats["25%"]
    stats["Коэффициент асимметрии"] = df[numeric_cols].skew()
    stats = stats.rename(columns={
        "mean": "Среднее",
        "std": "Станд. откл.",
        "min": "Минимум",
        "max": "Максимум",
    })
    cols = [
        "Среднее", "Станд. откл.", "Медиана", "Межквартильный размах (IQR)",
        "Минимум", "Максимум", "Коэффициент асимметрии",
    ]
    return stats[cols]


def load_prepared():
    """Та же подготовка, что в ноутбуке: 80/20, стандартизация только по train, PCA на 85% дисперсии."""
    df = load_frame()
    feature_names = [c for c in df.columns if c != TARGET]
    X = df[feature_names]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=feature_names, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=feature_names, index=X_test.index
    )

    vif = pd.DataFrame({
        "Признак": feature_names,
        "VIF": [
            variance_inflation_factor(X_train_scaled.values, i)
            for i in range(len(feature_names))
        ],
    }).sort_values("VIF", ascending=False).reset_index(drop=True)

    pca_full = PCA().fit(X_train_scaled)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    k_opt = int(np.searchsorted(cum_var, PCA_VARIANCE) + 1)
    pca = PCA(n_components=k_opt).fit(X_train_scaled)
    loadings = pd.DataFrame(
        pca.components_.T,
        index=feature_names,
        columns=[f"ГК {i + 1}" for i in range(k_opt)],
    )
    X_train_pca = pca.transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    return {
        "df": df,
        "feature_names": feature_names,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "vif": vif,
        "pca_full": pca_full,
        "pca": pca,
        "k_opt": k_opt,
        "loadings": loadings,
        "X_train_pca": X_train_pca,
        "X_test_pca": X_test_pca,
    }


def main():
    prepared = load_prepared()
    df = prepared["df"]
    feature_names = prepared["feature_names"]

    print("=" * 80)
    print(f"Размерность: {df.shape[0]} районов, {df.shape[1]} столбцов")
    print(f"Пропуски: {int(df.isnull().sum().sum())}, дубликаты: {int(df.duplicated().sum())}")
    print(f"Районов с ценой 50 тыс. $: {int((df[TARGET] == 50).sum())}")
    print("=" * 80)
    print("\nОПИСАТЕЛЬНАЯ СТАТИСТИКА")
    print(descriptive_table(df).round(2).to_string())

    corr = df.corr(numeric_only=True)
    pairs = []
    for i, left in enumerate(feature_names):
        for right in feature_names[i + 1:]:
            pairs.append((abs(corr.loc[left, right]), left, right, corr.loc[left, right]))
    pairs.sort(reverse=True)
    print("\nСамые сильные связи между признаками:")
    for _, left, right, value in pairs[:8]:
        print(f"  {left} — {right}: {value:+.3f}")
    print("\nСвязь с ценой жилья:")
    print(corr[TARGET].drop(TARGET).sort_values().round(3).to_string())

    print("\nVIF (стандартизированный train):")
    print(prepared["vif"].round(3).to_string(index=False))

    pca_full = prepared["pca_full"]
    pca_table = pd.DataFrame({
        "Компонента": [f"ГК {i + 1}" for i in range(len(pca_full.explained_variance_))],
        "Собственное значение": pca_full.explained_variance_,
        "Доля дисперсии": pca_full.explained_variance_ratio_,
        "Накопленная доля": np.cumsum(pca_full.explained_variance_ratio_),
    })
    print("\nPCA:")
    print(pca_table.round(4).to_string(index=False))
    print(f"Компонент с lambda >= 1: {int((pca_full.explained_variance_ >= 1).sum())}")
    print(f"Компонент для 85% дисперсии и больше: {prepared['k_opt']}")
    print("\nФакторные нагрузки:")
    print(prepared["loadings"].round(3).to_string())

    vif_pca = [
        variance_inflation_factor(prepared["X_train_pca"], i)
        for i in range(prepared["k_opt"])
    ]
    print("\nVIF главных компонент:", [round(v, 3) for v in vif_pca])
    print(f"\nTrain: {len(prepared['y_train'])}, test: {len(prepared['y_test'])}")


if __name__ == "__main__":
    main()
