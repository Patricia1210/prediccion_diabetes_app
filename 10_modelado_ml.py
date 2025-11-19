# ================================================================
# 10_modelado_ml.py — PASO 10: Entrenamiento de modelos supervisados
# ================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

RUTA_PASO9 = "PASO9_dataset_ml_modelmatrix.csv"
SALIDA_METRICAS = "PASO10_metricas_modelos.csv"


# ================================================================
# Función: cargar dataset
# ================================================================
def cargar_dataset():
    print("[INFO] Leyendo matriz de modelado (PASO 9)...")
    df = pd.read_csv(RUTA_PASO9)

    print(f"[INFO] Dataset PASO 9: {df.shape[0]} filas x {df.shape[1]} columnas\n")

    return df


# ================================================================
# Función: preparar X y y
# ================================================================
def preparar_xy(df):

    target = "diabetes_dx"

    print("[INFO] Distribución de la variable objetivo (diabetes_dx):")
    print(df[target].value_counts(), "\n")

    X = df.drop(columns=[target])
    y = df[target]

    return X, y


# ================================================================
# Función: imputación segura de NaN
# ================================================================
def imputar_nans(X):
    """
    Reemplaza:
      - NaN numéricos → 0
      - NaN categóricos (0/1 dummies) → 0
    """
    antes = X.isna().sum().sum()
    print(f"[INFO] NaNs totales en X ANTES de corregir: {antes}")

    X = X.fillna(0)

    despues = X.isna().sum().sum()
    print(f"[INFO] NaNs totales en X DESPUÉS de corregir: {despues}\n")

    return X


# ================================================================
# Función: evaluar modelos
# ================================================================
def evaluar_modelo(nombre, modelo, X_train, y_train, X_test, y_test):

    print(f"[INFO] Entrenando modelo: {nombre}")

    # --- Cross Validation ---
    try:
        cv_scores = cross_val_score(modelo, X_train, y_train, cv=5, scoring="roc_auc")
        cv_mean = cv_scores.mean()
    except Exception as e:
        print(f"[ERROR] Falló el CV para {nombre}: {e}")
        cv_mean = np.nan

    # --- Fit del modelo ---
    modelo.fit(X_train, y_train)

    # --- Predicciones ---
    y_pred = modelo.predict(X_test)
    y_prob = modelo.predict_proba(X_test)[:, 1] if hasattr(modelo, "predict_proba") else None

    results = {
        "modelo": nombre,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan,
        "cv_auc": cv_mean
    }

    return results, modelo


# ================================================================
# Función principal
# ================================================================
def main():

    df = cargar_dataset()

    # ----------------------
    # Preparar X y y
    # ----------------------
    X, y = preparar_xy(df)

    # ----------------------
    # Imputar NaN
    # ----------------------
    X = imputar_nans(X)

    # ----------------------
    # Train/Test split
    # ----------------------
    print("[INFO] Realizando partición train/test...\n")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    print(f"[INFO] Partición completada. Train: {X_train.shape[0]} filas, Test: {X_test.shape[0]} filas.\n")

    # ================================================================
    # Modelos a evaluar
    # ================================================================
    modelos = {
        "logistic_regression_l1": Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("model", LogisticRegression(
                penalty="l1",
                solver="liblinear",
                class_weight="balanced"
            ))
        ]),

        "logistic_regression_l2": Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("model", LogisticRegression(
                penalty="l2",
                solver="liblinear",
                class_weight="balanced"
            ))
        ]),

        "random_forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=10,
            min_samples_split=5,
            class_weight="balanced",
            random_state=42
        ),

        "gradient_boosting": GradientBoostingClassifier()
    }

    # ================================================================
    # Entrenar y evaluar
    # ================================================================
    resultados = []
    modelos_entrenados = {}

    for nombre, modelo in modelos.items():
        res, modelo_fit = evaluar_modelo(nombre, modelo, X_train, y_train, X_test, y_test)
        resultados.append(res)
        modelos_entrenados[nombre] = modelo_fit

    # Convertir a DF
    resultados_df = pd.DataFrame(resultados)

    # Ordenar por métrica prioritaria
    resultados_df = resultados_df.sort_values(
        by="roc_auc",
        ascending=False
    )

    resultados_df.to_csv(SALIDA_METRICAS, index=False, encoding="utf-8")

    print("\n=== RESULTADOS FINALES ===")
    print(resultados_df)

    print(f"\n[OK] Archivo guardado en: {SALIDA_METRICAS}")


if __name__ == "__main__":
    main()
