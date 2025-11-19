# 09_preprocesamiento_ml.py
# Preprocesa el dataset con ingeniería de características (PASO 8)
# para dejarlo listo como matriz de modelado:
# - Separa ID y variable target
# - Detecta columnas numéricas y categóricas
# - Imputa faltantes (mediana / categoría 'Desconocido')
# - Codifica variables categóricas con one-hot encoding
# - Genera un CSV final listo para modelos

from pathlib import Path
import pandas as pd
import numpy as np

# ================== 0) RUTAS Y CONSTANTES ==================
BASE = Path(__file__).parent

DATASET_PASO8 = BASE / "PASO8_dataset_ml_features.csv"
SALIDA_PASO9   = BASE / "PASO9_dataset_ml_modelmatrix.csv"

ID_COL      = "FOLIO_I"
TARGET_COL  = "diabetes_dx"


def detectar_separador(path_csv: Path, encoding: str = "latin-1") -> str:
    """
    Detecta si el CSV está separado por coma o punto y coma
    leyendo sólo la primera línea.
    """
    with open(path_csv, "r", encoding=encoding) as f:
        head = f.readline()
    return ";" if head.count(";") > head.count(",") else ","


def main():
    # =============== 1) CARGAR DATASET PASO 8 ===============
    sep = detectar_separador(DATASET_PASO8)
    print(f"[INFO] Leyendo dataset PASO 8 con separador '{sep}'...")

    df = pd.read_csv(DATASET_PASO8, sep=sep, low_memory=False, encoding="latin-1")
    print(f"[INFO] Dataset PASO 8: {df.shape[0]} filas x {df.shape[1]} columnas")

    # Comprobamos que existan ID y target
    if ID_COL not in df.columns:
        raise ValueError(f"No se encontró la columna ID '{ID_COL}' en el dataset PASO 8.")
    if TARGET_COL not in df.columns:
        raise ValueError(f"No se encontró la columna target '{TARGET_COL}' en el dataset PASO 8.")

    print("\n[INFO] Distribución de la variable objetivo (diabetes_dx):")
    print(df[TARGET_COL].value_counts(dropna=False))

    # =============== 2) SEPARAR ID, TARGET Y FEATURES ===============
    id_series     = df[ID_COL].copy()
    target_series = df[TARGET_COL].copy()

    feature_cols = [c for c in df.columns if c not in [ID_COL, TARGET_COL]]
    X = df[feature_cols].copy()

    print(f"\n[INFO] Número de columnas de características (sin ID ni target): {len(feature_cols)}")

    # =============== 3) DETECTAR TIPOS (NUMÉRICAS / CATEGÓRICAS) ===============
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    # Todo lo demás lo tratamos como categórico
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    print(f"[INFO] Columnas numéricas detectadas: {len(numeric_cols)}")
    print(f"[INFO] Columnas categóricas detectadas: {len(categorical_cols)}")

    # =============== 4) IMPUTACIÓN DE FALTANTES ===============
    # --- Numéricas: mediana ---
    if numeric_cols:
        X_num = X[numeric_cols].astype(float)
        medians = X_num.median()
        X_num = X_num.fillna(medians)
    else:
        X_num = pd.DataFrame(index=X.index)
        medians = pd.Series(dtype=float)

    # --- Categóricas: categoría 'Desconocido' ---
    if categorical_cols:
        X_cat = X[categorical_cols].astype("string")
        X_cat = X_cat.fillna("Desconocido")
    else:
        X_cat = pd.DataFrame(index=X.index)

    print("\n[INFO] Imputación terminada.")
    if len(medians) > 0:
        print("[INFO] Ejemplo de medianas usadas para numéricas:")
        print(medians.head())

    # =============== 5) ONE-HOT ENCODING PARA CATEGÓRICAS ===============
    if not X_cat.empty:
        X_cat_dummies = pd.get_dummies(
            X_cat,
            prefix=categorical_cols,
            prefix_sep="__",
            drop_first=True,  # para evitar colinealidad perfecta
            dtype=int,
        )
        print(f"[INFO] One-hot encoding: {X_cat_dummies.shape[1]} columnas dummy generadas.")
    else:
        X_cat_dummies = pd.DataFrame(index=X.index)

    # =============== 6) UNIR TODO EN UNA MATRIZ FINAL ===============
    X_processed = pd.concat([X_num, X_cat_dummies], axis=1)

    final_df = pd.concat(
        [id_series.reset_index(drop=True),
         target_series.reset_index(drop=True),
         X_processed.reset_index(drop=True)],
        axis=1
    )

    print(f"\n[INFO] Dimensiones finales de la matriz de modelado:")
    print(f"[INFO] {final_df.shape[0]} filas x {final_df.shape[1]} columnas")
    print(f"[INFO] Columnas totales de características (excluyendo ID y target): {X_processed.shape[1]}")

    # =============== 7) GUARDAR =================
    final_df.to_csv(SALIDA_PASO9, index=False, encoding="utf-8")
    print(f"\n[DONE] Dataset de modelado (matriz final) guardado en:\n       {SALIDA_PASO9}")


if __name__ == "__main__":
    main()
