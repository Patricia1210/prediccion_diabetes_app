# 11_detectar_columnas_sospechosas.py
# Busca columnas que casi “adivinan” el target (posible fuga de información)

from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(__file__).parent
DATA_PASO8 = BASE / "PASO8_dataset_ml_features.csv"      # features “legibles”
DATA_PASO9 = BASE / "PASO9_dataset_ml_modelmatrix.csv"   # matriz one-hot
TARGET_COL = "diabetes_dx"


def analizar_corr_numerica(df):
    """Correlación absoluta entre cada numérica y el target."""
    print("\n=== CORRELACIONES NUMÉRICAS ALTAS CON EL TARGET (|r| >= 0.95) ===")
    num_cols = [
        c for c in df.columns
        if c not in ["FOLIO_I", TARGET_COL] and np.issubdtype(df[c].dtype, np.number)
    ]

    corrs = []
    for c in num_cols:
        if df[c].nunique() <= 1:
            continue
        r = df[[c, TARGET_COL]].corr().iloc[0, 1]
        corrs.append((c, r))

    corrs = sorted(corrs, key=lambda x: abs(x[1]), reverse=True)
    for c, r in corrs:
        if abs(r) >= 0.95:
            print(f"{c:30s}  r = {r: .4f}")


def analizar_binarias_perfectas(df_features, y, origen):
    """
    En la matriz one-hot (PASO9) revisa columnas binarias donde
    alguna combinación es perfecta: p( diabetes_dx=1 | X=1 ) = 1 o 0.
    """
    print(f"\n=== COLUMNAS BINARIAS SOSPECHOSAS EN {origen} ===")
    sospechosas = []

    for col in df_features.columns:
        if col in ["FOLIO_I", TARGET_COL]:
            continue

        serie = df_features[col]
        # Sólo miramos columnas 0/1
        vals = set(serie.dropna().unique())
        if not vals.issubset({0, 1}):
            continue

        ct = pd.crosstab(serie, y)
        if ct.shape[0] < 2:  # sólo 0 o sólo 1
            continue

        # proporción de positivos cuando X=1
        if 1 in ct.index:
            total_1 = ct.loc[1].sum()
            if total_1 > 0:
                prop_pos = ct.loc[1, 1] / total_1 if 1 in ct.columns else 0.0
                if prop_pos == 0.0 or prop_pos == 1.0:
                    sospechosas.append((col, prop_pos, int(total_1)))

    sospechosas.sort(key=lambda x: abs(x[1] - 0.5), reverse=True)

    if not sospechosas:
        print("No se encontraron columnas binarias perfectamente separadoras.")
    else:
        print("Columna\t\tprop_pos(X=1)\tN con X=1")
        for col, prop, n1 in sospechosas:
            print(f"{col}\t{prop:.3f}\t\t{n1}")


def main():
    # ---------- 1) Analizar PASO 8 (features “humanos”) ----------
    print("[INFO] Leyendo PASO 8...")
    df8 = pd.read_csv(DATA_PASO8, sep=",", low_memory=False, encoding="latin-1")
    print(f"[INFO] PASO 8: {df8.shape[0]} filas x {df8.shape[1]} columnas")

    if TARGET_COL not in df8.columns:
        raise ValueError(f"No se encontró la columna {TARGET_COL} en PASO8.")

    analizar_corr_numerica(df8)

    # Ejemplo: revisar a0301, a0307 si existen
    print("\n=== EJEMPLO DE CRUCES PARA VARIABLES CLAVE (SI EXISTEN) ===")
    for col in ["a0301", "a0307", "a0310", "imc_calc"]:
        if col in df8.columns:
            print(f"\n[INFO] Cruce {col} vs {TARGET_COL}:")
            print(pd.crosstab(df8[col], df8[TARGET_COL], normalize="index"))

    # ---------- 2) Analizar PASO 9 (matriz one-hot) ----------
    print("\n[INFO] Leyendo PASO 9 (matriz de modelado)...")
    df9 = pd.read_csv(DATA_PASO9, sep=",", low_memory=False, encoding="latin-1")
    print(f"[INFO] PASO 9: {df9.shape[0]} filas x {df9.shape[1]} columnas")

    if TARGET_COL not in df9.columns:
        raise ValueError(f"No se encontró la columna {TARGET_COL} en PASO9.")

    y = df9[TARGET_COL]
    X = df9.drop(columns=[TARGET_COL])

    analizar_binarias_perfectas(X, y, origen="PASO9")


if __name__ == "__main__":
    main()
