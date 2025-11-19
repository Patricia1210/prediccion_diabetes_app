# 07_dataset_modelo_ml.py
# A partir de PASO6_dataset_modelo_target.csv:
# - Filtra sólo filas con target válido (0/1)
# - Asegura tipo entero en diabetes_dx
# - Guarda PASO7_dataset_modelo_ml.csv listo para modelado

from pathlib import Path
import pandas as pd

# ================== 0) RUTAS ==================
BASE = Path(__file__).parent

DATASET_PASO6 = BASE / "PASO6_dataset_modelo_target.csv"
SALIDA_PASO7  = BASE / "PASO7_dataset_modelo_ml.csv"

TARGET_COL = "diabetes_dx"


# ================== 1) LEER DATASET PASO 6 ==================
# Detectar separador coma vs punto y coma
with open(DATASET_PASO6, "r", encoding="latin-1") as f:
    head = f.readline()
sep = ";" if head.count(";") > head.count(",") else ","

print(f"[INFO] Leyendo dataset PASO 6 con separador '{sep}'...")
df = pd.read_csv(DATASET_PASO6, sep=sep, low_memory=False, encoding="latin-1")
print(f"[INFO] Dataset PASO 6: {df.shape[0]} filas x {df.shape[1]} columnas\n")

if TARGET_COL not in df.columns:
    raise ValueError(f"No se encontró la columna de target '{TARGET_COL}' en el dataset PASO 6.")


# ================== 2) REVISAR DISTRIBUCIÓN ORIGINAL DEL TARGET ==================
print("[INFO] Distribución original de 'diabetes_dx' (incluyendo NaN):")
print(df[TARGET_COL].value_counts(dropna=False))
print()

print("[INFO] Proporciones originales de 'diabetes_dx' (incluyendo NaN):")
print(df[TARGET_COL].value_counts(dropna=False, normalize=True))
print()


# ================== 3) FILTRAR SÓLO TARGET VÁLIDO (0 ó 1) ==================
mask_valid = df[TARGET_COL].isin([0, 1])
df_valid = df[mask_valid].copy()

n_original = df.shape[0]
n_valid = df_valid.shape[0]
n_eliminadas = n_original - n_valid

print(f"[INFO] Filas con target válido (0/1): {n_valid}")
print(f"[INFO] Filas eliminadas por target NaN u otro valor: {n_eliminadas}\n")

# Aseguramos tipo entero para el target
df_valid[TARGET_COL] = df_valid[TARGET_COL].astype(int)

# Resumen de la distribución ya filtrada
print("[INFO] Distribución de 'diabetes_dx' después del filtrado (0/1):")
print(df_valid[TARGET_COL].value_counts())
print()

print("[INFO] Proporciones de 'diabetes_dx' después del filtrado (0/1):")
print(df_valid[TARGET_COL].value_counts(normalize=True))
print()

# Vista rápida
print("[INFO] Vista rápida de las primeras filas del dataset listo para modelado:")
print(df_valid.head())
print(f"\n[INFO] Dataset final de modelado: {df_valid.shape[0]} filas x {df_valid.shape[1]} columnas\n")


# ================== 4) GUARDAR SALIDA ==================
df_valid.reset_index(drop=True, inplace=True)
df_valid.to_csv(SALIDA_PASO7, index=False, encoding="utf-8")

print(f"[DONE] Dataset final para modelado guardado en: {SALIDA_PASO7}")
