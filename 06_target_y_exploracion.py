# 06_target_y_exploracion.py
# Crea la variable objetivo de diabetes y hace una exploración rápida del dataset del modelo

from pathlib import Path
import pandas as pd

# ================== 0) RUTAS ==================
BASE = Path(__file__).parent

DATASET_PASO5 = BASE / "PASO5_dataset_modelo.csv"
SALIDA_PASO6   = BASE / "PASO6_dataset_modelo_target.csv"

# ================== 1) LEER DATASET DEL MODELO (PASO 5) ==================
# Detectar separador coma vs punto y coma (por si acaso)
with open(DATASET_PASO5, "r", encoding="latin-1") as f:
    head = f.readline()
sep = ";" if head.count(";") > head.count(",") else ","

print(f"[INFO] Leyendo dataset del modelo (PASO 5) con separador '{sep}'...")
df = pd.read_csv(DATASET_PASO5, sep=sep, low_memory=False, encoding="latin-1")
print(f"[INFO] Dataset PASO5: {df.shape[0]} filas x {df.shape[1]} columnas")

# ================== 2) CREAR VARIABLE OBJETIVO: diabetes_dx ==================
# a0301: “¿Algún médico le ha dicho que tiene diabetes (o alta el azúcar en la sangre)?”

if "a0301" not in df.columns:
    raise ValueError("La columna 'a0301' no está en el dataset. Revisa PASO5_dataset_modelo.csv")

# Mapeo simple:
#  1 = Sí  → 1 (tiene diabetes)
#  2 = No  → 0 (no tiene diabetes)
# Otros códigos (8 = No sabe, 9 = No responde, etc.) se dejan como NaN
map_diabetes = {1: 1, 2: 0}

df["diabetes_dx"] = df["a0301"].map(map_diabetes)

print("\n[INFO] Distribución de 'a0301' (código original):")
print(df["a0301"].value_counts(dropna=False))

print("\n[INFO] Distribución de 'diabetes_dx' (target binario):")
print(df["diabetes_dx"].value_counts(dropna=False))
print("\n[INFO] Proporciones de 'diabetes_dx':")
print(df["diabetes_dx"].value_counts(normalize=True, dropna=False))

# ================== 3) EXPLORACIÓN RÁPIDA ADICIONAL (OPCIONAL) ==================
print("\n[INFO] Vista rápida de las primeras filas del dataset con el target:")
print(df.head())

print("\n[INFO] Resumen general del dataset:")
print(df.describe(include="all").transpose().head(15))  # solo para no saturar la consola

# ================== 4) GUARDAR ==================
df.to_csv(SALIDA_PASO6, index=False, encoding="utf-8")
print(f"\n[DONE] Dataset con variable objetivo guardado en: {SALIDA_PASO6}")
