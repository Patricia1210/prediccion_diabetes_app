# 05_dataset_modelo.py
# Construye el DATASET FINAL DEL MODELO a partir de:
# - ENSANUT_2023_Adultos_Salud+Antro_merged.csv
# - PASO4_diccionario_modelo.csv
#
# Resultado: PASO5_dataset_modelo.csv
# (solo variables seleccionadas para el modelo, con filtros básicos)

from pathlib import Path
import pandas as pd

# ================== 0) RUTAS ==================
BASE = Path(__file__).parent

MERGED_CSV    = BASE / "ENSANUT_2023_Adultos_Salud+Antro_merged.csv"
DICC_MODELO   = BASE / "PASO4_diccionario_modelo.csv"
SALIDA_DATASET = BASE / "PASO5_dataset_modelo.csv"

# Opcional: nombre de la variable objetivo (ajústalo si decides otra)
TARGET_VAR = "a0301"     # "¿Algún médico le ha dicho que tiene diabetes...?"
ID_VAR     = "FOLIO_I"   # identificador del individuo (está en ambos módulos)


# ================== 1) DETECTAR SEPARADOR Y CARGAR BASE ==================
with open(MERGED_CSV, "r", encoding="latin-1") as f:
    head = f.readline()

sep = ";" if head.count(";") > head.count(",") else ","
print(f"[INFO] Leyendo base fusionada con separador '{sep}'...")

df = pd.read_csv(MERGED_CSV, sep=sep, low_memory=False, encoding="latin-1")
print(f"[INFO] Base fusionada: {df.shape[0]} filas x {df.shape[1]} columnas")


# ================== 2) CARGAR DICCIONARIO DEL MODELO ==================
dicc = pd.read_csv(DICC_MODELO, encoding="utf-8")
print(f"[INFO] Diccionario del modelo: {dicc.shape[0]} variables (filas)")

# Lista de variables del modelo
vars_modelo = dicc["Variable"].astype(str).str.strip().tolist()

# Nos aseguramos de que solo tomamos las que realmente existen en la base
cols_presentes = [v for v in vars_modelo if v in df.columns]

faltantes = sorted(set(vars_modelo) - set(cols_presentes))
if faltantes:
    print("[WARN] Hay variables definidas en el diccionario que NO están en la base fusionada:")
    for v in faltantes:
        print("   -", v)

print(f"[INFO] Variables del modelo presentes en la base: {len(cols_presentes)}")

if not cols_presentes:
    raise ValueError("Ninguna de las variables del modelo está presente en la base fusionada.")


# ================== 3) SUBSETEAR SOLO LAS VARIABLES DEL MODELO ==================
df_modelo = df[cols_presentes].copy()
print(f"[INFO] Dataset filtrado a variables del modelo: {df_modelo.shape[0]} filas x {df_modelo.shape[1]} columnas")


# ================== 4) LIMPIEZA BÁSICA Y FILTROS ==================
# 4.1) Eliminar filas completamente vacías en las variables del modelo
df_modelo = df_modelo.dropna(how="all")
print(f"[INFO] Tras eliminar filas completamente vacías: {df_modelo.shape[0]} filas")

# 4.2) Filtrar por disponibilidad de la variable objetivo (si existe)
if TARGET_VAR in df_modelo.columns:
    antes = df_modelo.shape[0]
    df_modelo = df_modelo[df_modelo[TARGET_VAR].notna()].copy()
    despues = df_modelo.shape[0]
    print(f"[INFO] Filas con {TARGET_VAR} no nulo: {despues} (se eliminaron {antes - despues})")
else:
    print(f"[WARN] La variable objetivo '{TARGET_VAR}' no está en el dataset. No se aplicó filtro por objetivo.")

# 4.3) Eliminar duplicados por ID (si la variable existe)
if ID_VAR in df_modelo.columns:
    antes = df_modelo.shape[0]
    df_modelo = df_modelo.drop_duplicates(subset=[ID_VAR])
    despues = df_modelo.shape[0]
    print(f"[INFO] Tras eliminar duplicados por {ID_VAR}: {despues} filas (se eliminaron {antes - despues})")
else:
    print(f"[WARN] La variable ID '{ID_VAR}' no está en el dataset. No se revisaron duplicados por ID.")


# ================== 5) GUARDAR ==================
df_modelo.to_csv(SALIDA_DATASET, index=False, encoding="utf-8")
print(f"[DONE] Dataset final del modelo guardado en: {SALIDA_DATASET}")
