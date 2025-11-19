# 04_diccionario_modelo.py
# A partir de PASO3_diccionario_presentes.csv
# crea un diccionario filtrado solo con las variables del proyecto/modelo

from pathlib import Path
import pandas as pd

BASE = Path(__file__).parent

DICC_PASO3 = BASE / "PASO3_diccionario_presentes.csv"
SALIDA_DICC_MODELO = BASE / "PASO4_diccionario_modelo.csv"

# ================== 1) LISTA DE VARIABLES DEL PROYECTO ==================
# (Esta lista está basada exactamente en el archivo que me mostraste)
VARIABLES_MODELO = [
    # --------- Antropometría / TA ----------
    "FOLIO_I",
    "an01_1",
    "an01_2",
    "an03",
    "an04_1",
    "an04_2",
    "an05",
    "an08_1",
    "an08_2",
    "an09",
    "an10",
    "an11",
    "entidad",
    "imc",
    "meses",
    "municipio",
    "reporte_imc",
    "x_region",

    # --------- Salud adultos ----------
    "a0107",
    "a0108",
    "a0109",
    "a0301",
    "a0301a",
    "a0302",
    "a0303num",
    "a0304a",
    "a0304d",
    "a0304m",
    "a0305",
    "a0305esp",
    "a0306a",
    "a0306b",
    "a0306c",
    "a0306e",
    "a0306f",
    "a0307",
    "a0308a",
    "a0308m",
    "a0309a",
    "a0309m",
    "a0310",
    "a0310a",
    "a0313",
    "a0313a",
    "a0314",
    "a0316a",
    "a0316b",
    "a0316d",
]

# Si más adelante quieres agregar edad/sexo, solo agrégalas aquí, por ejemplo:
# VARIABLES_MODELO += ["edad", "sexo"]   # si esas son las columnas correctas


# ================== 2) CARGAR DICCIONARIO COMPLETO ==================
print(f"[INFO] Cargando diccionario de variables presentes desde {DICC_PASO3.name}...")
dicc_presentes = pd.read_csv(DICC_PASO3, encoding="utf-8")

print(f"[INFO] Diccionario completo: {dicc_presentes.shape[0]} filas")

# ================== 3) FILTRAR SOLO VARIABLES DEL MODELO ==================
set_vars = set(VARIABLES_MODELO)
dicc_modelo = dicc_presentes[dicc_presentes["Variable"].isin(set_vars)].copy()

# Opcional: ordenar por módulo y variable
dicc_modelo = dicc_modelo.sort_values(["Módulo", "Variable"])

print(f"[INFO] Variables definidas en VARIABLES_MODELO: {len(VARIABLES_MODELO)}")
print(f"[INFO] Variables encontradas en el diccionario: {dicc_modelo.shape[0]}")

# ================== 4) GUARDAR ==================
dicc_modelo.to_csv(SALIDA_DICC_MODELO, index=False, encoding="utf-8")
print(f"[DONE] Diccionario del modelo guardado en: {SALIDA_DICC_MODELO}")
