# 03_diccionario_ensanut.py
# Construye un diccionario de variables solo con las columnas
# que están presentes en ENSANUT_2023_Adultos_Salud+Antro_merged.csv

from pathlib import Path
import pandas as pd

# ================== 0) RUTAS ==================
BASE = Path(__file__).parent

MERGED_CSV = BASE / "ENSANUT_2023_Adultos_Salud+Antro_merged.csv"
SALUD_CAT  = BASE / "adultos_ensanut2023_w_n.Catlogo.xlsx"
ANTRO_CAT  = BASE / "Antropometria_HTA_4mar24.Catlogo.xlsx"

SALIDA_DICC = BASE / "PASO3_diccionario_presentes.csv"


# ================== 1) FUNCIÓN PARA LEER CATÁLOGOS ==================
def extraer_catalogo_robusto(path_xlsx: Path):
    """
    Busca en todas las hojas del catálogo la fila donde aparecen las
    columnas 'Variable' y 'Etiqueta/Descripción/Label' y devuelve
    un DataFrame con esas dos columnas estandarizadas.
    """
    xls = pd.ExcelFile(path_xlsx)

    for sheet in xls.sheet_names:
        # Leemos sin encabezado para poder detectar en qué fila están los nombres
        tmp = pd.read_excel(path_xlsx, sheet_name=sheet, header=None)

        header_row = None
        # Revisamos las primeras 40 filas buscando 'Variable' + 'Etiqueta'
        for i in range(min(40, len(tmp))):
            row_vals = (
                tmp.iloc[i]
                .astype(str)
                .str.strip()
                .str.lower()
                .tolist()
            )
            if "variable" in row_vals and any(
                x in row_vals for x in ["etiqueta", "descripción", "descripcion", "label"]
            ):
                header_row = i
                break

        if header_row is not None:
            # Releemos la hoja usando esa fila como encabezado
            df = pd.read_excel(path_xlsx, sheet_name=sheet, header=header_row)

            cols_lower = [str(c).strip().lower() for c in df.columns]

            # Columna de variable
            try:
                var_idx = cols_lower.index("variable")
                var_col = df.columns[var_idx]
            except ValueError:
                continue

            # Columna de etiqueta / descripción
            etq_col = None
            for cand in ["etiqueta", "descripción", "descripcion", "label"]:
                if cand in cols_lower:
                    etq_col = df.columns[cols_lower.index(cand)]
                    break

            if etq_col is None:
                continue

            clean = df[[var_col, etq_col]].dropna()
            clean.columns = ["Variable", "Etiqueta"]
            clean["Variable"] = clean["Variable"].astype(str).str.strip()
            clean["Etiqueta"] = clean["Etiqueta"].astype(str).str.strip()

            print(f"[OK] Catálogo leído de la hoja '{sheet}' en {path_xlsx.name}")
            return sheet, clean

    print(f"[WARN] No se encontró tabla de 'Variable/Etiqueta' en {path_xlsx.name}")
    return None, pd.DataFrame(columns=["Variable", "Etiqueta"])


# ================== 2) CARGAR BASE FUSIONADA ==================
# Detectar separador simple: coma o punto y coma
with open(MERGED_CSV, "r", encoding="latin-1") as f:
    head = f.readline()
sep = ";" if head.count(";") > head.count(",") else ","

print(f"[INFO] Leyendo base fusionada con separador '{sep}'...")
df = pd.read_csv(MERGED_CSV, sep=sep, low_memory=False, encoding="latin-1")
print(f"[INFO] Base fusionada: {df.shape[0]} filas x {df.shape[1]} columnas")


# ================== 3) LEER CATÁLOGOS ==================
sheet_salud, cat_salud = extraer_catalogo_robusto(SALUD_CAT)
sheet_antro, cat_antro = extraer_catalogo_robusto(ANTRO_CAT)

# Agregamos campo módulo para identificarlos
if not cat_salud.empty:
    cat_salud["Módulo"] = "Salud adultos"
if not cat_antro.empty:
    cat_antro["Módulo"] = "Antropometría/TA"

# Unimos catálogos (puede haber variables que no estén en la base)
dicc_total = pd.concat([cat_salud, cat_antro], ignore_index=True)


# ================== 4) FILTRAR SOLO VARIABLES PRESENTES ==================
cols_presentes = set(df.columns)
dicc_presentes = dicc_total[dicc_total["Variable"].isin(cols_presentes)].copy()

# Reordenamos columnas
dicc_presentes = dicc_presentes[["Módulo", "Variable", "Etiqueta"]].sort_values(
    by=["Módulo", "Variable"]
)

print(f"[INFO] Variables en catálogo: {dicc_total.shape[0]}")
print(f"[INFO] Variables presentes en la base fusionada: {dicc_presentes.shape[0]}")


# ================== 5) GUARDAR ==================
dicc_presentes.to_csv(SALIDA_DICC, index=False, encoding="utf-8")
print(f"[DONE] Diccionario de variables presentes guardado en: {SALIDA_DICC}")


# ================== 6) OPCIONAL: DICCIONARIO FILTRADO PARA EL PROYECTO ==================
"""
En esta sección definimos una lista de variables CLAVE para tu análisis
(diabetes, hipertensión, obesidad, factores de riesgo, etc.) y creamos
un diccionario reducido solo con ellas.

Puedes modificar esta lista según lo que decidas usar al final.
"""

VARIABLES_PROYECTO = [
    # --- Identificación y datos básicos ---
    "FOLIO_I",
    "sexo",        # Sexo del seleccionado
    "edad",        # Edad del seleccionado (años)
    "meses",       # Meses edad (si lo usas)
    "entidad",
    "municipio",
    "x_region",

    # --- Antropometría básica / IMC / cintura ---
    "an01_1", "an01_2", "an03",       # Peso (primera, segunda medición, resultado)
    "an04_1", "an04_2", "an05",       # Talla/estatura (primera, segunda, resultado)
    "an08_1", "an08_2", "an09",       # Cintura (primera, segunda, resultado)
    "imc",                            # Índice de Masa Corporal
    "reporte_imc",                    # Categoría de IMC si la trae ENSANUT

    # --- Historia reciente de peso ---
    "an10",                           # ¿Ha perdido peso recientemente?
    "an11",                           # Amputaciones
    "a0107", "a0108", "a0109",        # Ganancia/pérdida de peso en 12 meses, intencionalidad

    # --- Diabetes mellitus ---
    "a0301",                          # ¿Médico le ha dicho que tiene diabetes?
    "a0301a",                         # ¿Le han dicho que tiene prediabetes?
    "a0302",                          # Edad al diagnóstico
    "a0303num",                       # Número/frecuencia de consultas
    "a0304a", "a0304d", "a0304m",     # Tiempo desde última consulta (años/meses)
    "a0305", "a0305esp",              # Dónde se atiende
    "a0306a", "a0306b", "a0306c",     # Revisión glucosa, TA, peso
    "a0306e", "a0306f",               # Consejería alimentación, actividad física
    "a0307",                          # ¿Toma pastillas/insulina?
    "a0308a", "a0308m",               # Tiempo tomando pastillas
    "a0309a", "a0309m",               # Tiempo con insulina
    "a0310", "a0310a",                # Frecuencia de insulina / gasto mensual
    "a0313", "a0313a",                # Adherencia (suspensión de medicamentos)
    "a0314",                          # Causa de suspensión
    # Complicaciones por diabetes
    "a0316a", "a0316b", "a0316d",
    "a0316e", "a0316f", "a0316g",
    "a0316h", "a0316i", "a0316j",
    "a0316k", "a0316l",

    # --- Hipertensión arterial ---
    "a0401",                          # ¿Médico le ha dicho que tiene presión alta?
    "a0402a",                         # Tiempo con diagnóstico (años)
    "a0404", "a0404a",                # Uso de medicamentos y gasto mensual
    "a0405a", "a0405m",               # Tiempo tomando medicamento
    "a0405aa", "a0405b", "a0405c",    # Adherencia y causas de suspensión
    "a0406", "a0406b", "a0406e",      # Consultas, gasto, hospitalización
    "a0407", "a0407esp",              # Dónde se atiende
    "a0409", "a0409a",                # Frecuencia de toma de presión
    "a0410a", "a0410b", "a0410c",
    "a0410d", "a0410e", "a0410ev",
    "a0410f", "a0410fd", "a0410fv",   # Daño de órgano blanco / urgencias

    # --- Lípidos y triglicéridos ---
    "a0603",                          # ¿Le han medido colesterol/triglicéridos?
    "a0604",                          # Colesterol alto
    "a0606",                          # Triglicéridos altos
    "a0607a", "a0607b", "a0607c", "a0607d",  # Manejo de triglicéridos

    # --- Antecedentes familiares ---
    "a0701m", "a0701p", "a0701h",     # DM en madre/padre/hermanos
    "a0702m", "a0702p", "a0702h",     # HTA en madre/padre/hermanos
    "a0703m", "a0703p", "a0703h",     # Infarto en madre/padre/hermanos
    "a0704m", "a0704p", "a0704h",     # Edad al primer infarto en familiares

    # --- Tabaco y alcohol ---
    "a1301", "a1302",                 # Fuma actualmente / edad de inicio
    "a1303", "a1304",                 # Cigarrillos por día / semana
    "a1305", "a1306p", "a1306t",      # Exfumador y tiempo desde que dejó
    "a1308", "a1309", "a1310",        # Frecuencia de consumo de alcohol / atracones
    "a1311", "a1312",                 # Binge drinking en 30 días
    "a1313", "a1314",                 # Tamizaje y consejería por alcohol

    # --- Función / discapacidad básica que quizá uses en análisis complementario ---
    "a1401", "a1402", "a1403a", "a1403b",
    "a1404a", "a1404b",
    "a1405", "a1406", "a1407", "a1408",

    # Puedes seguir agregando otras variables que te interesen...
]

# Nos quedamos solo con las variables de interés que además estén presentes
dicc_proyecto = dicc_presentes[dicc_presentes["Variable"].isin(VARIABLES_PROYECTO)].copy()

SALIDA_DICC_PROY = BASE / "PASO3_diccionario_proyecto.csv"
dicc_proyecto.to_csv(SALIDA_DICC_PROY, index=False, encoding="utf-8")

print(f"[DONE] Diccionario FILTRADO para el proyecto guardado en: {SALIDA_DICC_PROY}")
print(f"[INFO] Variables incluidas en el diccionario filtrado: {dicc_proyecto.shape[0]}")
