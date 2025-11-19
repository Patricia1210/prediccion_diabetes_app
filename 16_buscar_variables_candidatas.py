# 16_buscar_variables_candidatas.py
# Buscar automáticamente en ENSANUT variables candidatas
# para el modelo de riesgo de diabetes

from pathlib import Path
import pandas as pd

BASE = Path(__file__).parent

# ====== 1. Rutas de archivos (ajusta si tu nombre es distinto) ======
RUTA_DATOS = BASE / "ENSANUT_2023_Adultos_Salud+Antro_merged.csv"
RUTA_DICC  = BASE / "ENSANUT_variables_para_diabetes.csv"  # diccionario Variable / Etiqueta

print("📂 Leyendo estructura de la base grande...")
# Solo leemos unas filas para obtener los nombres de columnas (nrows para que sea rápido)
df = pd.read_csv(RUTA_DATOS, sep=";", low_memory=False, nrows=200)
columnas = df.columns.tolist()

print(f"  - Columnas detectadas en {RUTA_DATOS.name}: {len(columnas)}")

dicc = None
if RUTA_DICC.exists():
    print(f"📂 Leyendo diccionario: {RUTA_DICC.name}")
    dicc = pd.read_csv(RUTA_DICC, encoding="latin-1")
    # Normalizamos nombres esperados
    dicc.columns = [c.strip() for c in dicc.columns]
else:
    print("⚠️ No se encontró el diccionario ENSANUT_variables_para_diabetes.csv.")
    print("   Solo se buscará por nombre de variable.\n")

# ====== 2. Definir conceptos y palabras clave ======

conceptos = {
    "Region":        ["region", "área", "area", "urbano", "rural", "estrato"],
    "Edad":          ["edad"],
    "Sexo":          ["sexo", "genero", "género"],
    "Peso":          ["peso", "imc", "indice de masa", "índice de masa"],
    "Depresion":     ["depresion", "depresión", "depresivo", "estado emocional", "cuestionario de depresion"],
    "Prediabetico":  ["prediabet", "pre-diabet", "azucar alta", "azúcar alta", "glucosa elevada"],
    "Presion":       ["presion arterial", "presión arterial", "hipertens", "tension arterial", "tensión arterial"],
    "Infarto":       ["infarto", "ataque al corazon", "ataque al corazón", "cardiaco"],
    "Colesterol":    ["colesterol"],
    "Trigliceridos": ["triglicerid"],
    "Padre_diabetico": ["padre", "diabet", "antecedente paterno"],
    "Madre_diabetica": ["madre", "diabet", "antecedente materno"],
    "Fumar":         ["fuma", "fumador", "tabaco", "cigarro", "cigarros"],
    "Alcohol":       ["alcohol", "bebidas alcoh"],
    "Outcome":       ["diabet", "diagnostico", "diagnóstico", "resultado de la entrevista", "resultado de cuestionario"],
}

# ====== 3. Función de búsqueda ======

def buscar_en_columnas(columnas, keywords):
    """Devuelve lista de nombres de columna que contienen alguna palabra clave."""
    cols_match = []
    for col in columnas:
        col_low = col.lower()
        if any(kw in col_low for kw in keywords):
            cols_match.append(col)
    return cols_match


def buscar_en_diccionario(dicc, keywords):
    """
    Devuelve DataFrame con variables cuya etiqueta contiene alguna palabra clave.
    Espera que el diccionario tenga columnas tipo: 'Variable' y 'Etiqueta'
    """
    if dicc is None:
        return pd.DataFrame(columns=["Variable", "Etiqueta"])

    # Detectar posibles nombres de columnas para variable y etiqueta
    col_var = None
    col_lab = None
    for c in dicc.columns:
        cl = c.lower()
        if "var" in cl:
            col_var = c
        if "etiqueta" in cl or "label" in cl or "descripcion" in cl or "descripción" in cl:
            col_lab = c

    if col_var is None or col_lab is None:
        print("⚠️ No se pudieron identificar las columnas de Variable/Etiqueta en el diccionario.")
        return pd.DataFrame(columns=["Variable", "Etiqueta"])

    mask = dicc[col_lab].astype(str).str.lower().apply(
        lambda txt: any(kw in txt for kw in keywords)
    )
    return dicc.loc[mask, [col_var, col_lab]].rename(
        columns={col_var: "Variable", col_lab: "Etiqueta"}
    )


# ====== 4. Búsqueda concepto por concepto ======

resultados_global = []

print("\n================ BÚSQUEDA DE VARIABLES CANDIDATAS ================\n")

for concepto, keywords in conceptos.items():
    print(f"🔹 Concepto: {concepto}")
    print(f"   Palabras clave: {keywords}")

    # 4.1. Coincidencias por nombre de columna
    por_nombre = buscar_en_columnas(columnas, keywords)

    # 4.2. Coincidencias por etiqueta en diccionario
    por_dicc = buscar_en_diccionario(dicc, keywords) if dicc is not None else pd.DataFrame(columns=["Variable", "Etiqueta"])

    # ---- Mostrar en pantalla ----
    if por_nombre:
        print("   ✔ Coincidencias por NOMBRE de variable:")
        for col in por_nombre:
            print(f"      - {col}")
    else:
        print("   ✖ No hubo coincidencias por NOMBRE de variable.")

    if not por_dicc.empty:
        print("   ✔ Coincidencias por ETIQUETA en diccionario:")
        for _, row in por_dicc.iterrows():
            print(f"      - {row['Variable']}: {row['Etiqueta']}")
    else:
        print("   ✖ No hubo coincidencias por ETIQUETA.")

    print("   ------------------------------------------------------\n")

    # ---- Guardar en estructura global para CSV ----
    # Por nombre
    for col in por_nombre:
        resultados_global.append({
            "concepto": concepto,
            "origen": "nombre_columna",
            "variable": col,
            "etiqueta": ""
        })

    # Por diccionario
    for _, row in por_dicc.iterrows():
        resultados_global.append({
            "concepto": concepto,
            "origen": "diccionario",
            "variable": row["Variable"],
            "etiqueta": row["Etiqueta"]
        })

# ====== 5. Guardar resumen en CSV ======

if resultados_global:
    df_res = pd.DataFrame(resultados_global).drop_duplicates()
    out_path = BASE / "ENSANUT_variables_candidatas_para_modelo.csv"
    df_res.to_csv(out_path, sep=";", index=False, encoding="latin-1")
    print("✅ Resumen de variables candidatas guardado en:")
    print(f"   {out_path}")
else:
    print("⚠️ No se encontraron variables candidatas con las palabras clave definidas.")
