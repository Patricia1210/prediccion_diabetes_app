# Extraccion_Limpieza_Ensanut.py
# =========================================
# Extracción y limpieza de variables ENSANUT
# para modelo extendido de riesgo de diabetes
# =========================================

import pandas as pd

RUTA_CSV = "ENSANUT_2023_Adultos_Salud+Antro_merged.csv"

print(f"\n📂 Cargando base fusionada desde: {RUTA_CSV}")

df = pd.read_csv(
    RUTA_CSV,
    sep=",",
    encoding="latin-1",
    low_memory=False
)

print("\n✅ Base cargada.")
print("Dimensiones de la base completa:", df.shape)

print("\nEjemplo de columnas disponibles (primeras 40):")
print(list(df.columns[:40]))

# -----------------------------------------
# 1. Lista de variables que queremos usar
# -----------------------------------------
columnas_deseadas = [
    "FOLIO_INT",
    "an01_1", "an04_1", "an08_1", "an12_1", "an15_1", "an21_1",
    "Sistolica", "Diastolica",
    "desc_ent", "desc_mun",
    "edad", "sexo",
    "A0104", "A0213", "A0301A", "A0301",
    "A0401", "A0502A",
    "A0701P", "A0701M",
    "A0604", "A0606",
    "A1305", "A1308",
]

# -----------------------------------------
# 2. Ver cuáles de esas columnas existen
# -----------------------------------------
columnas_presentes = [c for c in columnas_deseadas if c in df.columns]
columnas_faltantes = [c for c in columnas_deseadas if c not in df.columns]

print("\n✅ Columnas seleccionadas que SÍ existen en la base:")
print(columnas_presentes)

print("\n⚠️ Estas columnas NO se encontraron en el archivo y se omitirán:")
print(columnas_faltantes)

if len(columnas_presentes) == 0:
    print("\n❌ No se encontró NINGUNA de las columnas deseadas.")
    print("   Revisa en el listado de columnas cómo se llaman exactamente.")
else:
    # -----------------------------------------
    # 3. Subconjunto y guardado
    # -----------------------------------------
    df_sub = df[columnas_presentes].copy()
    print("\nDimensiones después de seleccionar variables:", df_sub.shape)

    SALIDA = "PASO3_dataset_modelo_extendido.csv"
    df_sub.to_csv(SALIDA, index=False)
    print(f"\n💾 Archivo limpio guardado como: {SALIDA}")
    print("\nListo. Ya tienes tu base_modelo_extendido con más variables.")


# 16_crear_dataset_modelo_extendido.py
# Extracción y limpieza de variables ENSANUT para modelo extendido

import pandas as pd

# 1. Ruta de la base fusionada (la que ya usaste antes)
RUTA_CSV = "ENSANUT_2023_Adultos_Salud+Antro_merged.csv"

print(f"\n📂 Cargando base fusionada desde: {RUTA_CSV}")
df = pd.read_csv(RUTA_CSV, encoding="latin-1")

print("\n✅ Base cargada.")
print("Dimensiones de la base completa:", df.shape)

# 2. Definir mapeo: nombre_limpio -> nombre_real_en_csv
#    (los nombres reales son los que te mostró la búsqueda aproximada)
mapeo_columnas = {
    # Identificador
    "FOLIO_INT": None,  # lo rellenamos dinámicamente abajo

    # Variables que ya usabas
    "an01_1": "an01_1",
    "an04_1": "an04_1",
    "an08_1": "an08_1",
    "an12_1": "an12_1",
    "an15_1": "an15_1",
    "an21_1": "an21_1",
    "desc_ent": "desc_ent",
    "desc_mun": "desc_mun",
    "edad": "edad",
    "sexo": "sexo",

    # Nuevas variables clínicas / de riesgo (en minúsculas en el CSV)
    "A0104": "a0104",
    "A0213": "a0213",
    "A0301A": "a0301a",
    "A0301": "a0301",
    "A0401": "a0401",
    "A0502A": "a0502a",
    "A0701P": "a0701p",
    "A0701M": "a0701m",
    "A0604": "a0604",
    "A0606": "a0606",
    "A1305": "a1305",
    "A1308": "a1308",
}

# 3. Detectar automáticamente la columna FOLIO_INT (hay caracteres raros por el BOM)
folio_cols = [c for c in df.columns if "FOLIO_INT" in c]
if folio_cols:
    mapeo_columnas["FOLIO_INT"] = folio_cols[0]
    print(f"\n🔑 Columna de FOLIO_INT detectada como: {folio_cols[0]}")
else:
    print("\n⚠ No se encontró ninguna columna que contenga 'FOLIO_INT'.")
    mapeo_columnas.pop("FOLIO_INT", None)

# 4. Ver qué columnas del mapeo realmente existen en el DataFrame
presentes = {}
faltantes = []

for nombre_limpio, nombre_real in mapeo_columnas.items():
    if nombre_real in df.columns:
        presentes[nombre_limpio] = nombre_real
    else:
        faltantes.append((nombre_limpio, nombre_real))

print("\n✅ Columnas que SÍ se encontraron y se usarán:")
for k, v in presentes.items():
    print(f"  - {k}  <-  {v}")

if faltantes:
    print("\n⚠ Estas columnas del mapeo no se encontraron y se omitirán:")
    for k, v in faltantes:
        print(f"  - {k}  (esperada como '{v}')")

# 5. Construir el dataset final con esas columnas
df_out = df[list(presentes.values())].copy()
df_out.columns = list(presentes.keys())  # renombramos a los nombres limpios

print("\n📏 Dimensiones del dataset extendido:", df_out.shape)

# 6. Guardar
SALIDA = "PASO3_dataset_modelo_extendido.csv"
df_out.to_csv(SALIDA, index=False)
print(f"\n💾 Archivo limpio guardado como: {SALIDA}")

print("\nListo. Ya tienes tu dataset extendido con más variables para el modelo. 🚀")

