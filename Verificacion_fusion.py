import pandas as pd

# Cargar la base fusionada
df = pd.read_csv("ENSANUT2023_adultos_con_antropometria.csv", sep=";", low_memory=False)

# Mostrar las columnas totales
print(f"\n📊 Total de columnas en la base fusionada: {len(df.columns)}")

# Buscar coincidencias por palabra clave (sin depender de adultos)
keywords = ["PESO", "TALL", "TALLA", "ESTAT", "IMC", "CINT", "CIN", "TA", "SIS", "DIA", "PRES"]
for keyword in keywords:
    cols_found = [col for col in df.columns if keyword.upper() in col.upper()]
    print(f"\n🔎 Columnas que contienen '{keyword}':")
    if cols_found:
        print(cols_found)
    else:
        print("❌ No encontradas")

# Detectar columnas agregadas desde antropometría
# (Se asume que las columnas nuevas están al final del archivo)
umbral = 760  # aproximado: columnas originales de adultos (758) + 2 claves
cols_extra = df.columns[umbral:].tolist()

print("\n🆕 Columnas agregadas desde ANTROPOMETRÍA:")
for col in cols_extra:
    print("-", col)

