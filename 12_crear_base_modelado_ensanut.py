import pandas as pd

# 1. Cargar la base fusionada ENSANUT (ajusta el nombre de archivo)
df = pd.read_csv("ENSANUT2023_adultos_con_antropometria.csv", sep=";", low_memory=False)

# 2. Elegir variables clave para el modelo (ajusta la lista con lo que viste en ENSANUT_variables_para_diabetes.csv)
vars_basicas = [
    "FOLIO_INT",      # ID persona (ajusta si se llama distinto)
    "edad",           # edad numérica
    "sexo",           # sexo (o 'asexo', según cuál prefieras)
    "imc",            # si en tu base de antro ya viene IMC
    # agrega aquí más predictores que te interesen:
    # "a0107", "a0109", "A0110A", ...
]

# 3. Variable objetivo (target) para el modelo
#    Reemplaza 'NOMBRE_VAR_DIABETES' por la variable de diagnóstico de diabetes (ej. 'mc_v01' o similar)
var_objetivo = "a0301"

vars_modelo = vars_basicas + [var_objetivo]

# Verificar que todas existan
faltantes = [v for v in vars_modelo if v not in df.columns]
if faltantes:
    print("⚠ Variables no encontradas en el CSV:", faltantes)
else:
    print("✅ Todas las variables del modelo están en la base.")

# 4. Crear la sub-base solo con las columnas del modelo
df_modelo = df[vars_modelo].copy()

# 5. Limpieza básica de la variable objetivo (ejemplo para variable codificada 1 = sí, 2 = no)
# Ajusta según los códigos reales de ENSANUT
df_modelo["y_diabetes"] = df_modelo[var_objetivo].map({
    1: 1,   # tiene diabetes
    2: 0,   # no tiene
}).astype("Int64")

# 6. Guardar base final para modelado
df_modelo.to_csv("ENSANUT_diabetes_base_modelo.csv", sep=";", index=False)
print("📁 Base para modelado guardada como 'ENSANUT_diabetes_base_modelo.csv'")


import pandas as pd

df_modelo = pd.read_csv("ENSANUT_diabetes_base_modelo.csv")
print(df_modelo.head(10))
print(df_modelo.shape)

