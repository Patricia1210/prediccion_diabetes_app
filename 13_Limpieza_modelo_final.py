import pandas as pd

# 1. Cargar la base para modelado (con separador correcto)
df = pd.read_csv("ENSANUT_diabetes_base_modelo.csv", sep=";", low_memory=False)
print("Columnas del archivo:", df.columns.tolist())

# 2. Eliminar registros sin diagnóstico claro (NaN en y_diabetes)
df = df.dropna(subset=['y_diabetes'])

# 3. Asegurar que el target es numérico entero
df['y_diabetes'] = df['y_diabetes'].astype(int)

# 4. (Opcional) Convertir sexo a categórico legible
df['sexo'] = df['sexo'].replace({1: 'Hombre', 2: 'Mujer'})

print(df['y_diabetes'].value_counts())
print(df.shape)

# 5. Guardar versión limpia (si quieres)
df.to_csv("ENSANUT_diabetes_base_modelo_limpia.csv", sep=";", index=False)
print("Base limpia guardada como 'ENSANUT_diabetes_base_modelo_limpia.csv'")
