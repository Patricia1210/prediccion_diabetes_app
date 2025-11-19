# 14_preparacion_datos_modelo.py
# Preparación de ENSANUT para modelado de riesgo de diabetes (ANTES de entrenar modelos)

import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# =============================
# 1. Cargar base limpia
# =============================

RUTA_CSV = "ENSANUT_diabetes_base_modelo_limpia.csv"

print(f"Cargando base desde: {RUTA_CSV}")
df = pd.read_csv(RUTA_CSV, sep=";", low_memory=False)

print("\nColumnas disponibles:")
print(df.columns.tolist())

print("\nDimensiones de la base completa:", df.shape)

print("\nDistribución original de y_diabetes (0=No, 1=Sí):")
print(df["y_diabetes"].value_counts())
print(df["y_diabetes"].value_counts(normalize=True) * 100)

# =============================
# 1.1. Arreglar columnas numéricas con coma decimal
# =============================
columnas_numericas = ["edad", "imc"]

for col in columnas_numericas:
    df[col] = df[col].astype(str).str.replace(",", ".", regex=False)
    df[col] = pd.to_numeric(df[col], errors="coerce")

print("\nValores faltantes después de la conversión:")
print(df[columnas_numericas].isnull().sum())

print("\nTipos de datos después de limpieza:")
print(df.dtypes)

print("\nEjemplo de valores en imc:")
print(df["imc"].head())

# =============================
# 1.2. Imputar NaN en IMC (mediana)
# =============================
imc_mediana = df["imc"].median()
df["imc"] = df["imc"].fillna(imc_mediana)

# (si hubiera NaN en edad, también podrías imputar)
# edad_mediana = df["edad"].median()
# df["edad"] = df["edad"].fillna(edad_mediana)

print("\nNaN restantes en edad/imc después de imputación:")
print(df[["edad", "imc"]].isnull().sum())

# =============================
# 2. Definir X (predictores) e y (target)
# =============================

features = ["edad", "sexo", "imc"]
target = "y_diabetes"

X = df[features].copy()
y = df[target].copy()

print("\nPrimeras filas de X:")
print(X.head())

print("\nPrimeras filas de y:")
print(y.head())

# =============================
# 3. Codificar variables categóricas (sexo)
# =============================

X = pd.get_dummies(X, columns=["sexo"], drop_first=True)

print("\nColumnas de X después de get_dummies:")
print(X.columns.tolist())

print("\nVista rápida de X después de codificación:")
print(X.head())

# Asegurarnos de que NO queden NaN en X
print("\nNaN en X por columna:")
print(X.isnull().sum())

# Si quedara algún NaN residual, lo eliminamos
X = X.dropna()
y = y.loc[X.index]  # sincronizar índices por si se eliminaron filas

# =============================
# 4. Separar en Train y Test
# =============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

print("\nTamaño de los conjuntos:")
print("X_train:", X_train.shape)
print("X_test: ", X_test.shape)

print("\nDistribución en y_train antes de SMOTE:")
print(y_train.value_counts())
print(y_train.value_counts(normalize=True) * 100)

# =============================
# 5. Aplicar SMOTE (balanceo) en el TRAIN
# =============================

smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

print("\nDistribución en y_train DESPUÉS de SMOTE:")
print(y_train_bal.value_counts())
print(y_train_bal.value_counts(normalize=True) * 100)

print("\nTamaño de los conjuntos balanceados:")
print("X_train_bal:", X_train_bal.shape)
print("y_train_bal:", y_train_bal.shape)

# =============================
# 6. Guardar conjuntos listos para modelado
# =============================

X_train_bal.to_csv("X_train_bal.csv", sep=";", index=False)
y_train_bal.to_csv("y_train_bal.csv", sep=";", index=False)
X_test.to_csv("X_test.csv", sep=";", index=False)
y_test.to_csv("y_test.csv", sep=";", index=False)

print("\n✅ Archivos guardados:")
print(" - X_train_bal.csv")
print(" - y_train_bal.csv")
print(" - X_test.csv")
print(" - y_test.csv")
print("\nListo. Ya puedes usar estos archivos para entrenar y evaluar tus modelos.")
