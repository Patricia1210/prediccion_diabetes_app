# 17_preparacion_datos_modelo_extendido.py
# Preparación de ENSANUT extendido para modelo de riesgo de diabetes

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# =====================================================
# 1. Rutas de archivos de entrada / salida
# =====================================================

RUTA_BASE_SIMPLE = "ENSANUT_diabetes_base_modelo_limpia.csv"   # tiene y_diabetes
RUTA_BASE_EXT = "PASO3_dataset_modelo_extendido.csv"           # nuevas variables

# archivos de salida
OUT_X_TRAIN_BAL = "X_train_ext_bal.csv"
OUT_Y_TRAIN_BAL = "y_train_ext_bal.csv"
OUT_X_TEST = "X_test_ext.csv"
OUT_Y_TEST = "y_test_ext.csv"


# =====================================================
# 2. Cargar bases
# =====================================================

print("📂 Cargando base simple (con y_diabetes)...")
df_simple = pd.read_csv(RUTA_BASE_SIMPLE, sep=";", low_memory=False)
print("Dimensiones base simple:", df_simple.shape)
print("Columnas base simple:", df_simple.columns.tolist(), "\n")

print("📂 Cargando base extendida (más variables)...")
df_ext = pd.read_csv(RUTA_BASE_EXT, sep=",", low_memory=False)
print("Dimensiones base extendida:", df_ext.shape)
print("Columnas base extendida (primeras 20):",
      df_ext.columns.tolist()[:20], "\n")

# =====================================================
# 3. Limpiar duplicados antes de fusionar
#    (evitar tener dos columnas edad/sexo/a0301)
# =====================================================

cols_drop_ext = ["edad", "sexo", "a0301"]   # ya vienen en la base simple
df_ext = df_ext.drop(columns=cols_drop_ext, errors="ignore")

# Asegurar que FOLIO_INT esté como string en ambas
df_simple["FOLIO_INT"] = df_simple["FOLIO_INT"].astype(str)
df_ext["FOLIO_INT"] = df_ext["FOLIO_INT"].astype(str)

# =====================================================
# 4. Fusionar por FOLIO_INT (inner: sólo casos con y_diabetes)
# =====================================================

print("🔗 Haciendo merge por FOLIO_INT...")
df = pd.merge(df_simple, df_ext, on="FOLIO_INT", how="inner")
print("Dimensiones después del merge:", df.shape, "\n")

# =====================================================
# 5. Renombrar variables nuevas a nombres más claros
#    (ajusta si en tu diccionario tienen otro significado)
# =====================================================

rename_dict = {
    "an01_1": "peso_kg",
    "an04_1": "talla_cm",
    "an08_1": "cintura_cm",
    "an12_1": "imc_ext",                  # IMC calculado en ENSANUT (referencia)
    "an15_1": "clasificacion_imc",
    "an21_1": "categoria_presion",
    "desc_ent": "entidad",
    "desc_mun": "municipio",
    "A0104": "diagnostico_diabetes_autorep",
    "A0213": "medicacion_diabetes",
    "A0301A": "diagnostico_diabetes_medico",
    "A0401": "diagnostico_hipertension",
    "A0502A": "colesterol_alto",
    "A0701P": "padre_diabetico",
    "A0701M": "madre_diabetica",
    "A0604": "fuma_actual",
    "A0606": "consume_alcohol",
    "A1305": "depresion_diagnosticada",
    "A1308": "infarto_cardiaco",
}

df = df.rename(columns=rename_dict)

print("Columnas después de renombrar (primeras 30):")
print(df.columns.tolist()[:30], "\n")

# =====================================================
# 6. Conversión de tipos y tratamiento de valores faltantes
# =====================================================

# --- 6.1. Variables numéricas continuas ---
numeric_cols = [
    "edad",
    "imc",        # de la base simple
    "peso_kg",
    "talla_cm",
    "cintura_cm",
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

print("Valores faltantes en numéricas antes de imputar:")
print(df[numeric_cols].isna().sum(), "\n")

# --- 6.2. Variables binarias tipo 1=Sí, 2=No, 8/9=NS/NR ---
binary_cols = [
    "diagnostico_hipertension",
    "colesterol_alto",
    "padre_diabetico",
    "madre_diabetica",
    "fuma_actual",
    "consume_alcohol",
    "depresion_diagnosticada",
    "infarto_cardiaco",
    # si quieres incluirlas como predictores, puedes activar estas:
    # "diagnostico_diabetes_autorep",
    # "medicacion_diabetes",
    # "diagnostico_diabetes_medico",
]

for col in binary_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].map({1: 1, 2: 0}).astype("float")
        # Otros códigos (3, 8, 9, 99...) quedan como NaN

print("Valores faltantes en binarias antes de imputar:")
print(df[binary_cols].isna().sum(), "\n")

# --- 6.3. Imputación ---
# Numéricas -> mediana
for col in numeric_cols:
    if col in df.columns:
        med = df[col].median()
        df[col] = df[col].fillna(med)

# Binarias -> moda (valor más frecuente, 0 o 1)
for col in binary_cols:
    if col in df.columns:
        moda = df[col].mode(dropna=True)
        if len(moda) > 0:
            df[col] = df[col].fillna(moda.iloc[0])
        else:
            # si no hay moda (caso extremo), asumir 0
            df[col] = df[col].fillna(0)

print("NaN restantes en numéricas después de imputación:")
print(df[numeric_cols].isna().sum(), "\n")

print("NaN restantes en binarias después de imputación:")
print(df[binary_cols].isna().sum(), "\n")

# =====================================================
# 7. Definir X (predictores) e y (target)
# =====================================================

# y = variable objetivo (ya viene limpia de la base simple)
y = df["y_diabetes"].astype(int)

# Puedes ajustar esta lista según el modelo que quieras construir
feature_cols = [
    "edad",
    "imc",
    "peso_kg",
    "talla_cm",
    "cintura_cm",
    "diagnostico_hipertension",
    "colesterol_alto",
    "padre_diabetico",
    "madre_diabetica",
    "fuma_actual",
    "consume_alcohol",
    "depresion_diagnosticada",
    "infarto_cardiaco",
    # Si decides incluirlas (ojo con posible fuga de información):
    # "diagnostico_diabetes_autorep",
    # "medicacion_diabetes",
    # "diagnostico_diabetes_medico",
]

# Asegurarnos de que todas existan
feature_cols = [c for c in feature_cols if c in df.columns]

# Incluimos sexo como categórica (luego one-hot)
feature_cols_with_sex = feature_cols + ["sexo"]

X = df[feature_cols_with_sex].copy()

print("Columnas usadas como predictores antes de dummies:")
print(X.columns.tolist(), "\n")

# =====================================================
# 8. Codificar sexo (one-hot)
# =====================================================

X = pd.get_dummies(X, columns=["sexo"], drop_first=True)
print("Columnas de X después de get_dummies:")
print(X.columns.tolist(), "\n")

# Verificar que no haya NaN en X
print("NaN totales en X:", X.isna().sum().sum(), "\n")

# =====================================================
# 9. Train/Test split
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y,
)

print("Tamaño de los conjuntos:")
print("  X_train:", X_train.shape)
print("  X_test :", X_test.shape, "\n")

print("Distribución original en y_train (0=No, 1=Sí):")
print(y_train.value_counts())
print(y_train.value_counts(normalize=True) * 100, "\n")

# =====================================================
# 🔄 10. SMOTE para balancear la clase en entrenamiento
# =====================================================

smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

print("Distribución en y_train DESPUÉS de SMOTE:")
print(y_train_bal.value_counts())
print(y_train_bal.value_counts(normalize=True) * 100, "\n")

print("Tamaño de los conjuntos balanceados:")
print("  X_train_bal:", X_train_bal.shape)
print("  y_train_bal:", y_train_bal.shape, "\n")

# =====================================================
# 11. Guardar resultados
# =====================================================

X_train_bal.to_csv(OUT_X_TRAIN_BAL, index=False)
y_train_bal.to_csv(OUT_Y_TRAIN_BAL, index=False)
X_test.to_csv(OUT_X_TEST, index=False)
y_test.to_csv(OUT_Y_TEST, index=False)

print("✅ Archivos guardados:")
print("  -", OUT_X_TRAIN_BAL)
print("  -", OUT_Y_TRAIN_BAL)
print("  -", OUT_X_TEST)
print("  -", OUT_Y_TEST)
print("\nListo. Ya puedes usar estos archivos para entrenar tus modelos extendidos. 🚀")
