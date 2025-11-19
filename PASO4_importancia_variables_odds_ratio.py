# PASO 4: Importancia clínica - Coeficientes y Odds Ratios del modelo logístico

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ================== CONFIGURACIÓN ==================
BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "modelos_entrenados_ext" / "logistic_regression_ext.joblib"

# Archivo correcto según tu carpeta:
XTRAIN_PATH = BASE / "X_train_ext_bal.csv"

OUT_CSV = BASE / "analisis_modelo_ext_odds_ratios.csv"
OUT_HTML = BASE / "analisis_modelo_ext_odds_ratios.html"


# ================== 1. CARGAR MODELO Y DATOS ==================
print("🔍 Cargando modelo y variables...")
model = joblib.load(MODEL_PATH)
X_train = pd.read_csv(XTRAIN_PATH)

feature_names = X_train.columns.tolist()

print(f"Variables utilizadas ({len(feature_names)}):")
print(feature_names)

# ================== 2. EXTRAER COEFICIENTES ==================
coef = model.coef_[0]  
intercept = model.intercept_[0]

df_coef = pd.DataFrame({
    'variable': feature_names,
    'coeficiente': coef,
})

# ================== 3. CALCULAR ODDS RATIOS ==================
df_coef['odds_ratio'] = np.exp(df_coef['coeficiente'])

# ================== 4. INTERPRETACIÓN AUTOMÁTICA ==================
def interpretar_variable(row):
    beta = row['coeficiente']
    odds = row['odds_ratio']
    
    if odds > 1:
        return f"Aumenta el riesgo: cada unidad incrementa {((odds - 1) * 100):.1f}% el riesgo."
    elif odds < 1:
        return f"Disminuye el riesgo: reduce el riesgo en {(100 - odds * 100):.1f}%."
    else:
        return "No tiene impacto relevante en el riesgo."

df_coef['interpretacion_clinica'] = df_coef.apply(interpretar_variable, axis=1)

# ================== 5. ORDENAR POR IMPACTO ==================
df_coef['impacto_absoluto'] = df_coef['coeficiente'].abs()
df_coef = df_coef.sort_values(by='impacto_absoluto', ascending=False).drop(columns=['impacto_absoluto'])

# ================== 6. GUARDAR RESULTADOS ==================
df_coef.to_csv(OUT_CSV, index=False)
df_coef.to_html(OUT_HTML, index=False)

print("\n📄 Resultados guardados como:")
print(f"   - CSV:  {OUT_CSV}")
print(f"   - HTML: {OUT_HTML}")

print("\n🧬 Análisis completado: tabla de coeficientes y odds ratios lista.")
