# 18_entrenamiento_modelos_ext.py
# Entrenamiento de modelos con ENSANUT extendido

from pathlib import Path
import os
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from joblib import dump

# =====================================================
# 1. Rutas
# =====================================================

BASE = Path(__file__).parent

X_TRAIN_PATH = BASE / "X_train_ext_bal.csv"
Y_TRAIN_PATH = BASE / "y_train_ext_bal.csv"
X_TEST_PATH  = BASE / "X_test_ext.csv"
Y_TEST_PATH  = BASE / "y_test_ext.csv"

RESULTADOS_CSV = BASE / "resultados_modelos_ext_diabetes.csv"
MODELOS_DIR = BASE / "modelos_entrenados_ext"

MODELOS_DIR.mkdir(exist_ok=True)


# =====================================================
# 2. Cargar datos
# =====================================================

print("📂 Cargando datos de entrenamiento y prueba...")

X_train = pd.read_csv(X_TRAIN_PATH)
y_train = pd.read_csv(Y_TRAIN_PATH).squeeze()   # serie 1D

X_test  = pd.read_csv(X_TEST_PATH)
y_test  = pd.read_csv(Y_TEST_PATH).squeeze()

print("Dimensiones:")
print("  X_train:", X_train.shape)
print("  y_train:", y_train.shape)
print("  X_test :", X_test.shape)
print("  y_test :", y_test.shape, "\n")

print("Columnas de X_train:")
print(X_train.columns.tolist(), "\n")

print("Distribución de y_train (0=No, 1=Sí):")
print(y_train.value_counts(), "\n")


# =====================================================
# 3. Definir modelos
# =====================================================

modelos = {
    "logistic_regression": LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        n_jobs=-1
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    ),
    "gradient_boosting": GradientBoostingClassifier(
        random_state=42
    ),
    "decision_tree": DecisionTreeClassifier(
        max_depth=None,
        random_state=42
    ),
    "knn": KNeighborsClassifier(
        n_neighbors=7
    ),
}


# =====================================================
# 4. Función auxiliar para AUC
# =====================================================

def obtener_probabilidades(modelo, X):
    """
    Devuelve las probabilidades para la clase positiva (1).
    Si el modelo no tiene predict_proba, intenta usar decision_function.
    """
    if hasattr(modelo, "predict_proba"):
        return modelo.predict_proba(X)[:, 1]
    elif hasattr(modelo, "decision_function"):
        scores = modelo.decision_function(X)
        # escalar a 0-1
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        return scaler.fit_transform(scores.reshape(-1, 1)).ravel()
    else:
        # último recurso: usar predicciones como 0/1
        return modelo.predict(X)


# =====================================================
# 5. Entrenamiento y evaluación
# =====================================================

resultados = []
mejor_modelo = None
mejor_nombre = None
mejor_auc = -np.inf

for nombre, modelo in modelos.items():
    print("🔹 Entrenando modelo:", nombre)

    modelo.fit(X_train, y_train)

    # predicciones
    y_pred = modelo.predict(X_test)
    y_scores = obtener_probabilidades(modelo, X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    auc  = roc_auc_score(y_test, y_scores)

    resultados.append({
        "modelo": nombre,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
    })

    print(f"  accuracy : {acc:.4f}")
    print(f"  precision: {prec:.4f}")
    print(f"  recall   : {rec:.4f}")
    print(f"  f1       : {f1:.4f}")
    print(f"  roc_auc  : {auc:.4f}\n")

    # actualizar mejor modelo según AUC
    if auc > mejor_auc:
        mejor_auc = auc
        mejor_modelo = modelo
        mejor_nombre = nombre


# =====================================================
# 6. Guardar tabla de resultados
# =====================================================

df_resultados = pd.DataFrame(resultados)
df_resultados = df_resultados.sort_values(by="roc_auc", ascending=False)

df_resultados.to_csv(RESULTADOS_CSV, index=False)

print("📊 Resumen de métricas guardado en:")
print(" ", RESULTADOS_CSV, "\n")

print("Tabla de resultados:")
print(df_resultados, "\n")


# =====================================================
# 7. Guardar mejor modelo
# =====================================================

if mejor_modelo is not None:
    ruta_modelo = MODELOS_DIR / f"{mejor_nombre}_ext.joblib"
    dump(mejor_modelo, ruta_modelo)
    print("💾 Mejor modelo guardado como:")
    print(" ", ruta_modelo)
    print(f"\nMejor modelo: {mejor_nombre} (AUC = {mejor_auc:.4f})")
else:
    print("⚠️ No se pudo determinar un mejor modelo.")
