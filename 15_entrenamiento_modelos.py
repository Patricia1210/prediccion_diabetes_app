# 15_entrenamiento_modelos.py
# Entrenamiento y evaluación de varios modelos de ML para riesgo de diabetes ENSANUT

from pathlib import Path
import os
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

import joblib

# =============================
# 1. Rutas y carga de datos
# =============================

BASE = Path(__file__).parent

X_train_path = BASE / "X_train_bal.csv"
y_train_path = BASE / "y_train_bal.csv"
X_test_path  = BASE / "X_test.csv"
y_test_path  = BASE / "y_test.csv"

print("Cargando archivos de entrenamiento y prueba...\n")

X_train = pd.read_csv(X_train_path, sep=";")
y_train = pd.read_csv(y_train_path, sep=";")["y_diabetes"]

X_test = pd.read_csv(X_test_path, sep=";")
y_test = pd.read_csv(y_test_path, sep=";")["y_diabetes"]

# Asegurar tipo entero para y
y_train = y_train.astype(int)
y_test = y_test.astype(int)

print("Dimensiones:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test :", X_test.shape)
print("y_test :", y_test.shape)
print("\nDistribución en y_train:")
print(y_train.value_counts())
print("\nDistribución en y_test:")
print(y_test.value_counts())
print("\n----------------------------------------\n")

# =============================
# 2. Definir modelos a comparar
# =============================

modelos = {
    "logistic_regression": LogisticRegression(max_iter=2000),
    "decision_tree": DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        random_state=42,
        n_jobs=-1
    ),
    "gradient_boosting": GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    ),
    "knn": KNeighborsClassifier(
        n_neighbors=7
    ),
}

# Carpeta para guardar modelos
modelos_dir = BASE / "modelos_entrenados"
os.makedirs(modelos_dir, exist_ok=True)

# Lista para ir guardando métricas
resultados = []

# =============================
# 3. Entrenamiento y evaluación
# =============================

for nombre, modelo in modelos.items():
    print(f"🔹 Entrenando modelo: {nombre} ...")
    modelo.fit(X_train, y_train)

    # Predicciones
    y_pred = modelo.predict(X_test)

    # Algunas métricas básicas
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # ROC-AUC (si el modelo soporta predict_proba)
    try:
        y_proba = modelo.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
    except AttributeError:
        auc = np.nan

    # Guardar métricas en la lista
    resultados.append({
        "modelo": nombre,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
    })

    print(f"\nResultados para {nombre}:")
    print(f"  Accuracy : {acc:.3f}")
    print(f"  Precisión: {prec:.3f}")
    print(f"  Recall   : {rec:.3f}")
    print(f"  F1-score : {f1:.3f}")
    if not np.isnan(auc):
        print(f"  ROC-AUC  : {auc:.3f}")
    else:
        print("  ROC-AUC  : no disponible (sin predict_proba)")
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    print("\nMatriz de confusión (filas = verdadero, columnas = predicho):")
    print(cm)

    # Reporte de clasificación
    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred, digits=3))

    print("\n----------------------------------------\n")

    # Guardar modelo entrenado
    model_path = modelos_dir / f"{nombre}.joblib"
    joblib.dump(modelo, model_path)
    print(f"💾 Modelo guardado en: {model_path}\n")
    print("========================================\n")

# =============================
# 4. Guardar resumen de métricas
# =============================

df_resultados = pd.DataFrame(resultados)
df_resultados = df_resultados.sort_values(by="roc_auc", ascending=False)

res_path = BASE / "resultados_modelos_diabetes.csv"
df_resultados.to_csv(res_path, sep=";", index=False)

print("✅ Resumen de métricas guardado en:")
print(f"   {res_path}\n")
print("Tabla de resultados:")
print(df_resultados)
