# -*- coding: utf-8 -*-
"""
23_ensamble_voting_classweight_sin_region.py

Ensambles Voting (soft) SIN variable Region
Compatible SOLO con modelos entrenados SIN Region
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

from joblib import load, dump


# =====================================================
# 1) Configuración general
# =====================================================
BASE = Path(__file__).parent
DATA_PATH = BASE / "BaseFusionada_2023.csv"
MODELOS_DIR = BASE / "modelos_entrenados_ext"

OUT_CSV = BASE / "resultados_ensamble_voting_classweight_sin_region.csv"
OUT_JSON = BASE / "metricas_ensamble_voting_classweight_sin_region.json"
ENSEMBLE_PATH = MODELOS_DIR / "ensemble_voting_classweight_sin_region.joblib"

TARGET = "Outcome"
OPTIMAL_THRESHOLD = 0.30

MODELS_TO_LOAD = [
    "class_weight_logistic_regression_best.joblib",
    "class_weight_gradient_boosting_best.joblib",
    "class_weight_random_forest_best.joblib",
]


# =====================================================
# 2) Carga de datos (ELIMINANDO REGION)
# =====================================================
df = pd.read_csv(DATA_PATH)

if TARGET not in df.columns:
    raise ValueError("❌ No existe la columna Outcome")

# 🔥 ELIMINAR REGION SI EXISTE
if "Region" in df.columns:
    print("🧹 Eliminando columna 'Region'")
    df = df.drop(columns=["Region"])

X = df.drop(columns=[TARGET])
y = df[TARGET]

print("✅ Base cargada SIN Region:", X.shape)


# =====================================================
# 3) Split
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print(f"🔀 Split -> Train: {X_train.shape} | Test: {X_test.shape}")


# =====================================================
# 4) Cargar modelos individuales
# =====================================================
print("\n📦 Cargando modelos entrenados SIN Region...")

models = {}

for fname in MODELS_TO_LOAD:
    path = MODELOS_DIR / fname
    if not path.exists():
        raise FileNotFoundError(f"❌ No encontrado: {fname}")

    name = fname.replace("class_weight_", "").replace("_best.joblib", "")
    models[name] = load(path)
    print(f"✅ Cargado: {name}")

if len(models) < 2:
    raise ValueError("❌ Se requieren al menos 2 modelos para Voting")


# =====================================================
# 5) Funciones auxiliares
# =====================================================
def get_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    scores = model.decision_function(X)
    return (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)


def eval_metrics(y_true, scores, thr):
    y_pred = (scores >= thr).astype(int)
    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    return {
        "threshold": thr,
        "auc_roc": roc_auc_score(y_true, scores),
        "accuracy": rep["accuracy"],
        "precision_1": rep["1"]["precision"],
        "recall_1": rep["1"]["recall"],
        "f1_1": rep["1"]["f1-score"],
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }


# =====================================================
# 6) Evaluar modelos individuales
# =====================================================
rows = {}
csv_rows = []

print("\n📊 Evaluación de modelos individuales")

for name, model in models.items():
    scores = get_scores(model, X_test)

    met_05 = eval_metrics(y_test, scores, 0.5)
    met_opt = eval_metrics(y_test, scores, OPTIMAL_THRESHOLD)

    rows[name] = met_opt

    print(f"🔹 {name}")
    print(f"   Recall@0.30: {met_opt['recall_1']:.3f} | AUC: {met_opt['auc_roc']:.3f}")

    for met in [met_05, met_opt]:
        csv_rows.append({
            "model_type": "individual",
            "model": name,
            **met
        })


# =====================================================
# 7) Ensamble Voting Soft
# =====================================================
print("\n🎭 Creando ENSAMBLE Voting Soft")

estimators = [(name, model) for name, model in models.items()]

ensemble = VotingClassifier(
    estimators=estimators,
    voting="soft",
    n_jobs=-1
)

# Hack controlado (no refit)
ensemble.estimators_ = [m for _, m in estimators]
ensemble.named_estimators_ = dict(estimators)
ensemble.classes_ = np.array([0, 1])

scores_ens = ensemble.predict_proba(X_test)[:, 1]

ens_05 = eval_metrics(y_test, scores_ens, 0.5)
ens_opt = eval_metrics(y_test, scores_ens, OPTIMAL_THRESHOLD)

print(f"\n🏆 ENSAMBLE @0.30 -> Recall: {ens_opt['recall_1']:.3f} | AUC: {ens_opt['auc_roc']:.3f}")

for met in [ens_05, ens_opt]:
    csv_rows.append({
        "model_type": "ensemble",
        "model": "voting_soft",
        **met
    })


# =====================================================
# 8) Guardado de resultados
# =====================================================
df_out = pd.DataFrame(csv_rows)
df_out.to_csv(OUT_CSV, index=False)

dump(ensemble, ENSEMBLE_PATH)

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump({
        "optimal_threshold": OPTIMAL_THRESHOLD,
        "ensemble": ens_opt,
        "individuals": rows
    }, f, indent=2, ensure_ascii=False)

print("\n💾 Resultados guardados correctamente")
print("✅ ENSAMBLE SIN REGION COMPLETADO")
