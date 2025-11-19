# -*- coding: utf-8 -*-
"""
PASO11_informe_clinico_ext.py

Genera:
1) Forest plot con odds ratios e IC 95 % (si están disponibles)
2) Tabla interpretativa automática (CSV)
3) Informe HTML listo para exportar a PDF
4) Texto de recomendaciones clínicas
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from textwrap import shorten

# --------------------------------------------------------------------
# Rutas
# --------------------------------------------------------------------
BASE = Path(__file__).parent

# 👉 Archivo que generaste en PASO4 (modelo extendido)
RUTA_ODDS = BASE / "analisis_modelo_ext_odds_ratios.csv"

OUT_TABLE   = BASE / "PASO11_tabla_interpretativa_ext.csv"
OUT_HTML    = BASE / "PASO11_reporte_clinico_ext.html"
OUT_FIG_PNG = BASE / "PASO11_forestplot_ext.png"
OUT_FIG_PDF = BASE / "PASO11_forestplot_ext.pdf"

# --------------------------------------------------------------------
# 1. Cargar tabla de coeficientes / odds ratios
# --------------------------------------------------------------------
print("📂 Cargando resultados del modelo desde:", RUTA_ODDS)
df = pd.read_csv(RUTA_ODDS)

print("   Columnas disponibles en el archivo:")
print("   ", list(df.columns))

# Tipos numéricos en lo que exista
for col in ["coeficiente", "odds_ratio", "ci_inf_95", "ci_sup_95", "p_value"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# --------------------------------------------------------------------
# 1.1 Crear columnas de IC 95 % si no existen
# --------------------------------------------------------------------
tiene_ci = ("ci_inf_95" in df.columns) and ("ci_sup_95" in df.columns)

if not tiene_ci:
    print("⚠️ La tabla NO tiene columnas 'ci_inf_95' y 'ci_sup_95'.")
    print("   Se crearán usando el mismo valor de 'odds_ratio' (sin IC reales).")
    df["ci_inf_95"] = df["odds_ratio"]
    df["ci_sup_95"] = df["odds_ratio"]

# Si no hay p_value, creamos columna vacía
if "p_value" not in df.columns:
    df["p_value"] = np.nan

# Orden por impacto (OR)
df = df.sort_values("odds_ratio", ascending=False).reset_index(drop=True)

# --------------------------------------------------------------------
# 2. Clasificación de fuerza del efecto
# --------------------------------------------------------------------
def clasificar_fuerza(or_, p):
    if pd.isna(or_):
        return "Efecto no interpretable"
    if p is not None and not pd.isna(p) and p >= 0.05:
        return "No significativo estadísticamente"
    if or_ >= 3:
        return "Aumento MUY marcado del riesgo"
    if or_ >= 2:
        return "Aumento moderado-alto del riesgo"
    if or_ > 1.2:
        return "Aumento leve del riesgo"
    if 0.8 <= or_ <= 1.2:
        return "Efecto neutro o mínimo"
    if or_ <= 0.33:
        return "Reducción MUY marcada del riesgo"
    if or_ <= 0.5:
        return "Reducción moderada del riesgo"
    return "Reducción leve del riesgo"

df["clasificacion_efecto"] = [
    clasificar_fuerza(or_, p) for or_, p in zip(df["odds_ratio"], df["p_value"])
]

# Interpretación clínica (si no venía ya)
if "interpretacion_clinica" in df.columns:
    interp = df["interpretacion_clinica"]
else:
    def interpretar(or_):
        if pd.isna(or_):
            return "Efecto no interpretable."
        if or_ > 1:
            return (
                f"Aumenta el riesgo: cada unidad se asocia con un "
                f"{100 * (or_ - 1):.1f}% más de odds de diabetes."
            )
        else:
            return (
                f"Disminuye el riesgo: cada unidad se asocia con un "
                f"{100 * (1 - or_):.1f}% menos de odds de diabetes."
            )
        interp = df["odds_ratio"].apply(interpretar)


df["interpretacion_clinica_auto"] = interp

# --------------------------------------------------------------------
# 3. Forest plot
# --------------------------------------------------------------------
print("📊 Generando forest plot...")

plt.figure(figsize=(8, 0.45 * len(df) + 2))

y_pos = np.arange(len(df))

or_vals = df["odds_ratio"].values
ci_low  = df["ci_inf_95"].values
ci_high = df["ci_sup_95"].values

# Intervalos de confianza (si no hay IC reales, las barras quedan sin longitud)
plt.hlines(y_pos, ci_low, ci_high, linewidth=1.6)
plt.scatter(or_vals, y_pos, zorder=3)

plt.axvline(1.0, linestyle="--")  # OR = 1

plt.yticks(
    y_pos,
    [shorten(str(v), width=35, placeholder="…") for v in df["variable"]],
)

plt.xlabel("Odds ratio (escala logarítmica)")
plt.xscale("log")
plt.grid(axis="x", linestyle=":", alpha=0.5)
plt.tight_layout()

plt.savefig(OUT_FIG_PNG, dpi=300)
plt.savefig(OUT_FIG_PDF)
plt.close()

print("✅ Forest plot guardado en:")
print("   -", OUT_FIG_PNG)
print("   -", OUT_FIG_PDF)

# --------------------------------------------------------------------
# 4. Tabla interpretativa para informe clínico
# --------------------------------------------------------------------
print("📄 Generando tabla interpretativa...")

df_tabla = df[
    [
        "variable",
        "coeficiente",
        "odds_ratio",
        "ci_inf_95",
        "ci_sup_95",
        "p_value",
        "clasificacion_efecto",
        "interpretacion_clinica_auto",
    ]
]

df_tabla.to_csv(OUT_TABLE, index=False)
print("✅ Tabla interpretativa CSV:", OUT_TABLE)

# --------------------------------------------------------------------
# 5. Recomendaciones clínicas / preventivas
# --------------------------------------------------------------------
print("🩺 Elaborando recomendaciones clínicas...")

recomendaciones = []
for _, fila in df.iterrows():
    var   = str(fila["variable"])
    or_   = fila["odds_ratio"]
    clasif = fila["clasificacion_efecto"]

    if pd.isna(or_):
        continue

    if or_ > 1.2:  # factor de riesgo
        recomendaciones.append(
            f"- **{var}** ({clasif}). Se recomienda priorizar tamizaje y control "
            f"intensivo en personas con este factor, con énfasis en educación para "
            f"la salud y modificación de estilos de vida."
        )
    elif or_ < 0.8:  # factor protector
        recomendaciones.append(
            f"- **{var}** ({clasif}). Puede considerarse un factor protector; es "
            f"recomendable promoverlo dentro de los programas de prevención cuando "
            f"sea modificable."
        )

if not recomendaciones:
    recomendaciones.append(
        "- No se identificaron factores con efecto clínicamente relevante "
        "tras el ajuste multivariado."
    )

# --------------------------------------------------------------------
# 6. Informe HTML listo para PDF
# --------------------------------------------------------------------
print("📝 Construyendo informe HTML...")

html_rows = df_tabla.copy()
html_rows["coeficiente"] = html_rows["coeficiente"].map(lambda x: f"{x:.3f}")
html_rows["odds_ratio"]  = html_rows["odds_ratio"].map(lambda x: f"{x:.3f}")
html_rows["ci_95%"]      = html_rows.apply(
    lambda r: f"{r['ci_inf_95']:.3f} – {r['ci_sup_95']:.3f}", axis=1
)
html_rows["p_value"]     = html_rows["p_value"].map(
    lambda x: f"{x:.3f}" if not pd.isna(x) else ""
)

html_table = html_rows[
    [
        "variable",
        "coeficiente",
        "odds_ratio",
        "ci_95%",
        "p_value",
        "clasificacion_efecto",
        "interpretacion_clinica_auto",
    ]
].to_html(
    index=False,
    border=0,
    justify="center",
    classes="tabla-resultados",
    escape=False,
)

nota_ci = (
    "Los intervalos de confianza al 95 % se muestran según la tabla original."
    if tiene_ci
    else "En este análisis no se contaba con intervalos de confianza al 95 %; "
         "las barras del forest plot corresponden al valor puntual del odds ratio."
)

html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>Informe clínico – Modelo extendido de riesgo de diabetes</title>
<style>
body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    margin: 30px;
    color: #111827;
    line-height: 1.5;
}}
h1, h2, h3 {{
    color: #0f172a;
}}
.tabla-resultados {{
    border-collapse: collapse;
    width: 100%;
    font-size: 13px;
}}
.tabla-resultados th, .tabla-resultados td {{
    border: 1px solid #e5e7eb;
    padding: 6px 8px;
}}
.tabla-resultados th {{
    background: #f3f4f6;
    text-align: left;
}}
.figure-container {{
    text-align: center;
    margin: 24px 0;
}}
small {{
    color: #6b7280;
}}
</style>
</head>
<body>

<h1>Informe clínico – Modelo extendido de riesgo de diabetes</h1>

<p>Este informe resume los coeficientes, odds ratios e intervenciones sugeridas a partir
del modelo de regresión logística multivariado ajustado con el dataset extendido de
ENSANUT 2023.</p>

<h2>1. Forest plot de odds ratios</h2>

<div class="figure-container">
    <img src="{OUT_FIG_PNG.name}" alt="Forest plot odds ratios" style="max-width:100%; height:auto;">
    <div><small>{nota_ci}</small></div>
</div>

<h2>2. Tabla de resultados del modelo</h2>

{html_table}

<h2>3. Síntesis para la práctica clínica</h2>

<ul>
{''.join(f'<li>{rec}</li>' for rec in recomendaciones)}
</ul>

<p>Estas conclusiones pueden utilizarse para priorizar grupos de riesgo, definir
estrategias de educación para la salud y apoyar la toma de decisiones en programas
de prevención primaria y secundaria de diabetes mellitus tipo 2.</p>

<hr>
<small>Archivo generado automáticamente por <code>PASO11_informe_clinico_ext.py</code>.</small>

</body>
</html>
"""

OUT_HTML.write_text(html, encoding="utf-8")

print("✅ Informe HTML generado en:", OUT_HTML)
print("   ➜ Ábrelo en el navegador y usa 'Imprimir → Guardar como PDF'.")
print("🎉 Listo: forest plot, tabla interpretativa, recomendaciones e informe clínico generados.")
