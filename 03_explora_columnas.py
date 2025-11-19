# 03_explora_columnas.py
from pathlib import Path
import pandas as pd
import re

# ---------- RUTAS ----------
BASE = Path(__file__).parent
MERGED = BASE / "ENSANUT_2023_Adultos_Salud+Antro_merged.csv"   # ajusta si tu nombre es distinto

# ---------- UTILIDADES ----------
def guess_sep(path: Path) -> str:
    with open(path, "r", encoding="latin-1") as f:
        head = f.readline()
    return ";" if head.count(";") > head.count(",") else ","

def keep_existing(cols, df):
    """Devuelve solo los nombres que existen en df.columns (case-sensitive)."""
    return [c for c in cols if c in df.columns]

def find_by_keywords(df, keywords):
    patt = re.compile("|".join([re.escape(k) for k in keywords]), re.I)
    return [c for c in df.columns if patt.search(c)]

# ---------- CARGA ----------
sep = guess_sep(MERGED)
df = pd.read_csv(MERGED, sep=sep, low_memory=False, encoding="latin-1")
print(f"[INFO] Archivo: {MERGED.name}  shape={df.shape}  sep='{sep}'")

# ===================== (1) LISTA COMPLETA DE COLUMNAS =====================
cols_all = list(df.columns)
print("\n=== (1) TODAS LAS COLUMNAS ({}): ===".format(len(cols_all)))
for i, c in enumerate(cols_all, 1):
    print(f"{i:>3}. {c}")

# Guardar a archivo de apoyo
(out_txt := BASE / "columnas_todas.txt").write_text("\n".join(cols_all), encoding="utf-8")
print(f"\n[OK] Lista completa guardada en: {out_txt}")

# ===================== (2) AGRUPACIÓN POR TIPO ===========================
# Candidatas de llaves y estructuras comunes en ENSANUT
id_keys = [
    "FOLIOVIV", "FOLIOVIV_", "FOLIO_VIV", "FOLIO_HOG", "FOLIOHOG", "FOLIO_I", "FOLIO_INT",
    "NUMREN", "NUMREN_", "ID_PER", "ID_PERSONA", "entidad", "estado"
]
ids_present = keep_existing(id_keys, df)

# Sociodemográficas típicas
soc_keywords = ["edad", "sexo", "género", "genero", "escolar", "educ", "trabaj", "ocup", "ingres", "urb", "rural"]
soc_cols = find_by_keywords(df, soc_keywords)

# Clínicas / diabetes / HTA
clin_keywords = [
    "diab", "gluc", "a1c", "hb", "hiper", "hta", "hipert", "presion", "tensi", "sistol", "diast",
    "medic", "tratam", "diagn"
]
clin_cols = find_by_keywords(df, clin_keywords)

# Antropometría
antro_keywords = ["peso", "talla", "estatura", "imc", "cint", "cadera", "circun"]
antro_cols = find_by_keywords(df, antro_keywords)

# Estilo de vida
lifestyle_keywords = ["taba", "fuma", "alcoh", "actividad", "ejerc", "sedent", "fruta", "verdura", "azucar", "refres"]
life_cols = find_by_keywords(df, lifestyle_keywords)

# Construir resumen
grouped = {
    "IDs/llaves": ids_present,
    "Sociodemográficas": sorted(set(soc_cols)),
    "Clínicas (DM2/HTA/glucosa)": sorted(set(clin_cols)),
    "Antropometría (peso/talla/imc/cintura)": sorted(set(antro_cols)),
    "Estilo de vida": sorted(set(life_cols)),
    "Otras (no clasificadas)": sorted(set(cols_all)
                                      - set(ids_present)
                                      - set(soc_cols)
                                      - set(clin_cols)
                                      - set(antro_cols)
                                      - set(life_cols)),
}

print("\n=== (2) AGRUPACIÓN POR TIPO ===")
for g, cols in grouped.items():
    print(f"\n[{g}]  ({len(cols)})")
    for c in cols:
        print(f"  - {c}")

# Guardar versión Markdown
lines = []
for g, cols in grouped.items():
    lines.append(f"## {g} ({len(cols)})")
    lines += [f"- {c}" for c in cols]
    lines.append("")
(out_md := BASE / "columnas_agrupadas.md").write_text("\n".join(lines), encoding="utf-8")
print(f"\n[OK] Agrupación guardada en: {out_md}")

# Vista mínima para modelado (IDs + edad + sexo + algo de clínico + algo de antro si existe)
mini = sorted(set(ids_present + keep_existing(["edad", "Edad", "sexo", "Sexo"], df)
                  + clin_cols[:6] + antro_cols[:6]))
df_mini = df[mini] if mini else df
(out_csv := BASE / "ENSANUT_2023_vista_minima.csv").write_text(
    df_mini.to_csv(index=False), encoding="utf-8"
)
print(f"[OK] Vista mínima guardada en: {out_csv}  (cols={len(df_mini.columns)})")
