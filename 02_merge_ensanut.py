# 02_merge_ensanut.py
# Fusiona ENSANUT 2023: Salud de adultos (20+) + Antropometría/TA
# Calcula IMC si hay peso y talla y genera una vista para modelado DM2.

from pathlib import Path
import pandas as pd
import re

# ========= 0) RUTAS (ajusta SOLO si tus archivos tienen otro nombre) =========
BASE = Path(__file__).parent
SALUD_CSV = BASE / "adultos_ensanut2023_w_n.csv"
ANTRO_CSV = BASE / "Antropometria_HTA_4mar24.csv"
SALIDA_CSV = BASE / "ENSANUT_2023_Adultos_Salud+Antro_merged.csv"
VISTA_CSV  = BASE / "ENSANUT_2023_vista_modelo.csv"

# ========= Utilidades =========
def guess_sep(path: Path) -> str:
    with open(path, "r", encoding="latin-1") as f:
        head = f.readline()
    return ";" if head.count(";") > head.count(",") else ","

def ensure_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", "."), errors="coerce")

def find_by_keywords(columns, keywords):
    patt = re.compile("|".join([re.escape(k) for k in keywords]), re.I)
    return [c for c in columns if patt.search(c)]

# ========= 1) Cargar =========
sep_salud = guess_sep(SALUD_CSV)
sep_antro = guess_sep(ANTRO_CSV)
salud = pd.read_csv(SALUD_CSV, sep=sep_salud, low_memory=False, encoding="latin-1")
antro  = pd.read_csv(ANTRO_CSV,  sep=sep_antro, low_memory=False, encoding="latin-1")

print(f"[INFO] Salud: {salud.shape} (sep='{sep_salud}')")
print(f"[INFO] Antro: {antro.shape} (sep='{sep_antro}')")

# ========= 2) Claves (IDs) =========
keys_salud = [c for c in ["FOLIO_I","entidad"] if c in salud.columns]
keys_antro = [c for c in ["FOLIO_I","entidad"] if c in antro.columns]
if keys_salud != ["FOLIO_I","entidad"] or keys_antro != ["FOLIO_I","entidad"]:
    raise RuntimeError("No están las claves esperadas 'FOLIO_I' y 'entidad' en ambas bases.")

# ========= 3) Merge (left join desde SALUD) =========
merged = salud.merge(
    antro,
    left_on=keys_salud,
    right_on=keys_antro,
    how="left",
    suffixes=("", "_antro")
)
print(f"[INFO] Merge completo: {merged.shape}")
print(f"[INFO] Filas con match en ANTRO: {merged['FOLIO_I_antro'].notna().sum() if 'FOLIO_I_antro' in merged.columns else 'n/a'}")

# ========= 4) Detectar nombres de peso/talla (robusto) =========
# Buscamos primero candidatos exactos y luego por patrón
def pick_first(cols, patterns):
    # patterns: lista de listas de patrones (cada sublista equivale a un grupo)
    for pats in patterns:
        cand = find_by_keywords(cols, pats)
        if cand:
            return cand[0]
    return None

cols = list(merged.columns)
peso_col  = pick_first(cols, [["^peso$", "peso_kg", r"\bpeso\b"]])
talla_col = pick_first(cols, [["^talla$", "talla_cm", "estatura_cm", r"\btalla\b", r"\bestatura\b"]])
cint_col  = pick_first(cols, [["cintura", "circ_cintura", r"\bcint\b"]])

print(f"[INFO] peso_col: {peso_col} | talla_col: {talla_col} | cintura_col: {cint_col}")

# ========= 5) IMC (si hay peso+talla) =========
if peso_col and talla_col and (peso_col in merged.columns) and (talla_col in merged.columns):
    merged[peso_col]  = ensure_numeric(merged[peso_col])
    merged[talla_col] = ensure_numeric(merged[talla_col])

    talla_m = merged[talla_col].copy()
    # Si parece estar en cm (mediana > 3), convertimos a metros
    if talla_m.dropna().median() and talla_m.dropna().median() > 3:
        talla_m = talla_m / 100.0

    merged["imc_calc"] = merged[peso_col] / (talla_m ** 2)
    print("[OK] IMC calculado y añadido en 'imc_calc'.")
else:
    print("[WARN] No se detectaron columnas claras de peso y/o talla para IMC.")

# ========= 6) Variables de interés para vista de modelado =========
# Diabetes (ajusta la lista si confirmas otras en el catálogo de SALUD)
cands_diab_salud = [c for c in find_by_keywords(salud.columns, ["diab", "gluc", "a1c", "hb"])][:6]
edad_col  = next((c for c in salud.columns if re.search(r"edad", c, re.I)), None)
sexo_col  = next((c for c in salud.columns if re.search(r"sexo|género|genero", c, re.I)), None)
tas_col   = next((c for c in merged.columns if re.search(r"tas|sistol", c, re.I)), None)
tad_col   = next((c for c in merged.columns if re.search(r"tad|diast",  c, re.I)), None)

sel = list(dict.fromkeys(
    ["FOLIO_I","entidad", edad_col, sexo_col] +
    cands_diab_salud +
    [peso_col, talla_col, "imc_calc", cint_col, tas_col, tad_col]
))
sel = [c for c in sel if c and c in merged.columns]
vista = merged[sel].copy()

# ========= 7) Guardar =========
merged.to_csv(SALIDA_CSV, index=False, encoding="utf-8")
vista.to_csv(VISTA_CSV,  index=False, encoding="utf-8")

print(f"[OK] Archivo fusionado guardado en: {SALIDA_CSV}")
print(f"[OK] Vista para modelado guardada en: {VISTA_CSV}")
print(f"[INFO] Columnas en vista: {vista.columns.tolist()}")
