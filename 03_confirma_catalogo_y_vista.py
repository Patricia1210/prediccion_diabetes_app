# 03_confirma_catalogo_y_vista.py
from pathlib import Path
import pandas as pd
import re

BASE = Path(__file__).parent

# Archivos que ya tienes en la carpeta
MERGED = BASE / "ENSANUT_2023_Adultos_Salud+Antro_merged.csv"
CAT_SALUD = BASE / "adultos_ensanut2023_w_n.Catlogo.xlsx"
CAT_ANTRO = BASE / "adultos_ensanut2023_w_n.Catlogo.xlsx"  # si tu catálogo de antro es otro, cambia aquí
# (si tienes "Antropometria_HTA_4mar24.Catlogo.xlsx", usa ese nombre)
if (BASE / "Antropometria_HTA_4mar24.Catlogo.xlsx").exists():
    CAT_ANTRO = BASE / "Antropometria_HTA_4mar24.Catlogo.xlsx"

OUT_MIN = BASE / "ENSANUT_2023_vista_minima.csv"

def load_catalog(path: Path) -> pd.DataFrame:
    # Catálogos de ENSANUT suelen tener columnas tipo: Variable / Etiqueta / Valor (a veces)
    # Probamos varias hojas comunes y nos quedamos con la que contenga "Variable"
    xls = pd.ExcelFile(path)
    for sheet in xls.sheet_names:
        df = pd.read_excel(path, sheet_name=sheet)
        cols = [c.strip() for c in df.columns.astype(str)]
        df.columns = cols
        if any(c.lower() in ("variable","nombre","var","campo") for c in cols):
            return df
    # fallback: primera hoja
    return pd.read_excel(path)

def search_vars(cat: pd.DataFrame, keywords, label_cols=("Etiqueta","Label","Descripción","Descripcion","Definition","Definición")):
    """Devuelve pares (Variable, Etiqueta) que coincidan con keywords en cualquiera de las columnas."""
    df = cat.copy()
    df.columns = [str(c) for c in df.columns]
    patt = re.compile("|".join([re.escape(k) for k in keywords]), re.I)

    mask = pd.Series(False, index=df.index)
    for c in df.columns:
        mask = mask | df[c].astype(str).str.contains(patt, regex=True, na=False)

    res = df.loc[mask].copy()

    # elegimos columna de etiqueta disponible
    label_col = None
    for c in label_cols:
        if c in res.columns:
            label_col = c
            break
    if label_col is None:
        label_col = res.columns[1] if res.shape[1] > 1 else res.columns[0]

    # intentamos deducir la columna "Variable"
    var_col = None
    for name in ("Variable","variable","Nombre","Var","Campo","Name"):
        if name in res.columns:
            var_col = name
            break
    if var_col is None:
        var_col = res.columns[0]

    out = res[[var_col, label_col]].dropna().copy()
    out.columns = ["Variable","Etiqueta"]
    out = out.sort_values("Variable")
    return out

def first_present(df: pd.DataFrame, *cands):
    for c in cands:
        if c in df.columns:
            return c
    return None

# 1) Cargar bases y catálogos
merged = pd.read_csv(MERGED, low_memory=False)
print(f"[INFO] Base fusionada: {merged.shape}")

cat_salud = load_catalog(CAT_SALUD)
print(f"[INFO] Catálogo SALUD: {CAT_SALUD.name} (hojas detectadas)")

cat_antro = load_catalog(CAT_ANTRO)
print(f"[INFO] Catálogo ANTRO: {CAT_ANTRO.name}")

# 2) Buscar variables clave en catálogos
bloques = {
    "DIABETES": ["diab", "diabetes"],
    "GLUCOSA": ["gluc","glucosa","glucose","a1c","hb"],
    "HIPERTENSION": ["hipert","hta","tensión","tension","presión","presion","sistol","diast"],
    "ANTROPOMETRÍA": ["peso","talla","estatura","cintur","imc"]
}

for titulo, keys in bloques.items():
    print("\n" + "="*70)
    print(f"{titulo} — SALUD")
    print(search_vars(cat_salud, keys).head(30).to_string(index=False))
    print(f"\n{titulo} — ANTRO")
    print(search_vars(cat_antro, keys).head(30).to_string(index=False))

# 3) Construir vista mínima de modelado
cols = list(merged.columns)

# IDs
id_col  = first_present(merged, "FOLIO_I", "FOLIO_INT", "Ã¯Â»Â¿FOLIO_INT")
ent_col = first_present(merged, "entidad","desc_ent","desc_ent_antro")

# Demografía
edad_col = first_present(merged, "edad","aedad","a09132edad","a09132ed")
sexo_col = first_present(merged, "sexo","asexo","a09131ed","a09131bd")

# Antropometría mínima
imc_col = first_present(merged, "imc")

# Etiqueta candidata de DM (prioridad)
y_col = first_present(merged, "a1006hbe","a1006hb")  # ajustaremos tras revisar catálogos

vista_cols = [c for c in [id_col, ent_col, edad_col, sexo_col, imc_col, y_col] if c]
vista = merged[vista_cols].copy()

# Renombrar a nombres amigables
rename_map = {}
if id_col:  rename_map[id_col]  = "id"
if ent_col: rename_map[ent_col] = "entidad"
if edad_col: rename_map[edad_col] = "edad"
if sexo_col: rename_map[sexo_col] = "sexo"
if imc_col: rename_map[imc_col] = "imc"
if y_col: rename_map[y_col] = "dm_label"

vista = vista.rename(columns=rename_map)

# 4) Reporte rápido de disponibilidad y distribución
def disponibilidad(df):
    n = len(df)
    return (1.0 - df.isna().mean()).round(4).sort_values(ascending=False)

print("\n" + "="*70)
print("[VISTA MÍNIMA] Columnas y disponibilidad:")
print(disponibilidad(vista).to_string())

if "dm_label" in vista.columns:
    print("\n[dm_label] distribución (valor -> conteo):")
    print(vista["dm_label"].value_counts(dropna=False).to_string())
else:
    print("\n[AVISO] No se detectó dm_label. Revisa qué variable de DM corresponde en el catálogo y ajusta y_col.")

vista.to_csv(OUT_MIN, index=False, encoding="utf-8")
print(f"\n[OK] Vista mínima guardada en: {OUT_MIN}")
