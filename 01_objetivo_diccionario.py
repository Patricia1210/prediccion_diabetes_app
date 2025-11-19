# 01_objetivo_diccionario.py
from pathlib import Path
import pandas as pd

BASE = Path(__file__).parent
CATALOGO = BASE / "adultos_ensanut2023_w_n.Catlogo.xlsx"
CSV = BASE / "adultos_ensanut2023_w_n.csv"

# ---------- utilidades ----------
def find_header_row(xlsx_path: Path, sheet_name: str, required=("Variable","Valor"), max_scan=60):
    """Devuelve el índice de la fila que contiene todos los 'required'."""
    tmp = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=None, dtype=str)
    for i in range(min(max_scan, len(tmp))):
        row = tmp.iloc[i].astype(str).str.strip().str.lower().tolist()
        if all(r.lower() in row for r in required):
            return i
    return None

def guess_sep(csv_path: Path):
    with open(csv_path, "r", encoding="latin-1") as f:
        head = f.readline()
    return ";" if head.count(";") > head.count(",") else ","

# ---------- 1) leer hoja 'Valores' ----------
hdr_vals = find_header_row(CATALOGO, "Valores", required=("Variable","Valor"))
if hdr_vals is None:
    raise RuntimeError(
        "No encontré la fila de encabezado en 'Valores'. "
        "Abre el Excel y confirma que existan las columnas 'Variable' y 'Valor'."
    )
vals = pd.read_excel(CATALOGO, sheet_name="Valores", header=hdr_vals, dtype=str)
print(f"[Valores] Usé la fila {hdr_vals} como encabezado.")
# normaliza encabezados y localiza la columna de etiqueta
vals.columns = [str(c).strip() for c in vals.columns]
etq_col = next((c for c in vals.columns if "etiqueta" in c.lower()), None)
if etq_col is None:
    raise RuntimeError("En 'Valores' no encuentro la columna de etiqueta (p. ej. 'Etiqueta' o 'Etiqueta de valor').")
vals = vals[["Variable", "Valor", etq_col]].rename(columns={etq_col: "Etiqueta"})
vals = vals.dropna(subset=["Variable"]).fillna("")

# ---------- 2) leer hoja 'Variables' ----------
hdr_vars = find_header_row(CATALOGO, "Variables", required=("Variable","Etiqueta"))
if hdr_vars is None:
    # a veces la hoja se llama distinto; intentamos solo 'Variable'
    hdr_vars = find_header_row(CATALOGO, "Variables", required=("Variable",))
vars_df = pd.read_excel(CATALOGO, sheet_name="Variables", header=hdr_vars, dtype=str)
print(f"[Variables] Usé la fila {hdr_vars} como encabezado.")
vars_df.columns = [str(c).strip() for c in vars_df.columns]
vars_df = vars_df[[c for c in vars_df.columns if c.lower() in ("variable","etiqueta")]]
vars_df = vars_df.rename(columns={"Variable":"Variable","Etiqueta":"Etiqueta"}).dropna(subset=["Variable"])

# ---------- 3) construir diccionario de códigos ----------
# dict: variable -> {codigo: etiqueta}
codebook = (
    vals.groupby("Variable")
        .apply(lambda g: dict(zip(g["Valor"].astype(str), g["Etiqueta"].astype(str))))
        .to_dict()
)

# ---------- 4) guardar diccionario largo y resumen ----------
out_dic_long = BASE / "ENSANUT_diccionario_valores.csv"
vals.to_csv(out_dic_long, index=False, encoding="utf-8")
print(f"Diccionario (largo) guardado en: {out_dic_long}")

out_vars = BASE / "ENSANUT_variables.csv"
vars_df.to_csv(out_vars, index=False, encoding="utf-8")
print(f"Variables+etiquetas guardado en: {out_vars}")

# ---------- 5) (opcional) decodificar una muestra del CSV ----------
sep = guess_sep(CSV)
df = pd.read_csv(CSV, sep=sep, low_memory=False, dtype=str)

# selecciona unas cuantas variables para mostrar (ajusta a tu gusto)
candidatas = []
for key in ("diab","gluc","a1c","hba1c","imc","cint","pres","hipert","taba","edad","sexo","peso","talla","actividad"):
    hits = vars_df[vars_df["Etiqueta"].str.lower().str.contains(key, na=False)]["Variable"].tolist()
    candidatas.extend(hits)
candidatas = [c for c in candidatas if c in df.columns][:30]  # primeras 30 para muestra

def decode_column(col):
    s = df[col]
    if col in codebook:
        return s.astype(str).map(codebook[col]).fillna(s)
    return s

preview = pd.DataFrame({col: decode_column(col) for col in candidatas}).head(30)
out_preview = BASE / "ENSANUT_preview_decodificado.csv"
preview.to_csv(out_preview, index=False, encoding="utf-8")
print(f"Preview decodificado (30 filas) guardado en: {out_preview}")
