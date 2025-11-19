# Exploración.py
from pathlib import Path
import pandas as pd

# ---------- Rutas (relativas a este script) ----------
BASE = Path(__file__).parent
CSV_PATH = BASE / "adultos_ensanut2023_w_n.csv"
CATALOG_PATH = BASE / "adultos_ensanut2023_w_n.Catlogo.xlsx"  # usa el nombre exacto que ves en VS Code

# ---------- Utilidades ----------
def guess_sep(csv_path: Path) -> str:
    with open(csv_path, "r", encoding="latin-1") as f:
        head = f.readline()
    return ";" if head.count(";") > head.count(",") else ","

def read_catalog_variables(xlsx_path: Path) -> pd.DataFrame:
    xls = pd.ExcelFile(xlsx_path)
    # La hoja correcta se llama "Variables"; tiene encabezado a partir de ~fila 16
    for skip in range(10, 30):
        df = pd.read_excel(xls, sheet_name="Variables", skiprows=skip)
        cols = [str(c).strip().lower() for c in df.columns]
        if "variable" in cols and "etiqueta" in cols:
            df.columns = [c.strip() for c in df.columns]
            return df[["Variable", "Etiqueta"]].dropna(how="all")
    raise RuntimeError("No pude encontrar encabezados en la hoja 'Variables' del catálogo.")

# ---------- MAIN ----------
def main():
    # 1) Cargar catálogo (nombres amigables)
    cat_vars = read_catalog_variables(CATALOG_PATH)

    # 2) Palabras clave para diabetes (ajústalas si quieres)
    keywords = [
        "diab", "gluc", "a1c", "hba1c", "azúcar", "azucar", "medic",
        "peso", "talla", "imc", "cint", "cintura",
        "presión", "presion", "hipert", "tas", "tad",
        "taba", "fuma", "cigar",
        "alcohol",
        "actividad", "ejercicio", "sedentar",
        "edad", "sexo", "escolaridad"
    ]

    cat_vars["Etiqueta_lc"] = cat_vars["Etiqueta"].astype(str).str.lower()
    mask = cat_vars["Etiqueta_lc"].apply(lambda x: any(k in x for k in keywords))
    vars_candidatas = cat_vars.loc[mask, ["Variable", "Etiqueta"]].dropna()

    # 3) Detectar separador y cargar CSV
    sep = guess_sep(CSV_PATH)
    df = pd.read_csv(CSV_PATH, sep=sep, low_memory=False, encoding="latin-1")

    # 4) Intersección con columnas reales
    cols_exist = [v for v in vars_candidatas["Variable"] if v in df.columns]
    diccionario_util = vars_candidatas[vars_candidatas["Variable"].isin(cols_exist)].copy()

    # 5) Renombrar para un preview legible
    rename_map = {row.Variable: row.Etiqueta for _, row in diccionario_util.iterrows()}
    df_preview = df[cols_exist].rename(columns=rename_map).head(30)

    # 6) Guardar salidas
    out_dic = BASE / "ENSANUT_variables_para_diabetes.csv"
    out_prev = BASE / "ENSANUT_preview_diabetes.csv"
    diccionario_util.to_csv(out_dic, index=False, encoding="utf-8")
    df_preview.to_csv(out_prev, index=False, encoding="utf-8")

    print(f"Separador detectado: '{sep}'")
    print(f"Variables candidatas encontradas en el CSV: {len(cols_exist)}")
    print(f"Diccionario guardado en: {out_dic}")
    print(f"Preview (30 filas) guardado en: {out_prev}")
    print("\nEjemplos de variables detectadas:")
    print(diccionario_util.head(15).to_string(index=False))

if __name__ == "__main__":
    main()
