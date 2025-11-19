# 08_ingenieria_caracteristicas.py
# Ingeniería de características sobre PASO 7

from pathlib import Path
import numpy as np
import pandas as pd

# ================== 0) RUTAS ==================
BASE = Path(__file__).parent

DATASET_IN = BASE / "PASO7_dataset_modelo_ml.csv"
DATASET_OUT = BASE / "PASO8_dataset_ml_features.csv"

TARGET_COL = "diabetes_dx"


# ================== 1) HELPERS ==================
def to_numeric_robusto(s: pd.Series) -> pd.Series:
    """
    Convierte una serie a numérico manejando:
    - Cadenas con comas como separador decimal o errores: '60,8568,7'
    - Cadenas vacías o 'NaN'
    - Mezcla de texto y números

    Devuelve float con NaN donde no se pueda convertir.
    """
    # Si ya es numérica, no hacemos nada
    if pd.api.types.is_numeric_dtype(s):
        return s

    s2 = (
        s.astype(str)
         .str.strip()
    )

    # Reemplazar vacíos y variantes de NaN por NaN real
    s2 = s2.replace(
        {"": np.nan, "nan": np.nan, "NaN": np.nan, "None": np.nan},
        regex=False
    )

    # Unificamos separador decimal: coma -> punto
    s2 = s2.str.replace(",", ".", regex=False)

    # Si quedaron cosas tipo "60.8568.7", extraemos SOLO el primer número
    # patrón: opcional signo, dígitos, punto opcional, dígitos
    s2 = s2.str.extract(r"([-+]?\d*\.?\d+)")[0]

    # Conversión final
    return pd.to_numeric(s2, errors="coerce")


def crear_columna_promedio(df: pd.DataFrame, cols, new_col: str) -> pd.DataFrame:
    """
    Crea una columna como promedio fila a fila de 'cols',
    usando conversión robusta a numérico.
    """
    presentes = [c for c in cols if c in df.columns]

    if not presentes:
        print(f"[WARN] No se encontraron columnas {cols} para crear '{new_col}'.")
        df[new_col] = np.nan
        return df

    # Convertimos cada columna a numérico robusto
    sub = df[presentes].apply(to_numeric_robusto)

    df[new_col] = sub.mean(axis=1, skipna=True)
    print(f"[INFO] Columna '{new_col}' creada a partir de {presentes}")
    return df


# ================== 2) SCRIPT PRINCIPAL ==================
def main():
    # ---- 2.1 Leer dataset de PASO 7 ----
    with open(DATASET_IN, "r", encoding="latin-1") as f:
        head = f.readline()
    sep = ";" if head.count(";") > head.count(",") else ","

    print(f"[INFO] Leyendo dataset PASO 7 con separador '{sep}'...")
    df = pd.read_csv(DATASET_IN, sep=sep, low_memory=False, encoding="latin-1")
    print(f"[INFO] Dataset PASO 7: {df.shape[0]} filas x {df.shape[1]} columnas")

    # ---- 2.2 Edad en años a partir de 'meses' ----
    if "meses" in df.columns:
        df["edad_anios"] = to_numeric_robusto(df["meses"]) / 12.0
        print("[INFO] Columna 'edad_anios' creada a partir de 'meses'.")
    else:
        df["edad_anios"] = np.nan
        print("[WARN] No se encontró la columna 'meses'; 'edad_anios' se llena con NaN.")

    # ---- 2.3 Promedios antropométricos ----
    df = crear_columna_promedio(df, ["an01_1", "an01_2"], "peso_prom_kg")
    df = crear_columna_promedio(df, ["an04_1", "an04_2"], "talla_prom_cm")
    df = crear_columna_promedio(df, ["an08_1", "an08_2"], "cintura_prom_cm")

    # ---- 2.4 IMC recalculado a partir de peso/talla ----
    if {"peso_prom_kg", "talla_prom_cm"}.issubset(df.columns):
        peso = to_numeric_robusto(df["peso_prom_kg"])
        talla_m = to_numeric_robusto(df["talla_prom_cm"]) / 100.0

        df["imc_calc"] = peso / (talla_m ** 2)
        print("[INFO] Columna 'imc_calc' (IMC calculado) creada.")
    else:
        df["imc_calc"] = np.nan
        print("[WARN] No se pudieron crear 'imc_calc' por falta de peso/talla.")

    # ---- 2.5 Pérdida / ganancia de peso en 12 meses ----
    # a0107: perdió / ganó / igual
    # a0108: ¿cuántos kilos ganó/perdió?
    if {"a0107", "a0108"}.issubset(df.columns):
        cambio_tipo = to_numeric_robusto(df["a0107"])
        cambio_kg = to_numeric_robusto(df["a0108"])

        df["perdida_peso_kg_12m"] = np.where(cambio_tipo == 1, cambio_kg, 0.0)
        df["ganancia_peso_kg_12m"] = np.where(cambio_tipo == 2, cambio_kg, 0.0)

        print("[INFO] Columnas 'perdida_peso_kg_12m' y 'ganancia_peso_kg_12m' creadas.")
    else:
        df["perdida_peso_kg_12m"] = np.nan
        df["ganancia_peso_kg_12m"] = np.nan
        print("[WARN] No se encontraron 'a0107'/'a0108'; columnas de cambio de peso se llenan con NaN.")

    # ---- 2.6 Pérdida intencional / no intencional ----
    # a0109: “¿La pérdida de peso fue intencional?”
    if {"a0107", "a0109"}.issubset(df.columns):
        cambio_tipo = to_numeric_robusto(df["a0107"])
        intencional = to_numeric_robusto(df["a0109"])

        # Solo tiene sentido preguntar si hubo pérdida (cambio_tipo == 1)
        cond_perdio = cambio_tipo == 1

        df["perdida_peso_intencional"] = np.where(
            cond_perdio & (intencional == 1), 1,
            np.where(cond_perdio & (intencional == 2), 0, np.nan)
        )

        df["perdida_peso_no_intencional"] = np.where(
            df["perdida_peso_intencional"] == 0, 1,
            np.where(df["perdida_peso_intencional"] == 1, 0, np.nan)
        )

        print("[INFO] Columnas 'perdida_peso_intencional' y 'perdida_peso_no_intencional' creadas.")
    else:
        df["perdida_peso_intencional"] = np.nan
        df["perdida_peso_no_intencional"] = np.nan
        print("[WARN] No se encontraron 'a0107'/'a0109'; columnas de intencionalidad se llenan con NaN.")

    # ---- 2.7 (Espacio para más features después) ----
    # Aquí más adelante podemos crear dummies, categorías de IMC, etc.

    # ---- 2.8 Guardar resultado ----
    df.to_csv(DATASET_OUT, index=False, encoding="utf-8")
    print(f"[DONE] Dataset con ingeniería de características guardado en: {DATASET_OUT}")
    print(f"[INFO] Dimensiones finales: {df.shape[0]} filas x {df.shape[1]} columnas")


if __name__ == "__main__":
    main()
