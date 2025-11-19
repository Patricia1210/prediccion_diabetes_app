import pandas as pd

# 1. Cargar las bases ENSANUT (ajusta la ruta si es necesario)
adultos = pd.read_csv(
    "adultos_ensanut_Base_original.csv",
    sep=";",
    encoding="latin-1",
    low_memory=False,
    decimal=","
)

antro = pd.read_csv(
    "Antropometria_Base_Original.csv",
    sep=";",
    encoding="latin-1",
    low_memory=False,
    decimal=","
)

# 2. Limpiar el BOM en los nombres de columnas (especialmente FOLIO_INT)
adultos.rename(columns=lambda c: c.replace("ï»¿", ""), inplace=True)
antro.rename(columns=lambda c: c.replace("ï»¿", ""), inplace=True)

# 3. Definir la llave de fusión (única por persona en ambas bases)
key = "FOLIO_INT"

# Revisar que realmente sea única
print("Duplicados en adultos por FOLIO_INT:", adultos[key].duplicated().sum())
print("Duplicados en antro   por FOLIO_INT:", antro[key].duplicated().sum())

# 4. Evitar columnas duplicadas (mismas variables en las dos bases)
#    Nos quedamos con TODAS las columnas de ADULTOS
#    y sólo agregamos de ANTRO las que no existan en ADULTOS
common = set(adultos.columns).intersection(antro.columns)
common_minus_key = common - {key}

print("Columnas en común (además de la llave):", len(common_minus_key))

# Nos quedamos de antro sólo las columnas EXTRAS que no están en adultos
antro_extra = antro.drop(columns=list(common_minus_key))

print("Columnas originales adultos:", adultos.shape[1])
print("Columnas originales antro:  ", antro.shape[1])
print("Columnas antro extra:       ", antro_extra.shape[1])

# 5. Fusionar (left join: conserva todos los adultos)
ensanut_merged = adultos.merge(
    antro_extra,
    on=key,
    how="left"
)

print("Filas adultos:        ", adultos.shape[0])
print("Filas antro:          ", antro.shape[0])
print("Filas fusionadas:     ", ensanut_merged.shape[0])
print("Columnas fusionadas:  ", ensanut_merged.shape[1])

# 6. Guardar la base fusionada
ensanut_merged.to_csv(
    "ENSANUT2023_adultos_con_antropometria.csv",
    sep=";",
    index=False
)

print("Listo. Archivo guardado como 'ENSANUT2023_adultos_con_antropometria.csv'")
