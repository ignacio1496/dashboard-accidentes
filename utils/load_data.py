# import pandas as pd
# import streamlit as st

# @st.cache_data
# def load_data():
#     df = pd.read_csv(r"data/US_Accidents_FL.csv")
#     return df

import pandas as pd
# import streamlit as st
import os
import glob

# 🔹 Función para dividir CSV
def dividir_csv_por_tamano(input_file, output_prefix, max_size_mb=30):
    max_size = max_size_mb * 1024 * 1024
    
    with open(input_file, "r", encoding="utf-8") as f:
        header = f.readline()
        
        part_num = 0
        current_size = 0
        
        out_path = f"{output_prefix}_{part_num:03d}.csv"
        out_file = open(out_path, "w", encoding="utf-8")
        out_file.write(header)

        for line in f:
            line_size = len(line.encode("utf-8"))
            
            if current_size + line_size > max_size:
                out_file.close()
                part_num += 1
                out_path = f"{output_prefix}_{part_num:03d}.csv"
                out_file = open(out_path, "w", encoding="utf-8")
                out_file.write(header)
                current_size = 0

            out_file.write(line)
            current_size += line_size

        out_file.close()

# 🔹 Función principal (Streamlit)
# @st.cache_data
def load_data():
    archivo_original = "data/US_Accidents_FL.csv"
    prefijo = "data/US_Accidents_part"

    # Buscar partes existentes
    partes = sorted(glob.glob(f"{prefijo}_*.csv"))
    print(partes)
    # 🔥 Si no existen → dividir
    if not partes:
        dividir_csv_por_tamano(archivo_original, prefijo, max_size_mb=30)
        partes = sorted(glob.glob(f"{prefijo}_*.csv"))

    # 🔹 Cargar y unir
    dfs = []
    for p in partes:
        df_part = pd.read_csv(p)
        dfs.append(df_part)

    df = pd.concat(dfs, ignore_index=True)

    return df

# df3 = load_data()
# print(df3)
