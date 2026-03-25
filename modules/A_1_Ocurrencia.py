import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import math
from utils.load_data import *

# st.set_page_config(layout="wide")
def show():
    # @st.cache_data
    df = load_data()
    # =========================
    # 1. CARGA
    # =========================
    # def load_data():
    #     df = pd.read_csv("data/US_Accidents_FL.csv")
    #     return df


    # =========================
    # 3. LIMPIEZA
    # =========================
    missing = df.isnull().sum()
    # st.subheader("Valores Nulos")
    # st.dataframe(missing[missing > 0])

    num_cols = df.select_dtypes(include=np.number).columns
    cat_cols = df.select_dtypes(include='object').columns

    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # st.write("Nulos restantes:", df.isnull().sum().sum())

    # =========================
    # 3.2 ÚNICOS
    # =========================
    unique_df = pd.DataFrame({
        'Columna': df.columns,
        'Unicos': [df[col].nunique() for col in df.columns]
    })

    # st.dataframe(unique_df)

    # =========================
    # 3.5 OUTLIERS IQR
    # =========================
    # st.subheader("Eliminación de Outliers (IQR)")

    clean_df = df.copy()

    for col in clean_df.select_dtypes(include=['float64','int64']).columns:
        Q1 = clean_df[col].quantile(0.25)
        Q3 = clean_df[col].quantile(0.75)
        IQR = Q3 - Q1

        clean_df = clean_df[
            (clean_df[col] >= Q1 - 1.5 * IQR) &
            (clean_df[col] <= Q3 + 1.5 * IQR)
        ]

    # st.write(f"Antes: {len(df):,}")
    # st.write(f"Después: {len(clean_df):,}")

    # =========================
    # 4.1 DISTRIBUCIÓN
    # =========================
    st.title("A. Análisis de Ocurrencia")

    st.markdown("""
    #### 1.1 Distribución de variables categóricas
    """)    
    # Distribución de variables categóricas (Top 10 por columna)
    dist_cat_columns = clean_df[[
        'Street', 'City', 'County',
        # 'State',  # comentado: dataset filtrado a Florida únicamente
        'Wind_Direction', 'Weather_Condition',
        'Sunrise_Sunset', 'Civil_Twilight',
        'Nautical_Twilight', 'Astronomical_Twilight'
    ]]

    all_frecuencias = {}
    for column in dist_cat_columns:
        top_10_freq = clean_df[column].str.strip().value_counts()[:10]
        all_frecuencias[column] = pd.DataFrame({
            'Columna': column,
            'Valor': top_10_freq.index,
            'Frecuencia': top_10_freq.values
        })

    frecuencias_df = pd.concat(all_frecuencias.values(), ignore_index=True)
    columnas = frecuencias_df['Columna'].unique()
    n_cols = 3
    n_rows = math.ceil(len(columnas) / n_cols)
    colores = px.colors.qualitative.Pastel

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=list(columnas),
        vertical_spacing=0.18,
        horizontal_spacing=0.08
    )

    for i, columna in enumerate(columnas):
        row = i // n_cols + 1
        col = i %  n_cols + 1

        df_col = frecuencias_df[frecuencias_df['Columna'] == columna].copy()
        ratio = df_col['Frecuencia'] / df_col['Frecuencia'].sum() * 100

        # Top 10 para columnas con muchos valores
        top_idx = ratio.nlargest(10).index
        df_col = df_col.loc[top_idx]
        ratio = ratio.loc[top_idx]

        # Color por label
        color_map = {val: colores[j % len(colores)] for j, val in enumerate(df_col['Valor'].astype(str))}
        marker_colors = df_col['Valor'].astype(str).map(color_map).tolist()

        fig.add_trace(
            go.Bar(
                x=df_col['Valor'].astype(str),
                y=ratio.round(1),
                text=ratio.round(1).astype(str) + '%',
                textposition='outside',
                marker_color=marker_colors,
                cliponaxis=True,
                showlegend=False
            ),
            row=row, col=col
        )

        fig.update_yaxes(
            title_text='Ratio (%)',
            range=[0, ratio.max() * 1.30],
            row=row, col=col
        )

    fig.update_xaxes(tickangle=90)

    fig.update_layout(
        height=420 * n_rows,
        width=1100,
        template='plotly_white',
        font=dict(size=11),
        title_text='Distribución de Variables',
        title_x=0.5
    )    
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    - Mayor concentración de accidentes en ciudades como Miami y Orlando, y en el condado Miami-Dade        
    - Vías principales concentran más incidentes → relación con alto flujo vehicular
    - Predominan condiciones climáticas favorables (Fair, Mostly Cloudy, Partly Cloudy)
    - Más del 75 % de los accidentes ocurren de día → asociado a mayor volumen de tráfico
    ---
    """)
