# =========================
# STREAMLIT APP COMPLETO
# =========================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import math

from PIL import Image
# ── MODELADO | Setup ──────────────────────────────────────────────────────────
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_auc_score, accuracy_score)
from utils.load_data import *
# st.set_page_config(layout="wide")
def show():
    df = load_data()

    st.markdown("""
    ####  Mapa de calor geográfico de la sección: 
    #####  1.6 Índice de Riesgo por Frecuencia Geográfica
    """) 
    def map_mutcd_to_severity(mutcd_category):
        if mutcd_category == 'Menor':
            return 1
        elif mutcd_category == 'Intermedio':
            return 2
        elif mutcd_category == 'Mayor':
            return 3
        return None
    # Función reutilizable de clasificación de riesgo por cuartiles
    def clasificar_ocurrencia(serie):
        p25 = serie.quantile(0.25)
        p50 = serie.quantile(0.50)
        p75 = serie.quantile(0.75)
        def nivel(val):
            if val <= p25:   return 'Bajo'
            elif val <= p50: return 'Moderado'
            elif val <= p75: return 'Alto'
            else:            return 'Crítico'
        return serie.apply(nivel)

    # Función para clasificar según tiempo real (no por severidad)
    def clasificar_mutcd_por_tiempo(duracion):
        """Clasificación MUTCD basada en duración real del evento"""
        if pd.isna(duracion):
            return 'Menor'  # Asignar por defecto si no hay duración
        elif duracion < 30:
            return 'Menor'
        elif duracion <= 120:  # 2 horas = 120 minutos
            return 'Intermedio'
        else:
            return 'Mayor'
    # Calculamos la duración de cada accidente desde Start_Time y End_Time
    df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
    df['End_Time'] = pd.to_datetime(df['End_Time'], errors='coerce')

    df['Duration_min'] = (df['End_Time'] - df['Start_Time']).dt.total_seconds() / 60        

    # Consideramos válidas duraciones entre 1 minuto y 24 horas
    duration_clean = df[
        (df['Duration_min'] >= 1) &
        (df['Duration_min'] <= 1440)
    ].copy() # Explicitly create a copy to avoid SettingWithCopyWarning

    # Crear MUTCD_Category basada en duración real, no en severidad
    df['MUTCD_Category'] = df['Duration_min'].apply(clasificar_mutcd_por_tiempo)

    # Crear MUTCD_Category usando .loc para evitar SettingWithCopyWarning
    duration_clean.loc[:, 'MUTCD_Category'] = duration_clean['Duration_min'].apply(clasificar_mutcd_por_tiempo)    
    
    df['MUTCD_Severity'] = (
        df['MUTCD_Category'].apply(map_mutcd_to_severity)
    )    
    # Riesgo por coordenadas (~1 km²)
    df['lat_round'] = df['Start_Lat'].round(2)
    df['lng_round'] = df['Start_Lng'].round(2)

    coord_risk = (
        df
        .groupby(['lat_round','lng_round'])
        .agg(ocurrencias=('MUTCD_Severity','count'), severidad_promedio_mutcd=('MUTCD_Severity','mean'))
        .reset_index()
        .sort_values('ocurrencias', ascending=False)
    )
    coord_risk['nivel_ocurrencia'] = clasificar_ocurrencia(coord_risk['ocurrencias'])

    # Mapa de calor geográfico
    top_coords = coord_risk[coord_risk['ocurrencias'] >= coord_risk['ocurrencias'].quantile(0.90)]

    fig6 = px.density_mapbox(
        top_coords, lat='lat_round', lon='lng_round', z='ocurrencias',
        radius=8, center={'lat': 27.5, 'lon': -81.5}, zoom=5,
        mapbox_style='carto-positron', color_continuous_scale='YlOrRd',
        title='Mapa de Calor — Zonas de Mayor Ocurrencia por Frecuencia (Florida)'
    )
    fig6.update_layout(height=600, width=1100, font=dict(size=13))
    st.plotly_chart(fig6, use_container_width=True)
    st.markdown("""
    Los accidentes no están distribuidos de forma uniforme, sino altamente concentrados en zonas específicas, lo que permite identificar puntos críticos de intervención.

    - Los niveles se clasifican según frecuencia: bajo (1), moderado (2–4), alto (5–13) y crítico (>14 accidentes)
    - Predomina el nivel moderado y bajo, aunque existe una cantidad significativa de casos críticos.
    - Se identifican puntos geográficos críticos con alta concentración de accidentes (hasta 3,800 eventos)
    - A nivel administrativo, Miami-Dade y Orange presentan la mayor cantidad de incidentes.

    ---
    """)
