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
    # =========================
    # MUTCD
    # =========================
    # Construir etiquetas dinámicas con rangos reales
    def formato_tiempo(minutos):
        minutos = round(minutos)  # ← redondear primero
        h = int(minutos // 60)
        m = int(minutos % 60)
        if h > 0:
            return f"{h}h {m:02d}m"
        return f"{m}m"
      
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
    # Crear MUTCD_Category basada en duración real, no en severidad
    df['MUTCD_Category'] = df['Duration_min'].apply(clasificar_mutcd_por_tiempo)
    resumen_mutcd = (df
        .groupby('MUTCD_Category')['Severity']
        .count()
        .reset_index(name='Registros')
    ) 

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

    color_ocurrencia  = {
        'Bajo':     '#a8d5a2',
        'Moderado': '#f6d860',
        'Alto':     '#f4a261',
        'Crítico':  '#e63946'
    } 

    # Consideramos válidas duraciones entre 1 minuto y 24 horas
    duration_clean = df[
        (df['Duration_min'] >= 1) &
        (df['Duration_min'] <= 1440)
    ].copy() # Explicitly create a copy to avoid SettingWithCopyWarning

    st.markdown("""
    ####   1.3 Calcula la mediana de duración por categoría MUTCD
    - Nivel de Severidad ajustado
    """) 
    # Crear MUTCD_Category usando .loc para evitar SettingWithCopyWarning
    duration_clean.loc[:, 'MUTCD_Category'] = duration_clean['Duration_min'].apply(clasificar_mutcd_por_tiempo)

    # Calcula la mediana de duración por categoría MUTCD
    duration_stats_mutcd = (
        duration_clean
        .groupby('MUTCD_Category')['Duration_min']
        .median()
        .reset_index()
        .rename(columns={'Duration_min': 'Mediana'})
    )

    # Visualización
    fig3 = px.bar(
        duration_stats_mutcd,
        x='MUTCD_Category', y='Mediana',
        color='MUTCD_Category',
        category_orders={'MUTCD_Category': ['Menor', 'Intermedio', 'Mayor']},  # <-- esto
        color_discrete_map={
            'Menor':      '#a8d5a2',
            'Intermedio': '#f6d860',
            'Mayor':      '#e63946'
        },
        text=duration_stats_mutcd['Mediana'].apply(formato_tiempo),
        template='plotly_white',
        labels={
            'MUTCD_Category': 'Categoría MUTCD',
            'Mediana': 'Mediana de Duración (min)'
        },
        title='Mediana de Duración por Categoría MUTCD — Florida'
    )
    fig3.update_layout(height=450, width=700, font=dict(size=13), showlegend=False)    

    st.plotly_chart(fig3, use_container_width=True)
    st.markdown("""
    - Se creó la variable categórica Duration_Category basada en criterios de la Federal Highway Administration (FHWA)
    - Clasificación: menor (< 30 min), intermedio (30 min – 2 h) y mayor (> 2 h)
    - Mejora la coherencia en la medición del impacto en el flujo vehicular
    ---
    """)  
              

