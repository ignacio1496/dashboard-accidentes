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
    st.markdown("""
    ####   2.2 Análisis de Severidad
    ##### Top 3 condados con mayor severidad promedio distribuidas por hora 
    """)  
    # Extraer variables temporales
    df['Start_Time'] = pd.to_datetime(
        df['Start_Time'], errors='coerce'
    )
    df['Hour']      = df['Start_Time'].dt.hour
    df['DayOfWeek'] = df['Start_Time'].dt.day_name()
    df['Month']     = df['Start_Time'].dt.month_name()
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
    # Top 3 condados con más ocurrencias
    top3_condados = (df
                    .groupby('County')
                    .agg(
                        Severidad_Promedio=('MUTCD_Severity', 'mean'),
                        Total_Accidentes=('MUTCD_Severity', 'count')
                    )
                    .query('Total_Accidentes >= 50')
                    .sort_values('Severidad_Promedio', ascending=False)
                    .head(3)
                    .index
                    .tolist())
    # Nuevo dataset: severidad promedio por hora y condado
    severity_hour_county = (df[df['County'].isin(top3_condados)]
                            .groupby(['County', 'Hour'])['MUTCD_Severity']
                            .mean()
                            .reset_index(name='Severidad_Promedio'))


    # Paleta de colores, uno por condado
    colores = ['#e63946', '#457b9d', '#2a9d8f']

    fig8 = make_subplots(
        rows=1, cols=3,
        subplot_titles=[f'{c} County' for c in top3_condados],
        horizontal_spacing=0.08
    )

    for i, condado in enumerate(top3_condados):
        df_condado = severity_hour_county[severity_hour_county['County'] == condado]

        fig8.add_trace(
            go.Scatter(
                x=df_condado['Hour'],
                y=df_condado['Severidad_Promedio'],
                mode='lines+markers',
                line=dict(color=colores[i], width=2),
                marker=dict(size=6),
                name=condado
            ),
            row=1, col=i + 1
        )

        fig8.update_xaxes(title_text='Hora del Día', tickmode='linear', dtick=1,
                        tickangle=0, row=1, col=i + 1)
        fig8.update_yaxes(title_text='Severidad Promedio', row=1, col=i + 1)

    fig8.update_layout(
        height=400, width=1800,
        template='plotly_white',
        font=dict(size=12),
        title_text='Severidad Promedio por Hora — Top 3 Condados',
        title_x=0.5
    )

    st.plotly_chart(fig8, use_container_width=True)

    st.markdown("""
    - Patrones consistentes entre condados
    - Picos en madrugada, mediodía y noche
    - Variaciones explicadas por entorno urbano y comportamiento
    ---
    """)              
