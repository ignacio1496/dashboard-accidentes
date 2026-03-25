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

    color_ocurrencia  = {
        'Bajo':     '#a8d5a2',
        'Moderado': '#f6d860',
        'Alto':     '#f4a261',
        'Crítico':  '#e63946'
    } 

    st.markdown("""
    ### Análisis de relación ocurrencia vs severidad por condado 
    ####  2.3 Análisis de Severidad  
    ##### Resumen integrado: Ocurrencia + Severidad por condado 
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

    # Resumen integrado: Ocurrencia + Severidad por condado
    resumen_county = (df
                    .groupby('County')
                    .agg(
                        Ocurrencias=('MUTCD_Severity','count'),
                        Severidad_Promedio=('MUTCD_Severity','mean')
                    )
                    .reset_index())

    resumen_county['Riesgo_Ocurrencia'] = clasificar_ocurrencia(resumen_county['Ocurrencias'])
    resumen_county['Nivel_Severidad']   = pd.cut(
        resumen_county['Severidad_Promedio'],
        bins=[0, 1.5, 2.0, 2.5, 4],
        labels=['Baja','Moderada','Alta','Crítica']
    )

    fig9 = px.scatter(resumen_county,
                    x='Ocurrencias', y='Severidad_Promedio',
                    color='Riesgo_Ocurrencia', size='Ocurrencias',
                    hover_name='County',
                    color_discrete_map=color_ocurrencia,
                    template='seaborn',
                    title='Relación Ocurrencia vs Severidad por Condado — Florida<br>'
                        '<sup>Cada punto es un condado. Tamaño = número de accidentes</sup>')
    fig9.update_layout(height=550, width=1000, font=dict(size=13))

    st.plotly_chart(fig9, use_container_width=True)

    st.markdown("""
    - No se observa una relación lineal fuerte entre ocurrencia y severidad
    - Condados con más accidentes tienden a severidad moderada–alta
    - Destaca Miami-Dade County con alto volumen y alta severidad (riesgo crítico)
    - La mayoría presenta menor volumen y severidad moderada/baja
    - El riesgo se concentra en regiones específicas
    ---
    """)   

