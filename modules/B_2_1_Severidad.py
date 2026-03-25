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
    ###  B. Subsección: Severidad promedio de accidentes en Florida según variables
    ####  2.1 Análisis de Severidad 
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
    # ── Datos ─────────────────────────────────────────────────────────────────────
    severity_hour = (df
                    .groupby('Hour')['MUTCD_Severity']
                    .mean().reset_index(name='Severidad_Promedio'))

    severity_weather = (df
                        .groupby('Weather_Condition')
                        .agg(
                            Severidad_Promedio=('MUTCD_Severity','mean'),
                            Total_Accidentes=('MUTCD_Severity','count')
                        )
                        .reset_index()
                        .query('Total_Accidentes >= 100')
                        .sort_values('Severidad_Promedio', ascending=False)
                        .head(15))

    severity_county = (df
                    .groupby('County')
                    .agg(
                        Severidad_Promedio=('MUTCD_Severity','mean'),
                        Total_Accidentes=('MUTCD_Severity','count')
                    )
                    .reset_index()
                    .query('Total_Accidentes >= 50')
                    .sort_values('Severidad_Promedio', ascending=False)
                    .head(20))

    # ── Subplots: 1 fila, 3 columnas ─────────────────────────────────────────────
    fig7 = make_subplots(
        rows=1, cols=3,
        subplot_titles=(
            'Severidad por Hora del Día',
            'Top 15 Condiciones Climáticas',
            'Top 20 Condados'
        ),
        horizontal_spacing=0.12
    )

    # ── Gráfica 1: Line chart por hora ───────────────────────────────────────────
    fig7.add_trace(
        go.Scatter(
            x=severity_hour['Hour'],
            y=severity_hour['Severidad_Promedio'],
            mode='lines+markers',
            line=dict(color='#e63946', width=2),
            marker=dict(size=6),
            name='Por Hora'
        ),
        row=1, col=1
    )

    # ── Gráfica 2: Bar horizontal por clima ──────────────────────────────────────
    fig7.add_trace(
        go.Bar(
            x=severity_weather['Severidad_Promedio'],
            y=severity_weather['Weather_Condition'],
            orientation='h',
            text=severity_weather['Severidad_Promedio'].round(2),
            textposition='outside',
            marker=dict(
                color=severity_weather['Severidad_Promedio'],
                colorscale='YlOrRd',
                showscale=False
            ),
            name='Por Clima'
        ),
        row=1, col=2
    )

    # ── Gráfica 3: Bar vertical por condado ──────────────────────────────────────
    fig7.add_trace(
        go.Bar(
            x=severity_county['County'],
            y=severity_county['Severidad_Promedio'],
            text=severity_county['Severidad_Promedio'].round(2),
            textposition='outside',
            marker=dict(
                color=severity_county['Severidad_Promedio'],
                colorscale='YlOrRd',
                showscale=False
            ),
            name='Por Condado'
        ),
        row=1, col=3
    )

    # ── Ejes ──────────────────────────────────────────────────────────────────────
    fig7.update_xaxes(title_text='Hora',            tickangle=90,
                    row=1, col=1)
    fig7.update_xaxes(title_text='Severidad Prom.', tickangle=90,
                    title_standoff=15, automargin=True, row=1, col=2)
    fig7.update_xaxes(title_text='Condado',         tickangle=90,
                    row=1, col=3)

    fig7.update_yaxes(title_text='Severidad Promedio', row=1, col=1)
    fig7.update_yaxes(title_text='',                   row=1, col=2)
    fig7.update_yaxes(title_text='Severidad Promedio', row=1, col=3)

    # ── Layout ────────────────────────────────────────────────────────────────────
    fig7.update_layout(
        height=500,
        width=1800,
        template='seaborn',
        font=dict(size=11),
        title_text='Análisis de Severidad — Florida',
        title_x=0.5,
        showlegend=False
    )

    st.plotly_chart(fig7, use_container_width=True)

    st.markdown("""
    -	Variable temporal: Mayor severidad en la madrugada, mediodía y medianoche. 
    -	Factores: fatiga, dismunicion de atención, alcohol, mayor cantidad de autos.
    -	Condiciones Climaticas: Mayor severidad en escenarios con viento & nubosidad/lluvia ligera
    -	Factores: baja visibilidad, menor adherencia, menor estabilidad del auto.
    -	Variable geografica: Variabilidad moderada entre condados. Se destaca Miami- Dade, Orange y Sarasota.
    -	La severidad no es aleatoria: depende del tiempo, el clima y la ubicación
    ---
    """)
