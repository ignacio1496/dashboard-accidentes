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
    ####   1.4 Análisis Temporal
    """) 
    # Extraer variables temporales
    df['Start_Time'] = pd.to_datetime(
        df['Start_Time'], errors='coerce'
    )
    df['Hour']      = df['Start_Time'].dt.hour
    df['DayOfWeek'] = df['Start_Time'].dt.day_name()
    df['Month']     = df['Start_Time'].dt.month_name()

    day_order   = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

    month_order = ['January','February','March','April','May','June',
                'July','August','September','October','November','December']

    # Prepara los datos
    hourly  = df.groupby('Hour').size().reset_index(name='Accidentes')

    daily   = (df.groupby('DayOfWeek').size()
            .reindex(day_order).reset_index(name='Accidentes'))

    monthly = (df.groupby('Month').size()
            .reindex(month_order).reset_index(name='Accidentes'))

    # Config de cada subplot
    plots = [
        {'df': hourly,   'x': 'Hour',      'title': 'Por Hora del Día'},
        {'df': daily,    'x': 'DayOfWeek', 'title': 'Por Día de la Semana'},
        {'df': monthly,  'x': 'Month',     'title': 'Por Mes'},
    ]

    fig4 = make_subplots(
        rows=1, cols=3,
        subplot_titles=[p['title'] for p in plots],
        horizontal_spacing=0.08
    )

    for i, p in enumerate(plots):
        col = i + 1
        accidentes = p['df']['Accidentes']

        # Escala de color YlOrRd mapeada a los valores
        min_val, max_val = accidentes.min(), accidentes.max()
        normed = (accidentes - min_val) / (max_val - min_val)
        import plotly.colors as pc
        colors = [pc.sample_colorscale('YlOrRd', v)[0] for v in normed]

        fig4.add_trace(
            go.Bar(
                x=p['df'][p['x']].astype(str),
                y=accidentes,
                marker_color=colors,
                showlegend=False
            ),
            row=1, col=col
        )
        fig4.update_yaxes(title_text='Accidentes', row=1, col=col)
        fig4.update_xaxes(title_text=p['x'], tickangle=45, row=1, col=col)
        fig4.update_xaxes(title_text=p['x'], tickangle=90, row=1, col=col)

    fig4.update_layout(
        height=450,
        width=1200,
        template='plotly_white',
        font=dict(size=12),
        title_text='Distribución Temporal de Accidentes — Florida',
        title_x=0.5
    )

    st.plotly_chart(fig4, use_container_width=True)

    st.markdown("""
    - Mayor frecuencia a las 8:00 hrs, los viernes y en diciembre
    - Asociado a horas pico, fin de semana laboral y mayor actividad vehicular de fin de año
    ---
    """)
