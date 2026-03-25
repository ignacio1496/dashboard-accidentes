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
    ####   1.5 Análisis Climático vs Ocurrencia
    """)    
    top_weather = (df['Weather_Condition']
                .value_counts().head(10).reset_index())
    top_weather.columns = ['Weather_Condition', 'Accidentes']

    fig5 = px.bar(top_weather, x='Accidentes', y='Weather_Condition',
                orientation='h',
                color='Accidentes', color_continuous_scale='PuBuGn',
                text='Accidentes', template='seaborn',
                title='Top 10 Condiciones Climáticas con Mayor Ocurrencia de Accidentes')
    fig5.update_layout(height=500, width=1000, font=dict(size=13))
    st.plotly_chart(fig5, use_container_width=True)
    
    st.markdown("""
    - Mayor ocurrencia de accidentes en clima “Fair”, seguido de “Mostly Cloudy” y “Partly Cloudy”
    - Sugiere mayor exposición al riesgo en climas estables
    ---
    """) 
